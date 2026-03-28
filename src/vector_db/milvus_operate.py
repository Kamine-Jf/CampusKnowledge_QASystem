"""Milvus 向量集合管理与检索模块。"""
from __future__ import annotations

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 匹配度优化：国内镜像必须最先设置
import warnings
warnings.filterwarnings(
    "ignore",
    message=r"`resume_download` is deprecated",
    category=FutureWarning,
    module=r"huggingface_hub\.file_download",
)
warnings.filterwarnings(
    "ignore",
    message=r"Redirects are currently not supported in Windows or MacOs",
    category=UserWarning,
    module=r"torch\.distributed\.elastic\.multiprocessing\.redirects",
)
warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API",
    category=UserWarning,
    module=r"pymilvus\.client\.__init__",
)
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pymilvus  # 修复：pymilvus2.4.4适配 + 原因说明
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, utility

print(
    f"📌 当前pymilvus版本：{pymilvus.__version__} | 适配Milvus2.6.9服务端"
)  # 修复：pymilvus2.4.4适配 + 原因说明
if pymilvus.__version__ == "2.4.4":
    print("✅ 已适配pymilvus2.4.4 API，开始向量检索...")  # 修复：pymilvus2.4.4适配 + 原因说明【毕设答辩注释】

try:
    from .milvus_config import INDEX_PARAMS, MILVUS_COLLECTION_NAME, VECTOR_DIM
    from .milvus_conn import connect_milvus
except ImportError:
    current_path = Path(__file__).resolve()
    project_root = current_path.parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.vector_db.milvus_config import INDEX_PARAMS, MILVUS_COLLECTION_NAME, VECTOR_DIM  # type: ignore
    from src.vector_db.milvus_conn import connect_milvus  # type: ignore

try:
    from src.config.settings import MilvusConfig, ModelConfig  # 匹配度优化：复用全局 Milvus 配置
except ModuleNotFoundError:  # 匹配度优化：兼容脚本独立运行
    current_path = Path(__file__).resolve()
    project_root = current_path.parents[2]
    if str(project_root) not in sys.path:  # 匹配度优化：检测导入路径是否已包含根目录
        sys.path.insert(0, str(project_root))  # 匹配度优化：补全导入路径
    from src.config.settings import MilvusConfig, ModelConfig  # type: ignore  # 匹配度优化：再次导入全局配置

_MILVUS_ALIAS = "default"
_COLLECTION_CACHE: dict[str, Collection] = {}  # 缓存已加载的集合对象，避免每次检索重连
_NLIST = getattr(MilvusConfig, "MILVUS_NLIST", INDEX_PARAMS["params"].get("nlist", 128))  # 匹配度优化：优先使用全局 nlist
_NPROBE = getattr(MilvusConfig, "MILVUS_NPROBE", 10)  # 匹配度优化：优先使用全局 nprobe
_METRIC_TYPE = getattr(MilvusConfig, "MILVUS_METRIC_TYPE", INDEX_PARAMS["metric_type"])  # 匹配度优化：统一距离度量
_COLLECTION_NAME = getattr(MilvusConfig, "MILVUS_COLLECTION_NAME", MILVUS_COLLECTION_NAME)  # 匹配度优化：统一集合名称
_VECTOR_DIM = getattr(MilvusConfig, "VECTOR_DIM", VECTOR_DIM)  # 匹配度优化：统一向量维度
_INDEX_PARAMS = {
    "metric_type": _METRIC_TYPE,  # 匹配度优化：显式指定度量方式
    "index_type": getattr(MilvusConfig, "IVF_INDEX_TYPE", INDEX_PARAMS["index_type"]),  # 匹配度优化：索引类型
    "params": {"nlist": _NLIST},  # 匹配度优化：提升聚类数量
}  # 匹配度优化：索引参数以精度优先方式重构
_SEARCH_PARAMS = {"metric_type": _METRIC_TYPE, "params": {"nprobe": _NPROBE}}  # 匹配度优化：检索参数优化


def _truncate_text(content: str, max_length: int) -> str:
    """截断过长文本，避免超出 Milvus VARCHAR 限制。"""
    return content if len(content) <= max_length else content[: max_length - 3] + "..."


def _normalize_vector(vector: Sequence[float]) -> List[float]:
    """将输入向量统一转换为浮点列表，并确保维度匹配。"""
    array = np.asarray(vector, dtype=float)
    if array.ndim != 1:
        raise ValueError("向量必须是一维序列。")
    if array.shape[0] != _VECTOR_DIM:
        raise ValueError(f"向量维度应为 {_VECTOR_DIM}，当前为 {array.shape[0]}。")
    return array.astype(float).tolist()


def create_milvus_collection(collection_name: str = _COLLECTION_NAME) -> Collection:
    """创建或加载 Milvus 向量集合，确保索引和加载状态正常。

    返回:
        Collection: 已准备好的集合实例。

    异常:
        ConnectionError: 当 Milvus 未启动或连接异常时抛出。
    """
    connect_milvus()  # 修复：pymilvus2.4.4 + 原因：确保连接初始化完成

    if utility.has_collection(collection_name, using=_MILVUS_ALIAS):
        print(f"✅ 集合{collection_name}已存在，跳过创建。")  # 修复：pymilvus2.4.4 + 原因：避免重复创建
        collection = Collection(name=collection_name, using=_MILVUS_ALIAS)
    else:
        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=True,
            ),  # 修复：pymilvus2.4.4适配 + 原因说明【毕设答辩注释】
            FieldSchema(
                name="vector",
                dtype=DataType.FLOAT_VECTOR,
                dim=_VECTOR_DIM,
            ),  # 修复：pymilvus2.4.4适配 + 原因说明【毕设答辩注释】
            FieldSchema(
                name="text",
                dtype=DataType.VARCHAR,
                max_length=2000,
            ),  # 修复：pymilvus2.4.4适配 + 原因说明【毕设答辩注释】
            FieldSchema(
                name="source",
                dtype=DataType.VARCHAR,
                max_length=200,
            ),  # 修复：pymilvus2.4.4适配 + 原因说明【毕设答辩注释】
        ]
        schema = CollectionSchema(fields=fields, description="校园问答系统向量集合")
        collection = Collection(
            name=collection_name,
            schema=schema,
            using=_MILVUS_ALIAS,
        )
        index_params = {  # 修复：pymilvus2.4.4 + 原因：索引参数严格指定 nlist
            "metric_type": _METRIC_TYPE,
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
        }  # 修复：pymilvus2.4.4适配 + 原因说明【毕设答辩注释】
        collection.create_index(field_name="vector", index_params=index_params)

    if not collection.has_index():
        index_params = {  # 修复：pymilvus2.4.4 + 原因：补齐缺失索引参数
            "metric_type": _METRIC_TYPE,
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
        }  # 修复：pymilvus2.4.4适配 + 原因说明
        collection.create_index(field_name="vector", index_params=index_params)

    collection.load()
    print(f"✅ Milvus 集合已加载：{collection.name}")  # 修复：pymilvus2.4.4适配 + 原因说明
    return collection


def create_collection() -> Collection:
    """兼容旧入口：创建或加载 Milvus 向量集合。"""
    return create_milvus_collection(_COLLECTION_NAME)


def clear_milvus_collection() -> bool:
    """一键清理 Milvus 集合，避免旧数据影响检索效果。"""  # 匹配度优化：新增清理函数
    try:
        connect_milvus()  # 匹配度优化：确保连接可用
    except Exception as connect_error:  # pylint: disable=broad-except
        print(f"❌ Milvus 连接失败，无法清理集合：{connect_error}")  # 匹配度优化：异常兜底
        return False

    if not utility.has_collection(_COLLECTION_NAME, using=_MILVUS_ALIAS):  # 匹配度优化：检查集合是否存在
        print("⚠️ Milvus 集合不存在，无需清理。")  # 匹配度优化：避免误报
        return True

    try:
        utility.drop_collection(_COLLECTION_NAME, using=_MILVUS_ALIAS)
        print(f"✅ 已清理 Milvus 集合：{_COLLECTION_NAME}")  # 匹配度优化：清理成功日志
        return True
    except Exception as drop_error:  # pylint: disable=broad-except
        print(f"❌ Milvus 集合清理失败：{drop_error}")  # 匹配度优化：异常兜底
        return False


def rebuild_milvus_index() -> bool:
    """重建 Milvus 索引以应用最新参数。"""  # 匹配度优化：新增索引重建函数
    try:
        collection = create_collection()  # 匹配度优化：确保集合存在
    except Exception as create_error:  # pylint: disable=broad-except
        print(f"❌ 无法重建索引：{create_error}")  # 匹配度优化：异常兜底
        return False

    try:
        if collection.has_index():  # 匹配度优化：检测旧索引
            collection.drop_index()  # 匹配度优化：先删除旧索引
        collection.create_index(field_name="vector", index_params=_INDEX_PARAMS)  # 匹配度优化：重建新索引
        collection.load()  # 匹配度优化：加载索引后集合
        print("✅ Milvus 索引已重建并加载完成。")  # 匹配度优化：完成提示
        return True
    except Exception as rebuild_error:  # pylint: disable=broad-except
        print(f"❌ 索引重建失败：{rebuild_error}")  # 匹配度优化：异常兜底
        return False


def insert_vectors_to_milvus(
    vectors: List[object],
    texts: List[str] | None = None,
    sources: List[str] | None = None,
    collection_name: str = "campus_qa_vector",
) -> int:
    """批量插入文本向量数据，返回成功写入数量。

    参数:
        vectors: 向量列表或包含 text/vector/source 的字典列表。
        texts: 文本列表（与 vectors 对齐）。
        sources: 来源列表（与 vectors 对齐）。
        collection_name: 目标集合名称。

    返回:
        int: 成功写入的记录数量，连接失败时返回 0。
    """
    if not vectors:  # 修复：pymilvus2.4.4 + 原因：入库前空数据保护
        print("⚠️ 无向量数据可写入，已跳过 Milvus 插入。")
        return 0

    resolved_vectors: List[List[float]] = []
    resolved_texts: List[str] = []
    resolved_sources: List[str] = []

    if texts is None and sources is None and isinstance(vectors, list):  # 修复：pymilvus2.4.4 + 原因：兼容旧调用
        if vectors and isinstance(vectors[0], dict):
            for item in vectors:  # type: ignore[assignment]
                resolved_texts.append(str(item.get("text", "")))
                resolved_sources.append(str(item.get("source", "")))
                resolved_vectors.append(item.get("vector", []))
        else:
            print("❌ 入库失败：未提供 texts/sources，请检查调用方式。")
            return 0
    else:
        resolved_vectors = list(vectors)  # type: ignore[list-item]
        resolved_texts = list(texts or [])
        resolved_sources = list(sources or [])

    if not (len(resolved_vectors) == len(resolved_texts) == len(resolved_sources)):
        print("❌ 入库失败：vectors/texts/sources 长度不一致，请检查数据对齐。")
        return 0

    if not resolved_vectors:
        print("⚠️ 待入库数据为空，已跳过。")
        return 0

    try:
        collection = create_milvus_collection(collection_name)  # 修复：pymilvus2.4.4 + 原因：统一集合初始化入口
    except Exception as create_error:  # pylint: disable=broad-except
        print(f"❌ Milvus 连接或集合创建失败，已跳过写入：{create_error}")
        return 0

    texts_buffer: List[str] = []
    sources_buffer: List[str] = []
    vectors_buffer: List[List[float]] = []
    seen_texts: set[str] = set()

    for index, (raw_vector, raw_text, raw_source) in enumerate(
        zip(resolved_vectors, resolved_texts, resolved_sources), start=1
    ):
        try:
            normalized_vector = _normalize_vector(raw_vector)  # type: ignore[arg-type]
        except Exception as normalize_error:  # pylint: disable=broad-except
            print(f"❌ 入库失败：向量维度不是{_VECTOR_DIM}，请检查向量生成逻辑。错误：{normalize_error}")
            return 0

        cleaned_text = str(raw_text).strip()
        if not cleaned_text:
            print(f"⚠️ 第{index}条文本为空，已跳过。")
            continue
        if cleaned_text in seen_texts:
            print(f"⚠️ 第{index}条文本重复，已跳过去重。")
            continue
        seen_texts.add(cleaned_text)

        texts_buffer.append(_truncate_text(cleaned_text, 2000))
        sources_buffer.append(_truncate_text(str(raw_source).strip() or "未提供来源", 200))
        vectors_buffer.append(np.asarray(normalized_vector, dtype=np.float32).tolist())  # 修复：pymilvus2.4.4 + 原因：确保向量类型匹配

    if not texts_buffer:
        print("⚠️ 所有待写入数据均无效，未向 Milvus 插入记录。")
        return 0

    print(f"📥 开始入库→待入库向量数：{len(texts_buffer)}条")
    try:
        insert_payload = {  # 修复：pymilvus2.4.4 + 原因：2.4.4 插入要求字段名映射
            "vector": vectors_buffer,
            "text": texts_buffer,
            "source": sources_buffer,
        }  # 修复：pymilvus2.4.4适配 + 原因说明【毕设答辩注释】
        insert_result = collection.insert(insert_payload)  # 修复：pymilvus2.4.4 + 原因：按字段字典入库
    except Exception as insert_error:  # pylint: disable=broad-except
        if "DataNotMatchException" in str(insert_error):
            try:
                insert_payload = [vectors_buffer, texts_buffer, sources_buffer]  # 修复：pymilvus2.4.4 + 原因：字段顺序入库回退
                insert_result = collection.insert(insert_payload)
            except Exception as retry_error:  # pylint: disable=broad-except
                print(f"❌ Milvus 插入失败：{retry_error}")
                return 0
        else:
            print(f"❌ Milvus 插入失败：{insert_error}")
            return 0

    try:
        collection.flush()  # 修复：pymilvus2.4.4 + 原因：立即刷盘确保可检索
        collection.load()
    except Exception as flush_error:  # pylint: disable=broad-except
        print(f"❌ Milvus 刷盘失败：{flush_error}")
        return 0

    primary_keys = getattr(insert_result, "primary_keys", [])  # 修复：pymilvus2.4.4 + 原因：返回主键适配
    inserted_count = len(primary_keys) if primary_keys else len(texts_buffer)
    print(f"✅ 入库成功→实际入库ID数：{inserted_count}条")
    print(f"📊 入库后集合总数据量：{collection.num_entities}条")
    return inserted_count


def check_milvus_data(collection_name: str = _COLLECTION_NAME) -> int:
    """检查 Milvus 集合数据量并输出字段信息。"""
    try:
        collection = create_milvus_collection(collection_name)  # 修复：pymilvus2.4.4 + 原因：复用集合初始化
    except Exception as create_error:  # pylint: disable=broad-except
        print(f"❌ Milvus 连接失败，无法检查集合数据：{create_error}")
        return 0

    fields = collection.schema.fields
    print("📊 集合字段列表：")
    for field in fields:
        print(f" - {field.name} | {field.dtype}")  # 修复：pymilvus2.4.4 + 原因：字段类型核验【毕设答辩注释】

    entity_count = getattr(collection, "num_entities", 0)
    print(f"📊 当前集合向量数量：{entity_count}")
    if entity_count == 0:  # 修复：pymilvus2.4.4 + 原因：空集合提示
        print("⚠️ 集合为空→请运行一键入库脚本：python src/vector_db/refresh_milvus_data.py")
    return entity_count


def _get_cached_collection(collection_name: str = _COLLECTION_NAME) -> Collection:
    """获取缓存的集合对象，首次访问时初始化并缓存。"""
    if collection_name not in _COLLECTION_CACHE:
        _COLLECTION_CACHE[collection_name] = create_milvus_collection(collection_name)
    return _COLLECTION_CACHE[collection_name]


def search_similar_vector(query_vector: Sequence[float], top_k: int = 3) -> List[Dict[str, object]]:
    """基于向量检索相似文本块，返回按距离升序排序的结果。"""
    if top_k <= 0:
        raise ValueError("top_k 必须为正整数。")

    normalized_query = _normalize_vector(query_vector)
    try:
        collection = _get_cached_collection(_COLLECTION_NAME)
    except Exception as create_error:  # pylint: disable=broad-except
        print(f"❌ 无法连接 Milvus，返回空的非结构化检索结果：{create_error}")
        return []

    entity_count = getattr(collection, "num_entities", 0)
    if entity_count == 0:  # 修复：pymilvus2.4.4 + 原因：空集合直接返回
        print("⚠️ 集合为空→请运行一键入库脚本：python src/vector_db/refresh_milvus_data.py")
        return []

    adjusted_nprobe = _SEARCH_PARAMS["params"].get("nprobe", _NPROBE)  # 匹配度优化：读取检索参数
    if adjusted_nprobe > _NLIST:  # 匹配度优化：防止 nprobe 超限
        adjusted_nprobe = _NLIST  # 匹配度优化：自动修正
        print(f"⚠️ nprobe 超过 nlist，已自动调整为 {adjusted_nprobe}")  # 匹配度优化：避免 Milvus 报错

    try:
        search_results = collection.search(
            data=[normalized_query],
            anns_field="vector",
            param={"metric_type": _METRIC_TYPE, "params": {"nprobe": adjusted_nprobe}},  # 修复：pymilvus2.4.4适配 + 原因说明【毕设答辩注释】
            limit=top_k,
            output_fields=["text", "source"],
        )
    except Exception as search_error:  # pylint: disable=broad-except
        print(f"❌ Milvus 检索失败，已降级返回空列表：{search_error}")
        return []

    hits = search_results[0] if search_results else []
    print(f"📌 检索命中数：{len(hits)}")  # 修复：pymilvus2.4.4适配 + 原因说明
    formatted: List[Dict[str, object]] = []
    for hit in hits:
        raw_score = getattr(hit, "distance", None)  # 修复：pymilvus2.4.4适配 + 原因说明【毕设答辩注释】
        if raw_score is None:  # 修复：pymilvus2.4.4适配 + 原因说明【毕设答辩注释】
            raw_score = getattr(hit, "score", 0.0)  # 修复：pymilvus2.4.4适配 + 原因说明【毕设答辩注释】
        text_value = ""
        source_value = ""
        entity = getattr(hit, "entity", None)
        if isinstance(entity, dict):
            text_value = entity.get("text") or ""  # 修复：pymilvus2.4.4适配 + 原因说明【毕设答辩注释】
            source_value = entity.get("source") or ""  # 修复：pymilvus2.4.4适配 + 原因说明【毕设答辩注释】
        elif hasattr(hit, "get"):
            try:
                text_value = hit.get("text") or ""  # 修复：pymilvus2.4.4适配 + 原因说明【毕设答辩注释】
                source_value = hit.get("source") or ""  # 修复：pymilvus2.4.4适配 + 原因说明【毕设答辩注释】
            except Exception:
                text_value = ""  # 修复：pymilvus2.4.4适配 + 原因说明
                source_value = ""  # 修复：pymilvus2.4.4适配 + 原因说明
        else:
            fields = getattr(hit, "fields", {})
            if isinstance(fields, dict):
                text_value = fields.get("text") or ""  # 修复：pymilvus2.4.4适配 + 原因说明【毕设答辩注释】
                source_value = fields.get("source") or ""  # 修复：pymilvus2.4.4适配 + 原因说明【毕设答辩注释】

        if text_value or source_value:
            formatted.append(
                {
                    "text": text_value,
                    "source": source_value,
                    "score": round(float(raw_score), 4),
                }
            )  # 修复：pymilvus2.4.4适配 + 原因说明【毕设答辩注释】

    formatted.sort(key=lambda item: item["score"])  # 修复：pymilvus2.4.4适配 + 原因说明
    return formatted  # 修复：pymilvus2.4.4适配 + 原因说明


def fetch_chunks_by_source_prefix(
    source_prefix: str, max_results: int = 50
) -> List[Dict[str, object]]:
    """按来源 PDF 文件名前缀查询该文件的所有文本块。

    用于人物档案等单文档查询场景：当向量检索仅命中某 PDF 的部分分块时，
    通过标量过滤补全同一文件的全部分块，避免因 top_k 限制遗漏晚近的工作经历等内容。

    参数:
        source_prefix: PDF 文件名前缀（如 "甘琤.pdf"），不含页码/子块后缀。
        max_results: 最多返回记录数，默认 50。

    返回:
        List[Dict]: 包含 text/source/score 键的记录列表，按来源排序，score 固定为 0.0。
    """
    if not source_prefix or not source_prefix.strip():
        return []

    try:
        collection = _get_cached_collection(_COLLECTION_NAME)
    except Exception as conn_error:  # pylint: disable=broad-except
        print(f"❌ Milvus 连接失败，无法按来源检索：{conn_error}")
        return []

    escaped_prefix = source_prefix.replace('"', '\\"')
    expr = f'source like "{escaped_prefix}%"'

    try:
        rows = collection.query(
            expr=expr,
            output_fields=["text", "source"],
            limit=max_results,
        )
    except Exception as query_error:  # pylint: disable=broad-except
        print(f"❌ Milvus 来源过滤查询失败（前缀: {source_prefix}）：{query_error}")
        return []

    results: List[Dict[str, object]] = []
    for row in rows or []:
        text_val = (row.get("text") or "").strip()
        src_val = row.get("source") or ""
        if text_val:
            results.append({"text": text_val, "source": src_val, "score": 0.0})

    results.sort(key=lambda r: str(r.get("source", "")))
    return results


if __name__ == "__main__":
    try:
        collection = create_milvus_collection(_COLLECTION_NAME)
        print(f"✅ 集合已就绪：{collection.name}")

        entity_count = check_milvus_data(_COLLECTION_NAME)
        if entity_count == 0:
            print("⚠️ 集合为空，未执行检索。请先运行一键入库脚本。")
        else:
            from sentence_transformers import SentenceTransformer  # 匹配度优化：测试时生成语义向量

            os.environ.setdefault("HF_ENDPOINT", ModelConfig.HF_ENDPOINT)
            model = SentenceTransformer(
                ModelConfig.MODEL_NAME,
                cache_folder=str(ModelConfig.MODEL_CACHE_PATH),
                device=ModelConfig.MODEL_DEVICE,
            )

            def _encode_test_text(text: str) -> List[float]:
                embedding = model.encode(
                    sentences=[text],
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    device=ModelConfig.MODEL_DEVICE,
                    show_progress_bar=False,
                )[0]
                if embedding.shape[0] < _VECTOR_DIM:
                    embedding = np.pad(embedding, (0, _VECTOR_DIM - embedding.shape[0]), mode="constant")
                elif embedding.shape[0] > _VECTOR_DIM:
                    embedding = embedding[:_VECTOR_DIM]
                return embedding.astype(float).tolist()

            query_text = "缓考"
            query_vector = _encode_test_text(query_text)
            search_results = search_similar_vector(query_vector, top_k=3)
            if not search_results:
                print("⚠️ 检索无结果，请确认集合中已有向量数据。")
            else:
                print(f"✅ 测试关键词：{query_text}")
                print(f"✅ 命中结果数：{len(search_results)}")
                for rank, item in enumerate(search_results, start=1):
                    label = "高匹配" if item["score"] < 0.5 else "低匹配"
                    print(
                        f"Top{rank} | Score:{item['score']:.4f} | {label} | Source:{item['source']}\n"
                        f"文本摘要：{item['text'][:60]}..."
                    )
    except Exception as exc:  # pylint: disable=broad-except
        print(f"❌ 测试流程异常：{exc}")
