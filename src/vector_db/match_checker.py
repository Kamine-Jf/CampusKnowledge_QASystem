"""匹配度校验工具：一键验证 PDF → 向量 → Milvus 检索效果。"""  # 匹配度优化：新增工具脚本说明

from __future__ import annotations  # 匹配度优化：保证类型注解兼容性

import os  # 匹配度优化：配置镜像环境变量
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 匹配度优化：国内镜像必须最先设置
import sys  # 匹配度优化：用于补全导入路径
from pathlib import Path  # 匹配度优化：跨平台路径处理
from typing import Dict, List  # 匹配度优化：类型提示

try:
    from src.config.settings import ModelConfig, MilvusConfig  # 匹配度优化：复用全局配置
    from src.vector_db.milvus_operate import (  # 匹配度优化：复用 Milvus 管理函数
        clear_milvus_collection,  # 匹配度优化：清理集合函数
        insert_vectors_to_milvus,  # 匹配度优化：入库函数
        search_similar_vector,  # 匹配度优化：检索函数
    )
    from src.vector_db.pdf2vector import pdf_to_vectors  # 匹配度优化：复用 PDF 解析入口
except ModuleNotFoundError:  # 匹配度优化：兼容脚本独立运行
    current_path = Path(__file__).resolve()  # 匹配度优化：定位脚本路径
    project_root = current_path.parents[2]  # 匹配度优化：回溯项目根目录
    if str(project_root) not in sys.path:  # 匹配度优化：检测导入路径
        sys.path.insert(0, str(project_root))  # 匹配度优化：补全导入路径
    from src.config.settings import ModelConfig, MilvusConfig  # type: ignore  # 匹配度优化：再次导入配置
    from src.vector_db.milvus_operate import (  # type: ignore  # 匹配度优化：再次导入 Milvus 管理函数
        clear_milvus_collection,  # 匹配度优化：清理集合函数
        insert_vectors_to_milvus,  # 匹配度优化：入库函数
        search_similar_vector,  # 匹配度优化：检索函数
    )
    from src.vector_db.pdf2vector import pdf_to_vectors  # type: ignore  # 匹配度优化：再次导入解析入口

from sentence_transformers import SentenceTransformer  # 匹配度优化：生成关键词查询向量
import numpy as np  # 匹配度优化：向量维度补齐


def _encode_query(keyword: str) -> List[float]:
    """将关键词编码为向量，用于检索测试。"""  # 匹配度优化：封装关键词向量生成
    os.environ.setdefault("HF_ENDPOINT", ModelConfig.HF_ENDPOINT)  # 匹配度优化：配置 Hugging Face 国内镜像
    model = SentenceTransformer(
        ModelConfig.MODEL_NAME,
        cache_folder=str(ModelConfig.MODEL_CACHE_PATH),
        device=ModelConfig.MODEL_DEVICE,
    )  # 匹配度优化：复用全局模型配置
    embedding = model.encode(
        sentences=[keyword],
        convert_to_numpy=True,
        normalize_embeddings=True,
        device=ModelConfig.MODEL_DEVICE,
        show_progress_bar=False,
    )[0]  # 匹配度优化：生成关键词向量
    if embedding.shape[0] < MilvusConfig.VECTOR_DIM:  # 匹配度优化：维度不足补齐
        embedding = np.pad(embedding, (0, MilvusConfig.VECTOR_DIM - embedding.shape[0]), mode="constant")
    elif embedding.shape[0] > MilvusConfig.VECTOR_DIM:  # 匹配度优化：维度过长截断
        embedding = embedding[:MilvusConfig.VECTOR_DIM]
    return embedding.astype(float).tolist()  # 匹配度优化：返回标准 float 列表


def check_match_quality(keyword: str = "缓考", top_k: int = 3) -> Dict[str, object]:
    """执行匹配度校验流程并返回统计结果。"""  # 匹配度优化：核心校验函数
    vectors_data = pdf_to_vectors()  # 匹配度优化：PDF 解析并生成向量数据
    if not vectors_data:
        print("⚠️ 未生成任何向量数据，无法继续校验。")  # 匹配度优化：异常提示
        return {
            "total_vectors": 0,
            "hit_count": 0,
            "high_match_count": 0,
            "high_match_rate": 0.0,
        }  # 匹配度优化：返回空统计

    clear_milvus_collection()  # 匹配度优化：清理旧集合，避免污染
    inserted_count = insert_vectors_to_milvus(vectors_data)  # 匹配度优化：写入 Milvus
    if inserted_count == 0:
        print("⚠️ Milvus 入库失败，无法继续校验。")  # 匹配度优化：异常提示
        return {
            "total_vectors": len(vectors_data),
            "hit_count": 0,
            "high_match_count": 0,
            "high_match_rate": 0.0,
        }  # 匹配度优化：返回入库失败统计

    query_vector = _encode_query(keyword)  # 匹配度优化：生成关键词查询向量
    results = search_similar_vector(query_vector, top_k=top_k)  # 匹配度优化：执行检索
    hit_count = len(results)  # 匹配度优化：命中数量
    high_match_count = sum(1 for item in results if item.get("score", 1.0) < 0.5)  # 匹配度优化：高匹配统计
    high_match_rate = round((high_match_count / hit_count) * 100, 2) if hit_count else 0.0  # 匹配度优化：高匹配率

    print("✅ 匹配度校验结果")  # 匹配度优化：输出统计标题
    print(f"总生成向量数：{len(vectors_data)}")  # 匹配度优化：输出向量数
    print(f"检索命中数：{hit_count}")  # 匹配度优化：输出命中数
    print(f"高匹配数(score<0.5)：{high_match_count}")  # 匹配度优化：输出高匹配数
    print(f"高匹配率：{high_match_rate}%")  # 匹配度优化：输出高匹配率

    for index, item in enumerate(results, start=1):  # 匹配度优化：遍历结果
        print(
            f"Top{index} | score={item.get('score')} | source={item.get('source')}\n"
            f"text={item.get('text', '')[:120]}..."
        )  # 匹配度优化：输出每条结果

    if high_match_rate < 100:  # 匹配度优化：未达标提示
        print("⚠️ 高匹配率未达 100%，建议检查以下方向：")  # 匹配度优化：输出建议
        print("- 检查 PDF 是否包含关键词所在页，确认文本解析是否完整")  # 匹配度优化：建议 1
        print("- 调整分块长度范围（300-500）以提升语义完整性")  # 匹配度优化：建议 2
        print("- 观察关键词命中的文本块是否被过滤或过短")  # 匹配度优化：建议 3

    clear_milvus_collection()  # 匹配度优化：校验完成后清理集合
    return {
        "total_vectors": len(vectors_data),
        "hit_count": hit_count,
        "high_match_count": high_match_count,
        "high_match_rate": high_match_rate,
    }  # 匹配度优化：返回统计结果


if __name__ == "__main__":
    check_match_quality()  # 匹配度优化：脚本直跑入口
