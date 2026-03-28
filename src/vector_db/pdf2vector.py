# -*- coding: utf-8 -*-
"""PDF 文本解析与向量生成模块。

该模块负责遍历 data/unstructured_data 目录下的 PDF 文档，
实现“按页拆分 + 文本分块 + 向量生成”的完整流程，
确保输出 768 维向量，并返回可直接入库的结构化数据。
"""

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
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

try:
    from src.config.settings import ModelConfig, PdfConfig, MilvusConfig  # 匹配度优化：统一从全局配置读取模型与PDF路径
except ModuleNotFoundError:  # 匹配度优化：兼容脚本独立运行
    current_path = Path(__file__).resolve()  # 匹配度优化：定位项目根目录
    project_root = current_path.parents[2]  # 匹配度优化：回溯到项目根
    if str(project_root) not in os.sys.path:  # 匹配度优化：检测导入路径是否已包含根目录
        os.sys.path.insert(0, str(project_root))  # 匹配度优化：补全导入路径
    from src.config.settings import ModelConfig, PdfConfig, MilvusConfig  # type: ignore  # 匹配度优化：再次导入全局配置

# ============================= 全局配置 =============================
# 关键配置：保持与 download_model.py 一致的模型名称与缓存目录，避免重复下载。
MODEL_NAME = ModelConfig.MODEL_NAME  # 匹配度优化：复用全局模型名称配置
# 关键配置：统一模型缓存路径，避免硬编码目录。
CACHE_DIR = Path(ModelConfig.MODEL_CACHE_PATH)  # 匹配度优化：复用全局模型缓存路径
# 关键配置：统一 PDF 数据目录，便于跨环境迁移。
PDF_DIR = Path(PdfConfig.PDF_DIR)  # 匹配度优化：复用全局 PDF 路径配置
MIN_CHARS_PER_CHUNK = PdfConfig.MIN_CHARS_PER_CHUNK  # 匹配度优化：分块最小字符数
MAX_CHARS_PER_CHUNK = PdfConfig.MAX_CHARS_PER_CHUNK  # 匹配度优化：分块最大字符数
MIN_VALID_CHARS = PdfConfig.MIN_VALID_CHARS  # 匹配度优化：过滤无意义短块阈值
EXPECTED_DIM = MilvusConfig.VECTOR_DIM  # 匹配度优化：向量维度与 Milvus 端保持一致
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[。？！!?；;.!])\s*")  # 匹配度优化：覆盖中英文句末符号

_LAST_STATS: Dict[str, int] = {  # 匹配度优化：记录统计信息用于测试验证
    "total_chunks": 0,  # 匹配度优化：总提取块数
    "filtered_chunks": 0,  # 匹配度优化：被过滤块数
    "valid_chunks": 0,  # 匹配度优化：有效块数
    "vector_count": 0,  # 匹配度优化：成功向量数
}


# ============================= 文本处理工具函数 =============================

def preprocess_text(text: str) -> str:
    """PDF 文本预处理：降噪、统一标点、清理页眉页脚。"""  # 匹配度优化：新增预处理入口提升语义质量
    if not text:  # 匹配度优化：空文本直接短路
        return ""  # 匹配度优化：空文本直接返回

    normalized = text.replace("\u3000", " ")  # 匹配度优化：统一全角空格
    normalized = normalized.replace("\r", "\n")  # 匹配度优化：统一换行符
    normalized = re.sub(r"[\t\f]+", " ", normalized)  # 匹配度优化：移除制表符/分页符
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)  # 匹配度优化：压缩多余空行

    # 匹配度优化：移除常见页眉页脚/页码噪声
    header_footer_patterns = [  # 匹配度优化：常见页眉页脚规则
        r"^\s*郑州轻工业大学学生手册\s*$",
        r"^\s*学生手册\s*$",
        r"^\s*第\s*\d+\s*页(\s*/\s*共\s*\d+\s*页)?\s*$",
    ]
    for pattern in header_footer_patterns:  # 匹配度优化：逐条清理噪声行
        normalized = re.sub(pattern, "", normalized, flags=re.MULTILINE)  # 匹配度优化：去除页眉页脚

    # 匹配度优化：过滤 PDF 解析产生的乱码字符
    normalized = re.sub(r"[�□]", "", normalized)  # 匹配度优化：清理乱码字符

    # 匹配度优化：全角转半角（ASCII 范围）
    normalized = normalized.translate({code: code - 0xFEE0 for code in range(0xFF01, 0xFF5F)})  # 匹配度优化：全角转半角

    # 匹配度优化：去除重复标点
    normalized = re.sub(r"([。！？；;,.!?])\1+", r"\1", normalized)  # 匹配度优化：去除重复标点

    normalized = re.sub(r"[ ]{2,}", " ", normalized)  # 匹配度优化：压缩多余空格
    normalized = re.sub(r"[ ]*\n[ ]*", "\n", normalized)  # 匹配度优化：清理换行两侧空格
    return normalized.strip()  # 匹配度优化：输出前再去除首尾空白


def _split_long_sentence(sentence: str, max_chars: int) -> List[str]:
    """超长句子分段处理：先按逗号拆分，必要时再按长度兜底。"""  # 匹配度优化：避免纯字数硬切
    segments: List[str] = []  # 匹配度优化：收集拆分片段
    comma_parts = [seg for seg in re.split(r"(?<=[，,、])\s*", sentence) if seg]  # 匹配度优化：按逗号切分
    for part in comma_parts:  # 匹配度优化：遍历拆分子句
        if len(part) <= max_chars:  # 匹配度优化：长度合规直接保留
            segments.append(part)  # 匹配度优化：添加子句片段
            continue
        for start in range(0, len(part), max_chars):  # 匹配度优化：超长子句继续分段
            segments.append(part[start : start + max_chars])  # 匹配度优化：超长兜底切分
    return segments


def _is_meaningful_text(text: str) -> bool:
    """判定文本块是否具有语义信息。"""  # 匹配度优化：过滤无意义文本块
    if not text or len(text) < MIN_VALID_CHARS:  # 匹配度优化：长度门槛过滤
        return False  # 匹配度优化：过滤过短文本
    if re.fullmatch(r"[\d\W_]+", text):  # 匹配度优化：纯数字或符号
        return False  # 匹配度优化：过滤无语义文本
    if not re.search(r"[A-Za-z\u4e00-\u9fff]", text):  # 匹配度优化：缺少文字信息
        return False  # 匹配度优化：无有效语言字符
    return True  # 匹配度优化：通过语义有效性校验


def _split_paragraphs(text: str) -> List[str]:
    """按段落拆分文本，优先保留自然语义段落。"""  # 匹配度优化：段落优先分块
    if "\n\n" in text:  # 匹配度优化：优先按空行划分段落
        paragraphs = [seg.strip() for seg in re.split(r"\n{2,}", text) if seg.strip()]  # 匹配度优化：段落拆分
    else:
        paragraphs = [seg.strip() for seg in text.split("\n") if seg.strip()]  # 匹配度优化：无空行时按行拆分
    return paragraphs  # 匹配度优化：返回段落列表


def _merge_sentences(sentences: List[str], min_chars: int, max_chars: int) -> List[str]:
    """在句子级别合并为 300-500 字文本块。"""  # 匹配度优化：控制分块长度
    chunks: List[str] = []  # 匹配度优化：文本块容器
    current = ""  # 匹配度优化：当前拼接中的文本块
    for sentence in sentences:  # 匹配度优化：逐句拼接
        if not sentence:
            continue  # 匹配度优化：跳过空句子
        if len(sentence) > max_chars:
            if current:
                chunks.append(current)  # 匹配度优化：提交当前块
                current = ""  # 匹配度优化：重置缓存
            chunks.extend(_split_long_sentence(sentence, max_chars))  # 匹配度优化：拆分超长句
            continue
        if len(current) + len(sentence) <= max_chars:
            current = f"{current}{sentence}" if current else sentence  # 匹配度优化：拼接句子
        else:
            if current:
                chunks.append(current)  # 匹配度优化：提交当前块
            current = sentence  # 匹配度优化：开启新块
    if current:
        chunks.append(current)  # 匹配度优化：提交尾部块

    # 匹配度优化：合并最后的短块，尽量落入目标区间
    if len(chunks) >= 2 and len(chunks[-1]) < min_chars:  # 匹配度优化：短尾块拼接
        if len(chunks[-2]) + len(chunks[-1]) <= max_chars:  # 匹配度优化：不超过上限时合并
            chunks[-2] = f"{chunks[-2]}{chunks[-1]}"  # 匹配度优化：合并短尾块
            chunks.pop()  # 匹配度优化：移除已合并块
    return chunks  # 匹配度优化：返回最终分块列表


def _chunk_page_text(page_text: str, min_chars: int, max_chars: int) -> Tuple[List[str], int, int]:
    """段落优先→句子拆分→合并为目标长度文本块。"""  # 匹配度优化：语义优先分块逻辑
    cleaned = preprocess_text(page_text)  # 匹配度优化：先降噪再分块
    if not cleaned:
        return [], 0, 0  # 匹配度优化：空文本直接返回

    paragraphs = _split_paragraphs(cleaned)  # 匹配度优化：获取段落列表
    candidate_chunks: List[str] = []  # 匹配度优化：候选块容器
    for paragraph in paragraphs:  # 匹配度优化：逐段处理
        sentences = [seg.strip() for seg in SENTENCE_SPLIT_PATTERN.split(paragraph) if seg.strip()]  # 匹配度优化：句子拆分
        if not sentences:
            continue  # 匹配度优化：空段落跳过
        candidate_chunks.extend(_merge_sentences(sentences, min_chars, max_chars))  # 匹配度优化：合并成块

    filtered_chunks = [chunk for chunk in candidate_chunks if _is_meaningful_text(chunk)]  # 匹配度优化：过滤无意义块
    filtered_count = len(candidate_chunks) - len(filtered_chunks)  # 匹配度优化：统计过滤数量
    return filtered_chunks, len(candidate_chunks), filtered_count  # 匹配度优化：返回块与统计


def _extract_pdf_chunks(pdf_path: Path) -> Tuple[List[Tuple[str, str]], int, int]:
    """解析单个 PDF，返回 (文本块, 来源标识) 列表及统计数据。"""  # 匹配度优化：返回统计数据用于分析
    try:
        reader = PdfReader(str(pdf_path))
    except FileNotFoundError as not_found_error:
        print(f"❌ 未找到 PDF 文件，已跳过：{pdf_path}")  # 匹配度优化：文件缺失时降级处理
        return [], 0, 0  # 匹配度优化：避免中断整体解析
    except PermissionError as perm_error:
        print(f"❌ 无法读取 PDF 文件（权限不足），已跳过：{pdf_path}")  # 匹配度优化：权限异常兜底
        return [], 0, 0  # 匹配度优化：避免中断整体解析
    except Exception as exc:  # pylint: disable=broad-except
        print(f"❌ PDF 打开失败，已跳过：{pdf_path}，错误详情：{exc}")
        return [], 0, 0  # 匹配度优化：异常时返回空结果

    if getattr(reader, "is_encrypted", False):
        print(f"⚠️ 检测到加密 PDF，无法解析，已跳过：{pdf_path.name}")
        return [], 0, 0  # 匹配度优化：加密PDF直接返回空

    chunks: List[Tuple[str, str]] = []
    total_chunks = 0  # 匹配度优化：统计原始块数量
    filtered_chunks = 0  # 匹配度优化：统计过滤块数量
    for page_index, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
        except Exception as exc:  # pylint: disable=broad-except
            print(f"⚠️ 页码 {page_index} 解析失败，已跳过：{pdf_path.name}，错误：{exc}")
            continue

        page_chunks, page_total, page_filtered = _chunk_page_text(
            page_text,
            MIN_CHARS_PER_CHUNK,
            MAX_CHARS_PER_CHUNK,
        )  # 匹配度优化：按段落+句子语义分块
        total_chunks += page_total  # 匹配度优化：累计总块数
        filtered_chunks += page_filtered  # 匹配度优化：累计过滤块数
        if not page_chunks:
            continue

        source_tag = f"{pdf_path.name}_第{page_index}页"  # 匹配度优化：保留页码溯源
        chunks.extend(
            (chunk, f"{source_tag}_子块{chunk_index}")
            for chunk_index, chunk in enumerate(page_chunks, start=1)
        )  # 匹配度优化：标注子块编号便于定位

    if not chunks:
        print(f"⚠️ PDF 无有效文本内容，已跳过：{pdf_path.name}")

    return chunks, total_chunks, filtered_chunks  # 匹配度优化：返回块及统计


# ============================= 向量生成核心函数 =============================

def _ensure_cache_dir(cache_dir: Path) -> None:
    """确保缓存目录存在，并设置环境变量指向自定义路径。"""
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError as perm_error:
        print(f"❌ 创建模型缓存目录失败，权限不足：{cache_dir}")
        raise perm_error
    except OSError as os_error:
        print(f"❌ 创建模型缓存目录失败，系统错误：{cache_dir}")
        raise os_error

    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(cache_dir)
    os.environ.setdefault("HF_ENDPOINT", ModelConfig.HF_ENDPOINT)  # 匹配度优化：配置 Hugging Face 国内镜像
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # 默认使用 CPU，避免显存不足


def _locate_cached_model(cache_dir: Path) -> Optional[Path]:
    """查找已缓存的模型目录，用于网络异常时的离线加载。"""
    candidates: List[Path] = []

    repo_dir = cache_dir / "sentence-transformers" / "all-MiniLM-L6-v2"
    if repo_dir.exists():
        candidates.append(repo_dir)

    hub_dir = cache_dir / "models--sentence-transformers--all-MiniLM-L6-v2"
    snapshots_dir = hub_dir / "snapshots"
    if snapshots_dir.exists():
        snapshot_folders = [p for p in snapshots_dir.iterdir() if p.is_dir()]
        if snapshot_folders:
            latest_snapshot = max(snapshot_folders, key=lambda item: item.stat().st_mtime)
            candidates.append(latest_snapshot)

    for config_path in cache_dir.glob("**/config.json"):
        parent = config_path.parent
        if "all-MiniLM-L6-v2" in parent.as_posix() and parent not in candidates:
            candidates.append(parent)

    for candidate in candidates:
        if (candidate / "config.json").exists():
            return candidate
    return None


def _load_embedding_model() -> SentenceTransformer:
    """加载 SentenceTransformer 模型，强制使用 CPU。"""  # 匹配度优化：统一模型加载入口
    _ensure_cache_dir(CACHE_DIR)
    local_model_path = _locate_cached_model(CACHE_DIR)
    if local_model_path is not None:
        try:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            model = SentenceTransformer(str(local_model_path), device=ModelConfig.MODEL_DEVICE)  # 匹配度优化：强制CPU
            return model
        except Exception as cache_error:  # pylint: disable=broad-except
            print(f"⚠️ 本地缓存模型加载失败，将尝试在线下载。错误详情：{cache_error}")
        finally:
            os.environ.pop("HF_HUB_OFFLINE", None)

    try:
        model = SentenceTransformer(
            MODEL_NAME,
            cache_folder=str(CACHE_DIR),
            device=ModelConfig.MODEL_DEVICE,
        )  # 匹配度优化：复用全局配置加载模型
    except requests.exceptions.RequestException as request_error:
        if local_model_path is not None:
            print("⚠️ 网络连接异常，请确认网络或保证本地缓存完整后重试。")
        print("❌ 模型下载失败，请检查网络连接或先运行 download_model.py 预下载模型。")
        raise request_error
    except RuntimeError as runtime_error:
        print("❌ 模型加载失败，可能与依赖或缓存文件损坏有关。")
        raise runtime_error
    except Exception as unknown_error:  # pylint: disable=broad-except
        print("❌ 未知错误导致模型加载失败。")
        raise unknown_error
    return model


def _ensure_expected_dim(vectors: np.ndarray) -> np.ndarray:
    """将模型输出向量调整为指定的 768 维。"""  # 匹配度优化：统一向量维度
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    current_dim = vectors.shape[1]
    if current_dim == EXPECTED_DIM:
        return vectors
    if current_dim < EXPECTED_DIM:
        padding = EXPECTED_DIM - current_dim
        return np.pad(vectors, ((0, 0), (0, padding)), mode="constant")
    return vectors[:, :EXPECTED_DIM]


def pdf_to_vectors(base_dir: Path = PDF_DIR) -> List[Dict[str, object]]:
    """核心入口：解析 PDF 并返回包含文本、向量、来源的字典列表。"""  # 匹配度优化：保持接口不变
    if not base_dir.exists():
        print(f"❌ PDF 目录不存在，请确认路径：{base_dir}")
        return []

    pdf_files = sorted(base_dir.rglob("*.pdf"))
    if not pdf_files:
        print(f"⚠️ 目录中未找到 PDF 文件：{base_dir}")
        return []

    model = _load_embedding_model()  # 匹配度优化：复用模型缓存
    text_blocks: List[str] = []
    sources: List[str] = []
    stats = {"total_chunks": 0, "filtered_chunks": 0, "valid_chunks": 0, "vector_count": 0}  # 匹配度优化：统计数据
    for pdf_path in pdf_files:
        extracted_pairs, total_chunks, filtered_chunks = _extract_pdf_chunks(pdf_path)
        stats["total_chunks"] += total_chunks  # 匹配度优化：累计总块数
        stats["filtered_chunks"] += filtered_chunks  # 匹配度优化：累计过滤块数
        if not extracted_pairs:
            continue
        for text, source in extracted_pairs:
            if not text.strip():
                continue  # 匹配度优化：空块直接跳过
            text_blocks.append(text)
            sources.append(source)
        stats["valid_chunks"] += len(extracted_pairs)  # 匹配度优化：累计有效块数

    if not text_blocks:
        print("⚠️ 未解析到任何有效文本块，流程结束。")
        return []

    try:
        embeddings = model.encode(
            sentences=text_blocks,
            batch_size=16,
            device=ModelConfig.MODEL_DEVICE,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    except Exception as encode_error:  # pylint: disable=broad-except
        print(f"❌ 向量生成失败，错误详情：{encode_error}")
        return []

    processed_embeddings = _ensure_expected_dim(embeddings)  # 匹配度优化：对齐向量维度

    vectors_data: List[Dict[str, object]] = []
    for text, source, vector in zip(text_blocks, sources, processed_embeddings):
        if not text.strip():
            continue  # 匹配度优化：防止空文本入库
        if vector.shape[0] != EXPECTED_DIM:
            print(f"❌ 向量维度异常，已跳过：{vector.shape[0]}维")  # 匹配度优化：维度不符直接剔除
            continue
        vectors_data.append(
            {
                "text": text,
                "vector": vector.astype(float).tolist(),
                "source": source,
            }
        )
        stats["vector_count"] += 1  # 匹配度优化：统计成功向量数

    _LAST_STATS.update(stats)  # 匹配度优化：保存统计信息供测试输出

    return vectors_data


def gen_pdf_vectors() -> Tuple[List[List[float]], List[str], List[str]]:
    """兼容入口：返回 vectors/texts/sources 三列表。"""  # 修复：pymilvus2.4.4 + 原因：入库函数适配
    vectors_data = pdf_to_vectors()
    vectors = [item["vector"] for item in vectors_data]
    texts = [item["text"] for item in vectors_data]
    sources = [item["source"] for item in vectors_data]
    return vectors, texts, sources


def parse_pdf_to_text(base_dir: Path = PDF_DIR) -> List[str]:
    """兼容入口：仅返回解析后的文本块列表。"""  # 修复：pymilvus2.4.4 + 原因：仅校验文本输出
    vectors_data = pdf_to_vectors(base_dir)
    return [item["text"] for item in vectors_data]


def generate_query_vector(query_text: str) -> List[float]:
    """生成查询向量（768维），用于检索测试。"""  # 修复：pymilvus2.4.4 + 原因：统一查询向量维度
    model = _load_embedding_model()
    embedding = model.encode(
        sentences=[query_text],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
        device=ModelConfig.MODEL_DEVICE,
    )[0]
    embedding = _ensure_expected_dim(np.asarray(embedding))  # 修复：pymilvus2.4.4 + 原因：维度对齐
    return embedding[0].astype(float).tolist()


# ============================= 测试入口 =============================
if __name__ == "__main__":
    try:
        vectors, texts, sources = gen_pdf_vectors()  # 修复：pymilvus2.4.4 + 原因：统一数据输出
    except Exception as exc:  # pylint: disable=broad-except
        print(f"❌ 解析流程异常终止：{exc}")
    else:
        pdf_files = sorted(PDF_DIR.rglob("*.pdf"))
        print(f"📄 解析PDF数：{len(pdf_files)}个")  # 修复：pymilvus2.4.4 + 原因：解析统计
        print(f"📝 生成文本块数：{len(texts)}条")
        print(f"📊 生成向量数：{len(vectors)}条")
        if vectors:
            print(f"🎯 向量维度：{len(vectors[0])}")

        if not vectors or not texts or not sources:
            print("⚠️ 未生成任何向量，请检查 PDF 文件或解析日志。")
        else:
            if len(vectors) != len(texts) or len(texts) != len(sources):  # 修复：pymilvus2.4.4 + 原因：数据对齐校验
                print("❌ 数据长度不一致，请检查解析逻辑。")
            if any(len(vector) != EXPECTED_DIM for vector in vectors):  # 修复：pymilvus2.4.4 + 原因：维度校验
                print("❌ 检测到向量维度异常，请检查向量生成逻辑。")
            if any(not text.strip() for text in texts):  # 修复：pymilvus2.4.4 + 原因：空文本校验
                print("❌ 检测到空文本块，请检查解析逻辑。")
            if any("_第" not in source for source in sources):  # 修复：pymilvus2.4.4 + 原因：来源完整性校验
                print("❌ 来源信息缺失页码，请检查解析逻辑。")

            for idx in range(min(2, len(texts))):
                print(f"示例来源{idx + 1}：{sources[idx]}")
                print(f"示例文本{idx + 1}：{texts[idx][:60]}...")
                print(f"示例向量维度{idx + 1}：{len(vectors[idx])}")

            print(
                "✅ 统计：有效文本块数/总提取块数="
                f"{_LAST_STATS.get('valid_chunks', 0)}/{_LAST_STATS.get('total_chunks', 0)}，"
                f"过滤无意义块数={_LAST_STATS.get('filtered_chunks', 0)}，"
                f"成功生成向量数={_LAST_STATS.get('vector_count', 0)}"
            )
