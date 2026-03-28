"""向量生成模块。"""
from __future__ import annotations

from typing import List

import torch
import numpy as np
from sentence_transformers import SentenceTransformer

from .milvus_config import VECTOR_DIM

_MODEL: SentenceTransformer | None = None
_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _detect_device() -> str:
    """判断最优推理设备。

    功能:
        在 CUDA 可用时优先使用 GPU，否则回落至 CPU。
    参数:
        无。
    返回值:
        str: "cuda" 或 "cpu"。
    异常:
        无。
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_model() -> SentenceTransformer:
    """加载 SentenceTransformer 模型。

    功能:
        延迟加载模型并缓存，捕获显存不足异常后回退至 CPU。
    参数:
        无。
    返回值:
        SentenceTransformer: 已加载的文本向量化模型实例。
    异常:
        RuntimeError: 模型加载失败时抛出，包含底层错误信息。
    """
    global _MODEL

    if _MODEL is not None:
        return _MODEL

    device = _detect_device()
    try:
        model = SentenceTransformer(_MODEL_NAME, device=device)
    except RuntimeError as exc:  # pylint: disable=broad-except
        if "CUDA" in str(exc).upper() and device == "cuda":
            print(f"GPU 显存不足，切换至 CPU 推理。错误信息: {exc}")
            model = SentenceTransformer(_MODEL_NAME, device="cpu")
        else:
            raise RuntimeError(f"SentenceTransformer 模型加载失败: {exc}") from exc
    except Exception as exc:  # pylint: disable=broad-except
        raise RuntimeError(f"SentenceTransformer 模型加载异常: {exc}") from exc

    model.max_seq_length = 256
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _MODEL = model
    return _MODEL


def text_to_vector(text_chunks: List[str]) -> List[List[float]]:
    """将文本片段批量转换为向量。

    功能:
        对输入的文本列表执行批量嵌入生成操作，输出 768 维向量序列。
    参数:
        text_chunks (List[str]): 待转换的文本片段列表。
    返回值:
        List[List[float]]: 与输入顺序对应的向量列表。
    异常:
        ValueError: 当输入列表为空或全为空串时抛出。
        RuntimeError: 当模型推理过程中出现异常时抛出。
    """
    if not text_chunks:
        raise ValueError("文本片段列表为空，无法生成向量。")

    filtered_chunks = [chunk.strip() for chunk in text_chunks if chunk and chunk.strip()]
    if not filtered_chunks:
        raise ValueError("文本片段内容均为空，无法生成向量。")

    model = _load_model()
    try:
        embeddings = model.encode(filtered_chunks, convert_to_numpy=True, batch_size=16, show_progress_bar=False)
    except RuntimeError as exc:  # pylint: disable=broad-except
        raise RuntimeError(f"向量生成失败: {exc}") from exc

    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)

    current_dim = embeddings.shape[1]
    if current_dim > VECTOR_DIM:
        embeddings = embeddings[:, :VECTOR_DIM]
    elif current_dim < VECTOR_DIM:
        pad_width = VECTOR_DIM - current_dim
        embeddings = np.pad(embeddings, ((0, 0), (0, pad_width)), mode="constant")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return embeddings.tolist()
