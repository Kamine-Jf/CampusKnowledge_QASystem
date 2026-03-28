"""PDF 解析与文本分片模块。"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Tuple

from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader


_TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", ";", ".", " "]
)


def _clean_text(text: str) -> str:
    """清洗 PDF 页面文本。

    功能:
        去除多余空白字符与控制字符，确保输出文本可供向量化处理。
    参数:
        text (str): 原始提取文本。
    返回值:
        str: 清洗后的文本内容。
    异常:
        无。
    """
    normalized = text.replace("\u3000", " ")
    normalized = re.sub(r"[\r\t]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def parse_pdf_to_text_chunks(pdf_path: Path) -> List[str]:
    """解析单个 PDF 文件为文本片段列表。

    功能:
        读取 PDF 文件文本，完成清洗与分片，适配校园问答场景。
    参数:
        pdf_path (Path): PDF 文件的绝对或相对路径。
    返回值:
        List[str]: 分片后的文本内容列表。
    异常:
        FileNotFoundError: 当文件不存在时抛出。
        ValueError: 当 PDF 文件解析失败或无可用文本时抛出。
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"未找到 PDF 文件: {pdf_path}")

    try:
        reader = PdfReader(str(pdf_path))
    except Exception as exc:  # pylint: disable=broad-except
        raise ValueError(f"PDF 文件解析失败: {pdf_path}，错误信息: {exc}") from exc

    pages_text: List[str] = []
    for page_index, page in enumerate(reader.pages):
        try:
            extracted = page.extract_text() or ""
        except Exception as exc:  # pylint: disable=broad-except
            print(f"PDF 页面解析失败，已跳过: {pdf_path} - 页码 {page_index + 1}，错误: {exc}")
            continue
        if not extracted.strip():
            continue
        pages_text.append(_clean_text(extracted))

    if not pages_text:
        raise ValueError(f"PDF 文件无有效文本内容: {pdf_path}")

    chunks = _TEXT_SPLITTER.split_text("\n".join(pages_text))
    return [chunk for chunk in chunks if chunk.strip()]


def batch_parse_pdfs(base_dir: Path = Path("data/unstructured_data")) -> List[Tuple[str, str]]:
    """批量解析指定目录下的 PDF 文件。

    功能:
        遍历目录内所有 PDF 文件，解析为文本片段并附带来源信息。
    参数:
        base_dir (Path): 存放 PDF 文件的目录路径，默认指向 data/unstructured_data。
    返回值:
        List[Tuple[str, str]]: 包含 (文本片段, 来源文件名) 的元组列表。
    异常:
        ValueError: 当目录不存在或不存在任何 PDF 文件时抛出。
    """
    if not base_dir.exists():
        raise ValueError(f"未找到 PDF 目录: {base_dir}")

    pdf_files: Iterable[Path] = base_dir.glob("**/*.pdf")
    all_chunks: List[Tuple[str, str]] = []
    pdf_count = 0
    for pdf_path in pdf_files:
        pdf_count += 1
        try:
            chunks = parse_pdf_to_text_chunks(pdf_path)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"PDF 解析失败，已跳过: {pdf_path}，错误: {exc}")
            continue
        source_name = pdf_path.name
        all_chunks.extend((chunk, source_name) for chunk in chunks)

    if pdf_count == 0:
        raise ValueError(f"目录内未找到任何 PDF 文件: {base_dir}")

    return all_chunks
