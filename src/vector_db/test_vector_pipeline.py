"""阶段 3 向量管线测试脚本。"""
from __future__ import annotations

from typing import List, Tuple

from src.vector_db.milvus_operate import insert_vectors, search_similar_vectors
from src.vector_db.pdf_processor import batch_parse_pdfs
from src.vector_db.rag_retriever import hybrid_retrieve
from src.vector_db.vector_generator import text_to_vector


TEST_KEYWORDS = ["缓考申请", "奖学金申请"]


def _prepare_texts() -> Tuple[List[str], List[str]]:
    """加载 PDF 文本并拆解来源。

    功能:
        调用 batch_parse_pdfs 获取文本片段，拆分文本内容与来源文件名。
    参数:
        无。
    返回值:
        Tuple[List[str], List[str]]: 文本片段列表与来源文件名列表。
    异常:
        RuntimeError: 当解析过程失败时抛出。
    """
    try:
        chunk_pairs = batch_parse_pdfs()
    except Exception as exc:  # pylint: disable=broad-except
        raise RuntimeError(f"PDF 解析阶段失败: {exc}") from exc

    if not chunk_pairs:
        raise RuntimeError("PDF 解析结果为空，无法继续测试。")

    texts, sources = zip(*chunk_pairs)
    return list(texts), list(sources)


def run_pipeline_test() -> None:
    """执行向量管线全流程测试。

    功能:
        串联执行 PDF 解析、文本分片、向量生成、Milvus 入库、相似度检索与 RAG 融合查询，打印各阶段结果。
    参数:
        无。
    返回值:
        None。
    异常:
        未显式抛出，内部打印错误信息。
    """
    print("[步骤 1] 开始解析 PDF 文档...")
    try:
        text_chunks, sources = _prepare_texts()
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[失败] {exc}")
        return
    print(f"[成功] 共获取 {len(text_chunks)} 条文本片段。")

    print("[步骤 2] 开始生成文本向量...")
    try:
        vectors = text_to_vector(text_chunks)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[失败] 向量生成异常: {exc}")
        return
    print("[成功] 向量生成完成。")

    print("[步骤 3] 写入向量至 Milvus...")
    try:
        inserted = insert_vectors(text_chunks, vectors, sources)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[失败] Milvus 入库异常: {exc}")
        return
    print(f"[成功] 向量库新增 {inserted} 条记录。")

    if vectors:
        print("[步骤 4] 执行示例向量检索...")
        try:
            search_results = search_similar_vectors(vectors[0])
            print(f"[成功] 返回 {len(search_results)} 条相似结果。")
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[失败] 向量检索异常: {exc}")
            return
    else:
        print("[警告] 无向量可供检索。")

    print("[步骤 5] 执行 RAG 融合检索...")
    for keyword in TEST_KEYWORDS:
        try:
            results = hybrid_retrieve(keyword)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[失败] 关键词 {keyword} 检索异常: {exc}")
            return
        print(f"关键词: {keyword} | 返回结果数: {len(results)}")
        for item in results:
            print(
                f"  - 来源: {item.get('data_source')} | 分值: {item.get('score')} | 内容: {item.get('content')}"
            )

    print("阶段3模块测试通过")


if __name__ == "__main__":
    run_pipeline_test()
