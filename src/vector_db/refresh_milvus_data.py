# -*- coding: utf-8 -*-
"""一键入库脚本：PDF解析→向量生成→Milvus入库→检索验证。

阶段4修复：适配 pymilvus2.4.4 入库API，确保非结构化数据可检索。
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")  # 修复：pymilvus2.4.4适配 + 原因说明
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

try:
    from src.vector_db.pdf2vector import gen_pdf_vectors, generate_query_vector
    from src.vector_db.milvus_operate import (
        connect_milvus,
        create_milvus_collection,
        insert_vectors_to_milvus,
        check_milvus_data,
        search_similar_vector,
    )
except ModuleNotFoundError:
    current_path = Path(__file__).resolve()
    project_root = current_path.parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.vector_db.pdf2vector import gen_pdf_vectors, generate_query_vector  # type: ignore
    from src.vector_db.milvus_operate import (  # type: ignore
        connect_milvus,
        create_milvus_collection,
        insert_vectors_to_milvus,
        check_milvus_data,
        search_similar_vector,
    )


def _print_step(message: str) -> None:
    print(message)


def main() -> None:
    _print_step("🚀 阶段4一键入库脚本启动（pymilvus2.4.4适配）")  # 修复：pymilvus2.4.4 + 原因：统一入口提示

    # 步骤2：连接 Milvus
    try:  # 修复：pymilvus2.4.4 + 原因：连接异常捕获
        connect_milvus()
        _print_step("✅ 步骤2：Milvus 连接成功（pymilvus2.4.4）")
    except Exception as exc:  # pylint: disable=broad-except
        _print_step(f"❌ 步骤2：Milvus 连接失败→请检查Docker容器是否启动：{exc}")
        return

    # 步骤3：初始化集合
    try:  # 修复：pymilvus2.4.4 + 原因：集合初始化异常兜底
        create_milvus_collection()
        _print_step("✅ 步骤3：集合初始化完成（campus_qa_vector）")
    except Exception as exc:  # pylint: disable=broad-except
        _print_step(f"❌ 步骤3：集合初始化失败→请检查Milvus服务状态：{exc}")
        return

    # 步骤4：PDF解析 + 向量生成
    try:  # 修复：pymilvus2.4.4 + 原因：解析异常捕获
        _print_step("🚀 步骤4：开始解析PDF文件（data/unstructured_data/）")
        vectors, texts, sources = gen_pdf_vectors()
        _print_step(
            f"📊 步骤4：解析完成→文本块数：{len(texts)}条 | 向量数：{len(vectors)}条"
        )
    except Exception as exc:  # pylint: disable=broad-except
        _print_step(f"❌ 步骤4：PDF解析失败→请检查PDF文件：{exc}")
        return

    # 步骤5：入库前校验
    try:  # 修复：pymilvus2.4.4 + 原因：入库前校验
        if not vectors or not texts or not sources:
            _print_step("❌ 步骤5：数据为空→请检查PDF目录是否存在可解析文件")
            return
        if not (len(vectors) == len(texts) == len(sources)):
            _print_step("❌ 步骤5：数据长度不一致→请检查解析逻辑")
            return
        if any(len(vector) != 768 for vector in vectors):
            _print_step("❌ 步骤5：向量维度异常→请检查向量生成逻辑")
            return
        _print_step("⚠️ 步骤5：校验通过→所有数据长度一致，向量维度768")
    except Exception as exc:  # pylint: disable=broad-except
        _print_step(f"❌ 步骤5：校验异常→{exc}")
        return

    # 步骤6：向量入库
    try:  # 修复：pymilvus2.4.4 + 原因：入库异常捕获
        inserted = insert_vectors_to_milvus(vectors, texts, sources)
        if inserted <= 0:
            _print_step("❌ 步骤6：入库失败→请检查Milvus服务与入库日志")
            return
        _print_step(f"✅ 步骤6：入库成功→写入{inserted}条向量")
    except Exception as exc:  # pylint: disable=broad-except
        _print_step(f"❌ 步骤6：入库失败→{exc}")
        return

    # 步骤7：入库后数据量校验
    try:  # 修复：pymilvus2.4.4 + 原因：数据量校验
        total = check_milvus_data()
        if total <= 0:
            _print_step("❌ 步骤7：集合仍为空→请检查入库逻辑")
            return
        _print_step(f"✅ 步骤7：集合数据量校验通过→当前总量：{total}条")
    except Exception as exc:  # pylint: disable=broad-except
        _print_step(f"❌ 步骤7：数据量校验失败→{exc}")
        return

    # 步骤8：检索测试
    try:  # 修复：pymilvus2.4.4 + 原因：检索验证
        _print_step("🚀 步骤8：执行检索测试（关键词：缓考）")
        query_vector = generate_query_vector("缓考")
        results = search_similar_vector(query_vector, top_k=3)
        high_matches = [item for item in results if item.get("score", 1.0) < 0.5]
        if not results:
            _print_step("⚠️ 步骤8：检索无结果→请确认非结构化数据是否入库")
        else:
            for idx, item in enumerate(results, start=1):
                label = "高匹配" if item.get("score", 1.0) < 0.5 else "低匹配"
                text_preview = str(item.get("text", ""))[:60]
                _print_step(
                    f"Top{idx} | Score:{item.get('score', 0.0):.4f} | {label} | Source:{item.get('source', '')}\n"
                    f"文本摘要：{text_preview}..."
                )
            if len(high_matches) >= 3:
                _print_step("🎉 修复全流程成功！非结构化数据检索正常。")
            else:
                _print_step("⚠️ 高匹配结果不足3条，请检查PDF内容覆盖范围。")
    except Exception as exc:  # pylint: disable=broad-except
        _print_step(f"❌ 步骤8：检索失败→{exc}")
        return

    # 步骤9：总结
    _print_step("✅ 步骤9：一键入库流程完成，可开始RAG检索演示。")


if __name__ == "__main__":
    main()
