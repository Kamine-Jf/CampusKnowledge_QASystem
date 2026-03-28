# -*- coding: utf-8 -*-
"""
郑州轻工业大学校园知识问答系统 - 毕设演示入口脚本

功能说明：
    本脚本是毕设现场演示的唯一入口，支持命令行传参直接提问，
    实现端到端的RAG（检索增强生成）问答流程。

使用方式：
    python src/rag/rag_chat_demo.py 缓考申请流程
    python src/rag/rag_chat_demo.py 奖学金申请条件
    !!! python src/rag/rag_chat_demo.py 医保报销需要什么材料

运行前提：
    1. MySQL数据库服务已启动，且已导入校园事务数据
    2. Milvus向量数据库服务已启动，且已完成PDF向量化
    3. 网络可正常访问OpenRouter API（线上大模型）

作者：谢嘉峰
日期：2026年2月
"""

from __future__ import annotations

import os
import sys

# ===================== 全局路径配置（确保模块导入正常） =====================
# 说明：添加项目根目录到sys.path，确保在项目根目录执行时能正常导入所有模块
# 这是毕设演示中避免ImportError的关键配置
sys.path.append(os.getcwd())

# 获取脚本所在目录，向上两级找到项目根目录
from pathlib import Path
current_path = Path(__file__).resolve()
project_root = current_path.parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ===================== 导入RAG核心模块 =====================
try:
    from src.rag.rag_core import stage4_rag_query
except ModuleNotFoundError as e:
    print(f"❌ 模块导入失败：{e}")
    print("💡 提示：请确保在项目根目录下运行此脚本")
    print("   示例：python src/rag/rag_chat_demo.py 缓考申请流程")
    sys.exit(1)


def print_usage() -> None:
    """
    打印使用说明（未传入参数时显示）。
    
    输出示例命令，帮助用户快速上手毕设演示。
    """
    print()
    print("🎓 郑州轻工业大学校园知识问答系统 - 毕设演示")
    print("⚠️ 未检测到提问参数，请使用以下格式运行：")
    print("📌 使用方法：")
    print("  python src/rag/rag_chat_demo.py <你的问题>")
    print("📝 示例命令：")
    print("  python src/rag/rag_chat_demo.py 缓考申请流程")
    print("  python src/rag/rag_chat_demo.py 奖学金申请条件")
    print("  python src/rag/rag_chat_demo.py 医保报销需要什么材料")
    print("  python src/rag/rag_chat_demo.py 学生证补办需要哪些步骤")


def print_header(question: str) -> None:
    """
    打印演示头部信息（格式化输出，适合答辩展示）。
    
    Args:
        question: 用户提问的问题
    """
    print()
    print("🎓 郑州轻工业大学校园知识问答系统 - RAG端到端演示")
    print(f"📌 用户提问：{question}")


def print_answer(answer: str) -> None:
    """
    打印问答结果（格式化输出，回答内容分行清晰）。
    
    Args:
        answer: 大模型生成的回答文本
    """
    print()
    print("🤖 智能回答")
    print("-" * 24)

    # 将回答按行分割并按需缩进，保持结构紧凑
    answer_lines = answer.split("\n")
    for line in answer_lines:
        if line.strip():
            print(f"  {line}")
        else:
            print("  ")


def print_footer() -> None:
    """打印演示尾部信息。"""
    print()
    print("✅ RAG问答流程已完成 | 郑州轻工业大学毕业设计作品")


def main() -> None:
    """
    主函数：解析命令行参数并执行RAG问答流程。
    
    流程说明：
        1. 解析命令行参数，获取用户提问
        2. 调用stage4_rag_query执行端到端RAG流程
        3. 格式化输出问答结果
    """
    # ===== 解析命令行参数 =====
    # 支持多个参数拼接为一个问题（如：python demo.py 缓考 申请 流程）
    if len(sys.argv) < 2:
        print_usage()
        return
    
    # 将所有参数拼接为完整问题
    question = " ".join(sys.argv[1:]).strip()
    
    if not question:
        print_usage()
        return
    
    # ===== 打印演示头部 =====
    print_header(question)
    
    # ===== 调用RAG核心函数执行端到端问答 =====
    # 说明：stage4_rag_query是RAG系统的唯一对外接口
    #       内部自动完成：向量生成 → 双源检索 → 上下文融合 → Prompt拼接 → 线上模型生成
    try:
        answer = stage4_rag_query(question)
    except Exception as e:
        print(f"\n❌ RAG问答执行异常：{e}")
        print("💡 提示：请检查以下服务是否正常运行：")
        print("   1. MySQL数据库服务")
        print("   2. Milvus向量数据库服务")
        print("   3. 网络连接（用于调用线上大模型）")
        return
    
    # ===== 打印问答结果 =====
    print_answer(answer)
    
    # ===== 打印演示尾部 =====
    print_footer()


if __name__ == "__main__":
    main()
