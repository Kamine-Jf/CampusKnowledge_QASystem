# -*- coding: utf-8 -*-
"""
郑州轻工业大学校园知识问答RAG系统 - 线上大模型调用模块

功能说明：
    通过 OpenRouter API 调用大模型，为校园问答系统提供智能回答生成能力。

依赖安装：
    pip install openai
    （仅需安装 openai 库即可，无需 torch/transformers/auto-gptq 等本地模型依赖）

使用方式：
    1. 将 OPENROUTER_API_KEY 替换为您的 OpenRouter API Key
    2. RAG层直接调用 generate_answer(prompt) 函数即可

"""

from __future__ import annotations

import re
import warnings

# ============================= 依赖导入 =============================
# 强制使用 OpenAI 官方库调用 OpenRouter API（OpenRouter 兼容 OpenAI API 格式）
from openai import OpenAI
import httpx

# 关闭无关警告，保证毕设答辩演示输出简洁
warnings.filterwarnings("ignore")

# 缓存 OpenAI 客户端实例，避免每次请求重新创建 TCP 连接
_CLIENT_CACHE: dict[tuple[str, str], OpenAI] = {}

def _get_cached_client(api_base: str, api_key: str) -> OpenAI:
    """获取或创建缓存的 OpenAI 客户端，复用 HTTP 连接池。"""
    cache_key = (api_base, api_key)
    if cache_key not in _CLIENT_CACHE:
        _CLIENT_CACHE[cache_key] = OpenAI(
            base_url=api_base,
            api_key=api_key,
            timeout=httpx.Timeout(60.0, connect=10.0),
        )
    return _CLIENT_CACHE[cache_key]


# ============================= 配置区域 =============================

# API Key 
OPENROUTER_API_KEY = "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"

# 项目信息（用于 OpenRouter 统计来源）
YOUR_SITE_URL = "https://your-campus-qa-system.edu.cn"  # 您的网站URL，毕设可填学校官网或项目地址
# 注意：HTTP Header 仅支持 ASCII 字符，必须使用英文，否则会报编码错误
YOUR_SITE_NAME = "ZZULI Campus QA System"  # 您的网站/项目名称（必须英文）

# OpenRouter API 基础配置
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# 使用的模型
MODEL_NAME = "arcee-ai/trinity-large-preview:free"


# ============================= 系统提示词（校园问答专属） =============================

# 校园智能问答助手专属系统提示词，引导模型严格基于校园资料回答
SYSTEM_PROMPT = (
    "你是郑州轻工业大学校园智能问答助手，严格基于提供的校园官方资料回答问题。"
    "不编造任何信息，充分利用资料中的所有相关内容，复杂问题分点详细说明，不遗漏关键信息；"
    "无相关资料时直接回复：未查询到该问题的校园官方资料。"
    "用户可能会在同一会话中连续追问，请结合之前的对话上下文理解用户意图，"
    "当用户使用代词（如'它''这个''那个'）或省略主语时，请根据对话历史推断所指内容。"
)


# ============================= OpenAI Client 初始化 =============================

def _create_openai_client() -> OpenAI:
    """
    创建 OpenAI 客户端实例，配置 OpenRouter API 端点。
    
    Returns:
        OpenAI: 配置好的 OpenAI 客户端实例
    """
    client = OpenAI(
        base_url=OPENROUTER_BASE_URL,  # OpenRouter API 端点
        api_key=OPENROUTER_API_KEY,    # OpenRouter API Key
    )
    return client


# ============================= 核心生成函数 =============================

def generate_answer(prompt: str, model_config: dict | None = None, history: list[dict[str, str]] | None = None) -> str:
    """
    调用线上大模型生成回答（RAG层主调用接口）。
    
    本函数是 RAG 系统与大模型交互的唯一接口，RAG层将用户问题与检索到的
    上下文拼接后传入 prompt 参数，函数返回模型生成的纯文本回答。
    
    Args:
        prompt: RAG层拼接好的完整提示词（包含用户问题 + 检索到的上下文资料）
        model_config: 可选的模型配置字典，包含 model_name/api_base/api_key，
                      为 None 时使用默认配置
        history: 可选的多轮对话历史列表，每条为 {"role": "user"/"assistant", "content": "..."}，
                 用于支持用户在同一会话中连续追问
    
    Returns:
        str: 模型生成的回答文本（已清洗处理，可直接展示给用户）
    
    Raises:
        无异常抛出，所有异常均在内部捕获并返回兜底回复
    
    使用示例:
        >>> from src.llm.qwen_operate import generate_answer
        >>> answer = generate_answer("根据以下资料回答问题...缓考申请需要什么材料？")
        >>> print(answer)
    """
    
    # 解析模型配置：优先使用传入的配置，否则使用默认配置
    use_model = MODEL_NAME
    use_api_base = OPENROUTER_BASE_URL
    use_api_key = OPENROUTER_API_KEY
    if model_config:
        use_model = model_config.get("model_name") or MODEL_NAME
        use_api_base = model_config.get("api_base") or OPENROUTER_BASE_URL
        use_api_key = model_config.get("api_key") or OPENROUTER_API_KEY
    
    # 日志输出：提示用户正在调用线上模型（毕设演示友好）
    print(f"正在调用线上模型生成回答... 模型: {use_model}")
    
    try:
        # 复用缓存的 OpenAI 客户端，避免每次重新建立 TCP 连接
        client = _get_cached_client(use_api_base, use_api_key)
        
        # 构建对话消息列表
        # system: 设定助手角色和行为规范
        # history: 多轮对话历史（支持用户连续追问）
        # user: 传入RAG层拼接好的提示词（包含问题和上下文）
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})
        
        # 调用 OpenRouter API（兼容 OpenAI Chat Completions 格式）
        completion = client.chat.completions.create(
            model=use_model,
            messages=messages,
            max_tokens=3000,
            # extra_headers: OpenRouter 特有的请求头，用于统计请求来源
            extra_headers={
                "HTTP-Referer": YOUR_SITE_URL,   # 您的网站URL（可选，用于流量统计）
                "X-Title": YOUR_SITE_NAME,       # 您的网站名称（可选，用于来源标识）
            },
            # extra_body: 可传递额外参数（如需要可在此扩展）
            extra_body={},
        )
        
        # 提取模型回答内容（兼容异常/空返回）
        if completion is None or not getattr(completion, "choices", None):
            raise RuntimeError("OpenRouter 返回为空或未包含 choices")
        first_choice = completion.choices[0]
        message = getattr(first_choice, "message", None)
        raw_answer = getattr(message, "content", None)
        if raw_answer is None:
            raise RuntimeError("OpenRouter 返回的 message.content 为空")
        
        # 清洗回答文本：去除首尾空格、多余空行、特殊控制字符
        clean_answer = _clean_response_text(raw_answer)
        
        # 日志输出：回答生成成功
        print("✅ 回答生成成功！")
        
        return clean_answer
    
    except Exception as e:
        # ==================== 异常处理（毕设演示友好） ====================
        # 捕获所有可能的异常，返回友好的兜底回复，避免演示时程序崩溃
        
        error_message = str(e).lower()
        
        # 网络连接异常
        if "connection" in error_message or "network" in error_message or "timeout" in error_message:
            print(f"⚠️ 网络连接异常：{e}")
            print("💡 提示：请检查网络连接是否正常，或稍后重试")
            return "抱歉，当前网络连接异常，无法获取回答，请稍后重试。"
        
        # API Key 错误（未授权、无效Key等）
        if "401" in error_message or "unauthorized" in error_message or "invalid" in error_message and "key" in error_message:
            print(f"⚠️ API Key 验证失败：{e}")
            print("💡 提示：请检查 OPENROUTER_API_KEY 是否正确配置")
            return "抱歉，API 密钥验证失败，请联系管理员检查配置。"
        
        # 请求超时
        if "timeout" in error_message or "timed out" in error_message:
            print(f"⚠️ 请求超时：{e}")
            print("💡 提示：线上模型响应较慢，请稍后重试")
            return "抱歉，请求响应超时，请稍后重试。"
        
        # 速率限制（免费模型可能有调用频率限制）
        if "429" in error_message or "rate limit" in error_message or "too many" in error_message:
            print(f"⚠️ 请求频率超限：{e}")
            print("💡 提示：免费模型有调用频率限制，请稍等片刻后重试")
            return "抱歉，当前请求过于频繁，请稍等片刻后重试。"
        
        # 模型服务不可用
        if "503" in error_message or "service unavailable" in error_message or "model" in error_message and "unavailable" in error_message:
            print(f"⚠️ 模型服务暂时不可用：{e}")
            print("💡 提示：线上模型服务可能正在维护，请稍后重试")
            return "抱歉，模型服务暂时不可用，请稍后重试。"
        
        # 其他未知异常（兜底处理）
        print(f"⚠️ 调用线上模型时发生未知错误：{e}")
        print("💡 提示：请检查网络和配置，或联系管理员排查问题")
        return "抱歉，系统暂时无法生成回答，请稍后重试。"


def generate_answer_stream(prompt: str, model_config: dict | None = None, history: list[dict[str, str]] | None = None):
    """
    流式调用线上大模型生成回答（SSE流式输出接口）。
    
    与 generate_answer 功能一致，但使用 stream=True 逐块返回内容，
    前端可实时展示生成过程，用户可随时停止生成。
    
    Args:
        prompt: RAG层拼接好的完整提示词
        model_config: 可选的模型配置字典
        history: 可选的多轮对话历史列表，每条为 {"role": "user"/"assistant", "content": "..."}，
                 用于支持用户在同一会话中连续追问
    
    Yields:
        str: 每次生成的文本片段
    """
    use_model = MODEL_NAME
    use_api_base = OPENROUTER_BASE_URL
    use_api_key = OPENROUTER_API_KEY
    if model_config:
        use_model = model_config.get("model_name") or MODEL_NAME
        use_api_base = model_config.get("api_base") or OPENROUTER_BASE_URL
        use_api_key = model_config.get("api_key") or OPENROUTER_API_KEY

    print(f"正在流式调用线上模型生成回答... 模型: {use_model}")

    try:
        client = _get_cached_client(use_api_base, use_api_key)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})

        stream = client.chat.completions.create(
            model=use_model,
            messages=messages,
            stream=True,
            max_tokens=3000,
            extra_headers={
                "HTTP-Referer": YOUR_SITE_URL,
                "X-Title": YOUR_SITE_NAME,
            },
            extra_body={},
        )

        in_think_tag = False
        think_buffer = ""
        _first_chunk_sent = False

        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            if content is None:
                continue

            # 过滤 <think>...</think> 标签内容
            i = 0
            while i < len(content):
                if in_think_tag:
                    end_pos = content.find("</think>", i)
                    if end_pos != -1:
                        in_think_tag = False
                        i = end_pos + len("</think>")
                        think_buffer = ""
                    else:
                        break
                else:
                    start_pos = content.find("<think>", i)
                    if start_pos != -1:
                        before = content[i:start_pos]
                        if not _first_chunk_sent:
                            before = before.lstrip("\n\r ")
                        if before:
                            _first_chunk_sent = True
                            yield before
                        in_think_tag = True
                        i = start_pos + len("<think>")
                    else:
                        remaining = content[i:]
                        if not _first_chunk_sent:
                            remaining = remaining.lstrip("\n\r ")
                        if remaining:
                            _first_chunk_sent = True
                            yield remaining
                        break

        print("✅ 流式回答生成完成！")

    except Exception as e:
        error_message = str(e).lower()
        if "connection" in error_message or "network" in error_message or "timeout" in error_message:
            yield "抱歉，当前网络连接异常，无法获取回答，请稍后重试。"
        elif "401" in error_message or "unauthorized" in error_message:
            yield "抱歉，API 密钥验证失败，请联系管理员检查配置。"
        elif "429" in error_message or "rate limit" in error_message:
            yield "抱歉，当前请求过于频繁，请稍等片刻后重试。"
        else:
            yield "抱歉，系统暂时无法生成回答，请稍后重试。"
        print(f"⚠️ 流式调用线上模型时发生错误：{e}")


def _clean_response_text(text: str) -> str:
    """
    清洗模型返回的文本，确保输出整洁。
    
    处理内容：
        1. 去除首尾空白字符
        2. 去除多余的连续空行（保留单个换行）
        3. 去除特殊控制字符
        4. 去除可能的 think 标签内容（如 <think>...</think>，用户只需要最终答案）
    
    Args:
        text: 原始模型输出文本
    
    Returns:
        str: 清洗后的整洁文本
    """
    if not text:
        return ""
    
    # 去除模型的 <think>...</think> 思考过程标签（用户只需要最终答案）
    text = re.sub(r'<think>[\s\S]*?</think>', '', text)
    
    # 去除首尾空白
    text = text.strip()
    
    # 将多个连续空行替换为单个空行
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # 去除特殊控制字符（保留常见的换行和空格）
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    
    return text


# ============================= 测试入口 =============================

if __name__ == "__main__":
    """
    测试入口：单独运行本文件可直接测试线上 API 调用效果。
    
    运行方式：
        python src/llm/qwen_operate.py
    
    测试前请确保：
        1. 已安装 openai 库：pip install openai
        2. 已将 OPENROUTER_API_KEY 替换为真实的 API Key
        3. 网络可正常访问 OpenRouter API
    """
    
    print("=" * 60)
    print(" 郑州轻工业大学校园知识问答系统 - 线上模型测试")
    print("=" * 60)
    print(f" API 端点：{OPENROUTER_BASE_URL}")
    print(f" 使用模型：{MODEL_NAME}")
    print("=" * 60)
    
    # 测试问题（模拟 RAG 层传入的提示词）
    test_question = "缓考申请需要什么材料？"
    
    print(f"\n 测试问题：{test_question}\n")
    
    # 调用生成函数
    answer = generate_answer(test_question)
    
    print("\n" + "=" * 60)
    print(" 模型回答：")
    print("-" * 60)
    print(answer)
    print("=" * 60)
    print("\n 测试完成！如回答正常，说明线上 API 调用成功。")
