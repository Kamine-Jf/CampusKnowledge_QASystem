# -*- coding: utf-8 -*-
"""
RAG 核心模块：整合结构化与非结构化校园知识

功能说明：
    本模块是郑州轻工业大学校园知识问答系统的核心检索增强生成（RAG）模块，
    负责将MySQL结构化数据与Milvus向量数据进行融合检索，并调用线上大模型生成回答。

核心函数：
    - stage4_rag_query: 端到端RAG入口，毕设演示主调用函数
    - rag_query: 双源检索函数（阶段1-3实现）
    - fuse_context: 上下文融合函数，将检索结果转为Prompt可用格式

作者：谢嘉峰
日期：2026年2月
"""
from __future__ import annotations

import sys
from pathlib import Path

# ===================== 路径配置（确保模块导入正常） =====================
current_path = Path(__file__).resolve()
project_root = current_path.parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ===================== 线上大模型导入（阶段4核心） =====================
# 说明：仅使用线上模型调用，本地模型相关导入已注释
try:
    from src.llm.qwen_operate import generate_answer, generate_answer_stream  # 线上大模型生成回答
except ModuleNotFoundError:
    from src.llm.qwen_operate import generate_answer, generate_answer_stream  # type: ignore

# ===================== RAG配置导入（集中管理参数） =====================
try:
    from src.config.settings import RAGConfig
except ModuleNotFoundError:
    from src.config.settings import RAGConfig  # type: ignore

# 核心：添加Hugging Face国内镜像（必须最先设置）
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 匹配度优化：国内镜像必须最先设置

# 修复：pymilvus2.4.4 + 原因：屏蔽第三方弃用/平台提示
import logging
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
warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API",
    category=UserWarning,
)  # 修复：pymilvus2.4.4 + 原因：覆盖不同模块路径

logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
logging.getLogger("torch.distributed.elastic.multiprocessing").setLevel(logging.ERROR)

import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

# ===================== 模块级线程池（RAG双源检索效率优化） =====================
# 复用线程池避免每次查询都创建/销毁 ThreadPoolExecutor 的开销
_RAG_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="rag_retrieval")

try:
    from src.database.db_operate import query_by_keyword
    from src.vector_db.milvus_operate import insert_vectors_to_milvus, search_similar_vector, fetch_chunks_by_source_prefix
    from src.vector_db.pdf2vector import (
        CACHE_DIR as MODEL_CACHE_DIR,
        EXPECTED_DIM,
        MODEL_NAME,
        pdf_to_vectors,
    )
except ModuleNotFoundError:
    current_path = Path(__file__).resolve()
    project_root = current_path.parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.database.db_operate import query_by_keyword  # type: ignore
    from src.vector_db.milvus_operate import insert_vectors_to_milvus, search_similar_vector, fetch_chunks_by_source_prefix  # type: ignore
    from src.vector_db.pdf2vector import (  # type: ignore
        CACHE_DIR as MODEL_CACHE_DIR,
        EXPECTED_DIM,
        MODEL_NAME,
        pdf_to_vectors,
    )

_MODEL_CACHE: Optional[SentenceTransformer] = None


def _ensure_model() -> SentenceTransformer:
    """懒加载向量模型，强制绑定 CPU，避免显存不足。"""
    global _MODEL_CACHE  # pylint: disable=global-statement
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # 关键配置：沿用模块一的缓存目录，避免毕设演示时出现路径不一致。
    os.environ["TRANSFORMERS_CACHE"] = str(MODEL_CACHE_DIR)
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(MODEL_CACHE_DIR)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    try:
        _MODEL_CACHE = SentenceTransformer(MODEL_NAME, cache_folder=str(MODEL_CACHE_DIR), device="cpu")
    except Exception as exc:  # pylint: disable=broad-except
        raise RuntimeError(f"向量模型加载失败，请检查依赖与缓存目录：{exc}") from exc
    return _MODEL_CACHE


def _vectorize_text(query_text: str) -> List[float]:
    """将查询文本编码为 768 维向量。

    参数:
        query_text: 用户输入的自然语言问题。

    返回:
        List[float]: 标准化后的 768 维向量列表。
    """
    model = _ensure_model()
    embeddings = model.encode(
        sentences=[query_text],
        device="cpu",
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    vector = embeddings[0]
    current_dim = vector.shape[0]
    if current_dim < EXPECTED_DIM:
        vector = np.pad(vector, (0, EXPECTED_DIM - current_dim), mode="constant")
    elif current_dim > EXPECTED_DIM:
        vector = vector[:EXPECTED_DIM]
    return vector.astype(float).tolist()


def sync_pdf_vectors_to_milvus() -> int:
    """触发 PDF → 向量 → Milvus 的同步流程，便于初始化数据。

    返回:
        int: 实际写入 Milvus 的向量条数。
    """
    vectors_data = pdf_to_vectors()
    if not vectors_data:
        print("⚠️ 未获取到可用向量数据，Milvus 未更新。")
        return 0
    return insert_vectors_to_milvus(vectors_data)


def _strip_stopwords(text: str) -> str:
    """移除中文常见虚词/疑问词，保留实义关键词以提升 MySQL LIKE 命中率。

    例如：
        "软件学院的专业有哪些" → "软件学院专业"
        "怎么申请助学金呢"     → "申请助学金"
    """
    import re
    _STOPWORDS = (
        "的", "了", "在", "是", "我", "有", "和", "就",
        "不", "人", "都", "一", "一个", "上", "也", "很",
        "到", "说", "要", "去", "你", "会", "着", "没有",
        "看", "好", "自己", "这", "他", "她", "它",
        "吗", "吧", "呢", "啊", "呀", "哦", "嗯",
        "什么", "哪些", "哪个", "哪里", "怎么", "怎样", "如何",
        "为什么", "多少", "几个", "能否", "是否", "可以",
        "请问", "请", "想", "想要", "需要", "可以",
        "告诉", "一下", "关于",
        # 保留"介绍"等词，因其可能出现在"学校简介"等复合词中
        # 仅保留纯功能词，移除可能与校园专有名词构成复合词的词汇
        "相关", "全部", "所有", "具体",
    )
    # 按长度降序排列，优先匹配长词（如"为什么"优先于"为"）
    sorted_words = sorted(_STOPWORDS, key=len, reverse=True)
    pattern = "|".join(re.escape(w) for w in sorted_words)
    result = re.sub(pattern, "", text)
    return result.strip()


# ===================== jieba 分词初始化（模块级，只执行一次） =====================
_JIEBA_INITIALIZED = False


def _ensure_jieba():
    """懒加载 jieba 并注册校园领域词典，确保只初始化一次。"""
    global _JIEBA_INITIALIZED  # pylint: disable=global-statement
    if _JIEBA_INITIALIZED:
        return

    import jieba
    import jieba.posseg

    # 校园领域专有词汇：确保 jieba 不会把这些词拆散
    _CAMPUS_DICT_WORDS = [
        # 院系与机构
        "教务处", "学生处", "招生办", "就业指导中心", "后勤处", "保卫处", "财务处",
        "图书馆", "实验室", "研究生院", "国际交流处", "团委", "学工部", "科研处",
        "软件学院", "计算机学院", "电气学院", "机电学院", "食品学院", "材料学院",
        "经济管理学院", "艺术设计学院", "外国语学院", "数学学院", "物理学院",
        "马克思主义学院", "体育学院", "能源学院", "建筑环境工程学院", "法学院",
        "郑州轻工业大学", "轻工大", "轻院",
        # 学校概况
        "学校简介", "学院简介", "专业简介", "学校概况", "学校历史",
        # 教务与考试
        "缓考", "补考", "重修", "选课", "退课", "休学", "复学", "转专业",
        "学分", "绩点", "成绩查询", "成绩单", "考试安排", "教学评价",
        "毕业设计", "毕业论文", "毕业答辩", "学位证", "毕业证",
        "教务系统", "教务管理", "课程表", "培养方案",
        # 学生事务
        "助学金", "奖学金", "国家助学金", "国家奖学金", "励志奖学金",
        "助学贷款", "学费减免", "勤工助学", "困难认定",
        "学生证", "校园卡", "一卡通", "宿舍", "寝室",
        "入学", "报到", "注册", "档案", "户口",
        "社团", "学生会", "志愿者", "社会实践", "创新创业",
        # 校园生活
        "校医院", "心理咨询", "体育场", "食堂", "澡堂", "快递",
        "校园网", "VPN", "正方教务",
        # 常见操作
        "申请流程", "办理流程", "报名", "审批", "提交材料",
    ]
    for word in _CAMPUS_DICT_WORDS:
        jieba.add_word(word, freq=50000)

    _JIEBA_INITIALIZED = True


# jieba 分词时保留的词性集合（名词、动词、专名等实义词）
_USEFUL_POS_TAGS = frozenset({
    "n", "nr", "ns", "nt", "nz", "nl", "ng",   # 名词类
    "v", "vn", "vd", "vg",                       # 动词类
    "a", "an", "ag",                              # 形容词类
    "j",                                           # 简称
    "i",                                           # 成语/习语
    "l",                                           # 习惯用语
    "eng",                                         # 英文
    "x",                                           # 非语素字（兜底）
})


# 停用词集合（用于 jieba 分词后的 token 级过滤，避免正则破坏"学校简介"等复合词）
_STOPWORDS_CORE_SET = frozenset({
    "的", "了", "在", "是", "我", "有", "和", "就",
    "不", "人", "都", "一", "一个", "上", "也", "很",
    "到", "说", "要", "去", "你", "会", "着", "没有",
    "看", "好", "自己", "这", "他", "她", "它",
    "吗", "吧", "呢", "啊", "呀", "哦", "嗯",
    "什么", "哪些", "哪个", "哪里", "怎么", "怎样", "如何",
    "为什么", "多少", "几个", "能否", "是否", "可以",
    "请问", "请", "想", "想要", "需要",
    "告诉", "介绍", "一下", "关于",
    "知道", "了解", "查找", "搜索", "找",
    "老师", "同学", "教授", "导师", "学生", "学校",
    "资料", "信息", "详情", "情况", "内容", "简介",
    "相关", "全部", "所有", "具体", "查询", "查看",
})


def _extract_keywords(text: str) -> List[str]:
    """从用户查询中提取多个有意义的关键词，用于全面检索结构化数据。

    策略（已优化）：
        1. 使用 jieba 带词性标注的精确分词，替代原始滑动窗口
        2. 按词性过滤，只保留名词/动词/形容词等实义词
        3. 生成相邻实义词的组合（bigram），覆盖复合短语检索
        4. 去虚词后的整体文本作为首选关键词
        5. 原始文本兜底
    """
    import re
    import jieba.posseg as pseg

    if not text or not text.strip():
        return []

    _ensure_jieba()

    compact = text.strip().replace(" ", "")
    # 使用 jieba token 级停用词过滤，避免正则直接替换破坏"学校简介"等复合领域词
    _core_tokens = [w for w, _ in pseg.lcut(compact) if w.strip() and w not in _STOPWORDS_CORE_SET]
    core = "".join(_core_tokens).strip() or compact

    keywords: List[str] = []
    seen: set[str] = set()

    def _add(kw: str) -> None:
        kw = kw.strip()
        if len(kw) >= 2 and kw not in seen:
            seen.add(kw)
            keywords.append(kw)

    # 1. 去虚词后的核心文本整体作为首选关键词（精确匹配优先）
    _add(core)

    # 2. 按常见连接词/标点拆分为子短语
    _SPLIT_PATTERN = r"[，,。.、；;！!？?和与及或还有以及跟同]"
    segments = re.split(_SPLIT_PATTERN, core)
    for seg in segments:
        _add(seg.strip())

    # 3. 用 jieba 词性标注分词，提取实义词
    #    关键优化：合并连续单字符 token 为复合词（处理人名等未登录词）
    meaningful_words: List[str] = []
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        seg_words = []
        raw_tokens = pseg.lcut(seg)

        # 3a. 合并连续单字 token —— jieba 对未登录词（人名等）常拆成单字
        #     先收集所有实义词到 seg_words，统一在下方添加（bigram 优先于单词）
        i = 0
        while i < len(raw_tokens):
            word, flag = raw_tokens[i]
            word = word.strip()
            if not word:
                i += 1
                continue
            if len(word) == 1 and not word.isascii():
                # 收集连续单字
                merged = word
                j = i + 1
                while j < len(raw_tokens):
                    nw, _ = raw_tokens[j]
                    nw = nw.strip()
                    if len(nw) == 1 and not nw.isascii():
                        merged += nw
                        j += 1
                    else:
                        break
                if len(merged) >= 2:
                    seg_words.append(merged)
                i = j
            else:
                if len(word) >= 2:
                    if flag in _USEFUL_POS_TAGS or flag.startswith("n") or flag.startswith("v"):
                        seg_words.append(word)
                i += 1
        meaningful_words.extend(seg_words)

        # 4. 优先添加相邻实义词的 bigram 组合（如"学校简介"），再添加单个词
        #    确保复合短语关键词在 MySQL 检索时优先于宽泛的单字词
        for i in range(len(seg_words) - 1):
            _add(seg_words[i] + seg_words[i + 1])
        for word in seg_words:
            _add(word)

    # 5. 原始输入（含虚词）兜底
    _add(compact)

    # 6. 从原始文本（未过滤停用词）生成 bigram，覆盖跨停用词的复合短语
    #    例如 "学生档案查询" 中 "学生" 是停用词，过滤后 core 只剩 "档案查询"，
    #    此步骤确保 "学生档案" 作为关键词被单独检索到。
    orig_tokens_raw = [w for w, _ in pseg.lcut(compact) if w.strip()]
    for _i in range(len(orig_tokens_raw) - 1):
        _bigram = orig_tokens_raw[_i].strip() + orig_tokens_raw[_i + 1].strip()
        _add(_bigram)

    return keywords


def _structured_query_with_fallback(query_text: str) -> List[Dict[str, object]]:
    """执行结构化检索：多关键词全面检索 + 合并去重，无结果时自动降级。

    优化策略：
        1. 从用户问题中提取多个关键词
        2. 对每个关键词分别调用 query_by_keyword
        3. 按记录ID去重，合并所有命中结果，确保检索全面
        4. 若多关键词均无结果，降级为前后缀截断尝试
    """
    cleaned = (query_text or "").strip()
    if not cleaned:
        return []

    # ===== 阶段1：多关键词全面检索 =====
    keywords = _extract_keywords(cleaned)
    all_results: List[Dict[str, object]] = []
    seen_ids: set = set()

    soft_cap = getattr(RAGConfig, "STRUCTURED_MAX_RESULTS", 20) * 3
    for kw in keywords:
        if len(all_results) >= soft_cap:
            break
        try:
            records = query_by_keyword(kw, enable_log=False)
        except Exception as mysql_error:  # pylint: disable=broad-except
            print(f"❌ MySQL 查询失败（关键词: {kw}）：{mysql_error}")
            continue
        for record in records:
            record_id = record.get("id")
            if record_id is not None:
                dedup_key = (record.get("_src", "s"), record_id)
                if dedup_key not in seen_ids:
                    seen_ids.add(dedup_key)
                    all_results.append(record)
            else:
                all_results.append(record)

    if all_results:
        # 按关键词命中数降序重排，保证最相关记录出现在 top-20 截断之前
        if len(keywords) >= 2:
            def _kw_score(record: Dict[str, object]) -> int:
                hit_text = " ".join(
                    str(record.get(f, "")) for f in ("item", "category", "operation", "channel")
                )
                return sum(1 for kw in keywords if kw in hit_text)
            all_results.sort(key=_kw_score, reverse=True)
        print(f"ℹ️ 结构化多关键词检索命中 {len(all_results)} 条（关键词: {keywords[:5]}）")
        return all_results

    # ===== 阶段2：降级 - 基于去虚词核心文本的前后缀截断（兜底） =====
    # 关键：用去虚词后的核心文本做截断，避免"介绍一下"等虚词短语误命中
    compact = cleaned.replace(" ", "")
    core = _strip_stopwords(compact) if compact else ""
    fallback_base = core if core and len(core) >= 2 else compact

    if fallback_base:
        for span in range(len(fallback_base) - 1, 1, -1):
            prefix = fallback_base[:span]
            if len(prefix) < 2:
                continue
            try:
                records = query_by_keyword(prefix, enable_log=False)
            except Exception:  # pylint: disable=broad-except
                continue
            if records:
                print(f"ℹ️ 结构化检索降级命中，使用前缀关键词[{prefix}]")
                return records
        for span in range(len(fallback_base) - 1, 1, -1):
            suffix = fallback_base[-span:]
            if len(suffix) < 2:
                continue
            try:
                records = query_by_keyword(suffix, enable_log=False)
            except Exception:  # pylint: disable=broad-except
                continue
            if records:
                print(f"ℹ️ 结构化检索降级命中，使用后缀关键词[{suffix}]")
                return records

    return []


def _expand_vector_results_by_source(
    vector_results: List[Dict[str, object]],
    max_source_pdfs: int = 4,
) -> List[Dict[str, object]]:
    """补全同一 PDF 文件的所有分块，解决人物档案等单文档查询遗漏问题。

    当向量检索（top_k 限制）仅命中某 PDF 的部分分块时，
    通过 Milvus 标量过滤把该文件的其余分块（如近期工作经历）追加到结果中。
    仅对命中 PDF 数量 ≤ max_source_pdfs 的查询执行扩展，避免泛化查询膨胀上下文。
    """
    if not vector_results:
        return vector_results

    # 提取命中分块对应的 PDF 文件名（来源格式："文件名.pdf_第N页_子块M"）
    # 使用正则匹配 .pdf 后跟 _第N页，避免文件名本身含 _第 时误切割
    _PDF_NAME_RE = re.compile(r"^(.+\.pdf)(?:_第\d+页|$)", re.IGNORECASE)
    source_pdf_map: dict[str, bool] = {}
    for r in vector_results:
        src = str(r.get("source", ""))
        m = _PDF_NAME_RE.match(src)
        pdf_name = m.group(1) if m else src
        if pdf_name:
            source_pdf_map[pdf_name] = True

    if not source_pdf_map or len(source_pdf_map) > max_source_pdfs:
        return vector_results

    seen_texts: set[str] = {str(r.get("text", "")) for r in vector_results}
    expanded = list(vector_results)

    for pdf_name in source_pdf_map:
        try:
            extra_chunks = fetch_chunks_by_source_prefix(pdf_name)
        except Exception:  # pylint: disable=broad-except
            continue
        for chunk in extra_chunks:
            chunk_text = str(chunk.get("text", ""))
            if chunk_text and chunk_text not in seen_texts:
                seen_texts.add(chunk_text)
                expanded.append(chunk)

    added = len(expanded) - len(vector_results)
    if added > 0:
        print(f"ℹ️ 来源扩展：补充 {added} 个同源分块（PDF: {list(source_pdf_map.keys())}）")
    return expanded


def rag_query(query_text: str, top_k: int = 3) -> Dict[str, List[Dict[str, object]]]:
    """执行 RAG 查询，融合 MySQL 结构化数据与 Milvus 向量数据。

    参数:
        query_text: 用户查询文本，将被自动编码为向量。
        top_k: 非结构化检索返回的最大结果数量，默认 3。

    返回:
        Dict[str, List[Dict[str, object]]]: 包含 structured 与 unstructured 两类结果的字典。
    """
    normalized_query = (query_text or "").strip()
    if not normalized_query:
        print("⚠️ 查询文本为空，请输入有效问题。")
        return {"structured": [], "unstructured": []}

    try:
        query_vector = _vectorize_text(normalized_query)
    except RuntimeError as model_error:
        print(f"❌ 向量生成失败：{model_error}")
        return {"structured": [], "unstructured": []}

    # 并行执行 MySQL 结构化检索和 Milvus 向量检索，显著减少等待时间
    # 效率优化：复用模块级线程池 _RAG_EXECUTOR，避免每次查询创建/销毁线程池
    future_milvus = _RAG_EXECUTOR.submit(search_similar_vector, query_vector, top_k)
    future_mysql = _RAG_EXECUTOR.submit(_structured_query_with_fallback, normalized_query)

    try:
        vector_results = future_milvus.result(timeout=15)
    except Exception as milvus_error:  # pylint: disable=broad-except
        print(f"❌ Milvus 检索失败：{milvus_error}")
        vector_results = []

    try:
        structured_results = future_mysql.result(timeout=15)
    except Exception as mysql_error:  # pylint: disable=broad-except
        print(f"❌ MySQL 检索失败：{mysql_error}")
        structured_results = []

    # 来源扩展：若向量结果集中在少数 PDF，补全该文件的所有分块
    vector_results = _expand_vector_results_by_source(vector_results)

    return {"structured": structured_results, "unstructured": vector_results}


def _print_structured_section(records: List[Dict[str, object]]) -> None:
    """以易读格式打印结构化查询结果。"""
    if not records:
        print("- 未检索到结构化数据。")
        return
    for index, record in enumerate(records, start=1):
        print(f"- 结构化数据{index}：")
        print(f"  类别：{record.get('category', '-')}")
        print(f"  事项：{record.get('item', '-')}")
        print(f"  操作说明：{record.get('operation', '-')}")
        print(f"  时间要求：{record.get('time_requirement', '-')}")
        print(f"  办理渠道：{record.get('channel', '-')}")
        print(f"  信息来源：{record.get('source_note', '-')}")


def _print_unstructured_section(records: List[Dict[str, object]]) -> None:
    """以易读格式打印 Milvus 向量检索结果。"""
    if not records:
        print("- 未检索到非结构化数据。")
        return
    for index, record in enumerate(records, start=1):
        preview = str(record.get("text", ""))[:80]
        score = record.get("score", "N/A")
        formatted_score = f"{score:.4f}" if isinstance(score, (int, float)) else score
        print(
            f"- 向量数据{index}：来源 {record.get('source', '-')}; "
            f"相似度得分（越小越相似） {formatted_score}"
        )
        print(f"  文本摘录：{preview}...")


# ===================== 阶段4：上下文融合函数 =====================
def fuse_context(
    structured_data: List[Dict[str, object]],
    unstructured_data: List[Dict[str, object]]
) -> str:
    """
    融合结构化与非结构化检索结果，生成Prompt可用的上下文文本。

    功能说明：
        将MySQL结构化数据（事务办理指南）与Milvus向量数据（PDF文档片段）
        整合为统一格式的上下文字符串，供大模型理解和回答。

    参数:
        structured_data: MySQL结构化检索结果列表，每条包含category/item/operation等字段
        unstructured_data: Milvus向量检索结果列表，每条包含text/source/score字段

    返回:
        str: 融合后的上下文文本，若无数据则返回空字符串
    """
    context_parts = []  # 存储所有上下文片段

    # ===== 优先处理非结构化数据（Milvus向量检索的PDF片段） =====
    # Milvus 结果按语义相似度排序，通常包含更完整、更相关的内容（如教师简历、手册原文）
    if unstructured_data:
        context_parts.append("【校园文档资料（向量检索）】")

        # 将同一 PDF 的多个分块合并为一段连续文本，避免 LLM 因块边界误解上下文
        # 例如：简历中"2026.01-至今："与职位名称跨块时，合并后 LLM 可读到完整信息
        _pdf_src_re = re.compile(r"^(.+\.pdf)(?:_第\d+页|$)", re.IGNORECASE)

        def _numeric_sort_key(record: Dict[str, object]) -> tuple:
            """提取来源字符串中的所有数字用于数值排序（保证第2页 < 第10页）。"""
            nums = re.findall(r"\d+", str(record.get("source", "")))
            return tuple(int(n) for n in nums)

        _pdf_groups: dict[str, List[Dict[str, object]]] = {}
        _other_records: List[Dict[str, object]] = []
        for record in unstructured_data:
            src = str(record.get("source", ""))
            m = _pdf_src_re.match(src)
            if m:
                base_pdf = m.group(1)
                _pdf_groups.setdefault(base_pdf, []).append(record)
            else:
                _other_records.append(record)

        idx = 1
        for base_pdf, chunks in _pdf_groups.items():
            chunks.sort(key=_numeric_sort_key)
            merged_text = "\n".join(
                str(c.get("text", "")).strip() for c in chunks if str(c.get("text", "")).strip()
            )
            context_parts.append(f"\n{idx}. 来源：{base_pdf}\n   内容：{merged_text}")
            idx += 1

        for record in _other_records:
            text = record.get("text", "")
            source = record.get("source", "未知来源")
            context_parts.append(f"\n{idx}. 来源：{source}\n   内容：{text}")
            idx += 1

    # ===== 补充结构化数据（MySQL事务办理指南 + 校园新闻） =====
    if structured_data:
        context_parts.append("\n【校园事务办理指南与新闻（结构化检索）】")
        max_struct = getattr(RAGConfig, "STRUCTURED_MAX_RESULTS", len(structured_data))
        # 操作说明截断上限：事务流程保留更多细节，新闻正文截取摘要即可
        _MAX_OP_PROCEDURE = 1500  # campus_struct_data：完整流程说明
        _MAX_OP_NEWS = 500        # campus_structured_data：新闻正文只取摘要
        for idx, record in enumerate(structured_data[:max_struct], start=1):
            # 提取各字段，缺失字段使用默认值
            category = record.get("category", "未分类")
            item = record.get("item", "未知事项")
            operation = record.get("operation", "暂无操作说明")
            time_req = record.get("time_requirement", "暂无时间要求")
            channel = record.get("channel", "暂无办理渠道")
            source = record.get("source_note", "暂无来源")

            # 按记录类型选择截断上限：新闻正文内容较长，只取摘要
            _max_op = _MAX_OP_NEWS if category == "校园新闻" else _MAX_OP_PROCEDURE
            if len(operation) > _max_op:
                operation = operation[:_max_op] + "…"

            # 格式化单条结构化数据
            structured_text = (
                f"\n{idx}. 【{category}】{item}\n"
                f"   操作说明：{operation}\n"
                f"   时间要求：{time_req}\n"
                f"   办理渠道：{channel}\n"
                f"   信息来源：{source}"
            )
            context_parts.append(structured_text)

    # 拼接所有上下文片段
    fused_context = "\n".join(context_parts)
    return fused_context.strip()


# ===================== 阶段4：端到端RAG入口函数（毕设核心） =====================
def stage4_rag_query(query: str, model_config: dict | None = None, history: list | None = None) -> str:
    """
    端到端RAG查询入口（毕设演示主调用函数）。

    功能说明：
        本函数是RAG系统的唯一对外接口，实现完整的「检索-融合-生成」流程：
        1. 生成查询向量（复用阶段1-3的向量化逻辑）
        2. 双源检索（MySQL结构化 + Milvus向量）
        3. 上下文融合（将检索结果转为Prompt可用格式）
        4. Prompt拼接（使用RAGConfig中的模板）
        5. 调用线上大模型生成回答

    参数:
        query: 用户输入的自然语言问题（如"缓考申请流程"）
        model_config: 可选的模型配置字典
        history: 可选的多轮对话历史列表，支持用户连续追问

    返回:
        str: 大模型生成的回答文本，或兜底提示信息

    异常处理：
        所有异常均在内部捕获，返回友好的兜底提示，确保演示不中断
    """
    t_pipeline_start = time.time()
    print("\n" + "=" * 60)
    print("🚀 【阶段4】端到端RAG问答流程启动")
    print("=" * 60)

    # ===== Step 0: 输入校验 =====
    normalized_query = (query or "").strip()
    if not normalized_query:
        print("⚠️ 用户提问为空，返回兜底提示")
        return RAGConfig.FALLBACK_MESSAGE

    print(f"📝 用户提问：{normalized_query}")

    try:
        # ===== Step 1: 双源检索（复用rag_query函数） =====
        print("\n🔍 【Step 1/4】开始双源检索...")
        print(f"   - MySQL结构化检索：关键词匹配")
        print(f"   - Milvus向量检索：Top-{RAGConfig.VECTOR_TOP_K}相似文档")

        # 调用阶段1-3实现的检索函数
        t_retrieval = time.time()
        rag_results = rag_query(normalized_query, top_k=RAGConfig.VECTOR_TOP_K)
        t_retrieval_cost = time.time() - t_retrieval

        structured_data = rag_results.get("structured", [])
        unstructured_data = rag_results.get("unstructured", [])

        # 打印检索结果统计
        print(f"   ✅ 检索到 {len(structured_data)} 条结构化数据（MySQL）")
        print(f"   ✅ 检索到 {len(unstructured_data)} 条非结构化数据（Milvus）")
        print(f"   ⏱️ 双源检索耗时：{t_retrieval_cost:.3f}s")

        # ===== Step 2: 上下文融合 =====
        print("\n📋 【Step 2/4】融合检索结果为上下文...")
        t_fuse = time.time()
        fused_context = fuse_context(structured_data, unstructured_data)
        t_fuse_cost = time.time() - t_fuse

        # 检索兜底：无任何数据时直接返回兜底提示，不调用线上模型
        if not fused_context or not fused_context.strip():
            print("⚠️ 检索结果为空，无相关校园资料，返回兜底提示")
            print("   💡 提示：不调用线上模型，避免无依据回答")
            return RAGConfig.FALLBACK_MESSAGE

        print(f"   ✅ 上下文融合完成，共 {len(fused_context)} 字符，耗时：{t_fuse_cost:.3f}s")

        # ===== Step 3: Prompt拼接 =====
        print("\n📝 【Step 3/4】拼接Prompt模板...")
        t_prompt = time.time()
        try:
            # 使用RAGConfig中的Prompt模板，替换占位符
            final_prompt = RAGConfig.PROMPT_TEMPLATE.format(
                context=fused_context,
                question=normalized_query
            )
            print(f"   ✅ Prompt拼接完成，共 {len(final_prompt)} 字符")
        except KeyError as template_error:
            print(f"❌ Prompt模板占位符错误：{template_error}")
            return RAGConfig.FALLBACK_MESSAGE
        t_prompt_cost = time.time() - t_prompt
        print(f"   ⏱️ Prompt拼接耗时：{t_prompt_cost:.3f}s")

        # ===== Step 4: 调用线上大模型生成回答 =====
        print("\n🤖 【Step 4/4】调用线上大模型生成回答...")
        t_llm = time.time()
        answer = generate_answer(final_prompt, model_config=model_config, history=history)
        t_llm_cost = time.time() - t_llm

        t_total = time.time() - t_pipeline_start
        print("\n" + "=" * 60)
        print("✅ 【阶段4】端到端RAG问答流程完成")
        print(f"   ⏱️ 双源检索: {t_retrieval_cost:.3f}s | 上下文融合: {t_fuse_cost:.3f}s | Prompt拼接: {t_prompt_cost:.3f}s | LLM生成: {t_llm_cost:.3f}s | 总耗时: {t_total:.3f}s")
        print("=" * 60)

        return answer

    except Exception as e:
        # ===== 全局异常捕获（确保演示不中断） =====
        print(f"\n❌ RAG流程执行异常：{e}")
        print("   💡 提示：请检查Milvus/MySQL服务是否正常运行")
        return RAGConfig.FALLBACK_MESSAGE


def stage4_rag_query_stream(query: str, model_config: dict | None = None, history: list | None = None):
    """
    流式端到端RAG查询入口。
    
    与 stage4_rag_query 功能一致，但第4步使用流式生成，逐块 yield 文本。
    检索阶段（Step 1-3）同步完成，生成阶段（Step 4）流式输出。

    参数:
        query: 用户输入的自然语言问题
        model_config: 可选的模型配置字典
        history: 可选的多轮对话历史列表，支持用户连续追问

    Yields:
        str: 每次生成的文本片段
    """
    t_pipeline_start = time.time()
    print("\n" + "=" * 60)
    print("🚀 【阶段4-流式】端到端RAG问答流程启动")
    print("=" * 60)

    normalized_query = (query or "").strip()
    if not normalized_query:
        print("⚠️ 用户提问为空，返回兜底提示")
        yield RAGConfig.FALLBACK_MESSAGE
        return

    print(f"📝 用户提问：{normalized_query}")

    try:
        # ===== Step 1: 双源检索 =====
        print("\n🔍 【Step 1/4】开始双源检索...")
        t_retrieval = time.time()
        rag_results = rag_query(normalized_query, top_k=RAGConfig.VECTOR_TOP_K)
        t_retrieval_cost = time.time() - t_retrieval
        structured_data = rag_results.get("structured", [])
        unstructured_data = rag_results.get("unstructured", [])
        print(f"   ✅ 检索到 {len(structured_data)} 条结构化数据（MySQL）")
        print(f"   ✅ 检索到 {len(unstructured_data)} 条非结构化数据（Milvus）")
        print(f"   ⏱️ 双源检索耗时：{t_retrieval_cost:.3f}s")

        # ===== Step 2: 上下文融合 =====
        print("\n📋 【Step 2/4】融合检索结果为上下文...")
        fused_context = fuse_context(structured_data, unstructured_data)

        if not fused_context or not fused_context.strip():
            print("⚠️ 检索结果为空，返回兜底提示")
            yield RAGConfig.FALLBACK_MESSAGE
            return

        print(f"   ✅ 上下文融合完成，共 {len(fused_context)} 字符")

        # ===== Step 3: Prompt拼接 =====
        print("\n📝 【Step 3/4】拼接Prompt模板...")
        try:
            final_prompt = RAGConfig.PROMPT_TEMPLATE.format(
                context=fused_context,
                question=normalized_query
            )
            print(f"   ✅ Prompt拼接完成，共 {len(final_prompt)} 字符")
        except KeyError as template_error:
            print(f"❌ Prompt模板占位符错误：{template_error}")
            yield RAGConfig.FALLBACK_MESSAGE
            return

        # ===== Step 4: 流式调用线上大模型 =====
        print("\n🤖 【Step 4/4】流式调用线上大模型生成回答...")
        for chunk in generate_answer_stream(final_prompt, model_config=model_config, history=history):
            yield chunk

        t_total = time.time() - t_pipeline_start
        print("\n" + "=" * 60)
        print("✅ 【阶段4-流式】端到端RAG问答流程完成")
        print(f"   ⏱️ 双源检索: {t_retrieval_cost:.3f}s | 总耗时: {t_total:.3f}s")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ RAG流程执行异常：{e}")
        yield RAGConfig.FALLBACK_MESSAGE


if __name__ == "__main__":
    def _parse_cli_query(args: List[str]) -> str:
        """辅助函数：解析命令行参数为查询文本。"""
        if len(args) <= 1:
            return ""
        return " ".join(part for part in args[1:] if part.strip()).strip()

    cli_query = _parse_cli_query(sys.argv)
    if not cli_query:
        try:
            cli_query = input("请输入查询关键词（示例：缓考申请流程）：").strip()
        except EOFError:
            cli_query = ""

    if not cli_query:
        print("⚠️ 未提供查询关键词，已结束脚本。")
        sys.exit(0)

    print(f"===== RAG 查询演示：{cli_query} =====")
    try:
        rag_results = rag_query(cli_query)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"❌ RAG 查询执行失败：{exc}")
    else:
        structured = rag_results.get("structured", [])
        unstructured = rag_results.get("unstructured", [])

        if not structured and not unstructured:
            print("⚠️ 未检索到相关数据，请检查数据库与向量库是否已初始化。")
        else:
            print("\n【结构化数据】")
            _print_structured_section(structured)

            print("\n【非结构化数据】")
            _print_unstructured_section(unstructured)

    print("===== 查询结束 =====")
