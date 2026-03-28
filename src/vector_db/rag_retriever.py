"""RAG 检索融合模块。"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

from src.database.db_operate import query_by_keyword

from .milvus_operate import search_similar_vector as search_similar_vectors
from .vector_generator import text_to_vector

# 效率优化：模块级线程池，复用线程避免每次检索创建/销毁开销
_HYBRID_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="hybrid_retrieval")

# 中文常见虚词/疑问词，用于从用户查询中提取实义关键词
_STOPWORDS = (
    "的", "了", "在", "是", "我", "有", "和", "就",
    "不", "人", "都", "一", "一个", "上", "也", "很",
    "到", "说", "要", "去", "你", "会", "着", "没有",
    "看", "好", "自己", "这", "他", "她", "它",
    "吗", "吧", "呢", "啊", "呀", "哦", "嗯",
    "什么", "哪些", "哪个", "哪里", "怎么", "怎样", "如何",
    "为什么", "多少", "几个", "能否", "是否", "可以",
    "请问", "请", "想", "想要", "需要", "可以",
    "告诉", "介绍", "一下", "关于",
    "知道", "了解", "查找", "搜索", "找",
    # 通用上下文词：单独检索时噪声大，应去除以突出核心实体
    "老师", "同学", "教授", "导师", "学生", "学校",
    "资料", "信息", "详情", "情况", "内容", "简介",
    "相关", "全部", "所有", "具体", "查询", "查看",
)
_STOPWORD_PATTERN = "|".join(re.escape(w) for w in sorted(_STOPWORDS, key=len, reverse=True))
_STOPWORDS_SET = frozenset(_STOPWORDS)

# ===================== jieba 分词初始化（模块级，只执行一次） =====================
_JIEBA_INITIALIZED = False


def _ensure_jieba():
    """懒加载 jieba 并注册校园领域词典，确保只初始化一次。"""
    global _JIEBA_INITIALIZED  # pylint: disable=global-statement
    if _JIEBA_INITIALIZED:
        return

    import jieba

    _CAMPUS_DICT_WORDS = [
        "教务处", "学生处", "招生办", "就业指导中心", "后勤处", "保卫处", "财务处",
        "图书馆", "实验室", "研究生院", "国际交流处", "团委", "学工部", "科研处",
        "软件学院", "计算机学院", "电气学院", "机电学院", "食品学院", "材料学院",
        "经济管理学院", "艺术设计学院", "外国语学院", "数学学院", "物理学院",
        "马克思主义学院", "体育学院", "能源学院", "建筑环境工程学院", "法学院",
        "郑州轻工业大学", "轻工大", "轻院",
        "学校简介", "学院简介", "专业简介", "学校概况", "学校历史",
        "缓考", "补考", "重修", "选课", "退课", "休学", "复学", "转专业",
        "学分", "绩点", "成绩查询", "成绩单", "考试安排", "教学评价",
        "毕业设计", "毕业论文", "毕业答辩", "学位证", "毕业证",
        "教务系统", "教务管理", "课程表", "培养方案",
        "助学金", "奖学金", "国家助学金", "国家奖学金", "励志奖学金",
        "助学贷款", "学费减免", "勤工助学", "困难认定",
        "学生证", "校园卡", "一卡通", "宿舍", "寝室",
        "入学", "报到", "注册", "档案", "户口",
        "社团", "学生会", "志愿者", "社会实践", "创新创业",
        "校医院", "心理咨询", "体育场", "食堂", "澡堂", "快递",
        "校园网", "VPN", "正方教务",
        "申请流程", "办理流程", "报名", "审批", "提交材料",
    ]
    for word in _CAMPUS_DICT_WORDS:
        jieba.add_word(word, freq=50000)

    _JIEBA_INITIALIZED = True


# jieba 分词时保留的词性集合（名词、动词、专名等实义词）
_USEFUL_POS_TAGS = frozenset({
    "n", "nr", "ns", "nt", "nz", "nl", "ng",
    "v", "vn", "vd", "vg",
    "a", "an", "ag",
    "j", "i", "l", "eng", "x",
})


def _multi_keyword_query(text: str) -> List[Dict[str, object]]:
    """从用户查询中提取多个关键词，分别检索MySQL结构化数据并合并去重。

    使用 jieba 词性标注分词提取实义关键词，替代原始滑动窗口方式，
    大幅减少无意义碎片关键词，提升检索精度。
    """
    import jieba.posseg as pseg

    _ensure_jieba()

    compact = text.strip().replace(" ", "")
    # 使用 jieba token 级停用词过滤，避免正则直接替换破坏"学校简介"等复合领域词
    _core_tokens = [w for w, _ in pseg.lcut(compact) if w.strip() and w not in _STOPWORDS_SET]
    core = "".join(_core_tokens).strip() or compact

    keywords: List[str] = []
    seen_kw: set[str] = set()

    def _add(kw: str) -> None:
        kw = kw.strip()
        if len(kw) >= 2 and kw not in seen_kw:
            seen_kw.add(kw)
            keywords.append(kw)

    # 1. 去虚词后的核心文本整体作为首选关键词
    _add(core)

    # 2. 按标点/连接词拆分为子短语
    segments = re.split(r"[，,。.、；;！!？?和与及或还有以及跟同]", core)
    for seg in segments:
        _add(seg.strip())

    # 3. 用 jieba 词性标注分词，提取实义词
    #    关键优化：合并连续单字符 token 为复合词（处理人名等未登录词）
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

        # 4. 优先添加 bigram 复合词，再添加单词（确保"学校简介"优先于"学校""简介"）
        for i in range(len(seg_words) - 1):
            _add(seg_words[i] + seg_words[i + 1])
        for word in seg_words:
            _add(word)

    # 5. 原始输入兜底
    _add(compact)

    # 6. 从原始文本（未过滤停用词）生成 bigram，覆盖跨停用词的复合短语
    orig_tokens_raw = [w for w, _ in pseg.lcut(compact) if w.strip()]
    for _i in range(len(orig_tokens_raw) - 1):
        _bigram = orig_tokens_raw[_i].strip() + orig_tokens_raw[_i + 1].strip()
        _add(_bigram)

    all_results: List[Dict[str, object]] = []
    seen_ids: set = set()
    soft_cap = 60
    for kw in keywords:
        if len(all_results) >= soft_cap:
            break
        try:
            records = query_by_keyword(kw, enable_log=False)
        except Exception:  # pylint: disable=broad-except
            continue
        for record in records:
            rid = record.get("id")
            if rid is not None:
                dedup_key = (record.get("_src", "s"), rid)
                if dedup_key not in seen_ids:
                    seen_ids.add(dedup_key)
                    all_results.append(record)
            else:
                all_results.append(record)
    return all_results


def hybrid_retrieve(query_text: str, top_k: int = 5) -> List[Dict[str, object]]:
    """执行 RAG 双源融合检索。

    功能:
        将查询文本向量化并在 Milvus 中检索非结构化片段，同时调用 MySQL 结构化查询，最终融合两类结果。
    参数:
        query_text (str): 用户输入的检索文本。
        top_k (int): 向量检索返回的最大数量，默认 5。
    返回值:
        List[Dict[str, object]]: 按内容去重后的融合结果列表，统一包含 data_source、content、source、score、metadata 键。
    异常:
        ValueError: 当查询文本为空时抛出。
        RuntimeError: 当向量检索或 MySQL 查询失败时抛出。
    """
    if not query_text or not query_text.strip():
        raise ValueError("查询文本不能为空。")

    cleaned_text = query_text.strip()

    # 效率优化：先向量化（必须串行），再并行执行 Milvus + MySQL 检索
    try:
        query_vector = text_to_vector([cleaned_text])[0]
    except Exception as exc:  # pylint: disable=broad-except
        raise RuntimeError(f"向量化失败: {exc}") from exc

    # 并行执行双源检索，显著减少等待时间
    future_milvus = _HYBRID_EXECUTOR.submit(search_similar_vectors, query_vector, top_k)
    future_mysql = _HYBRID_EXECUTOR.submit(_multi_keyword_query, cleaned_text)

    try:
        vector_results = future_milvus.result(timeout=15)
    except Exception as exc:  # pylint: disable=broad-except
        raise RuntimeError(f"Milvus 向量检索执行失败: {exc}") from exc

    try:
        mysql_records = future_mysql.result(timeout=15)
    except Exception as exc:  # pylint: disable=broad-except
        raise RuntimeError(f"MySQL 结构化查询执行失败: {exc}") from exc

    formatted_vector_results: List[Dict[str, object]] = []
    for record in vector_results:
        formatted_vector_results.append(
            {
                "data_source": "milvus",
                "content": record.get("text", ""),
                "source": record.get("source", ""),
                "score": float(record.get("score", 0.0)),
                "metadata": {
                    "match_type": "vector",
                },
            }
        )

    formatted_mysql_results: List[Dict[str, object]] = []
    for record in mysql_records:
        summary = f"{record.get('item', '')}: {record.get('operation', '')}".strip(": ")
        formatted_mysql_results.append(
            {
                "data_source": "mysql",
                "content": summary,
                "source": record.get("category", ""),
                "score": 0.0,
                "metadata": record,
            }
        )

    # 去重并融合
    merged_results: Dict[str, Dict[str, object]] = {}
    for item in formatted_vector_results + formatted_mysql_results:
        content_key = item.get("content", "")
        if not content_key:
            continue
        if content_key not in merged_results:
            merged_results[content_key] = item
        else:
            existing = merged_results[content_key]
            if item["score"] > existing.get("score", 0.0):
                merged_results[content_key] = item

    return list(merged_results.values())
