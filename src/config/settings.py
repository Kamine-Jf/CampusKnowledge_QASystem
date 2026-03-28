"""项目全局配置汇总（模型/数据路径/Milvus参数）。"""  # 匹配度优化：新增全局配置文件，集中管理关键参数

from __future__ import annotations  # 匹配度优化：保证类型注解在低版本运行一致

from dataclasses import dataclass  # 匹配度优化：使用数据类承载配置项
from pathlib import Path  # 匹配度优化：统一路径处理，兼容 Windows

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # 匹配度优化：自动定位项目根目录，避免硬编码路径


@dataclass(frozen=True)
class ModelConfig:
    """向量模型相关配置。"""  # 匹配度优化：集中管理模型配置

    MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"  # 匹配度优化：统一模型名称
    MODEL_CACHE_PATH: Path = PROJECT_ROOT / "models"  # 匹配度优化：统一模型缓存目录
    MODEL_DEVICE: str = "cpu"  # 匹配度优化：强制 CPU 推理，适配显存限制
    HF_ENDPOINT: str = "https://hf-mirror.com"  # 匹配度优化：使用 Hugging Face 国内镜像


@dataclass(frozen=True)
class PdfConfig:
    """PDF 解析与分块配置。"""  # 匹配度优化：集中管理 PDF 路径与分块阈值

    PDF_DIR: Path = PROJECT_ROOT / "data" / "unstructured_data"  # 匹配度优化：统一 PDF 数据目录
    MIN_CHARS_PER_CHUNK: int = 300  # 匹配度优化：分块最小字符数，保证语义完整
    MAX_CHARS_PER_CHUNK: int = 500  # 匹配度优化：分块最大字符数，控制向量精度
    MIN_VALID_CHARS: int = 20  # 匹配度优化：过滤无意义短文本块


@dataclass(frozen=True)
class MilvusConfig:
    """Milvus 连接与检索配置。"""  # 匹配度优化：集中管理 Milvus 参数

    MILVUS_HOST: str = "localhost"  # 匹配度优化：统一 Milvus 主机配置
    MILVUS_PORT: int = 19530  # 匹配度优化：统一 Milvus 端口配置
    MILVUS_COLLECTION_NAME: str = "campus_qa_vector"  # 匹配度优化：统一集合名称
    VECTOR_DIM: int = 768  # 匹配度优化：向量维度保持 768
    MILVUS_NLIST: int = 256  # 匹配度优化：小数据量场景提升索引精度
    MILVUS_NPROBE: int = 128  # 匹配度优化：提升检索覆盖率，减少漏召
    MILVUS_METRIC_TYPE: str = "L2"  # 匹配度优化：与检索距离度量保持一致
    IVF_INDEX_TYPE: str = "IVF_FLAT"  # 匹配度优化：精度优先的索引类型



# ===================== RAG检索增强生成配置 =====================
@dataclass(frozen=True)
class RAGConfig:
    """RAG（检索增强生成）核心配置，集中管理检索参数与Prompt模板。"""

    # 检索参数配置
    VECTOR_TOP_K: int = 10  # Milvus向量检索返回的最大结果数量（提升至10覆盖多维度问题）
    STRUCTURED_MAX_RESULTS: int = 20  # MySQL结构化检索最大结果数量（多关键词全面检索后取Top-N）
    STRUCTURED_PER_KEYWORD_MAX: int = 50  # 单关键词最多贡献记录数上限（实际由总量 STRUCTURED_MAX_RESULTS 控制）

    # 兜底提示语（检索无结果时直接返回，不调用线上模型）
    FALLBACK_MESSAGE: str = "未查询到该问题的校园官方资料，无法为你解答"

    # ===================== Prompt模板（校园问答专属） =====================
    # 模板说明：
    #   - {context}: 由fuse_context函数生成的上下文（结构化+非结构化数据融合）
    #   - {question}: 用户原始提问
    # 设计原则：
    #   - 明确告知模型只能基于提供的资料回答
    #   - 要求分点简洁说明，适合毕设演示
    #   - 无相关资料时明确告知用户
    PROMPT_TEMPLATE: str = """你是郑州轻工业大学校园智能问答助手，严格基于以下校园官方资料回答问题。

【校园资料】
{context}

【用户问题】
{question}

【回答要求】
1. 仔细阅读全部校园资料，不得跳过或忽略任何与问题相关的段落
2. 优先依据"校园文档资料"给出详细回答，它通常包含最完整的信息
3. 同时参考"结构化检索"中的事务指南和新闻，作为补充信息一并纳入回答
4. 资料中明确提到的具体数字、日期、步骤、条件、联系方式，必须完整引用，不得省略
5. 多维度问题需逐一分点回答每个维度，每点须充分展开，不因内容较多而压缩或合并
6. 仅当资料中确实没有相关信息时，才回复"未查询到该问题的校园官方资料"，不可因回答较长就提前截断
7. 涉及人物时，必须严格依据资料中明确标注的性别信息使用正确的人称代词（男性用"他"，女性用"她"），资料未注明性别时直接使用姓名，禁止自行猜测性别

请基于以上全部资料，给出完整、准确的回答："""


# ===================== MySQL数据库配置 =====================
@dataclass(frozen=True)
class MySQLConfig:
    """MySQL数据库连接配置。"""

    HOST: str = "localhost"
    PORT: int = 3306
    USER: str = "root"
    PASSWORD: str = "123456"
    DATABASE: str = "campus_qa"
    CHARSET: str = "utf8mb4"