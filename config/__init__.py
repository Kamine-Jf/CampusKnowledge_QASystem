"""配置模块初始化，方便统一导入日志工具等配置项。"""

from .log_config import setup_logging

__all__ = ["setup_logging"]
