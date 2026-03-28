"""项目配置包初始化。"""  # 匹配度优化：新增配置包说明，便于统一导入

from .settings import (  # 匹配度优化：导出全局配置，避免散落硬编码
	MilvusConfig,
	ModelConfig,
	PdfConfig,
)

__all__ = [
	"MilvusConfig",
	"ModelConfig",
	"PdfConfig",
]  # 匹配度优化：显式暴露配置类
