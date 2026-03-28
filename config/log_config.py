import logging
import os
from logging.handlers import RotatingFileHandler


def setup_logging(log_dir: str = "logs",
                  app_log_name: str = "app.log",
                  error_log_name: str = "error.log",
                  level: int = logging.INFO) -> None:
    """
    配置项目的日志记录。

    功能说明：
    - 在项目根目录下的 logs/ 目录输出运行日志(app.log)与错误日志(error.log)。
    - 采用 RotatingFileHandler 按大小轮转，避免日志文件过大。
    - 控制台输出与文件输出均启用，便于开发与部署排查问题。

    参数:
        log_dir (str): 日志目录，默认 "logs"。
        app_log_name (str): 运行日志文件名，默认 "app.log"。
        error_log_name (str): 错误日志文件名，默认 "error.log"。
        level (int): 根日志级别，默认 logging.INFO。

    适配说明：
    - 目录结构已固定为 CampusKnowledge_QASystem/logs/，本函数会在目录不存在时自动创建。
    - 仅使用 Python 标准库 logging，无额外依赖，适配本科毕设规范。
    """

    # 若重复调用，避免重复添加 Handler
    if getattr(setup_logging, "_configured", False):
        return

    os.makedirs(log_dir, exist_ok=True)

    # 日志格式：包含时间、级别、模块、消息，便于回溯问题
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 控制台输出（适合开发调试）
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    # 运行日志文件，记录 INFO 及以上
    app_log_path = os.path.join(log_dir, app_log_name)
    app_file_handler = RotatingFileHandler(
        app_log_path, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    app_file_handler.setLevel(level)
    app_file_handler.setFormatter(formatter)

    # 错误日志文件，记录 WARNING 及以上（包含 ERROR、CRITICAL）
    error_log_path = os.path.join(log_dir, error_log_name)
    error_file_handler = RotatingFileHandler(
        error_log_path, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    error_file_handler.setLevel(logging.WARNING)
    error_file_handler.setFormatter(formatter)

    # 根记录器配置
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(app_file_handler)
    root_logger.addHandler(error_file_handler)

    setup_logging._configured = True
