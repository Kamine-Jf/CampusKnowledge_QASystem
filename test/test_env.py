import gradio as gr
from dotenv import load_dotenv
from loguru import logger
from prettytable import PrettyTable

# 测试各库基础功能
logger.info("✅ 阶段4新增依赖加载成功")
tb = PrettyTable(["模块", "状态"])
tb.add_row(["Gradio", "可用"])
tb.add_row(["python-dotenv", "可用"])
tb.add_row(["loguru", "可用"])
tb.add_row(["prettytable", "可用"])
print(tb)
load_dotenv()  # 测试配置文件加载
print("✅ 环境层准备完成！")