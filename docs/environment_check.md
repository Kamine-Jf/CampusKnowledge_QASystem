# CampusKnowledge_QASystem 环境巡检指引

## Python 与核心依赖
- 推荐使用 Python 3.11.8 版本，与项目调试环境一致。
- 关键依赖版本（详见 requirements.txt）：
  - pymysql==1.1.0
  - pymilvus==2.2.16、milvus==2.2.16
  - langchain==0.1.10、langchain-community==0.0.28
  - torch==2.2.1+cu121（RTX 3050Ti 4GB 适配版）
  - python-dotenv==1.0.1

## MySQL
- 默认配置位于 src/database/mysql_config.py，可通过 .env 覆盖。
- 环境巡检脚本会自动读取配置并执行 SELECT 1 验证。

## Milvus 服务
- 在本地或服务器上启动 Milvus 2.2.16（默认监听 19530）。
- env_check.py 会连接目标服务，创建临时集合完成插入/搜索并自动清理。

## LangChain
- 巡检脚本通过 LCEL 管线验证核心组件加载是否正常。

## CUDA/GPU
- 脚本会输出 CUDA 可用性、显卡型号与显存信息。
- 至少保证一块显存≥4GB 的 GPU（RTX 3050Ti 4G）以运行量化模型。

## 运行环境巡检
```
python test/env_check.py
```
- 全部检测通过时返回 0，并显示“全部核心检测通过”。
- 若出现失败提示，按输出的中文指引排查对应模块。
