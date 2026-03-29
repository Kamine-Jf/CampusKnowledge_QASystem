# 运行指南 — 校园知识库智能问答系统

> 基于 LangChain + Milvus + MySQL + FastAPI 的 RAG 智能问答系统

---

## 目录

1. [环境要求](#1-环境要求)
2. [依赖安装](#2-依赖安装)
3. [基础服务启动（MySQL & Milvus）](#3-基础服务启动mysql--milvus)
4. [配置说明](#4-配置说明)
5. [数据初始化](#5-数据初始化)
6. [启动后端服务](#6-启动后端服务)
7. [访问系统](#7-访问系统)
8. [Docker 一键部署（可选）](#8-docker-一键部署可选)
9. [常见问题排查](#9-常见问题排查)
10. [项目结构速览](#10-项目结构速览)

---

## 1. 环境要求

| 组件 | 要求 |
|------|------|
| Python | 3.10 或 3.11（推荐 3.11） |
| CUDA | 12.1（使用 GPU 时，对应 `torch==2.2.1+cu121`） |
| MySQL | 8.0+ |
| Milvus | 2.6.x（推荐 Docker 部署） |
| Docker & Docker Compose | 启动 Milvus 独立节点时需要 |
| Git | 克隆仓库用 |

---

## 2. 依赖安装

### 2.1 克隆仓库

```bash
git clone https://github.com/Kamine-Jf/CampusKnowledge_QASystem.git
cd CampusKnowledge_QASystem
```

### 2.2 创建并激活虚拟环境

```bash
# 创建虚拟环境
python -m venv .venv

# Windows 激活
.venv\Scripts\activate

# Linux / macOS 激活
source .venv/bin/activate
```

### 2.3 安装 Python 依赖

```bash
# 标准依赖（使用清华镜像加速）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 如需使用量化本地大模型（Qwen1.5-1.8B-Chat-GPTQ-Int4），额外安装
pip install -r requirements_stage4.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> **注意**：`torch==2.2.1+cu121` 为 CUDA 12.1 版本，仅适配 NVIDIA 显卡。若使用 CPU 环境，请将 `requirements.txt` 中对应行替换为 `torch==2.2.1`。

### 2.4 下载 Embedding 模型

```bash
python download_model.py
```

模型缓存默认存放于 `models/` 目录（`all-MiniLM-L6-v2`），向量维度为 768。

---

## 3. 基础服务启动（MySQL & Milvus）

### 3.1 启动 MySQL

确保本机 MySQL 8.0+ 已运行，默认端口 `3306`。若使用 Docker：

```bash
docker run -d \
  --name campus_mysql \
  -e MYSQL_ROOT_PASSWORD=123456 \
  -e MYSQL_DATABASE=campus_qa_db \
  -p 3306:3306 \
  mysql:8.0
```

### 3.2 启动 Milvus（Docker Standalone 模式）

```bash
# 下载官方独立部署脚本
wget https://github.com/milvus-io/milvus/releases/download/v2.6.9/milvus-standalone-docker-compose.yml \
     -O milvus-standalone.yml

# 启动
docker-compose -f milvus-standalone.yml up -d
```

或直接使用 Docker 命令（若已有 etcd/minio 环境）：

```bash
# 验证 Milvus 是否就绪
docker ps | grep milvus
# 默认服务端口：19530
```

---

## 4. 配置说明

### 4.1 MySQL 连接配置

修改 `src/database/mysql_config.py`：

```python
MYSQL_HOST = "localhost"   # 数据库地址
MYSQL_PORT = 3306          # 端口
MYSQL_USER = "root"        # 用户名
MYSQL_PWD  = "123456"      # 密码（改为实际密码）
MYSQL_DB   = "campus_qa_db"
```

### 4.2 Milvus 连接配置

修改 `src/vector_db/milvus_config.py`：

```python
MILVUS_HOST            = "localhost"
MILVUS_PORT            = 19530
MILVUS_COLLECTION_NAME = "campus_qa_vector"
VECTOR_DIM             = 768
```

### 4.3 LLM / API Key 配置

在项目根目录创建或编辑 `.env` 文件，填入大模型的 API 地址与 Key（如 OpenAI 兼容接口）：

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
```

> 若使用本地量化模型（Qwen1.5-1.8B-Chat-GPTQ-Int4），`.env` 中的 Key 配置可留空，后台通过 `src/llm/` 模块加载本地权重。

---

## 5. 数据初始化

按以下顺序执行（每步确认无报错后再继续）：

### 步骤一：初始化 MySQL 数据表

```bash
# 方法 A：在 MySQL 客户端执行建表 SQL
mysql -u root -p campus_qa_db < src/database/create_tables.sql

# 方法 B：启动后端后自动执行（lifespan 钩子会自动调用 ensure_*_schema()）
```

### 步骤二：导入结构化教务数据（Excel → MySQL）

```bash
python src/database/excel2mysql.py
```

> 将 `data/` 目录下的 Excel 校园数据批量写入 MySQL，可重复执行（幂等）。

### 步骤三：解析 PDF/Word 并生成向量

```bash
python src/vector_db/pdf2vector.py
```

> 解析 `data/unstructured_data/` 下的文档（PDF/Word），输出向量到本地缓存。

### 步骤四：将向量同步至 Milvus

```bash
python -c "from src.rag.rag_core import sync_pdf_vectors_to_milvus; sync_pdf_vectors_to_milvus()"
```

### 步骤五：验证 Milvus 集合与检索

```bash
python src/vector_db/milvus_operate.py
```

### 步骤六（可选）：执行 RAG 检索冒烟测试

```bash
python src/rag/rag_core.py 缓考申请流程
# 替换为任意问题关键词进行测试
```

---

## 6. 启动后端服务

```bash
# 方式一：直接用 Python 启动（开发模式，支持热重载）
python main.py

# 方式二：使用 uvicorn 启动
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

服务启动后控制台输出类似：

```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Database schema initialisation completed.
INFO:     Background warmup triggered at server startup.
```

---

## 7. 访问系统

| 地址 | 说明 |
|------|------|
| `http://localhost:8000` | Web 问答主界面（HTML 前端） |
| `http://localhost:8000/health` | 健康检查接口，返回 `{"status":"ok"}` |
| `http://localhost:8000/docs` | FastAPI 自动生成的 Swagger API 文档 |
| `http://localhost:8000/redoc` | ReDoc 风格 API 文档 |

---

## 8. Docker 一键部署（可选）

项目提供 `docker-compose.yml`，可将后端与前端一并容器化运行：

```bash
# 构建并后台启动
docker-compose up -d --build

# 查看服务状态
docker-compose ps

# 查看后端日志
docker logs campus_qa_backend -f

# 停止所有服务
docker-compose down
```

> **注意**：`docker-compose.yml` 中 MySQL 与 Milvus 默认注释掉，需自行启动宿主机对应服务，或取消注释并补充完整 Milvus standalone 配置后再使用。

---

## 9. 常见问题排查

### Milvus 连接失败

```
ConnectionError: connect to Milvus server failed
```

- 检查 Docker 容器是否运行：`docker ps | grep milvus`
- 检查端口是否占用：`netstat -ano | findstr 19530`（Windows）
- 查看 Milvus 日志：`docker logs milvus-standalone`

### MySQL 连接失败

```
OperationalError: Can't connect to MySQL server
```

- 确认 MySQL 服务已启动（Windows：`net start mysql`；Linux：`systemctl start mysql`）
- 确认 `src/database/mysql_config.py` 中账号密码与实际一致
- 确认数据库 `campus_qa_db` 已创建：`CREATE DATABASE campus_qa_db DEFAULT CHARACTER SET utf8mb4;`

### PDF 解析为空或乱码

- 确保使用文字版 PDF（非扫描图片版）
- 扫描件请先用 OCR 工具（如 Tesseract）转换为文字 PDF

### 模型下载超时

- 离线环境请先在联网环境执行 `python download_model.py` 预拉取
- 可配置 HuggingFace 镜像：`HF_ENDPOINT=https://hf-mirror.com python download_model.py`

### 端口 8000 已被占用

```bash
# Windows：查找占用进程
netstat -ano | findstr 8000
# 换端口启动
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

---

## 10. 项目结构速览

```
CampusKnowledge_QASystem/
├── main.py                      # FastAPI 项目入口
├── requirements.txt             # Python 依赖（主）
├── requirements_stage4.txt      # 量化大模型额外依赖
├── docker-compose.yml           # Docker 部署配置
├── Dockerfile                   # 镜像构建文件
├── .env                         # API Key / 环境变量（勿提交）
├── src/
│   ├── api/                     # FastAPI 路由与应用入口
│   │   ├── main.py              # app 实例、路由注册、lifespan
│   │   └── routes/              # auth / chat / history / upload 等路由
│   ├── database/                # MySQL 连接、建表、数据操作
│   │   ├── mysql_config.py      # 连接配置常量
│   │   ├── create_tables.sql    # 建表 SQL
│   │   ├── excel2mysql.py       # Excel 数据导入
│   │   └── db_operate.py        # 数据 CRUD 操作
│   ├── vector_db/               # Milvus 向量库操作
│   │   ├── milvus_config.py     # 连接配置常量
│   │   ├── pdf2vector.py        # PDF/Word 解析 → 向量生成
│   │   ├── milvus_operate.py    # 集合管理 & 检索测试
│   │   └── rag_retriever.py     # RAG 检索逻辑
│   ├── rag/                     # RAG 核心管道
│   │   └── rag_core.py          # 双源检索 + Prompt 组装 + LLM 调用
│   ├── llm/                     # 大模型加载与推理
│   └── service/                 # 业务服务层
├── static/                      # 前端静态资源（HTML/CSS/JS）
├── data/
│   └── unstructured_data/       # PDF/Word 校园资料存放目录
├── models/                      # Embedding 模型缓存（all-MiniLM-L6-v2）
├── config/                      # 日志等全局配置
├── test/                        # 单元测试 & 集成测试
└── logs/                        # 运行日志输出目录
```

---

> 如遇其他问题，欢迎提 Issue：https://github.com/Kamine-Jf/CampusKnowledge_QASystem/issues
