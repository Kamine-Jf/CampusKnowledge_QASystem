# 基于 LangChain + Milvus 的校园知识库智能问答系统




**针对校园政策查询繁琐、信息分散的问题，搭建智能问答系统**，整合教务规则、学生手册等资料，实现精准高效检索，为师生提供便捷信息服务，提升校园管理信息化水平。

本系统采用 **RAG（Retrieval-Augmented Generation）** 架构，基于 LangChain 框架 + Milvus 向量数据库 + MySQL 结构化存储，实现结构化教务数据与非结构化文档的双源融合检索。已完成端到端开发，支持 Web 可视化交互，功能稳定、检索准确，可为同类校园智能问答系统提供参考方案。

**这是郑州轻工业大学软件工程专业毕业设计（课题名称：基于LangChain+Milvus的校园知识库智能问答系统的设计与实现）**，指导教师：甘琤。

---

## ✨ 主要功能

- **自然语言智能问答**：支持“缓考流程是什么？”、“奖学金评定条件”、“学校简介”等校园相关问题
- **双源检索机制**：MySQL 精准查询结构化教务规则 + Milvus 向量相似性检索非结构化资料（PDF/Word/Excel）
- **RAG 完整流程**：文档解析 → 向量入库 → 双源检索融合 → Prompt 工程 → 大模型生成
- **现代 Web 界面**：支持浅色/暗黑模式切换，侧边栏历史会话与快捷入口
- **数据自动处理**：支持 Excel、Word、PDF 校园资料自动解析与入库
- **Docker 一键部署**：Milvus + MySQL + FastAPI 全部容器化
- **异常兜底**：无检索结果、网络异常、非法提问均有友好提示

---

## 🛠 技术栈

- **后端**：Python + FastAPI
- **AI 框架**：LangChain（RAG 管道）
- **向量数据库**：Milvus（Docker 部署）
- **结构化数据库**：MySQL
- **大模型**：可自行后台添加
- **部署**：Docker + Docker Compose
- **前端**：HTML + CSS + JavaScript（`static/` 目录）
- **文档解析**：支持 PDF、Word、Excel
- **其他**：requirements.txt 依赖管理

---

## 🚀 快速开始

### 1. 克隆仓库
```bash
git clone https://github.com/Kamine-Jf/CampusKnowledge_QASystem.git
cd CampusKnowledge_QASystem
2. 环境准备

安装 Docker 和 Docker Compose
（可选）配置 .env 文件（若仓库提供 .env.example）

3. 启动服务
Bashdocker-compose up -d --build


4. 访问系统
打开浏览器访问 http://localhost:8000（端口以 docker-compose.yml 配置为准）
系统启动后即可在界面中提问或点击侧边栏快捷入口。

📸 系统截图
欢迎界面（浅色模式）
<img src="screenshots/welcome-light-1.png" alt="欢迎界面 - 浅色模式">
欢迎界面（暗黑模式）
<img src="screenshots/welcome-dark-1.png" alt="欢迎界面 - 暗黑模式">
欢迎界面（浅色模式 - 另一种布局）
<img src="screenshots/welcome-light-2.png" alt="欢迎界面 - 浅色模式2">
欢迎界面（暗黑模式 - 另一种布局）
<img src="screenshots/welcome-dark-2.png" alt="欢迎界面 - 暗黑模式2">
学校简介查询示例（浅色模式）
<img src="screenshots/school-intro-light.png" alt="学校简介 - 浅色模式">
学校简介查询示例（暗黑模式）
<img src="screenshots/school-intro-dark.png" alt="学校简介 - 暗黑模式">
如何添加截图：
在仓库根目录新建 screenshots/ 文件夹
将本消息中提供的6张图片上传到该文件夹
按上方文件名重命名（welcome-light-1.png、welcome-dark-1.png 等）
提交后 README 中的图片即可正常显示


📁 项目结构（核心目录）
textCampusKnowledge_QASystem/
├── src/                  # 核心源码（RAG 管道、检索逻辑）
├── static/               # 前端静态资源（HTML/CSS/JS）
├── config/               # 配置文件
├── data/                 # 校园资料（PDF/Excel 等）
├── docs/                 # 文档
├── test/                 # 测试脚本
├── docker-compose.yml    # Docker 部署配置
├── Dockerfile
├── main.py               # FastAPI 入口
├── requirements.txt      # Python 依赖
└── stage3_run_guide.md   # 运行指南

📄 相关文档

毕业设计开题报告：见仓库 docs/开题报告/ 或本地提供的 谢嘉峰开题报告表.docx
研究背景与意义、技术路线、时间安排等详见开题报告
测试报告与论文：后续开发阶段会持续更新


🎯 项目意义

实践意义：为郑州轻工业大学师生提供7×24小时智能校园咨询服务，减轻人工负担
理论意义：验证 LangChain + Milvus 在轻量级校园 RAG 系统中的可行性
应用价值：可扩展至其他高校，提供标准化校园知识库智能问答方案


👤 作者

姓名：谢嘉峰（Kamine）
学号：542213330641
专业：软件工程
指导教师：甘琤
学校：郑州轻工业大学



欢迎 Star ⭐、Issue 与 PR，一起完善校园智能问答系统！

项目链接：https://github.com/Kamine-Jf/CampusKnowledge_QASystem
