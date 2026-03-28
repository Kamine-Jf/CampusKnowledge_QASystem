# 阶段3运行说明（CampusKnowledge_QASystem）

## 依赖顺序
1. **模型预下载**：执行 `python download_model.py`，确保 `models/` 目录存在 `all-MiniLM-L6-v2` 模型缓存。
2. **PDF 向量生成**：运行 `python src/vector_db/pdf2vector.py`，验证 PDF 解析与向量生成流程正常（已在 data/unstructured_data/ 下提供示例 test.pdf）。
3. **Milvus 向量入库**：在 Milvus Docker 服务运行的前提下执行 `python src/vector_db/milvus_operate.py`，确认集合创建、插入与检索通路正常。
4. **RAG 联动测试**：执行 `python src/rag/rag_core.py 缓考申请流程`（或其它关键词），观察结构化与非结构化搜索结果是否符合预期；若未传参脚本会提示输入查询词。

## 运行命令
- 预下载模型：`python download_model.py`
- 解析 PDF 并生成向量：`python src/vector_db/pdf2vector.py`
- 测试 Milvus 集合与检索：`python src/vector_db/milvus_operate.py`
- 同步向量到 Milvus：`python -c "from src.rag.rag_core import sync_pdf_vectors_to_milvus; sync_pdf_vectors_to_milvus()"`
- 执行 RAG 查询演示：`python src/rag/rag_core.py 缓考申请流程`（可替换为任意关键词）

## 常见问题与解决方案
- **Milvus 连接失败**：确认 Docker 容器 `milvus-standalone` 已启动，端口 `19530` 未被占用；可通过 `docker ps` 与 `docker logs` 检查服务状态。
- **PDF 解析乱码或为空**：请使用文字版 PDF；如为扫描件，可先使用 OCR 转换；同时查看脚本输出的页码跳过提示。
- **模型下载超时或显存不足**：无网络时先执行 `python download_model.py` 在联网环境预拉取；所有脚本默认在 CPU 上运行，若显存影响请确认未手动启用 GPU。
- **MySQL 查询失败**：检查 MySQL 服务是否启动、账号密码配置是否正确； `src/database/mysql_config.py` 中的连接参数需与实际一致。

## 毕设演示步骤建议
1. 启动 Milvus 与 MySQL 服务，激活虚拟环境 `.venv`。
2. 展示运行 `python download_model.py` 的过程，说明模型缓存目录位于 D 盘。
3. 演示 `python src/vector_db/pdf2vector.py`，展示解析日志与向量维度输出。
4. 运行 `python -c "from src.rag.rag_core import sync_pdf_vectors_to_milvus; sync_pdf_vectors_to_milvus()"`，讲解向量入库成功提示。
5. 执行 `python src/rag/rag_core.py`，展示结构化与非结构化结果，重点说明 "结构化优先 + 向量补充" 的融合策略。
6. 若需要补充说明，可运行 `python src/vector_db/milvus_operate.py` 展示集合索引与检索的单体演示。
