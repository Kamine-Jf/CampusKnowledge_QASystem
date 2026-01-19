"""
环境自检脚本（Python 3.11.8）
==================================================
检测项：
1) MySQL 连接可用性（读取环境变量连接，执行 SELECT 1）
2) Milvus Lite 本地启动与基础读写（自动拉起轻量版服务、建表/插入/搜索/清理）
3) LangChain 初始化（基于 LCEL 组装最小可运行管线并执行）

使用说明：
- 可在命令行直接运行：
    python -m src.env_check

环境变量（可选，若未设置将使用默认值尝试连接）：
- MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DB

硬件与模型约束：
- 适配 NVIDIA RTX 3050Ti 4GB 显存；本项目推荐使用 ChatGLM-6B-int4 量化版离线推理（无外部API）。
- 本脚本将输出 CUDA 可用性与显存信息以供参考，但不会加载大模型。

注意：
- 本脚本仅做连通性与初始化校验，不会改动业务数据。
"""

from __future__ import annotations

import os
import sys
import time
import traceback
import logging
from typing import Dict, Any

from dotenv import load_dotenv

# 日志配置：输出到 logs/ 与控制台
try:
    # 相对导入项目内的日志配置
    from config.logging_config import setup_logging
    setup_logging()
except Exception:  # 若日志初始化失败，也保证脚本可运行
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

logger = logging.getLogger("env_check")


def check_cuda() -> Dict[str, Any]:
    """检测 CUDA/GPU 情况，仅用于环境参考，不影响功能性校验。
    返回包含可用性、设备名、总显存(GB)等信息。
    """
    info: Dict[str, Any] = {
        "cuda_available": False,
        "device_count": 0,
        "devices": [],  # 每个元素包含 name、total_memory_gb
        "recommendation": "建议使用 ChatGLM-6B-int4 量化版，本地离线推理；4GB 显存可运行但需控制上下文长度。",
    }
    try:
        import torch
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["device_count"] = int(torch.cuda.device_count())
        if info["cuda_available"]:
            for i in range(info["device_count"]):
                name = torch.cuda.get_device_name(i)
                mem_bytes = torch.cuda.get_device_properties(i).total_memory
                info["devices"].append({
                    "index": i,
                    "name": name,
                    "total_memory_gb": round(mem_bytes / (1024 ** 3), 2)
                })
    except Exception as e:
        info["error"] = f"检测CUDA异常: {e}"
    return info


def check_mysql_connection() -> Dict[str, Any]:
    """检测 MySQL 连接：尝试建立连接并执行 SELECT 1。
    链接信息优先读取环境变量，若未设置将使用默认值。
    """
    load_dotenv(override=False)
    host = os.getenv("MYSQL_HOST", "127.0.0.1")
    port = int(os.getenv("MYSQL_PORT", "3306"))
    user = os.getenv("MYSQL_USER", "root")
    password = os.getenv("MYSQL_PASSWORD", "")
    database = os.getenv("MYSQL_DB", "campus_qa")

    result: Dict[str, Any] = {
        "passed": False,
        "host": host,
        "port": port,
        "user": user,
        "database": database,
        "error": None,
    }

    try:
        import pymysql
        conn = pymysql.connect(
            host=host, port=port, user=user, password=password,
            database=database, connect_timeout=3, read_timeout=3, write_timeout=3,
            cursorclass=pymysql.cursors.Cursor,
        )
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                _ = cur.fetchone()
            result["passed"] = True
        finally:
            conn.close()
    except Exception as e:
        result["error"] = f"MySQL 连接失败: {e}"
        logger.warning(result["error"])  # 记录到错误日志

    return result


def check_milvus_lite() -> Dict[str, Any]:
    """检测 Milvus Lite：
    - 启动本地轻量版服务
    - 连接、创建测试集合、插入少量数据并向量搜索
    - 清理资源并停止服务
    """
    result: Dict[str, Any] = {
        "passed": False,
        "error": None,
        "server_port": None,
        "search_results": None,
    }

    try:
        # 启动本地 Milvus Lite 服务器（由 milvus-lite 提供）
        try:
            from milvus import default_server  # milvus-lite 暴露的默认内置服务
        except Exception as e:
            raise RuntimeError(
                "未找到 milvus-lite 的默认服务入口，请确认已安装 milvus-lite，并使用与 pymilvus 匹配的版本。"
            ) from e

        default_server.start()
        time.sleep(0.3)  # 略等服务就绪

        listen_port = default_server.listen_port
        result["server_port"] = listen_port

        from pymilvus import (
            connections, FieldSchema, CollectionSchema, DataType, Collection, utility
        )

        # 建立连接
        connections.connect(alias="default", host="127.0.0.1", port=str(listen_port))

        # 若集合已存在，先删除（保证幂等）
        col_name = "_env_check_collection_"
        if utility.has_collection(col_name):
            utility.drop_collection(col_name)

        # 定义最小化字段结构：主键 + 向量
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="emb", dtype=DataType.FLOAT_VECTOR, dim=8),
        ]
        schema = CollectionSchema(fields=fields, description="env check collection")
        coll = Collection(name=col_name, schema=schema)

        # 插入 3 条测试数据
        entities = [
            [0, 1, 2],
            [
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            ],
        ]
        coll.insert(entities)
        coll.flush()

        # 创建索引并加载
        coll.create_index(field_name="emb", index_params={"index_type": "IVF_FLAT", "params": {"nlist": 8}, "metric_type": "L2"})
        coll.load()

        # 向量搜索
        query_vec = [[0.1] * 8]
        res = coll.search(
            data=query_vec, anns_field="emb", param={"nprobe": 4},
            limit=2, output_fields=["id"], consistency_level="Strong"
        )
        result["search_results"] = [[{"id": hit.entity.get("id"), "distance": float(hit.distance)} for hit in hits] for hits in res]

        # 清理
        utility.drop_collection(col_name)
        connections.disconnect("default")
        default_server.stop()
        result["passed"] = True
    except Exception as e:
        result["error"] = f"Milvus Lite 检测失败: {e}\n{traceback.format_exc()}"
        logger.warning(result["error"])  # 记录异常详情
        try:
            # 尝试停止已启动的服务，避免驻留进程
            from milvus import default_server
            default_server.stop()
        except Exception:
            pass

    return result


def check_langchain_init() -> Dict[str, Any]:
    """检测 LangChain 初始化：
    - 使用 LCEL（LangChain Expression Language）构建最小可跑通的管线
    - 不加载大模型，验证核心组件可用
    """
    result: Dict[str, Any] = {
        "passed": False,
        "output": None,
        "error": None,
    }
    try:
        from langchain_core.prompts import PromptTemplate
        from langchain_core.runnables import RunnableLambda
        from langchain_core.output_parsers import StrOutputParser

        # 构建最小管线：Prompt -> 透传函数 -> 字符串解析
        prompt = PromptTemplate.from_template("你好，{name}！欢迎使用校园知识库问答系统。")
        chain = prompt | RunnableLambda(lambda s: s) | StrOutputParser()

        # 运行一次
        output = chain.invoke({"name": "同学"})
        result["output"] = output
        result["passed"] = True
    except Exception as e:
        result["error"] = f"LangChain 初始化失败: {e}\n{traceback.format_exc()}"
        logger.warning(result["error"])  # 记录异常详情

    return result


def main() -> int:
    print("=" * 68)
    print("CampusKnowledge_QASystem 环境检测 | Python 3.11.8")
    print("=" * 68)

    # 1) CUDA/GPU 信息（参考）
    cuda_info = check_cuda()
    print("[CUDA/GPU] 可用:", cuda_info.get("cuda_available"))
    print("[CUDA/GPU] 数量:", cuda_info.get("device_count"))
    for dev in cuda_info.get("devices", []):
        print(f"  - GPU{dev['index']}: {dev['name']} | 显存 {dev['total_memory_gb']} GB")
    if not cuda_info.get("cuda_available"):
        print("[提示] CUDA 不可用，将以CPU运行；如需GPU请安装对应CUDA版本的PyTorch。")

    # 2) MySQL 连接
    print("\n[1/3] 开始检测 MySQL 连接...")
    mysql_res = check_mysql_connection()
    print("    目标:", f"{mysql_res['user']}@{mysql_res['host']}:{mysql_res['port']}/{mysql_res['database']}")
    if mysql_res["passed"]:
        print("    结果: 通过 ✅")
    else:
        print("    结果: 未通过 ❌")
        print("    详情:", mysql_res.get("error"))
        print("    建议: 设置 MYSQL_HOST/PORT/USER/PASSWORD/MYSQL_DB 环境变量后重试。")

    # 3) Milvus Lite
    print("\n[2/3] 开始检测 Milvus Lite 本地服务与向量操作...")
    milvus_res = check_milvus_lite()
    if milvus_res["passed"]:
        print(f"    结果: 通过 ✅ (端口 {milvus_res['server_port']})")
        print("    示例搜索结果:", milvus_res.get("search_results"))
    else:
        print("    结果: 未通过 ❌")
        print("    详情:", milvus_res.get("error"))
        print("    建议: 确认已安装 milvus-lite 与 pymilvus 版本匹配，并重新运行。")

    # 4) LangChain 初始化
    print("\n[3/3] 开始检测 LangChain 初始化...")
    lc_res = check_langchain_init()
    if lc_res["passed"]:
        print("    结果: 通过 ✅")
        print("    运行输出:", lc_res.get("output"))
    else:
        print("    结果: 未通过 ❌")
        print("    详情:", lc_res.get("error"))
        print("    建议: 确认 langchain 及 langchain-core 版本兼容，或重新安装。")

    print("\n— 完成 —")

    # 以整体通过与否返回码区分（非强制）
    passed_all = mysql_res["passed"] and milvus_res["passed"] and lc_res["passed"]
    return 0 if passed_all else 1


if __name__ == "__main__":
    sys.exit(main())
