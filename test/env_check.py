"""
CampusKnowledge_QASystem 阶段1环境巡检脚本
==================================================
检测范围：
1) Python 运行时与核心依赖版本
2) MySQL 连接连通性（默认读取 .env 或 src/database/mysql_config.py 配置）
3) Milvus 2.2.16 服务连接/建表/插入/向量搜索/清理
4) LangChain 最小化 LCEL 管线执行
5) CUDA 显卡信息（确保 RTX 3050Ti 4GB 兼容性）

使用方式：
    python test/env_check.py

脚本确保在依赖齐全、MySQL/Milvus 正常的情况下输出“全部检测通过”。
如检测失败，会给出中文指引，便于快速定位问题。

执行 python -m src.vector_db.test_vector_pipeline 
进行全流程验证，观察日志确认“阶段3模块测试通过”。
"""

from __future__ import annotations

import logging
import os
import platform
import sys
import time
import traceback
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv

# -------------------------------------------------
# 日志初始化：优先使用项目自带配置，失败则回退到基础配置。
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from config.log_config import setup_logging

    setup_logging()
except Exception:  # pylint: disable=broad-except
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

logger = logging.getLogger("env_check")


def _read_mysql_defaults() -> Dict[str, Any]:
    """读取 src/database/mysql_config.py 中的默认配置，便于在未设环境变量时回退。"""
    defaults = {
        "host": "127.0.0.1",
        "port": 3306,
        "user": "root",
        "password": "123456",
        "database": "campus_qa",
    }
    try:
        mysql_cfg = import_module("src.database.mysql_config")
        defaults["host"] = getattr(mysql_cfg, "MYSQL_HOST", defaults["host"])
        defaults["port"] = int(getattr(mysql_cfg, "MYSQL_PORT", defaults["port"]))
        defaults["user"] = getattr(mysql_cfg, "MYSQL_USER", defaults["user"])
        defaults["password"] = getattr(mysql_cfg, "MYSQL_PWD", defaults["password"])
        defaults["database"] = getattr(mysql_cfg, "MYSQL_DB", defaults["database"])
    except Exception:  # pylint: disable=broad-except
        logger.debug("读取 mysql_config 默认值失败，使用内置兜底", exc_info=True)
    return defaults


def check_python_runtime(expected_version: str = "3.11.8") -> Dict[str, Any]:
    """校验 Python 版本是否满足项目要求。"""
    current = platform.python_version()
    passed = current.startswith(expected_version)
    return {
        "passed": passed,
        "current": current,
        "expected": expected_version,
        "msg": "Python 版本匹配" if passed else f"当前版本为 {current}，请使用 {expected_version}",
    }


def check_required_packages() -> Dict[str, Any]:
    """逐一确认核心依赖是否已安装且版本正确。"""
    expected = {
        "pymysql": "1.1.0",
        "pymilvus": "2.2.16",
        "milvus": "2.2.16",
        "langchain": "0.1.10",
        "langchain-community": "0.0.28",
        "torch": "2.2.1",
        "python-dotenv": "1.0.1",
    }
    details: List[Tuple[str, str, str, bool]] = []
    passed = True
    for package, expected_version in expected.items():
        try:
            installed_version = version(package)
            version_match = installed_version == expected_version or (
                package == "torch" and installed_version.startswith(expected_version)
            )
        except PackageNotFoundError:
            installed_version = "未安装"
            version_match = False
        details.append((package, expected_version, installed_version, version_match))
        if not version_match:
            passed = False
    return {
        "passed": passed,
        "details": details,
    }


def check_mysql_connection() -> Dict[str, Any]:
    """尝试连接 MySQL 并执行 SELECT 1。"""
    load_dotenv(override=False)
    defaults = _read_mysql_defaults()
    host = os.getenv("MYSQL_HOST", defaults["host"])
    port = int(os.getenv("MYSQL_PORT", defaults["port"]))
    user = os.getenv("MYSQL_USER", defaults["user"])
    password = os.getenv("MYSQL_PASSWORD", defaults["password"])
    database = os.getenv("MYSQL_DB", defaults["database"])

    result: Dict[str, Any] = {
        "passed": False,
        "host": host,
        "port": port,
        "user": user,
        "database": database,
        "msg": "",
    }

    try:
        import pymysql

        connection = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            connect_timeout=5,
            read_timeout=5,
            write_timeout=5,
            cursorclass=pymysql.cursors.Cursor,
        )
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                cursor.fetchone()
            result["passed"] = True
            result["msg"] = "MySQL 连接成功并完成 SELECT 1 测试"
        finally:
            connection.close()
    except Exception as exc:  # pylint: disable=broad-except
        result["msg"] = f"MySQL 连接失败: {exc}"
        logger.exception("MySQL 连接异常", exc_info=True)
    return result


def check_milvus_server() -> Dict[str, Any]:
    """连接 Milvus 2.2.16 服务并执行建表/插入/搜索验证。"""
    result: Dict[str, Any] = {
        "passed": False,
        "host": os.getenv("MILVUS_HOST", "127.0.0.1"),
        "port": int(os.getenv("MILVUS_PORT", "19530")),
        "collection": "env_check_collection",
        "entity_count": 0,
        "search_results": [],
        "msg": "",
    }

    alias = "env_check"
    collection = None

    try:
        from pymilvus import Collection, CollectionSchema, DataType, FieldSchema
        from pymilvus import connections as milvus_connections
        from pymilvus import utility

        uri = os.getenv("MILVUS_URI")
        username = os.getenv("MILVUS_USER", "")
        password = os.getenv("MILVUS_PASSWORD", "")
        secure_flag = os.getenv("MILVUS_SECURE", "false").strip().lower() == "true"

        def _has_alias(connection_list: Any) -> bool:
            for item in connection_list:
                if isinstance(item, (list, tuple)) and item:
                    if item[0] == alias:
                        return True
                if isinstance(item, dict) and item.get("alias") == alias:
                    return True
            return False

        existing_connections = milvus_connections.list_connections()
        if _has_alias(existing_connections):
            milvus_connections.disconnect(alias)

        connect_kwargs = {
            "alias": alias,
            "secure": secure_flag,
        }

        if username:
            connect_kwargs["user"] = username
        if password:
            connect_kwargs["password"] = password

        if uri:
            result["host"] = uri
            result["port"] = None
            connect_kwargs["uri"] = uri
        else:
            connect_kwargs["host"] = result["host"]
            connect_kwargs["port"] = str(result["port"])

        milvus_connections.connect(**connect_kwargs)

        collection_name = result["collection"]
        if utility.has_collection(collection_name, using=alias):
            temp_collection = Collection(collection_name, using=alias)
            temp_collection.release()
            temp_collection.drop()

        pk_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False)
        vector_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=8)
        schema = CollectionSchema(fields=[pk_field, vector_field], description="env check collection")
        collection = Collection(name=collection_name, schema=schema, using=alias)

        vectors = [[float(i * j) / 10.0 for j in range(1, 9)] for i in range(1, 6)]
        ids = list(range(1, 6))
        collection.insert([ids, vectors])
        collection.flush()
        result["entity_count"] = collection.num_entities

        index_params = {"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 16}}
        collection.create_index(field_name="embedding", index_params=index_params, using=alias, timeout=120)

        collection.load()
        search_params = {"metric_type": "L2", "params": {"nprobe": 8}}
        search_output = collection.search(
            data=[[0.1 for _ in range(8)]],
            anns_field="embedding",
            param=search_params,
            limit=3,
            output_fields=["id"],
        )
        hits = []
        for hit in search_output[0]:
            hits.append({"id": int(hit.id), "distance": float(hit.distance)})
        result["search_results"] = hits

        collection.release()
        collection.drop()
        collection = None

        result["passed"] = True
        result["msg"] = "Milvus 服务连接成功并完成基础 CRUD 测试"
    except Exception as exc:  # pylint: disable=broad-except
        result["msg"] = f"Milvus 服务检测失败: {exc}"
        result["traceback"] = traceback.format_exc()
        logger.exception("Milvus 检测出现异常", exc_info=True)
    finally:
        try:
            if collection is not None:
                collection.release()
                collection.drop()
        except Exception:  # pylint: disable=broad-except
            logger.debug("Milvus 集合清理失败", exc_info=True)
        try:
            from pymilvus import connections as milvus_connections  # 动态导入，规避未安装情况

            existing_connections = milvus_connections.list_connections()
            alias_found = False
            for item in existing_connections:
                if isinstance(item, (list, tuple)) and item and item[0] == alias:
                    alias_found = True
                    break
                if isinstance(item, dict) and item.get("alias") == alias:
                    alias_found = True
                    break
            if alias_found:
                milvus_connections.disconnect(alias)
        except Exception:  # pylint: disable=broad-except
            logger.debug("Milvus 连接关闭异常", exc_info=True)
    return result


def check_langchain_pipeline() -> Dict[str, Any]:
    """构建最小 LangChain LCEL 管线，确保核心组件可用。"""
    result: Dict[str, Any] = {
        "passed": False,
        "output": None,
        "msg": "",
    }
    try:
        from langchain_core.prompts import PromptTemplate
        from langchain_core.runnables import RunnableLambda

        prompt = PromptTemplate.from_template("你好 {name}，校园知识库已就绪。")
        chain = prompt | RunnableLambda(lambda text: text)
        output = chain.invoke({"name": "同学"})
        result["passed"] = True
        result["output"] = output
        result["msg"] = "LangChain 管线校验成功"
    except Exception as exc:  # pylint: disable=broad-except
        result["msg"] = f"LangChain 初始化失败: {exc}"
        logger.exception("LangChain 检测异常", exc_info=True)
    return result


def check_cuda() -> Dict[str, Any]:
    """输出 CUDA 环境信息，确认显存是否满足 4GB 要求。"""
    info: Dict[str, Any] = {
        "cuda_available": False,
        "device_count": 0,
        "devices": [],
        "meets_requirement": False,
        "msg": "",
    }
    try:
        import torch

        info["cuda_available"] = bool(torch.cuda.is_available())
        info["device_count"] = int(torch.cuda.device_count())
        if info["cuda_available"]:
            meets_requirement = False
            for index in range(info["device_count"]):
                name = torch.cuda.get_device_name(index)
                total_bytes = torch.cuda.get_device_properties(index).total_memory
                total_gb = round(total_bytes / (1024 ** 3), 2)
                info["devices"].append({
                    "index": index,
                    "name": name,
                    "total_memory_gb": total_gb,
                })
                if total_gb >= 4.0:
                    meets_requirement = True
            info["meets_requirement"] = meets_requirement
            info["msg"] = "满足 RTX 3050Ti 4GB 要求" if meets_requirement else "显存不足 4GB，建议压缩模型或改用 CPU 模式"
        else:
            info["msg"] = "未检测到可用 CUDA，默认使用 CPU"
    except Exception as exc:  # pylint: disable=broad-except
        info["msg"] = f"CUDA 信息获取失败: {exc}"
        logger.debug("CUDA 检测异常", exc_info=True)
    return info


def render_package_table(details: List[Tuple[str, str, str, bool]]) -> None:
    """以表格样式打印依赖版本对比，便于人工核对。"""
    header = f"{'依赖包':<24}{'期望版本':<16}{'已安装版本':<20}{'状态'}"
    print(header)
    print("-" * len(header))
    for package, expected_version, installed_version, version_match in details:
        status = "通过" if version_match else "需修复"
        print(f"{package:<24}{expected_version:<16}{installed_version:<20}{status}")


def main() -> int:
    start = time.time()
    print("=" * 72)
    print("CampusKnowledge_QASystem 阶段1 环境巡检 | Python 3.11.8")
    print("=" * 72)

    python_res = check_python_runtime()
    package_res = check_required_packages()
    mysql_res = check_mysql_connection()
    milvus_res = check_milvus_server()
    langchain_res = check_langchain_pipeline()
    cuda_info = check_cuda()

    print(f"\n[Python] 当前版本: {python_res['current']} | 目标: {python_res['expected']} | 状态: {'通过' if python_res['passed'] else '需修复'}")

    print("\n[依赖检查] 核心三方库版本对比")
    render_package_table(package_res["details"])
    print(f"状态: {'全部匹配' if package_res['passed'] else '存在版本差异，需修复'}")

    print("\n[MySQL] 连接目标: {user}@{host}:{port}/{db}".format(
        user=mysql_res["user"],
        host=mysql_res["host"],
        port=mysql_res["port"],
        db=mysql_res["database"],
    ))
    print(f"结果: {'通过 ✅' if mysql_res['passed'] else '未通过 ❌'} | {mysql_res['msg']}")

    print("\n[Milvus] 服务地址: {host}:{port}".format(
        host=milvus_res["host"],
        port=milvus_res["port"] or '未知',
    ))
    if milvus_res["passed"]:
        print(f"结果: 通过 ✅ | 向量条数: {milvus_res['entity_count']}")
        print(f"示例搜索: {milvus_res['search_results']}")
    else:
        print(f"结果: 未通过 ❌ | {milvus_res['msg']}")
        if milvus_res.get("traceback"):
            print("详细异常:\n" + milvus_res["traceback"])

    print("\n[LangChain] {status} | 输出: {output}".format(
        status="通过 ✅" if langchain_res["passed"] else "未通过 ❌",
        output=langchain_res["output"] or langchain_res["msg"],
    ))

    print("\n[CUDA] 可用: {avail} | 显卡数量: {count}".format(
        avail=cuda_info["cuda_available"],
        count=cuda_info["device_count"],
    ))
    for device in cuda_info["devices"]:
        print("  - GPU{idx}: {name} | 显存 {mem} GB".format(
            idx=device["index"],
            name=device["name"],
            mem=device["total_memory_gb"],
        ))
    print(f"说明: {cuda_info['msg']}")

    overall_passed = all(
        result["passed"]
        for result in (python_res, package_res, mysql_res, milvus_res, langchain_res)
    )

    duration = time.time() - start
    print("\n总耗时: {:.2f}s".format(duration))
    if overall_passed:
        print("\n✅ 全部核心检测通过，CampusKnowledge_QASystem 阶段1环境已就绪。")
        exit_code = 0
    else:
        print("\n❌ 核心检测存在异常，请根据提示逐项排查。")
        exit_code = 1

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
