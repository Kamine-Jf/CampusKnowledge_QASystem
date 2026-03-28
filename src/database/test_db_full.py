# -*- coding: utf-8 -*-
"""阶段 2 MySQL 模块全流程测试脚本。"""

from __future__ import annotations

import os
import sys
from typing import Callable, List, Optional

from .mysql_conn import MysqlConnection
from .excel2mysql import excel_to_mysql
from .db_operate import (
    add_struct_data,
    del_struct_data,
    update_struct_data,
    get_struct_data_by_id,
    list_struct_data,
    add_query_history,
    del_query_history,
    update_query_history,
    get_query_history_by_id,
    list_query_history,
    add_pdf_info,
    del_pdf_info,
    update_pdf_info,
    get_pdf_info_by_id,
    list_pdf_info,
    query_by_keyword,
)


def _print_success(message: str) -> None:
    """统一的成功提示格式。"""
    print(f"{message} ✔️")


def _print_failure(message: str, error: Exception) -> None:
    """统一的失败提示格式，输出详细异常信息。"""
    print(f"{message} ✖️")
    print(f"详细错误：{error}")


def _run_step(step_message: str, handler: Callable[[], None]) -> None:
    """执行测试步骤的统一入口，出现异常时立即退出脚本。"""
    try:
        handler()
        _print_success(step_message)
    except Exception as step_error:  # pylint: disable=broad-except
        _print_failure(step_message, step_error)
        sys.exit(1)


def _execute_sql_file(sql_file_path: str) -> None:
    """执行建库建表 SQL 文件，确保数据库结构符合要求。"""
    if not os.path.exists(sql_file_path):
        raise FileNotFoundError(f"未找到建表文件：{sql_file_path}")

    mysql_client = MysqlConnection()
    connection_tuple = mysql_client.connect_db(use_database=False)
    if connection_tuple is None:
        raise ConnectionError("无法连接到 MySQL 服务，请检查 mysql_config.py 配置。")
    connection, cursor = connection_tuple

    statements: List[str] = []
    with open(sql_file_path, "r", encoding="utf-8") as sql_file:
        buffer: List[str] = []
        for raw_line in sql_file:
            stripped_line = raw_line.strip()
            if not stripped_line or stripped_line.startswith("--"):
                continue
            buffer.append(stripped_line)
            if stripped_line.endswith(";"):
                statement = " ".join(buffer).rstrip(";")
                statements.append(statement)
                buffer.clear()
        if buffer:
            statements.append(" ".join(buffer).rstrip(";"))

    try:
        for statement in statements:
            cursor.execute(statement)
        connection.commit()
    finally:
        mysql_client.close_db()


def _verify_mysql_service() -> None:
    """检测 MySQL 服务可用性，连接成功后立即释放资源。"""
    mysql_client = MysqlConnection()
    connection_tuple = mysql_client.connect_db(use_database=False)
    if connection_tuple is None:
        raise ConnectionError("无法连接到 MySQL 服务，请检查 mysql_config.py 配置。")
    mysql_client.close_db()


def _reset_business_tables() -> None:
    """清理三张业务表，保证测试数据不会重复堆积。"""
    mysql_client = MysqlConnection()
    connection_tuple = mysql_client.connect_db()
    if connection_tuple is None:
        raise ConnectionError("无法连接至数据库，无法重置数据表。")
    connection, cursor = connection_tuple
    try:
        cursor.execute("DELETE FROM user_query_history")
        cursor.execute("DELETE FROM pdf_doc_info")
        cursor.execute("DELETE FROM campus_struct_data")
        connection.commit()
    except Exception as reset_error:  # pylint: disable=broad-except
        connection.rollback()
        raise reset_error
    finally:
        mysql_client.close_db()


def _test_struct_table_crud() -> None:
    """验证 campus_struct_data 表的增删改查流程。"""
    test_category = "测试分类-结构化"
    test_item = "测试事项-结构化"
    test_operation = "这是用于验证结构化数据 CRUD 的测试操作步骤。"

    if not add_struct_data(test_category, test_item, test_operation, "测试时间", "测试渠道", "测试备注"):
        raise RuntimeError("结构化数据新增失败。")

    records = list_struct_data()
    inserted = next((item for item in records if item["category"] == test_category and item["item"] == test_item), None)
    if inserted is None:
        raise RuntimeError("未查询到刚刚插入的结构化数据。")

    record_id = inserted["id"]
    if not update_struct_data(record_id, test_category, test_item, "更新后的操作步骤", "更新后的时间", "更新渠道", "更新备注"):
        raise RuntimeError("结构化数据更新失败。")

    fetched = get_struct_data_by_id(record_id)
    if fetched is None or fetched.get("operation") != "更新后的操作步骤":
        raise RuntimeError("结构化数据更新未生效。")

    if not del_struct_data(record_id):
        raise RuntimeError("结构化数据删除失败。")

    if get_struct_data_by_id(record_id) is not None:
        raise RuntimeError("结构化数据删除后仍能查询到记录。")


def _test_query_history_crud() -> None:
    """验证 user_query_history 表的增删改查流程。"""
    if not add_query_history("测试问题", "测试回答"):
        raise RuntimeError("用户历史新增失败。")

    records = list_query_history()
    inserted = next((item for item in records if item["query_content"] == "测试问题"), None)
    if inserted is None:
        raise RuntimeError("未查询到刚插入的用户历史。")

    record_id = inserted["id"]
    if not update_query_history(record_id, "测试问题更新", "测试回答更新"):
        raise RuntimeError("用户历史更新失败。")

    fetched = get_query_history_by_id(record_id)
    if fetched is None or fetched.get("answer_content") != "测试回答更新":
        raise RuntimeError("用户历史更新未生效。")

    if not del_query_history(record_id):
        raise RuntimeError("用户历史删除失败。")

    if get_query_history_by_id(record_id) is not None:
        raise RuntimeError("用户历史删除后仍存在记录。")


def _test_pdf_info_crud() -> None:
    """验证 pdf_doc_info 表的增删改查流程。"""
    if not add_pdf_info("测试文档.pdf", "/tmp/test.pdf"):
        raise RuntimeError("PDF 信息新增失败。")

    records = list_pdf_info()
    inserted = next((item for item in records if item["doc_name"] == "测试文档.pdf"), None)
    if inserted is None:
        raise RuntimeError("未查询到刚插入的 PDF 信息。")

    record_id = inserted["id"]
    if not update_pdf_info(record_id, "测试文档-更新.pdf", "/tmp/test-update.pdf"):
        raise RuntimeError("PDF 信息更新失败。")

    fetched = get_pdf_info_by_id(record_id)
    if fetched is None or fetched.get("doc_name") != "测试文档-更新.pdf":
        raise RuntimeError("PDF 信息更新未生效。")

    if not del_pdf_info(record_id):
        raise RuntimeError("PDF 信息删除失败。")

    if get_pdf_info_by_id(record_id) is not None:
        raise RuntimeError("PDF 信息删除后仍存在记录。")


def _test_keyword_queries() -> None:
    """针对指定关键词执行查询，并打印结果。"""
    for keyword in ("缓考", "奖学金"):
        results = query_by_keyword(keyword)
        print(f"关键词 '{keyword}' 命中 {len(results)} 条记录。")
        for item in results[:3]:
            print(f"  · {item['category']} - {item['item']}")


def main() -> None:
    """执行阶段 2 MySQL 功能的完整验证流程。"""
    base_dir = os.path.dirname(__file__)
    sql_path = os.path.join(base_dir, "create_tables.sql")

    _run_step("MySQL 服务连接验证", _verify_mysql_service)
    _run_step("数据库与数据表创建", lambda: _execute_sql_file(sql_path))
    _run_step("业务表数据重置", _reset_business_tables)
    _run_step("Excel 数据导入", lambda: print(f"Excel 数据导入成功，共导入 {excel_to_mysql()} 条。"))
    _run_step("campus_struct_data CRUD 功能", _test_struct_table_crud)
    _run_step("user_query_history CRUD 功能", _test_query_history_crud)
    _run_step("pdf_doc_info CRUD 功能", _test_pdf_info_crud)
    _run_step("关键词模糊查询验证", _test_keyword_queries)

    print("阶段2 MySQL模块开发完成，所有功能验证通过。")


if __name__ == "__main__":
    main()
