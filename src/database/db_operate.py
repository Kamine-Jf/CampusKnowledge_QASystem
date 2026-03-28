# -*- coding: utf-8 -*-
"""数据库核心操作函数集合，提供 CRUD 与关键词查询能力。
运行python src/database/db_operate.py 缓考等关键词可直接执行命令行查询测试。
"""

from __future__ import annotations

import sys

from typing import Any, Dict, List, Optional, Sequence

try:
    from .mysql_conn import MysqlConnection
except ImportError:
    from pathlib import Path

    current_path = Path(__file__).resolve()
    project_root = current_path.parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        from src.database.mysql_conn import MysqlConnection
    except ImportError as _exc:
        raise ImportError(
            "无法导入 MysqlConnection，请检查项目目录结构或确保 src 目录已加入 PYTHONPATH。"
        ) from _exc


def _execute_write(sql: str, params: Sequence[Any] | Dict[str, Any]) -> bool:
    """通用写操作执行函数，负责提交或回滚事务。"""
    mysql_client = MysqlConnection()
    connection_tuple = mysql_client.connect_db()
    if connection_tuple is None:
        print("[连接失败] 数据库连接异常，写操作取消。")
        return False
    connection, cursor = connection_tuple
    try:
        cursor.execute(sql, params)
        connection.commit()
        return cursor.rowcount > 0
    except Exception as write_error:  # pylint: disable=broad-except
        connection.rollback()
        print(f"[写入异常] {write_error}")
        return False
    finally:
        mysql_client.close_db()


def _execute_fetchone(sql: str, params: Sequence[Any] | Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """通用单条查询函数，返回字典格式结果。"""
    mysql_client = MysqlConnection()
    connection_tuple = mysql_client.connect_db()
    if connection_tuple is None:
        print("[连接失败] 数据库连接异常，查询取消。")
        return None
    _, cursor = connection_tuple
    try:
        cursor.execute(sql, params)
        result = cursor.fetchone()
        return result
    except Exception as query_error:  # pylint: disable=broad-except
        print(f"[查询异常] {query_error}")
        return None
    finally:
        mysql_client.close_db()


def _execute_fetchall(sql: str, params: Sequence[Any] | Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    """通用多条查询函数，返回字典结果列表。"""
    mysql_client = MysqlConnection()
    connection_tuple = mysql_client.connect_db()
    if connection_tuple is None:
        print("[连接失败] 数据库连接异常，查询取消。")
        return []
    _, cursor = connection_tuple
    try:
        if params is None:
            cursor.execute(sql)
        else:
            cursor.execute(sql, params)
        results = cursor.fetchall()
        return list(results) if results else []
    except Exception as query_error:  # pylint: disable=broad-except
        print(f"[查询异常] {query_error}")
        return []
    finally:
        mysql_client.close_db()


def ensure_user_auth_schema() -> None:
    """Ensure user_auth table and unique index exist for auth APIs."""
    mysql_client = MysqlConnection()
    connection_tuple = mysql_client.connect_db()
    if connection_tuple is None:
        print("[连接失败] 数据库连接异常，无法检查 user_auth 表结构。")
        return

    connection, cursor = connection_tuple
    try:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS user_auth (
                id INT PRIMARY KEY AUTO_INCREMENT,
                name VARCHAR(50) NOT NULL,
                student_id VARCHAR(30) NOT NULL,
                phone VARCHAR(20) NOT NULL,
                password VARCHAR(128) NOT NULL DEFAULT '123456',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE KEY uk_student_id (student_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='用户登录注册信息表'
            """
        )
        connection.commit()
    except Exception as schema_error:  # pylint: disable=broad-except
        connection.rollback()
        print(f"⚠️ user_auth 表结构检查失败：{schema_error}")

    try:
        cursor.execute(
            "ALTER TABLE user_auth ADD COLUMN password VARCHAR(128) NOT NULL DEFAULT '123456'"
        )
        connection.commit()
        print("✅ user_auth 已新增 password 字段。")
    except Exception as col_error:  # pylint: disable=broad-except
        msg = str(col_error)
        if "Duplicate column name" not in msg:
            connection.rollback()
            print(f"⚠️ password 字段检查失败：{col_error}")
    finally:
        mysql_client.close_db()


def ensure_feedback_schema() -> None:
    """Ensure feedback table exists."""
    mysql_client = MysqlConnection()
    connection_tuple = mysql_client.connect_db()
    if connection_tuple is None:
        print("[连接失败] 数据库连接异常，无法检查 feedback 表结构。")
        return

    connection, cursor = connection_tuple
    try:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id INT PRIMARY KEY AUTO_INCREMENT,
                user_id INT DEFAULT NULL,
                user_name VARCHAR(50) DEFAULT '',
                query_content TEXT,
                answer_content TEXT,
                feedback_text TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='用户反馈信息表'
            """
        )
        connection.commit()
    except Exception as schema_error:  # pylint: disable=broad-except
        connection.rollback()
        print(f"⚠️ feedback 表结构检查失败：{schema_error}")
    finally:
        mysql_client.close_db()


# -------------------------- campus_struct_data 表操作 --------------------------


def add_struct_data(
    category: str,
    item: str,
    operation: str,
    time_requirement: str = "",
    channel: str = "",
    source_note: str = "",
) -> bool:
    """新增校园结构化数据记录。"""
    insert_sql = (
        "INSERT INTO campus_struct_data (category, item, operation, time_requirement, channel, source_note) "
        "VALUES (%s, %s, %s, %s, %s, %s)"
    )
    return _execute_write(insert_sql, (category, item, operation, time_requirement, channel, source_note))


def del_struct_data(record_id: int) -> bool:
    """按主键删除校园结构化数据记录。"""
    delete_sql = "DELETE FROM campus_struct_data WHERE id = %s"
    return _execute_write(delete_sql, (record_id,))


def update_struct_data(
    record_id: int,
    category: str,
    item: str,
    operation: str,
    time_requirement: str = "",
    channel: str = "",
    source_note: str = "",
) -> bool:
    """更新校园结构化数据记录的全部字段。"""
    update_sql = (
        "UPDATE campus_struct_data SET category = %s, item = %s, operation = %s, "
        "time_requirement = %s, channel = %s, source_note = %s WHERE id = %s"
    )
    return _execute_write(
        update_sql,
        (category, item, operation, time_requirement, channel, source_note, record_id),
    )


def get_struct_data_by_id(record_id: int) -> Optional[Dict[str, Any]]:
    """按主键查询校园结构化数据，返回字典结果。"""
    query_sql = (
        "SELECT id, category, item, operation, time_requirement, channel, source_note "
        "FROM campus_struct_data WHERE id = %s"
    )
    return _execute_fetchone(query_sql, (record_id,))


def list_struct_data() -> List[Dict[str, Any]]:
    """查询校园结构化数据的全量列表，按主键升序排列。"""
    query_sql = (
        "SELECT id, category, item, operation, time_requirement, channel, source_note "
        "FROM campus_struct_data ORDER BY id ASC"
    )
    return _execute_fetchall(query_sql)


# -------------------------- user_auth 表操作 --------------------------


def add_user_auth(name: str, student_id: str, phone: str, password: str = "123456") -> bool:
    """新增用户注册信息。"""
    insert_sql = (
        "INSERT INTO user_auth (name, student_id, phone, password) VALUES (%s, %s, %s, %s)"
    )
    return _execute_write(insert_sql, (name, student_id, phone, password))


def get_user_auth_by_student_id(student_id: str) -> Optional[Dict[str, Any]]:
    """按学号查询用户信息。"""
    query_sql = (
        "SELECT id, name, student_id, phone, created_at FROM user_auth WHERE student_id = %s"
    )
    return _execute_fetchone(query_sql, (student_id,))


def get_user_auth_for_login(student_id: str, phone: str) -> Optional[Dict[str, Any]]:
    """按学号和手机号查询登录用户信息。"""
    query_sql = (
        "SELECT id, name, student_id, phone, created_at FROM user_auth "
        "WHERE student_id = %s AND phone = %s"
    )
    return _execute_fetchone(query_sql, (student_id, phone))


def get_user_auth_by_account_and_password(account: str, password: str) -> Optional[Dict[str, Any]]:
    """按账号(学号或手机号)和密码查询登录用户信息。"""
    query_sql = (
        "SELECT id, name, student_id, phone, created_at FROM user_auth "
        "WHERE (student_id = %s OR phone = %s) AND password = %s"
    )
    return _execute_fetchone(query_sql, (account, account, password))


def get_user_auth_by_phone(phone: str) -> Optional[Dict[str, Any]]:
    """按手机号查询用户信息。"""
    query_sql = (
        "SELECT id, name, student_id, phone, created_at FROM user_auth WHERE phone = %s"
    )
    return _execute_fetchone(query_sql, (phone,))


def list_all_users(limit: int = 200) -> List[Dict[str, Any]]:
    """查询所有注册用户列表，按注册时间倒序。"""
    safe_limit = max(1, min(int(limit), 500))
    query_sql = (
        "SELECT id, name, student_id, phone, created_at "
        "FROM user_auth ORDER BY created_at DESC LIMIT %s"
    )
    return _execute_fetchall(query_sql, (safe_limit,))


def delete_user_by_id(user_id: int) -> bool:
    """按用户ID注销（删除）用户账号。"""
    delete_sql = "DELETE FROM user_auth WHERE id = %s"
    return _execute_write(delete_sql, (user_id,))


def delete_user_history_by_user_id(user_id: int) -> bool:
    """按用户ID删除该用户的全部对话历史记录。"""
    delete_sql = "DELETE FROM user_query_history WHERE user_id = %s"
    return _execute_write(delete_sql, (user_id,))


def list_user_sessions(user_id: int, limit: int = 50) -> List[Dict[str, Any]]:
    """查询某用户的所有会话列表及每个会话的消息数量。"""
    safe_limit = max(1, min(int(limit), 200))
    query_sql = (
        "SELECT session_id, "
        "MIN(query_content) AS first_query, "
        "COUNT(*) AS message_count, "
        "MAX(create_time) AS last_active "
        "FROM user_query_history "
        "WHERE user_id = %s AND session_id != 'default' "
        "GROUP BY session_id "
        "ORDER BY last_active DESC "
        "LIMIT %s"
    )
    return _execute_fetchall(query_sql, (user_id, safe_limit))


# -------------------------- feedback 表操作 --------------------------


def add_feedback(
    feedback_text: str,
    user_id: int | None = None,
    user_name: str = "",
    query_content: str = "",
    answer_content: str = "",
) -> bool:
    """新增用户反馈记录。"""
    insert_sql = (
        "INSERT INTO feedback (user_id, user_name, query_content, answer_content, feedback_text) "
        "VALUES (%s, %s, %s, %s, %s)"
    )
    return _execute_write(insert_sql, (user_id, user_name, query_content, answer_content, feedback_text))


def list_feedback() -> List[Dict[str, Any]]:
    """获取全部反馈记录，按时间倒序排列。"""
    query_sql = (
        "SELECT id, user_id, user_name, query_content, answer_content, feedback_text, created_at "
        "FROM feedback ORDER BY created_at DESC"
    )
    return _execute_fetchall(query_sql)


def del_feedback(record_id: int) -> bool:
    """按主键删除反馈记录。"""
    delete_sql = "DELETE FROM feedback WHERE id = %s"
    return _execute_write(delete_sql, (record_id,))


# -------------------------- user_query_history 表操作 --------------------------


def add_query_history(query_content: str, answer_content: str) -> bool:
    """新增用户问答历史记录。"""
    insert_sql = (
        "INSERT INTO user_query_history (query_content, answer_content) VALUES (%s, %s)"
    )
    return _execute_write(insert_sql, (query_content, answer_content))


def ensure_query_history_session_schema() -> None:
    """Ensure session_id and user_id columns and indexes exist for session history queries."""
    mysql_client = MysqlConnection()
    connection_tuple = mysql_client.connect_db()
    if connection_tuple is None:
        print("[连接失败] 数据库连接异常，无法检查 user_query_history 表结构。")
        return

    connection, cursor = connection_tuple
    try:
        cursor.execute(
            """
            ALTER TABLE user_query_history
            ADD COLUMN session_id VARCHAR(64) NOT NULL DEFAULT 'default'
            """
        )
        connection.commit()
        print("✅ user_query_history 已新增 session_id 字段。")
    except Exception as column_error:  # pylint: disable=broad-except
        message = str(column_error)
        if "Duplicate column name" not in message:
            connection.rollback()
            print(f"⚠️ session_id 字段检查失败：{column_error}")

    try:
        cursor.execute("CREATE INDEX idx_session_id ON user_query_history (session_id)")
        connection.commit()
        print("✅ user_query_history 已新增 idx_session_id 索引。")
    except Exception as index_error:  # pylint: disable=broad-except
        message = str(index_error)
        if "Duplicate key name" not in message:
            connection.rollback()
            print(f"⚠️ idx_session_id 索引检查失败：{index_error}")

    # Ensure user_id column exists
    try:
        cursor.execute(
            """
            ALTER TABLE user_query_history
            ADD COLUMN user_id INT DEFAULT NULL COMMENT '关联用户ID'
            """
        )
        connection.commit()
        print("✅ user_query_history 已新增 user_id 字段。")
    except Exception as uid_error:  # pylint: disable=broad-except
        message = str(uid_error)
        if "Duplicate column name" not in message:
            connection.rollback()
            print(f"⚠️ user_id 字段检查失败：{uid_error}")

    try:
        cursor.execute("CREATE INDEX idx_user_id ON user_query_history (user_id)")
        connection.commit()
        print("✅ user_query_history 已新增 idx_user_id 索引。")
    except Exception as idx_error:  # pylint: disable=broad-except
        message = str(idx_error)
        if "Duplicate key name" not in message:
            connection.rollback()
            print(f"⚠️ idx_user_id 索引检查失败：{idx_error}")
    finally:
        mysql_client.close_db()


def add_query_history_with_session(session_id: str, query_content: str, answer_content: str, user_id: int | None = None) -> bool:
    """新增带 session_id 和 user_id 的用户问答历史记录。"""
    normalized_session = (session_id or "").strip() or "default"
    insert_sql = (
        "INSERT INTO user_query_history (session_id, query_content, answer_content, user_id) VALUES (%s, %s, %s, %s)"
    )
    return _execute_write(insert_sql, (normalized_session, query_content, answer_content, user_id))


def del_query_history(record_id: int) -> bool:
    """按主键删除用户问答历史记录。"""
    delete_sql = "DELETE FROM user_query_history WHERE id = %s"
    return _execute_write(delete_sql, (record_id,))


def update_query_history(record_id: int, query_content: str, answer_content: str) -> bool:
    """更新用户问答历史的查询与回答内容。"""
    update_sql = (
        "UPDATE user_query_history SET query_content = %s, answer_content = %s WHERE id = %s"
    )
    return _execute_write(update_sql, (query_content, answer_content, record_id))


def get_query_history_by_id(record_id: int) -> Optional[Dict[str, Any]]:
    """按主键查询单条问答历史记录。"""
    query_sql = (
        "SELECT id, query_content, answer_content, create_time FROM user_query_history WHERE id = %s"
    )
    return _execute_fetchone(query_sql, (record_id,))


def list_query_history() -> List[Dict[str, Any]]:
    """获取全部问答历史记录，按时间倒序排列。"""
    query_sql = (
        "SELECT id, query_content, answer_content, create_time FROM user_query_history ORDER BY create_time DESC"
    )
    return _execute_fetchall(query_sql)


def list_query_history_by_session(session_id: str, limit: int = 100) -> List[Dict[str, Any]]:
    """按会话查询问答历史，按时间升序返回。"""
    normalized_session = (session_id or "").strip()
    if not normalized_session:
        return []

    safe_limit = max(1, min(int(limit), 500))
    query_sql = (
        "SELECT id, session_id, query_content, answer_content, create_time "
        "FROM user_query_history "
        "WHERE session_id = %s "
        "ORDER BY create_time ASC "
        "LIMIT %s"
    )
    return _execute_fetchall(query_sql, (normalized_session, safe_limit))


def list_sessions(limit: int = 50, user_id: int | None = None) -> List[Dict[str, Any]]:
    """获取会话列表，按用户过滤，每个会话返回第一条消息（按时间）作为标题，按最近活跃时间倒序。
    若会话有自定义标题，则优先返回自定义标题。"""
    safe_limit = max(1, min(int(limit), 200))
    if user_id is not None:
        query_sql = (
            "SELECT h.session_id, "
            "COALESCE("
            "  (SELECT ct.title FROM session_custom_title ct WHERE ct.session_id = h.session_id),"
            "  (SELECT hh.query_content FROM user_query_history hh "
            "   WHERE hh.session_id = h.session_id ORDER BY hh.create_time ASC LIMIT 1)"
            ") AS first_query, "
            "MAX(h.create_time) AS last_active "
            "FROM user_query_history h "
            "WHERE h.session_id != 'default' AND h.user_id = %s "
            "GROUP BY h.session_id "
            "ORDER BY last_active DESC "
            "LIMIT %s"
        )
        return _execute_fetchall(query_sql, (user_id, safe_limit))
    else:
        query_sql = (
            "SELECT h.session_id, "
            "COALESCE("
            "  (SELECT ct.title FROM session_custom_title ct WHERE ct.session_id = h.session_id),"
            "  (SELECT hh.query_content FROM user_query_history hh "
            "   WHERE hh.session_id = h.session_id ORDER BY hh.create_time ASC LIMIT 1)"
            ") AS first_query, "
            "MAX(h.create_time) AS last_active "
            "FROM user_query_history h "
            "WHERE h.session_id != 'default' "
            "GROUP BY h.session_id "
            "ORDER BY last_active DESC "
            "LIMIT %s"
        )
        return _execute_fetchall(query_sql, (safe_limit,))


def delete_session(session_id: str, user_id: int | None = None) -> bool:
    """删除某个会话的全部记录，可按 user_id 限制。"""
    normalized_session = (session_id or "").strip()
    if not normalized_session:
        return False
    if user_id is not None:
        delete_sql = "DELETE FROM user_query_history WHERE session_id = %s AND user_id = %s"
        return _execute_write(delete_sql, (normalized_session, user_id))
    else:
        delete_sql = "DELETE FROM user_query_history WHERE session_id = %s"
        return _execute_write(delete_sql, (normalized_session,))


# -------------------------- pdf_doc_info 表操作 --------------------------


def add_pdf_info(doc_name: str, doc_path: str) -> bool:
    """新增 PDF 文档元信息记录。"""
    insert_sql = (
        "INSERT INTO pdf_doc_info (doc_name, doc_path) VALUES (%s, %s)"
    )
    return _execute_write(insert_sql, (doc_name, doc_path))


def del_pdf_info(record_id: int) -> bool:
    """按主键删除 PDF 文档信息。"""
    delete_sql = "DELETE FROM pdf_doc_info WHERE id = %s"
    return _execute_write(delete_sql, (record_id,))


def update_pdf_info(record_id: int, doc_name: str, doc_path: str) -> bool:
    """更新 PDF 文档信息的名称与路径。"""
    update_sql = (
        "UPDATE pdf_doc_info SET doc_name = %s, doc_path = %s WHERE id = %s"
    )
    return _execute_write(update_sql, (doc_name, doc_path, record_id))


def get_pdf_info_by_id(record_id: int) -> Optional[Dict[str, Any]]:
    """按主键查询 PDF 文档信息。"""
    query_sql = (
        "SELECT id, doc_name, doc_path, upload_time FROM pdf_doc_info WHERE id = %s"
    )
    return _execute_fetchone(query_sql, (record_id,))


def list_pdf_info() -> List[Dict[str, Any]]:
    """获取全部 PDF 文档信息，按上传时间倒序排列。"""
    query_sql = (
        "SELECT id, doc_name, doc_path, upload_time FROM pdf_doc_info ORDER BY upload_time DESC"
    )
    return _execute_fetchall(query_sql)


# -------------------------- session_custom_title 表操作 --------------------------


def ensure_session_title_schema() -> None:
    """Ensure session_custom_title table exists for storing user-defined session titles."""
    mysql_client = MysqlConnection()
    connection_tuple = mysql_client.connect_db()
    if connection_tuple is None:
        print("[连接失败] 数据库连接异常，无法检查 session_custom_title 表结构。")
        return

    connection, cursor = connection_tuple
    try:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS session_custom_title (
                session_id VARCHAR(200) PRIMARY KEY COMMENT '会话ID',
                title VARCHAR(200) NOT NULL COMMENT '用户自定义会话标题',
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='用户自定义会话标题'
            """
        )
        connection.commit()
    except Exception as schema_error:  # pylint: disable=broad-except
        connection.rollback()
        print(f"⚠️ session_custom_title 表结构检查失败：{schema_error}")
    finally:
        mysql_client.close_db()


def set_session_custom_title(session_id: str, title: str) -> bool:
    """插入或更新会话的自定义标题（UPSERT）。"""
    upsert_sql = (
        "INSERT INTO session_custom_title (session_id, title) VALUES (%s, %s) "
        "ON DUPLICATE KEY UPDATE title = VALUES(title), updated_at = CURRENT_TIMESTAMP"
    )
    return _execute_write(upsert_sql, (session_id, title))


# -------------------------- llm_model 表操作 --------------------------


def ensure_model_schema() -> None:
    """Ensure llm_model table exists for admin model management."""
    mysql_client = MysqlConnection()
    connection_tuple = mysql_client.connect_db()
    if connection_tuple is None:
        print("[连接失败] 数据库连接异常，无法检查 llm_model 表结构。")
        return

    connection, cursor = connection_tuple
    try:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_model (
                id INT PRIMARY KEY AUTO_INCREMENT,
                model_name VARCHAR(100) NOT NULL COMMENT '模型显示名称',
                model_id VARCHAR(200) NOT NULL COMMENT 'API模型标识符',
                api_base VARCHAR(500) NOT NULL DEFAULT 'https://openrouter.ai/api/v1' COMMENT 'API基础地址',
                api_key VARCHAR(500) NOT NULL DEFAULT '' COMMENT 'API密钥',
                is_default TINYINT NOT NULL DEFAULT 0 COMMENT '是否为默认模型',
                enabled TINYINT NOT NULL DEFAULT 1 COMMENT '是否启用',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='管理员配置的LLM模型列表'
            """
        )
        connection.commit()
    except Exception as schema_error:  # pylint: disable=broad-except
        connection.rollback()
        print(f"⚠️ llm_model 表结构检查失败：{schema_error}")
    finally:
        mysql_client.close_db()


def add_model(
    model_name: str,
    model_id: str,
    api_base: str = "https://openrouter.ai/api/v1",
    api_key: str = "",
    is_default: int = 0,
) -> bool:
    """新增LLM模型配置。"""
    insert_sql = (
        "INSERT INTO llm_model (model_name, model_id, api_base, api_key, is_default) "
        "VALUES (%s, %s, %s, %s, %s)"
    )
    return _execute_write(insert_sql, (model_name, model_id, api_base, api_key, is_default))


def list_models(enabled_only: bool = False) -> List[Dict[str, Any]]:
    """获取LLM模型列表。"""
    if enabled_only:
        query_sql = (
            "SELECT id, model_name, model_id, api_base, api_key, is_default, enabled, created_at "
            "FROM llm_model WHERE enabled = 1 ORDER BY is_default DESC, id ASC"
        )
    else:
        query_sql = (
            "SELECT id, model_name, model_id, api_base, api_key, is_default, enabled, created_at "
            "FROM llm_model ORDER BY is_default DESC, id ASC"
        )
    return _execute_fetchall(query_sql)


def del_model(record_id: int) -> bool:
    """按主键删除模型配置。"""
    delete_sql = "DELETE FROM llm_model WHERE id = %s"
    return _execute_write(delete_sql, (record_id,))


def update_model(
    record_id: int,
    model_name: str,
    model_id: str,
    api_base: str,
    api_key: str,
    is_default: int = 0,
    enabled: int = 1,
) -> bool:
    """更新模型配置。"""
    update_sql = (
        "UPDATE llm_model SET model_name = %s, model_id = %s, api_base = %s, "
        "api_key = %s, is_default = %s, enabled = %s WHERE id = %s"
    )
    return _execute_write(
        update_sql,
        (model_name, model_id, api_base, api_key, is_default, enabled, record_id),
    )


def get_model_by_id(record_id: int) -> Optional[Dict[str, Any]]:
    """按主键查询模型配置。"""
    query_sql = (
        "SELECT id, model_name, model_id, api_base, api_key, is_default, enabled, created_at "
        "FROM llm_model WHERE id = %s"
    )
    return _execute_fetchone(query_sql, (record_id,))


# -------------------------- 关键词模糊查询核心函数 --------------------------


def query_by_keyword(keyword: str, enable_log: bool = False) -> List[Dict[str, Any]]:
    """针对校园结构化数据执行多字段联合模糊查询。

    功能:
        支持按校园结构化数据的所有核心字段进行模糊匹配，并返回匹配结果列表。
    参数:
        keyword (str): 待检索的关键词，支持中文与英文输入。
        enable_log (bool): 是否在控制台输出 SQL 语句与命中条数，默认关闭。
    返回值:
        List[Dict[str, Any]]: 查询命中的记录列表，每条记录包含六个字段。
    异常:
        无显式抛出，内部打印详细的异常信息并返回空列表。
    """
    sanitized_keyword = (keyword or "").strip()
    if not sanitized_keyword:
        if enable_log:
            print("[查询提示] 关键词为空，返回结果为空列表。")
        return []

    mysql_client = MysqlConnection()
    connection_tuple = mysql_client.connect_db()
    if connection_tuple is None:
        if enable_log:
            print("❌ 数据库连接失败，无法执行查询。")
        return []

    _, cursor = connection_tuple
    like_pattern = f"%{sanitized_keyword}%"

    results: List[Dict[str, Any]] = []

    primary_query_sql = (
        "SELECT id, category, item, operation, time_requirement, channel, source_note "
        "FROM campus_struct_data "
        "WHERE category LIKE %s OR item LIKE %s OR operation LIKE %s "
        "OR time_requirement LIKE %s OR channel LIKE %s OR source_note LIKE %s "
        "ORDER BY "
        "  CASE WHEN item LIKE %s THEN 1 "
        "       WHEN category LIKE %s THEN 2 "
        "       ELSE 3 END ASC, "
        "  id ASC"
    )
    primary_params = (like_pattern,) * 8

    crawler_query_sql = (
        "SELECT id, title, contributor, publisher, publish_date, content, url, "
        "(CASE WHEN title LIKE %s THEN 1 ELSE 0 END) AS title_hit "
        "FROM campus_structured_data "
        "WHERE title LIKE %s OR contributor LIKE %s OR publisher LIKE %s "
        "OR CAST(publish_date AS CHAR) LIKE %s OR content LIKE %s OR url LIKE %s "
        "ORDER BY title_hit DESC, id DESC "
        "LIMIT 10"
    )
    crawler_params = (like_pattern,) * 7

    normalized_primary_sql = " ".join(primary_query_sql.split())  # 【优化点：SQL 日志】规避换行影响日志展示
    normalized_crawler_sql = " ".join(crawler_query_sql.split())
    try:
        cursor.execute("SET NAMES utf8mb4")  # 【编码保障】避免连接池复用时中文模糊匹配乱码
        cursor.execute(primary_query_sql, primary_params)
        primary_fetched = cursor.fetchall()
        # 添加 _src 标记以便调用方区分主表与新闻表（两表 id 独立自增，直接用 id 去重会碰撞）
        results.extend([{**row, "_src": "s"} for row in primary_fetched] if primary_fetched else [])

        # 爬虫新闻表字段与业务表不同，统一映射为 RAG 结构化上下文字段。
        cursor.execute(crawler_query_sql, crawler_params)
        crawler_fetched = cursor.fetchall()
        for row in list(crawler_fetched) if crawler_fetched else []:
            contributor = str(row.get("contributor") or "")
            publisher = str(row.get("publisher") or "")
            publish_date = row.get("publish_date")
            publish_date_text = str(publish_date) if publish_date is not None else ""
            source_note = f"URL: {row.get('url') or '-'}"
            if contributor:
                source_note += f" | 供稿单位: {contributor}"

            results.append(
                {
                    "_src": "n",
                    "id": row.get("id"),
                    "category": "校园新闻",
                    "item": row.get("title") or "未命名新闻",
                    "operation": row.get("content") or "暂无正文",
                    "time_requirement": publish_date_text,
                    "channel": publisher or "官网发布",
                    "source_note": source_note,
                }
            )

        if enable_log:
            print(f"✅ 数据库连接成功，执行查询SQL（业务表）：{normalized_primary_sql}")
            print(f"✅ 数据库连接成功，执行查询SQL（新闻表）：{normalized_crawler_sql}")
            print(f"✅ 共查询到{len(results)}条相关数据：")
    except Exception as query_error:  # pylint: disable=broad-except
        print(f"[查询异常] {query_error}")
        if enable_log:
            print("❌ SQL 执行失败，请检查关键词或数据库状态。")
    finally:
        mysql_client.close_db()

    return results


# -------------------------- 命令行查询工具函数 --------------------------


def _parse_cli_keyword(args: Sequence[str]) -> Optional[str]:
    """解析命令行参数并返回关键词字符串。"""
    if len(args) <= 1:
        return None
    keyword = " ".join(arg for arg in args[1:] if arg.strip())
    return keyword.strip() if keyword else None


def _print_struct_records(records: List[Dict[str, Any]]) -> None:
    """按照既定格式打印结构化数据查询结果。"""
    for index, record in enumerate(records, start=1):
        print(f"\n【数据{index}】")  # 【优化点：输出格式化】清晰呈现每条记录
        print(f"大类分类：{record.get('category') or '-'}")
        print(f"具体事项：{record.get('item') or '-'}")
        print(f"详细操作：{record.get('operation') or '-'}")
        print(f"时间要求：{record.get('time_requirement') or '-'}")
        print(f"办理渠道：{record.get('channel') or '-'}")
        print(f"信息来源/备注：{record.get('source_note') or '-'}")


def _run_cli_tool(args: Sequence[str]) -> None:
    """命令行入口：解析参数并执行关键词查询。"""
    keyword = _parse_cli_keyword(args)
    if not keyword:
        print("请传入查询关键词，示例：python src/database/db_operate.py 缓考")
        return

    print(f"===== 查询关键词：{keyword} =====")
    records = query_by_keyword(keyword, enable_log=True)
    if not records:
        print(f"未查询到【{keyword}】相关数据")
        print("===== 查询完成 =====")
        return

    _print_struct_records(records)
    print("\n===== 查询完成 =====")


def main() -> None:
    """脚本主入口，供命令行执行调用。"""
    _run_cli_tool(sys.argv)


if __name__ == "__main__":
    main()
