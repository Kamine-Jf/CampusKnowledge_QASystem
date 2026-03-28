# -*- coding: utf-8 -*-
"""MySQL 数据库连接封装模块。"""

from __future__ import annotations

import threading
from typing import Optional, Tuple

import pymysql
from pymysql.cursors import DictCursor
from pymysql.err import MySQLError, OperationalError, InterfaceError

from .mysql_config import MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PWD, MYSQL_DB


# ===================== 线程安全连接池（RAG效率优化） =====================
# 消除每次查询都创建/销毁 TCP 连接的开销，双源检索场景下可减少 ~60ms/次
_POOL_SIZE = 4
_pool_lock = threading.Lock()
_pool: list[pymysql.connections.Connection] = []
_pool_initialized = False


def _make_connection(use_database: bool = True) -> pymysql.connections.Connection:
    """创建一个新的 MySQL 物理连接。"""
    kwargs = {
        "host": MYSQL_HOST,
        "port": MYSQL_PORT,
        "user": MYSQL_USER,
        "password": MYSQL_PWD,
        "charset": "utf8mb4",
        "cursorclass": DictCursor,
        "autocommit": False,
        "init_command": "SET NAMES utf8mb4",
    }
    if use_database:
        kwargs["database"] = MYSQL_DB
    return pymysql.connect(**kwargs)


def get_pooled_connection(use_database: bool = True) -> pymysql.connections.Connection:
    """从连接池获取可用连接，池空时新建。连接已预设 utf8mb4 编码。"""
    with _pool_lock:
        while _pool:
            conn = _pool.pop()
            try:
                conn.ping(reconnect=False)
                return conn
            except (OperationalError, InterfaceError, MySQLError):
                try:
                    conn.close()
                except Exception:
                    pass
    return _make_connection(use_database)


def return_pooled_connection(conn: pymysql.connections.Connection) -> None:
    """归还连接到池中；池满则关闭连接。"""
    if conn is None:
        return
    try:
        conn.ping(reconnect=False)
    except (OperationalError, InterfaceError, MySQLError):
        try:
            conn.close()
        except Exception:
            pass
        return
    with _pool_lock:
        if len(_pool) < _POOL_SIZE:
            _pool.append(conn)
        else:
            try:
                conn.close()
            except Exception:
                pass


class MysqlConnection:
    """封装 MySQL 连接的创建、重连与关闭逻辑。"""

    def __init__(self) -> None:
        """初始化连接占位符，避免重复创建物理连接。"""
        self._connection: Optional[pymysql.connections.Connection] = None
        self._cursor: Optional[DictCursor] = None
        self._from_pool: bool = False

    def connect_db(self, use_database: bool = True) -> Optional[Tuple[pymysql.connections.Connection, DictCursor]]:
        """创建或复用数据库连接与游标。

        参数说明：
            use_database (bool): 是否在连接时选择默认数据库，当数据库尚未创建时设为 False。

        返回值说明：
            Optional[Tuple[Connection, DictCursor]]: 成功时返回连接对象与字典游标元组，失败时返回 None。

        异常处理：
            捕获 MySQLError 及其子类异常，输出详细报错信息并返回 None，确保调用方能够安全判定连接状态。
        """
        if self._connection:
            try:
                # 使用 ping 实现自动重连，避免连接长时间空闲后被 MySQL 服务断开。
                self._connection.ping(reconnect=True)
                if self._cursor:
                    return self._connection, self._cursor
                self._cursor = self._connection.cursor()
                return self._connection, self._cursor
            except (OperationalError, InterfaceError) as reconnect_error:
                print(f"[数据库连接失效] {reconnect_error}")
                self.close_db()

        try:
            self._connection = get_pooled_connection(use_database)
            self._from_pool = True
            self._cursor = self._connection.cursor()
            return self._connection, self._cursor
        except MySQLError as connect_error:
            print(f"[数据库连接错误] {connect_error}")
            self._connection = None
            self._cursor = None
            return None

    def close_db(self) -> None:
        """关闭游标与连接，防止资源泄露。"""
        if self._cursor:
            try:
                self._cursor.close()
            except MySQLError as cursor_close_error:
                print(f"[游标关闭异常] {cursor_close_error}")
        if self._connection:
            if self._from_pool:
                return_pooled_connection(self._connection)
            else:
                try:
                    self._connection.close()
                except MySQLError as conn_close_error:
                    print(f"[连接关闭异常] {conn_close_error}")
        self._cursor = None
        self._connection = None
        self._from_pool = False

    def __del__(self) -> None:
        """析构函数中自动释放数据库资源，避免忘记手动关闭。"""
        self.close_db()
