"""Milvus 连接封装模块。"""
from __future__ import annotations

import time
from typing import Optional

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

from .milvus_config import (
    INDEX_PARAMS,
    MILVUS_COLLECTION_NAME,
    MILVUS_HOST,
    MILVUS_PORT,
    VECTOR_DIM,
)


class MilvusConnection:
    """Milvus 连接管理类。

    功能:
        负责 Milvus 服务的连接管理与集合创建。
    属性:
        _alias (str): 连接别名，对应 pymilvus 的连接标识。
        _connected (bool): 当前是否已建立连接的标记。
    """

    def __init__(self) -> None:
        """初始化连接状态。

        功能:
            设置默认连接别名并初始化连接标记。
        参数:
            无。
        返回值:
            None。
        异常:
            无。
        """
        self._alias: str = "default"
        self._connected: bool = False

    def connect(self, retry: int = 3, interval: float = 2.0) -> None:
        """建立与 Milvus 服务的连接。

        功能:
            建立到 Milvus 的网络连接并检测服务可用性，失败时支持重试。
        参数:
            retry (int): 允许的最大重试次数。
            interval (float): 每次重试之间的等待秒数。
        返回值:
            None。
        异常:
            ConnectionError: 在重试次数耗尽后仍无法连通 Milvus 服务时抛出。
        """
        last_error: Optional[Exception] = None
        for attempt in range(1, retry + 1):
            try:
                connections.disconnect(self._alias)
            except Exception:
                # 忽略断开过程中可能出现的未连接异常
                pass

            try:
                connections.connect(
                    alias=self._alias,
                    host=MILVUS_HOST,
                    port=MILVUS_PORT,
                )
                utility.get_server_version(using=self._alias)
                self._connected = True
                return
            except Exception as exc:  # pylint: disable=broad-except
                last_error = exc
                print(f"Milvus 连接失败（第{attempt}次尝试）：{exc}")
                time.sleep(interval)

        raise ConnectionError(
            f"Milvus 服务连接失败，请检查 Docker 容器状态与端口占用。最后错误信息：{last_error}"
        ) from last_error

    def disconnect(self) -> None:
        """断开与 Milvus 服务的连接。

        功能:
            主动释放与 Milvus 的连接资源。
        参数:
            无。
        返回值:
            None。
        异常:
            无，内部吞并 pymilvus 抛出的未连接异常。
        """
        try:
            connections.disconnect(self._alias)
        finally:
            self._connected = False

    def get_collection(self) -> Collection:
        """获取或创建默认向量集合。

        功能:
            根据预设 schema 构建 Milvus 集合，并保证索引存在且已加载。
        参数:
            无。
        返回值:
            Collection: 已存在或新建的 Milvus 集合对象。
        异常:
            ConnectionError: 当前未建立连接时触发。
        """
        if not self._connected:
            raise ConnectionError("尚未连接 Milvus 服务，请先调用 connect().")

        if not utility.has_collection(MILVUS_COLLECTION_NAME, using=self._alias):
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2048),
                FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM),
            ]
            schema = CollectionSchema(fields=fields, description="校园问答向量集合")
            collection = Collection(
                name=MILVUS_COLLECTION_NAME,
                schema=schema,
                using=self._alias,
            )
            collection.create_index(
                field_name="vector",
                index_params=INDEX_PARAMS,
            )  # 匹配度优化：兼容 pymilvus 2.4.4，移除 using 参数
            collection.load()
            return collection

        collection = Collection(name=MILVUS_COLLECTION_NAME, using=self._alias)
        if not collection.has_index():
            collection.create_index(
                field_name="vector",
                index_params=INDEX_PARAMS,
            )  # 匹配度优化：兼容 pymilvus 2.4.4，移除 using 参数
        collection.load()
        return collection


_GLOBAL_CONNECTION: Optional[MilvusConnection] = None


def connect_milvus(retry: int = 3, interval: float = 2.0) -> MilvusConnection:
    """提供统一的 Milvus 连接入口，供其他模块复用。"""
    global _GLOBAL_CONNECTION  # pylint: disable=global-statement
    if _GLOBAL_CONNECTION is None:
        _GLOBAL_CONNECTION = MilvusConnection()
    _GLOBAL_CONNECTION.connect(retry=retry, interval=interval)
    return _GLOBAL_CONNECTION
