# -*- coding: utf-8 -*-

import os
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量（如果存在）
load_dotenv()

from pprint import pprint
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ConfigurationError


class MongoConfig:
    # ===================== 默认连接配置 =====================
    # 优先读取环境变量；未设置时使用本地默认值。
    _host = os.getenv("MONGO_HOST", "localhost")
    _port = int(os.getenv("MONGO_PORT", 27017))
    _db_name = os.getenv("MONGO_DB_NAME", "mydatabase")
    _username = os.getenv("MONGO_USERNAME")
    _password = os.getenv("MONGO_PASSWORD")
    _auth_source = os.getenv("MONGO_AUTH_SOURCE", "admin")

    # ===================== 连接参数 =====================
    # 连接池与超时参数，避免网络抖动导致长时间阻塞。
    _max_pool_size = 100
    _connect_timeout = 5000  # 毫秒
    _socket_timeout = 3000  # 毫秒

    # ===================== 单例状态 =====================
    # 进程内共享同一个 MongoClient，避免重复创建连接。
    _client = None
    _db = None

    @classmethod
    def _build_connection_uri(cls):
        """构建MongoDB连接URI"""
        # 如果配置了用户名密码，则走鉴权 URI
        if cls._username and cls._password:
            return f"mongodb://{cls._username}:{cls._password}@{cls._host}:{cls._port}/?authSource={cls._auth_source}"
        # 否则走无鉴权 URI（本地开发常见）
        return f"mongodb://{cls._host}:{cls._port}"

    @classmethod
    def initialize(cls):
        """初始化MongoDB连接"""
        # 只在第一次调用时初始化，后续复用单例连接
        if cls._client is None:
            try:
                # 创建 MongoClient（此时不一定立刻建立网络连接）
                cls._client = MongoClient(
                    cls._build_connection_uri(),
                    maxPoolSize=cls._max_pool_size,
                    connectTimeoutMS=cls._connect_timeout,
                    socketTimeoutMS=cls._socket_timeout,
                    serverSelectionTimeoutMS=5000
                )

                # 主动 ping 验证连接是否可用（失败会抛异常）
                cls._client.admin.command('ping')
                # 获取目标数据库句柄
                cls._db = cls._client[cls._db_name]
                print("Successfully connected to MongoDB")

            except ConfigurationError as e:
                raise RuntimeError(f"MongoDB configuration error: {str(e)}")
            except ConnectionFailure as e:
                raise RuntimeError(f"Failed to connect to MongoDB: {str(e)}")
            except Exception as e:
                raise RuntimeError(f"Unexpected MongoDB connection error: {str(e)}")

    @classmethod
    def get_db(cls):
        """获取数据库实例"""
        # 延迟初始化：首次访问时自动建连
        if cls._client is None:
            cls.initialize()
        return cls._db

    @classmethod
    def get_collection(cls, collection_name):
        """获取集合实例"""
        # 统一从单例数据库句柄中获取集合对象
        return cls.get_db()[collection_name]

    @classmethod
    def close(cls):
        """关闭所有连接"""
        # 关闭连接并清空单例状态，便于测试场景重建连接
        if cls._client:
            cls._client.close()
            cls._client = None
            cls._db = None
            print("MongoDB connection closed")


# 应用启动时初始化连接
MongoConfig.initialize()


if __name__ == "__main__":
    client = MongoConfig()
    collection = MongoConfig.get_collection("my_collection")
    dic = {'name':'serena',"id":1532}
    collection.insert_one(dic)
    list_of_records = [{'name': 'amy', 'id': 1798},{'name': 'bob', 'id': 1631}]
    collection.insert_many(list_of_records)
    for record in collection.find():
        pprint(record)
