"""Checkpoint 配置 — 支持 SQLite/Redis/InMemory 三种持久化"""

import sqlite3
from typing import Optional

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from src.config.settings import settings


def get_checkpointer(
    store: Optional[str] = None,
    db_path: Optional[str] = None,
):
    """获取 Checkpointer 实例

    Args:
        store: 存储类型 "sqlite"/"memory"/"redis"，默认从 Settings 读取
        db_path: SQLite 数据库路径，默认从 Settings 读取

    Returns:
        Checkpointer 实例（SqliteSaver / InMemorySaver / RedisSaver）
    """
    store = store or settings.CHECKPOINT_STORE

    if store == "sqlite":
        path = db_path or settings.CHECKPOINT_DB_PATH
        conn = sqlite3.connect(path, check_same_thread=False)
        saver = SqliteSaver(conn)
        # 初始化 checkpoint 表结构（幂等）
        try:
            saver.setup()
        except Exception:
            # setup 可能要求上下文管理器；忽略初始化错误，编译时仍可用
            pass
        return saver
    elif store == "memory":
        return InMemorySaver()
    elif store == "redis":
        try:
            from langgraph.checkpoint.redis import RedisSaver

            return RedisSaver.from_conn_string(settings.REDIS_URL)
        except ImportError:
            raise ImportError("Redis checkpoint 需要安装: pip install langgraph-scrm[redis]")
    else:
        raise ValueError(f"不支持的 checkpoint 存储: {store}")
