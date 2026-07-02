"""测试 Checkpoint 配置"""

from src.core.checkpoint import get_checkpointer


def test_get_sqlite_checkpointer(tmp_path):
    """SQLite checkpointer 正常创建"""
    db_path = str(tmp_path / "checkpoints.db")
    cp = get_checkpointer(store="sqlite", db_path=db_path)
    assert cp is not None


def test_get_memory_checkpointer():
    """InMemory checkpointer 正常创建（测试用）"""
    cp = get_checkpointer(store="memory")
    assert cp is not None
