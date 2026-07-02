"""共享测试 fixtures — mock LLM、测试 State、临时目录"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage


@pytest.fixture
def mock_llm():
    """Mock LLM，返回固定 AIMessage"""
    llm = MagicMock()
    llm.invoke = MagicMock(return_value=AIMessage(content="mock response"))
    llm.ainvoke = AsyncMock(return_value=AIMessage(content="mock response"))
    return llm


@pytest.fixture
def mock_llm_with_tool_calls():
    """Mock LLM，返回带 tool_calls 的 AIMessage"""
    llm = MagicMock()
    msg = AIMessage(
        content="",
        tool_calls=[{"name": "retriever", "args": {"query": "test"}, "id": "tc1"}],
    )
    llm.invoke = MagicMock(return_value=msg)
    llm.ainvoke = AsyncMock(return_value=msg)
    return llm


@pytest.fixture
def tmp_data_dir(tmp_path):
    """临时数据目录（用于 SQLite/Chroma 测试）"""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir
