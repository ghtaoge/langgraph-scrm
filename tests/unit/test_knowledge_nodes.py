"""知识库问答节点单元测试 — mock LLM 与 retriever"""
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

from src.modules.knowledge_qa.nodes import (
    grade_docs_node,
    generate_node,
    verify_node,
    route_after_grade,
    route_after_verify,
)


@patch("src.modules.knowledge_qa.nodes.get_llm")
def test_grade_docs_relevant(mock_get_llm):
    """grade_docs 文档相关时返回 docs_relevant"""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="relevant")
    mock_get_llm.return_value = mock_llm

    state = {"question": "Q", "documents": ["doc"], "web_results": [], "answer": "", "citations": [], "verification": "", "retries": 0, "error": None, "error_node": None}
    result = grade_docs_node(state)
    assert result["verification"] == "docs_relevant"


@patch("src.modules.knowledge_qa.nodes.get_llm")
def test_grade_docs_irrelevant(mock_get_llm):
    """grade_docs 文档不相关时返回 docs_irrelevant"""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="irrelevant")
    mock_get_llm.return_value = mock_llm

    state = {"question": "Q", "documents": ["doc"], "web_results": [], "answer": "", "citations": [], "verification": "", "retries": 0, "error": None, "error_node": None}
    result = grade_docs_node(state)
    assert result["verification"] == "docs_irrelevant"


@patch("src.modules.knowledge_qa.nodes.get_llm")
def test_generate_node_returns_answer(mock_get_llm):
    """generate_node 应返回回答并递增 retries"""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="基于文档的回答...")
    mock_get_llm.return_value = mock_llm

    state = {"question": "Q", "documents": ["doc"], "web_results": [], "answer": "", "citations": [], "verification": "", "retries": 0, "error": None, "error_node": None}
    result = generate_node(state)
    assert result["answer"] == "基于文档的回答..."
    assert result["retries"] == 1


@patch("src.modules.knowledge_qa.nodes.get_llm")
def test_verify_node_passed(mock_get_llm):
    """verify_node 自检通过返回 passed"""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="passed, 回答基于文档")
    mock_get_llm.return_value = mock_llm

    state = {"question": "Q", "documents": ["doc"], "web_results": [], "answer": "A", "citations": [], "verification": "", "retries": 1, "error": None, "error_node": None}
    result = verify_node(state)
    assert result["verification"] == "passed"


def test_route_after_grade_irrelevant():
    """文档不相关时路由到 web_search"""
    assert route_after_grade({"verification": "docs_irrelevant"}) == "web_search"


def test_route_after_grade_relevant():
    """文档相关时路由到 generate"""
    assert route_after_grade({"verification": "docs_relevant"}) == "generate"


def test_route_after_verify_failed_retry():
    """自检失败且未达上限时回到 generate"""
    assert route_after_verify({"verification": "failed", "retries": 1}) == "generate"


def test_route_after_verify_passed():
    """自检通过时到 respond"""
    assert route_after_verify({"verification": "passed", "retries": 1}) == "respond"


def test_route_after_verify_max_retries():
    """自检失败但达上限时到 respond"""
    assert route_after_verify({"verification": "failed", "retries": 3}) == "respond"
