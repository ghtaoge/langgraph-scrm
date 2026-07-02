"""知识库问答图集成测试 — mock LLM 与 retriever 验证 Corrective RAG 流程"""
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

from src.modules.knowledge_qa.graph import build_knowledge_qa_graph


@patch("src.modules.knowledge_qa.nodes.get_llm")
@patch("src.modules.knowledge_qa.nodes.get_retriever_tool")
def test_knowledge_qa_happy_path(mock_get_retriever, mock_get_llm):
    """知识库问答完整流程 — 文档相关 + 自检通过"""
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = "SCRM 支持微信客户管理、朋友圈营销、社群运营等功能。"
    mock_get_retriever.return_value = mock_retriever

    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = [
        AIMessage(content="relevant"),               # grade_docs
        AIMessage(content="SCRM 支持微信客户管理..."),  # generate
        AIMessage(content="passed, 回答基于文档"),     # verify
    ]
    mock_get_llm.return_value = mock_llm

    graph = build_knowledge_qa_graph()
    result = graph.invoke({
        "question": "SCRM 系统支持哪些微信功能？",
        "documents": [],
        "web_results": [],
        "answer": "",
        "citations": [],
        "verification": "",
        "retries": 0,
        "error": None,
        "error_node": None,
    })

    assert result["answer"] != ""
    assert result["verification"] == "passed"


@patch("src.modules.knowledge_qa.nodes.get_llm")
@patch("src.modules.knowledge_qa.nodes.get_retriever_tool")
def test_knowledge_qa_docs_irrelevant_triggers_web_search(mock_get_retriever, mock_get_llm):
    """文档不相关时触发 web_search 补充检索后再生成本回答"""
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = "补充搜索结果"
    mock_get_retriever.return_value = mock_retriever

    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = [
        AIMessage(content="irrelevant"),             # 第一次 grade_docs
        AIMessage(content="relevant"),               # 第二次 grade_docs（补充后）
        AIMessage(content="综合回答"),                # generate
        AIMessage(content="passed"),                 # verify
    ]
    mock_get_llm.return_value = mock_llm

    graph = build_knowledge_qa_graph()
    result = graph.invoke({
        "question": "某个冷门问题",
        "documents": [],
        "web_results": [],
        "answer": "",
        "citations": [],
        "verification": "",
        "retries": 0,
        "error": None,
        "error_node": None,
    })

    assert result["answer"] == "综合回答"
    assert result["verification"] == "passed"
