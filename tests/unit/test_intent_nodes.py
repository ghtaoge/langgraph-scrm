"""意图路由节点单元测试 — mock LLM，验证节点逻辑"""
import json
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

from src.modules.intent_router.nodes import (
    classify_node,
    respond_consult,
    route_by_intent,
)


@patch("src.modules.intent_router.nodes.get_llm")
def test_classify_node_returns_intent(mock_get_llm):
    """classify_node 应返回意图和置信度"""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(
        content=json.dumps({"intent": "consult", "confidence": 0.95})
    )
    mock_get_llm.return_value = mock_llm

    state = {"message": "我想咨询产品价格", "intent": "", "confidence": 0.0, "skill_group": "", "response": "", "error": None, "error_node": None}
    result = classify_node(state)

    assert result["intent"] == "consult"
    assert result["confidence"] == 0.95
    assert result["skill_group"] == "产品咨询组"


@patch("src.modules.intent_router.nodes.get_llm")
def test_classify_node_handles_json_error(mock_get_llm):
    """classify_node 应处理 JSON 解析失败"""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="not json")
    mock_get_llm.return_value = mock_llm

    state = {"message": "test", "intent": "", "confidence": 0.0, "skill_group": "", "response": "", "error": None, "error_node": None}
    result = classify_node(state)

    assert result["intent"] == "other"
    assert result["confidence"] == 0.0


def test_route_by_intent_returns_correct_node():
    """route_by_intent 应正确路由到对应节点"""
    assert route_by_intent({"intent": "consult"}) == "respond_consult"
    assert route_by_intent({"intent": "complaint"}) == "escalate_complaint"
    assert route_by_intent({"intent": "after_sale"}) == "handle_after_sale"
    assert route_by_intent({"intent": "other"}) == "handle_other"
    assert route_by_intent({"intent": "unknown"}) == "handle_other"


@patch("src.modules.intent_router.nodes.get_llm")
def test_respond_consult_returns_response(mock_get_llm):
    """respond_consult 应返回咨询回复"""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="我们的产品价格如下...")
    mock_get_llm.return_value = mock_llm

    state = {"message": "价格咨询", "intent": "consult", "confidence": 0.9, "skill_group": "产品咨询组", "response": "", "error": None, "error_node": None}
    result = respond_consult(state)

    assert "response" in result
