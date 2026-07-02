"""意图路由图集成测试 — 使用 fake LLM 验证完整图流程"""
import json
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

from src.modules.intent_router.graph import build_intent_router_graph


@patch("src.modules.intent_router.nodes.get_llm")
def test_intent_router_full_flow_consult(mock_get_llm):
    """意图路由完整流程 — 咨询类"""
    mock_llm = MagicMock()
    # classify 节点返回咨询意图
    mock_llm.invoke.side_effect = [
        AIMessage(content=json.dumps({"intent": "consult", "confidence": 0.9})),
        AIMessage(content="我们的产品价格如下..."),
    ]
    mock_get_llm.return_value = mock_llm

    graph = build_intent_router_graph()
    result = graph.invoke({
        "message": "我想咨询产品价格",
        "intent": "",
        "confidence": 0.0,
        "skill_group": "",
        "response": "",
        "error": None,
        "error_node": None,
    })

    assert result["intent"] == "consult"
    assert result["skill_group"] == "产品咨询组"
    assert result["response"] != ""


@patch("src.modules.intent_router.nodes.get_llm")
def test_intent_router_full_flow_complaint(mock_get_llm):
    """意图路由完整流程 — 投诉类"""
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = [
        AIMessage(content=json.dumps({"intent": "complaint", "confidence": 0.85})),
        AIMessage(content="非常抱歉给您带来不好的体验..."),
    ]
    mock_get_llm.return_value = mock_llm

    graph = build_intent_router_graph()
    result = graph.invoke({
        "message": "服务太差了我要投诉",
        "intent": "",
        "confidence": 0.0,
        "skill_group": "",
        "response": "",
        "error": None,
        "error_node": None,
    })

    assert result["intent"] == "complaint"
    assert result["skill_group"] == "投诉处理组"
