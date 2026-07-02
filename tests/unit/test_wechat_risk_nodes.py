"""微信风控节点单元测试"""
import json
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

from src.modules.wechat_risk.nodes import (
    receive_message_node,
    classify_node,
    route_after_classify,
    route_after_risk_assess,
)


def test_receive_message_node_generates_id():
    """receive_message_node 应生成消息 ID"""
    result = receive_message_node({})
    assert "message_id" in result and len(result["message_id"]) > 0


@patch("src.modules.wechat_risk.nodes.get_llm")
def test_classify_node_returns_type(mock_get_llm):
    """classify_node 应返回消息类型"""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(
        content=json.dumps({"message_type": "sensitive", "reason": "涉及客户数据"})
    )
    mock_get_llm.return_value = mock_llm

    state = {"sender": "员工A", "content": "我把客户名单发给你了", "message_id": "M1", "message_type": "", "risk_score": 0.0, "risk_category": "", "action": "", "escalation_decision": "", "log_entry": {}, "error": None, "error_node": None}
    result = classify_node(state)
    assert result["message_type"] == "sensitive"


@patch("src.modules.wechat_risk.nodes.get_llm")
def test_classify_node_handles_json_error(mock_get_llm):
    """classify_node JSON 解析失败时降级为 normal"""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="not json")
    mock_get_llm.return_value = mock_llm

    state = {"sender": "员工A", "content": "hi", "message_id": "M1", "message_type": "", "risk_score": 0.0, "risk_category": "", "action": "", "escalation_decision": "", "log_entry": {}, "error": None, "error_node": None}
    result = classify_node(state)
    assert result["message_type"] == "normal"


def test_route_after_classify_four_ways():
    """四路路由正确分流"""
    assert route_after_classify({"message_type": "normal"}) == "allow"
    assert route_after_classify({"message_type": "business"}) == "log_only"
    assert route_after_classify({"message_type": "sensitive"}) == "risk_assess"
    assert route_after_classify({"message_type": "violation"}) == "risk_assess"
    assert route_after_classify({"message_type": "unknown"}) == "allow"


def test_route_after_risk_assess_high():
    """高风险(≥80)路由到 escalate"""
    assert route_after_risk_assess({"risk_score": 85.0}) == "escalate"


def test_route_after_risk_assess_low():
    """低风险路由到 warn"""
    assert route_after_risk_assess({"risk_score": 50.0}) == "warn"
