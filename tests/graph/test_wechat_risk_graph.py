"""微信风控图集成测试 — 四路分支 + 高风险 interrupt 上报"""

import json
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from src.modules.wechat_risk.graph import build_wechat_risk_graph


@patch("src.modules.wechat_risk.nodes.get_llm")
def test_wechat_risk_normal_message(mock_get_llm):
    """正常消息直接放行"""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content=json.dumps({"message_type": "normal"}))
    mock_get_llm.return_value = mock_llm

    checkpointer = InMemorySaver()
    graph = build_wechat_risk_graph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "risk-normal"}}
    result = graph.invoke(
        {
            "sender": "员工A",
            "content": "今天天气不错",
            "message_id": "",
            "message_type": "",
            "risk_score": 0.0,
            "risk_category": "",
            "action": "",
            "escalation_decision": "",
            "log_entry": {},
            "error": None,
            "error_node": None,
        },
        config=config,
    )

    assert result["message_type"] == "normal"
    assert result["action"] == "allow"


@patch("src.modules.wechat_risk.nodes.get_llm")
def test_wechat_risk_high_risk_escalation(mock_get_llm):
    """高风险敏感消息 → interrupt 上报 → 主管批准拦截"""
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = [
        AIMessage(content=json.dumps({"message_type": "sensitive"})),  # classify
        AIMessage(content=json.dumps({"risk_score": 90.0, "risk_category": "info_leak"})),  # risk_assess
    ]
    mock_get_llm.return_value = mock_llm

    checkpointer = InMemorySaver()
    graph = build_wechat_risk_graph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "risk-001"}}

    # 第一轮 — 在 escalate 处 interrupt
    result = graph.invoke(
        {
            "sender": "员工E",
            "content": "我把客户名单发给你了，注意保密",
            "message_id": "",
            "message_type": "",
            "risk_score": 0.0,
            "risk_category": "",
            "action": "",
            "escalation_decision": "",
            "log_entry": {},
            "error": None,
            "error_node": None,
        },
        config=config,
    )

    assert result.get("risk_score") == 90.0

    # 恢复 — 主管批准拦截
    result = graph.invoke(Command(resume={"decision": "approve_block"}), config=config)
    assert result["action"] == "block"
