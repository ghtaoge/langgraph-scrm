"""售后工单节点单元测试"""

import json
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

from src.modules.after_sale.nodes import (
    analyze_node,
    create_ticket_node,
    route_after_approve,
    route_after_verify,
)


def test_create_ticket_node_generates_id():
    """create_ticket_node 应生成工单 ID 与初始状态"""
    result = create_ticket_node({})
    assert "ticket_id" in result and len(result["ticket_id"]) > 0
    assert result["status"] == "created"
    assert result["approval_status"] == "pending"


@patch("src.modules.after_sale.nodes.get_llm")
def test_analyze_node_returns_issue_type(mock_get_llm):
    """analyze_node 应返回问题类型和严重度"""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content=json.dumps({"issue_type": "refund", "severity": "high"}))
    mock_get_llm.return_value = mock_llm

    state = {
        "customer_request": "要求退款",
        "ticket_id": "T1",
        "issue_type": "",
        "severity": "",
        "approval_status": "",
        "approver_comment": "",
        "resolution": "",
        "customer_feedback": "",
        "status": "",
        "error": None,
        "error_node": None,
    }
    result = analyze_node(state)
    assert result["issue_type"] == "refund"
    assert result["severity"] == "high"


def test_route_after_approve_approved():
    assert route_after_approve({"approval_status": "approved"}) == "execute"


def test_route_after_approve_rejected():
    assert route_after_approve({"approval_status": "rejected"}) == "analyze"


def test_route_after_approve_needs_info():
    assert route_after_approve({"approval_status": "needs_info"}) == "analyze"


def test_route_after_verify_satisfied():
    assert route_after_verify({"customer_feedback": "满意"}) == "close"


def test_route_after_verify_unsatisfied():
    assert route_after_verify({"customer_feedback": "不满意"}) == "execute"
