"""售后工单图集成测试 — 双 interrupt + resume 完整流程"""

import json
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from src.modules.after_sale.graph import build_after_sale_graph


@patch("src.modules.after_sale.nodes.get_llm")
def test_after_sale_full_flow_approve_satisfied(mock_get_llm):
    """售后完整流程 — 审批通过 + 客户满意 → 关闭"""
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = [
        AIMessage(content=json.dumps({"issue_type": "refund", "severity": "high"})),  # analyze
        AIMessage(content="已为您办理退款"),  # execute
    ]
    mock_get_llm.return_value = mock_llm

    checkpointer = InMemorySaver()
    graph = build_after_sale_graph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "after-sale-001"}}

    initial = {
        "customer_request": "产品质量有问题，要求退款",
        "ticket_id": "",
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

    # 第一轮 — 在 approve 处 interrupt
    result = graph.invoke(initial, config=config)
    assert result.get("issue_type") == "refund"

    # 恢复 — 主管审批通过
    result = graph.invoke(Command(resume={"approval_status": "approved", "comment": "同意"}), config=config)
    # 在 verify 处 interrupt
    assert result.get("resolution") == "已为您办理退款"

    # 恢复 — 客户满意
    result = graph.invoke(Command(resume={"feedback": "满意"}), config=config)
    assert result["status"] == "closed"


@patch("src.modules.after_sale.nodes.get_llm")
def test_after_sale_unsatisfied_reexecutes(mock_get_llm):
    """客户不满意时回到 execute 重试"""
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = [
        AIMessage(content=json.dumps({"issue_type": "repair", "severity": "medium"})),  # analyze
        AIMessage(content="已安排维修"),  # 第一次 execute
        AIMessage(content="已重新处理"),  # 第二次 execute（重试）
    ]
    mock_get_llm.return_value = mock_llm

    checkpointer = InMemorySaver()
    graph = build_after_sale_graph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "after-sale-002"}}

    initial = {
        "customer_request": "产品需要维修",
        "ticket_id": "",
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

    graph.invoke(initial, config=config)  # interrupt at approve
    graph.invoke(Command(resume={"approval_status": "approved"}), config=config)  # interrupt at verify
    # 客户不满意 → 回到 execute，再次 interrupt at verify
    result = graph.invoke(Command(resume={"feedback": "不满意"}), config=config)
    assert result.get("resolution") == "已重新处理"
