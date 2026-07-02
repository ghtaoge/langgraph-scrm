"""售后工单模块节点函数 — 长流程 + 双审批"""

import json
import logging
import uuid

from langchain_core.messages import HumanMessage
from langgraph.types import interrupt

from src.config.prompts import AFTER_SALE_ANALYZE_PROMPT, AFTER_SALE_EXECUTE_PROMPT
from src.core.llm import get_llm, safe_llm_call

logger = logging.getLogger("langgraph-scrm.after_sale")


def create_ticket_node(state: dict) -> dict:
    """创建工单节点 — 生成工单 ID，设置初始状态"""
    return {
        "ticket_id": str(uuid.uuid4())[:8],
        "status": "created",
        "approval_status": "pending",
    }


@safe_llm_call
def analyze_node(state: dict) -> dict:
    """分析节点 — AI 分析问题类型和严重度"""
    llm = get_llm()
    prompt = AFTER_SALE_ANALYZE_PROMPT.format(customer_request=state["customer_request"])
    response = llm.invoke([HumanMessage(content=prompt)])

    try:
        result = json.loads(response.content)
        issue_type = result.get("issue_type", "complaint")
        severity = result.get("severity", "medium")
    except (json.JSONDecodeError, KeyError):
        issue_type = "complaint"
        severity = "medium"

    return {"issue_type": issue_type, "severity": severity, "status": "analyzing"}


def approve_node(state: dict) -> dict:
    """主管审批节点 — interrupt() 暂停等待人工决策"""
    decision = interrupt(
        {
            "question": "请审批售后工单",
            "ticket_id": state.get("ticket_id", ""),
            "issue_type": state.get("issue_type", ""),
            "severity": state.get("severity", ""),
            "customer_request": state.get("customer_request", ""),
        }
    )

    approval_status = decision.get("approval_status", "approved")
    approver_comment = decision.get("comment", "")

    return {"approval_status": approval_status, "approver_comment": approver_comment}


@safe_llm_call
def execute_node(state: dict) -> dict:
    """执行节点 — 根据审批结果执行处理方案"""
    llm = get_llm()
    prompt = AFTER_SALE_EXECUTE_PROMPT.format(
        ticket_id=state["ticket_id"],
        issue_type=state["issue_type"],
        severity=state["severity"],
        customer_request=state["customer_request"],
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"resolution": response.content, "status": "executing"}


def verify_node(state: dict) -> dict:
    """客户验证节点 — interrupt() 暂停等待客户反馈"""
    feedback = interrupt(
        {
            "question": "请确认客户满意度",
            "ticket_id": state.get("ticket_id", ""),
            "resolution": state.get("resolution", ""),
        }
    )

    return {"customer_feedback": feedback.get("feedback", "满意"), "status": "verifying"}


def close_node(state: dict) -> dict:
    """关闭工单节点"""
    return {"status": "closed"}


# ── 条件路由 ──
def route_after_approve(state: dict) -> str:
    """审批后路由"""
    if state.get("approval_status") == "approved":
        return "execute"
    elif state.get("approval_status") == "rejected":
        return "analyze"
    else:  # needs_info
        return "analyze"


def route_after_verify(state: dict) -> str:
    """验证后路由"""
    feedback = state.get("customer_feedback", "")
    if "不满意" in feedback or "不满" in feedback:
        return "execute"
    return "close"
