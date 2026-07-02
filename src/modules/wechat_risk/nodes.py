"""微信风控模块节点函数 — 四路条件分支 + interrupt 上报"""

import json
import logging
import uuid

from langchain_core.messages import HumanMessage
from langgraph.types import interrupt

from src.config.prompts import WECHAT_RISK_ASSESS_PROMPT, WECHAT_RISK_CLASSIFY_PROMPT
from src.core.llm import get_llm, safe_llm_call

logger = logging.getLogger("langgraph-scrm.wechat_risk")


def receive_message_node(state: dict) -> dict:
    """接收消息节点 — 设置初始状态"""
    return {"message_id": str(uuid.uuid4())[:8]}


@safe_llm_call
def classify_node(state: dict) -> dict:
    """消息分类节点 — 将消息分为 4 类"""
    llm = get_llm()
    prompt = WECHAT_RISK_CLASSIFY_PROMPT.format(content=state["content"], sender=state["sender"])
    response = llm.invoke([HumanMessage(content=prompt)])

    try:
        result = json.loads(response.content)
        message_type = result.get("message_type", "normal")
    except (json.JSONDecodeError, KeyError):
        message_type = "normal"
        logger.warning(f"消息分类 JSON 解析失败: {response.content}")

    return {"message_type": message_type}


def allow_node(state: dict) -> dict:
    """放行节点 — 正常消息直接放行"""
    return {"action": "allow", "log_entry": {"action": "allow", "message_id": state.get("message_id")}}


def log_only_node(state: dict) -> dict:
    """记录节点 — 业务消息记录后放行"""
    return {"action": "log_only", "log_entry": {"action": "log_only", "message_id": state.get("message_id")}}


@safe_llm_call
def risk_assess_node(state: dict) -> dict:
    """风险评估节点 — 评估敏感/违规消息的风险"""
    llm = get_llm()
    prompt = WECHAT_RISK_ASSESS_PROMPT.format(
        content=state["content"], message_type=state.get("message_type", "sensitive")
    )
    response = llm.invoke([HumanMessage(content=prompt)])

    try:
        result = json.loads(response.content)
        risk_score = float(result.get("risk_score", 50))
        risk_category = result.get("risk_category", "other")
    except (json.JSONDecodeError, KeyError, ValueError):
        risk_score = 50.0
        risk_category = "other"

    return {"risk_score": risk_score, "risk_category": risk_category}


def escalate_node(state: dict) -> dict:
    """上报节点 — interrupt() 暂停等待主管决策"""
    decision = interrupt(
        {
            "question": "高风险消息需主管审核",
            "message_id": state.get("message_id"),
            "content": state.get("content"),
            "risk_score": state.get("risk_score"),
            "risk_category": state.get("risk_category"),
        }
    )
    return {"escalation_decision": decision.get("decision", "approve_block")}


def block_node(state: dict) -> dict:
    """拦截节点 — 拦截消息 + 通知发送者"""
    return {"action": "block", "log_entry": {"action": "block", "message_id": state.get("message_id")}}


def warn_node(state: dict) -> dict:
    """提醒节点 — 低风险消息提醒发送者"""
    return {"action": "warn", "log_entry": {"action": "warn", "message_id": state.get("message_id")}}


def log_node(state: dict) -> dict:
    """日志记录节点 — 写入风控审计日志"""
    return {}


# ── 条件路由 ──
def route_after_classify(state: dict) -> str:
    """分类后四路路由"""
    message_type = state.get("message_type", "normal")
    route_map = {
        "normal": "allow",
        "business": "log_only",
        "sensitive": "risk_assess",
        "violation": "risk_assess",
    }
    return route_map.get(message_type, "allow")


def route_after_risk_assess(state: dict) -> str:
    """风险评估后路由"""
    risk_score = state.get("risk_score", 0)
    if risk_score >= 80:
        return "escalate"
    return "warn"
