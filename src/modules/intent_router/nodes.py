"""意图路由模块节点函数

LangGraph 知识点:
- 每个节点是一个函数，接收 state dict，返回 state 更新 dict
- 节点函数只更新需要修改的字段，其他字段保持不变
- 使用 @safe_llm_call 装饰器处理 LLM 调用异常
"""
import json
import logging

from langchain_core.messages import HumanMessage

from src.config.prompts import (
    INTENT_ROUTER_CLASSIFY_PROMPT,
    INTENT_ROUTER_RESPOND_CONSULT_PROMPT,
    INTENT_ROUTER_ESCALATE_COMPLAINT_PROMPT,
    INTENT_ROUTER_HANDLE_AFTER_SALE_PROMPT,
    INTENT_ROUTER_OTHER_PROMPT,
)
from src.core.llm import get_llm, safe_llm_call

logger = logging.getLogger("langgraph-scrm.intent_router")


# ── 意图分类节点 ──
@safe_llm_call
def classify_node(state: dict) -> dict:
    """意图分类节点 — 调用 LLM 将客户消息分为 4 类

    LangGraph 知识点: add_node("classify", classify_node)
    节点函数接收完整 state，返回需要更新的字段。
    """
    llm = get_llm()
    prompt = INTENT_ROUTER_CLASSIFY_PROMPT.format(message=state["message"])
    response = llm.invoke([HumanMessage(content=prompt)])

    # 解析 LLM 返回的 JSON
    try:
        result = json.loads(response.content)
        intent = result.get("intent", "other")
        confidence = result.get("confidence", 0.5)
    except (json.JSONDecodeError, KeyError):
        intent = "other"
        confidence = 0.0
        logger.warning(f"意图分类 JSON 解析失败，降级为 other: {response.content}")

    # 根据意图分配技能组
    skill_group_map = {
        "consult": "产品咨询组",
        "complaint": "投诉处理组",
        "after_sale": "售后服务组",
        "other": "通用客服组",
    }

    return {
        "intent": intent,
        "confidence": confidence,
        "skill_group": skill_group_map.get(intent, "通用客服组"),
    }


# ── 各意图处理节点 ──
@safe_llm_call
def respond_consult(state: dict) -> dict:
    """咨询类处理节点"""
    llm = get_llm()
    prompt = INTENT_ROUTER_RESPOND_CONSULT_PROMPT.format(
        message=state["message"], intent=state["intent"]
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"response": response.content}


@safe_llm_call
def escalate_complaint(state: dict) -> dict:
    """投诉类处理节点"""
    llm = get_llm()
    prompt = INTENT_ROUTER_ESCALATE_COMPLAINT_PROMPT.format(message=state["message"])
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"response": response.content}


@safe_llm_call
def handle_after_sale(state: dict) -> dict:
    """售后类处理节点"""
    llm = get_llm()
    prompt = INTENT_ROUTER_HANDLE_AFTER_SALE_PROMPT.format(message=state["message"])
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"response": response.content}


@safe_llm_call
def handle_other(state: dict) -> dict:
    """其他类处理节点"""
    llm = get_llm()
    prompt = INTENT_ROUTER_OTHER_PROMPT.format(message=state["message"])
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"response": response.content}


# ── 条件路由函数 ──
def route_by_intent(state: dict) -> str:
    """条件路由函数 — 根据意图分流到不同处理节点

    LangGraph 知识点: add_conditional_edges("classify", route_by_intent)
    条件路由函数返回下一个节点名（字符串）。
    """
    intent = state.get("intent", "other")
    route_map = {
        "consult": "respond_consult",
        "complaint": "escalate_complaint",
        "after_sale": "handle_after_sale",
        "other": "handle_other",
    }
    return route_map.get(intent, "handle_other")
