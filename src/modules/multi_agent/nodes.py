"""多Agent客服模块节点函数 — Supervisor 模式

LangGraph 知识点:
- Supervisor 节点使用 LLM 决定分派哪些 Agent
- 每个 Agent 是独立节点（并行 fan-out）
- 质量检查不合格时循环回到 Supervisor
- 并行 Agent 通过 dict reducer 安全合并 agent_responses
"""
import json
import logging

from langchain_core.messages import HumanMessage

from src.config.prompts import (
    MULTI_AGENT_SUPERVISOR_PROMPT,
    MULTI_AGENT_PRODUCT_PROMPT,
    MULTI_AGENT_POLICY_PROMPT,
    MULTI_AGENT_ORDER_PROMPT,
    MULTI_AGENT_SYNTHESIZE_PROMPT,
    MULTI_AGENT_QUALITY_PROMPT,
)
from src.core.llm import get_llm, safe_llm_call

logger = logging.getLogger("langgraph-scrm.multi_agent")


@safe_llm_call
def supervisor_node(state: dict) -> dict:
    """Supervisor 调度节点 — 使用 LLM 决定分派哪些 Agent

    LangGraph 知识点: Supervisor 模式的核心 — LLM 动态决定路由，
    而非硬编码条件分支。
    """
    llm = get_llm()
    prompt = MULTI_AGENT_SUPERVISOR_PROMPT.format(
        customer_question=state["customer_question"]
    )
    response = llm.invoke([HumanMessage(content=prompt)])

    try:
        result = json.loads(response.content)
        agents = result.get("assigned_agents", ["product_expert"])
    except (json.JSONDecodeError, KeyError):
        agents = ["product_expert"]
        logger.warning(f"Supervisor JSON 解析失败: {response.content}")

    return {"assigned_agents": agents}


@safe_llm_call
def product_expert_node(state: dict) -> dict:
    """产品专家 Agent 节点"""
    llm = get_llm()
    prompt = MULTI_AGENT_PRODUCT_PROMPT.format(
        customer_question=state["customer_question"]
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    # 仅返回本 Agent 的部分 dict，由 reducer 合并（并行安全）
    return {"agent_responses": {"product_expert": response.content}}


@safe_llm_call
def policy_expert_node(state: dict) -> dict:
    """政策专家 Agent 节点"""
    llm = get_llm()
    prompt = MULTI_AGENT_POLICY_PROMPT.format(
        customer_question=state["customer_question"]
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"agent_responses": {"policy_expert": response.content}}


@safe_llm_call
def order_handler_node(state: dict) -> dict:
    """订单处理 Agent 节点"""
    llm = get_llm()
    prompt = MULTI_AGENT_ORDER_PROMPT.format(
        customer_question=state["customer_question"]
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"agent_responses": {"order_handler": response.content}}


@safe_llm_call
def synthesize_node(state: dict) -> dict:
    """合成节点 — 将各 Agent 回答合成为最终回答"""
    llm = get_llm()
    responses_str = json.dumps(state.get("agent_responses", {}), ensure_ascii=False)
    prompt = MULTI_AGENT_SYNTHESIZE_PROMPT.format(
        customer_question=state["customer_question"],
        agent_responses=responses_str,
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"final_answer": response.content}


@safe_llm_call
def quality_check_node(state: dict) -> dict:
    """质量检查节点 — 评估合成回答质量"""
    llm = get_llm()
    prompt = MULTI_AGENT_QUALITY_PROMPT.format(
        customer_question=state["customer_question"],
        final_answer=state.get("final_answer", ""),
    )
    response = llm.invoke([HumanMessage(content=prompt)])

    try:
        result = json.loads(response.content)
        score = float(result.get("score", 0))
        feedback = result.get("feedback", "")
    except (json.JSONDecodeError, KeyError, ValueError):
        score = 5.0
        feedback = "质量检查 JSON 解析失败"

    return {"quality_score": score, "feedback": feedback, "iteration": state.get("iteration", 0) + 1}


# ── 条件路由函数 ──
def route_after_quality(state: dict) -> str:
    """质量检查后路由 — 不合格时回到 Supervisor 重试"""
    if state.get("quality_score", 0) < 7.0 and state.get("iteration", 0) < 3:
        return "supervisor"
    return "respond"


def respond_node(state: dict) -> dict:
    """最终回复节点"""
    return {}
