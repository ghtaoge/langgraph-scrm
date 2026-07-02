"""多Agent客服模块 State 定义"""

from typing import Annotated, Optional, TypedDict, operator


def merge_dicts(left: dict, right: dict) -> dict:
    """dict 合并 reducer — 用于并行 Agent 并发写 agent_responses 时安全合并"""
    result = dict(left or {})
    if right:
        result.update(right)
    return result


class MultiAgentState(TypedDict):
    """多Agent客服状态 — Supervisor 模式

    LangGraph 知识点:
    - agent_responses 使用 dict 合并 reducer，并行 Agent 并发写入时安全合并
    - assigned_agents 使用 reducer list 追加分派记录
    """

    customer_question: str  # 客户问题
    assigned_agents: Annotated[list[str], operator.add]  # supervisor 分派的 Agent 列表
    agent_responses: Annotated[dict, merge_dicts]  # {agent_name: response}
    final_answer: str  # 合成后的最终回答
    quality_score: float  # 质量评分 (0-10)
    feedback: str  # 质量检查反馈
    iteration: int  # 重试轮次
    error: Optional[str]
    error_node: Optional[str]
