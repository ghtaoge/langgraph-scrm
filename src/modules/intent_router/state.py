"""意图路由模块 State 定义"""

from typing import Optional, TypedDict


class IntentRouterState(TypedDict):
    """意图路由状态

    LangGraph 知识点: StateGraph 使用 TypedDict 定义状态，
    所有节点函数接收 state 并返回 state 更新 dict。
    """

    message: str  # 客户原始消息
    intent: str  # 分类结果: consult/complaint/after_sale/other
    confidence: float  # 分类置信度 (0-1)
    skill_group: str  # 分配的技能组
    response: str  # 最终回复
    error: Optional[str]  # 错误信息
    error_node: Optional[str]  # 发生错误的节点名
