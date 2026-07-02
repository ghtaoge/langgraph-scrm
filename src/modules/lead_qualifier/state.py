"""线索评级模块 State 定义"""
from typing import Annotated, Optional, TypedDict, operator


class LeadQualifierState(TypedDict):
    """线索评级状态

    LangGraph 知识点:
    - Annotated[list, operator.add] 用于列表追加（reducer）
    - 循环图中的状态需要在每轮迭代中累积
    """
    lead_info: dict                     # 线索基本信息（来源/公司/职位）
    questions_asked: Annotated[list[str], operator.add]    # 已提问列表（追加模式）
    answers_received: Annotated[list[str], operator.add]   # 已回答列表（追加模式）
    score: float                        # 当前评分 (0-100)
    score_history: Annotated[list[float], operator.add]    # 每轮评分记录（追加模式）
    qualification: str                  # 最终评级: hot/warm/cold
    human_decision: str                 # 人工审核结果: approve/reject/needs_info
    error: Optional[str]
    error_node: Optional[str]
