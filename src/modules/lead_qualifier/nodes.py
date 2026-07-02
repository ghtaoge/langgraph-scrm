"""线索评级模块节点函数

LangGraph 知识点:
- 循环图：节点之间形成循环（ask_question → evaluate → ask_question）
- interrupt()：在节点内暂停执行，等待外部输入
- Command(resume=...)：恢复执行时传入人工决策
"""

import json
import logging

from langchain_core.messages import HumanMessage
from langgraph.types import interrupt

from src.config.prompts import LEAD_QUALIFIER_EVALUATE_PROMPT, LEAD_QUALIFIER_QUESTION_PROMPT
from src.config.settings import settings
from src.core.llm import get_llm, safe_llm_call

logger = logging.getLogger("langgraph-scrm.lead_qualifier")


@safe_llm_call
def ask_question_node(state: dict) -> dict:
    """提问节点 — 根据线索信息提出评估问题

    LangGraph 知识点: 循环图中的节点，每次迭代生成新问题。
    questions_asked 使用 Annotated[list, operator.add] reducer，
    返回 {"questions_asked": [new_question]} 会追加到现有列表。
    """
    llm = get_llm()
    prompt = LEAD_QUALIFIER_QUESTION_PROMPT.format(
        lead_info=json.dumps(state["lead_info"], ensure_ascii=False),
        questions_asked=json.dumps(state.get("questions_asked", []), ensure_ascii=False),
        answers_received=json.dumps(state.get("answers_received", []), ensure_ascii=False),
        score=state.get("score", 0),
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"questions_asked": [response.content]}


@safe_llm_call
def evaluate_node(state: dict) -> dict:
    """评估节点 — 根据问答给出评分和评级

    LangGraph 知识点: 循环终止条件判断。
    """
    llm = get_llm()
    qa_pairs = "\n".join(
        f"Q: {q}\nA: {a}" for q, a in zip(state.get("questions_asked", []), state.get("answers_received", []))
    )
    prompt = LEAD_QUALIFIER_EVALUATE_PROMPT.format(
        lead_info=json.dumps(state["lead_info"], ensure_ascii=False),
        qa_pairs=qa_pairs,
    )
    response = llm.invoke([HumanMessage(content=prompt)])

    try:
        result = json.loads(response.content)
        score = float(result.get("score", 0))
        qualification = result.get("qualification", "cold")
    except (json.JSONDecodeError, KeyError, ValueError):
        score = 0.0
        qualification = "cold"
        logger.warning(f"评估结果 JSON 解析失败: {response.content}")

    return {
        "score": score,
        "score_history": [score],
        "qualification": qualification,
    }


def human_review_node(state: dict) -> dict:
    """人工审核节点 — 使用 interrupt() 暂停等待人工决策

    LangGraph 知识点:
    - interrupt() 在节点内部调用，暂停图执行
    - interrupt() 的返回值是 resume 时传入的值
    - 使用 Command(resume={"human_decision": "approve"}) 恢复
    """
    decision = interrupt(
        {
            "question": "请审核线索评级结果",
            "score": state.get("score", 0),
            "qualification": state.get("qualification", "cold"),
            "lead_info": state.get("lead_info", {}),
        }
    )

    # decision 是 Command(resume=...) 传入的值
    human_decision = decision.get("human_decision", "approve")
    return {"human_decision": human_decision}


def finalize_node(state: dict) -> dict:
    """最终节点 — 根据人工决策确定最终评级"""
    if state.get("human_decision") == "reject":
        return {"qualification": "cold", "score": 0.0}
    elif state.get("human_decision") == "needs_info":
        return {"qualification": "warm"}
    # approve — 保持原有评分和评级
    return {}


# ── 循环终止条件 ──
def should_continue_evaluation(state: dict) -> str:
    """循环路由函数 — 判断是否继续提问评估

    LangGraph 知识点: add_conditional_edges 中的循环终止条件。
    当评分达标或提问次数达到上限时，退出循环进入评分汇总。
    """
    score = state.get("score", 0)
    questions_count = len(state.get("questions_asked", []))
    min_score = settings.LEAD_QUALIFIER_MIN_SCORE
    max_questions = settings.LEAD_QUALIFIER_MAX_QUESTIONS

    if score >= min_score or questions_count >= max_questions:
        return "score_lead"

    # 评分未达标且提问次数未达上限，继续提问
    return "ask_question"
