"""线索评级节点单元测试"""
import json
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

from src.modules.lead_qualifier.nodes import (
    ask_question_node,
    evaluate_node,
    should_continue_evaluation,
)


@patch("src.modules.lead_qualifier.nodes.get_llm")
def test_ask_question_node_appends_question(mock_get_llm):
    """ask_question_node 应追加新问题"""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="贵公司目前使用什么 CRM 系统？")
    mock_get_llm.return_value = mock_llm

    state = {
        "lead_info": {"source": "官网", "company": "测试公司"},
        "questions_asked": ["你们团队规模多大？"],
        "answers_received": ["约50人"],
        "score": 30.0,
        "score_history": [30.0],
        "qualification": "warm",
        "human_decision": "",
        "error": None,
        "error_node": None,
    }
    result = ask_question_node(state)
    assert "questions_asked" in result
    assert len(result["questions_asked"]) == 1


@patch("src.modules.lead_qualifier.nodes.get_llm")
def test_evaluate_node_returns_score(mock_get_llm):
    """evaluate_node 应返回评分和评级"""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(
        content=json.dumps({"score": 75.0, "qualification": "hot"})
    )
    mock_get_llm.return_value = mock_llm

    state = {
        "lead_info": {"source": "官网"},
        "questions_asked": ["Q1"],
        "answers_received": ["A1"],
        "score": 0.0,
        "score_history": [],
        "qualification": "",
        "human_decision": "",
        "error": None,
        "error_node": None,
    }
    result = evaluate_node(state)
    assert result["score"] == 75.0
    assert result["qualification"] == "hot"


def test_should_continue_evaluation_loop():
    """评分未达标时应继续循环"""
    state = {"score": 30.0, "questions_asked": ["Q1", "Q2"]}
    assert should_continue_evaluation(state) == "ask_question"


def test_should_continue_evaluation_exit():
    """评分达标时应退出循环"""
    state = {"score": 75.0, "questions_asked": ["Q1", "Q2"]}
    assert should_continue_evaluation(state) == "score_lead"


def test_should_continue_max_questions():
    """提问次数达到上限时应退出循环"""
    state = {"score": 30.0, "questions_asked": ["Q1", "Q2", "Q3", "Q4", "Q5"]}
    assert should_continue_evaluation(state) == "score_lead"
