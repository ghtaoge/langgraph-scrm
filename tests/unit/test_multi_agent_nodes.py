"""多Agent客服节点单元测试"""

import json
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

from src.modules.multi_agent.nodes import (
    product_expert_node,
    quality_check_node,
    route_after_quality,
    supervisor_node,
    synthesize_node,
)


@patch("src.modules.multi_agent.nodes.get_llm")
def test_supervisor_node_returns_agents(mock_get_llm):
    """supervisor_node 应返回分派的 Agent 列表"""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(
        content=json.dumps({"assigned_agents": ["product_expert", "policy_expert"]})
    )
    mock_get_llm.return_value = mock_llm

    state = {
        "customer_question": "Q",
        "assigned_agents": [],
        "agent_responses": {},
        "final_answer": "",
        "quality_score": 0.0,
        "feedback": "",
        "iteration": 0,
        "error": None,
        "error_node": None,
    }
    result = supervisor_node(state)
    assert result["assigned_agents"] == ["product_expert", "policy_expert"]


@patch("src.modules.multi_agent.nodes.get_llm")
def test_supervisor_node_handles_json_error(mock_get_llm):
    """supervisor_node JSON 解析失败时降级为 product_expert"""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="not json")
    mock_get_llm.return_value = mock_llm

    state = {
        "customer_question": "Q",
        "assigned_agents": [],
        "agent_responses": {},
        "final_answer": "",
        "quality_score": 0.0,
        "feedback": "",
        "iteration": 0,
        "error": None,
        "error_node": None,
    }
    result = supervisor_node(state)
    assert result["assigned_agents"] == ["product_expert"]


@patch("src.modules.multi_agent.nodes.get_llm")
def test_product_expert_node_returns_partial_dict(mock_get_llm):
    """product_expert_node 应只返回自己的部分 dict（由 reducer 合并）"""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="产品角度回答")
    mock_get_llm.return_value = mock_llm

    state = {
        "customer_question": "Q",
        "assigned_agents": [],
        "agent_responses": {},
        "final_answer": "",
        "quality_score": 0.0,
        "feedback": "",
        "iteration": 0,
        "error": None,
        "error_node": None,
    }
    result = product_expert_node(state)
    assert result["agent_responses"] == {"product_expert": "产品角度回答"}


@patch("src.modules.multi_agent.nodes.get_llm")
def test_synthesize_node_returns_final_answer(mock_get_llm):
    """synthesize_node 应返回合成回答"""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="合成回答")
    mock_get_llm.return_value = mock_llm

    state = {
        "customer_question": "Q",
        "assigned_agents": ["product_expert"],
        "agent_responses": {"product_expert": "回答"},
        "final_answer": "",
        "quality_score": 0.0,
        "feedback": "",
        "iteration": 0,
        "error": None,
        "error_node": None,
    }
    result = synthesize_node(state)
    assert result["final_answer"] == "合成回答"


@patch("src.modules.multi_agent.nodes.get_llm")
def test_quality_check_node_returns_score(mock_get_llm):
    """quality_check_node 应返回评分并递增 iteration"""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content=json.dumps({"score": 8.5, "feedback": "good"}))
    mock_get_llm.return_value = mock_llm

    state = {
        "customer_question": "Q",
        "assigned_agents": [],
        "agent_responses": {},
        "final_answer": "A",
        "quality_score": 0.0,
        "feedback": "",
        "iteration": 0,
        "error": None,
        "error_node": None,
    }
    result = quality_check_node(state)
    assert result["quality_score"] == 8.5
    assert result["iteration"] == 1


def test_route_after_quality_retry():
    """质量不达标且未达上限时回到 supervisor"""
    assert route_after_quality({"quality_score": 5.0, "iteration": 1}) == "supervisor"


def test_route_after_quality_pass():
    """质量达标时到 respond"""
    assert route_after_quality({"quality_score": 8.0, "iteration": 1}) == "respond"


def test_route_after_quality_max_iteration():
    """质量不达标但达上限时到 respond"""
    assert route_after_quality({"quality_score": 5.0, "iteration": 3}) == "respond"
