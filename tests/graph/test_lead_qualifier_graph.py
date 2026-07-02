"""线索评级图集成测试"""

import json
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from src.modules.lead_qualifier.graph import build_lead_qualifier_graph


@patch("src.modules.lead_qualifier.nodes.get_llm")
def test_lead_qualifier_full_flow_with_interrupt(mock_get_llm):
    """线索评级完整流程 — 包含 interrupt 和 resume"""
    mock_llm = MagicMock()
    # 第一轮 ask_question
    mock_llm.invoke.side_effect = [
        AIMessage(content="贵公司目前使用什么 CRM 系统？"),  # ask_question
        AIMessage(content=json.dumps({"score": 75.0, "qualification": "hot"})),  # evaluate
    ]
    mock_get_llm.return_value = mock_llm

    checkpointer = InMemorySaver()
    graph = build_lead_qualifier_graph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "test-lead-001"}}

    initial_state = {
        "lead_info": {"source": "官网", "company": "测试公司", "position": "CTO"},
        "questions_asked": [],
        "answers_received": [],
        "score": 0.0,
        "score_history": [],
        "qualification": "",
        "human_decision": "",
        "error": None,
        "error_node": None,
    }

    # 第一轮 — 应在 human_review 处 interrupt
    result = graph.invoke(initial_state, config=config)

    # 评分已达标（75.0），应到达 human_review 并 interrupt
    assert result.get("score") == 75.0 or result.get("qualification") != ""

    # 恢复执行 — 人工审核通过
    resumed = graph.invoke(Command(resume={"human_decision": "approve"}), config=config)
    assert resumed["qualification"] == "hot"
