"""多Agent客服图集成测试 — mock LLM 验证 fan-out/fan-in + 质量检查"""
import json
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

from src.modules.multi_agent.graph import build_multi_agent_graph


@patch("src.modules.multi_agent.nodes.get_llm")
def test_multi_agent_happy_path(mock_get_llm):
    """多Agent完整流程 — 质量达标直接 respond"""
    mock_llm = MagicMock()
    # 调用顺序：supervisor → 3 个 agent（并行，回答相同）→ synthesize → quality_check
    mock_llm.invoke.side_effect = [
        AIMessage(content=json.dumps({"assigned_agents": ["product_expert", "policy_expert", "order_handler"]})),  # supervisor
        AIMessage(content="产品角度回答"),     # product_expert
        AIMessage(content="政策角度回答"),     # policy_expert
        AIMessage(content="订单角度回答"),     # order_handler
        AIMessage(content="合成统一回复"),     # synthesize
        AIMessage(content=json.dumps({"score": 9.0, "feedback": "优秀"})),  # quality_check
    ]
    mock_get_llm.return_value = mock_llm

    graph = build_multi_agent_graph()
    result = graph.invoke({
        "customer_question": "订单没发货且规格不符，想了解退换货政策",
        "assigned_agents": [],
        "agent_responses": {},
        "final_answer": "",
        "quality_score": 0.0,
        "feedback": "",
        "iteration": 0,
        "error": None,
        "error_node": None,
    })

    assert result["final_answer"] == "合成统一回复"
    assert result["quality_score"] == 9.0
    # 3 个 Agent 回答均合并进 agent_responses
    assert "product_expert" in result["agent_responses"]
    assert "policy_expert" in result["agent_responses"]
    assert "order_handler" in result["agent_responses"]
