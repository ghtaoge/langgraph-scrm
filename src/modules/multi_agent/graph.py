"""多Agent客服模块 StateGraph — Supervisor 模式"""

from langgraph.graph import END, START, StateGraph

from src.modules.multi_agent.nodes import (
    order_handler_node,
    policy_expert_node,
    product_expert_node,
    quality_check_node,
    respond_node,
    route_after_quality,
    supervisor_node,
    synthesize_node,
)
from src.modules.multi_agent.state import MultiAgentState


def build_multi_agent_graph():
    """构建多Agent客服 StateGraph

    图结构:
    START → supervisor → (fan-out 并行) product_expert / policy_expert / order_handler
      → (fan-in) synthesize → quality_check → (条件路由)
        ├── quality < 7 且 iter < 3 → supervisor (循环重试)
        └── quality >= 7 或达上限 → respond → END

    Agent 执行策略：
    supervisor 分派后并行执行所有 3 个 Agent 节点，
    fan-in 到 synthesize 合成（agent_responses 通过 dict reducer 合并）。
    """
    graph = StateGraph(MultiAgentState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("product_expert", product_expert_node)
    graph.add_node("policy_expert", policy_expert_node)
    graph.add_node("order_handler", order_handler_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("quality_check", quality_check_node)
    graph.add_node("respond", respond_node)

    graph.add_edge(START, "supervisor")

    # Supervisor → 3 个 Agent（fan-out 并行）
    graph.add_edge("supervisor", "product_expert")
    graph.add_edge("supervisor", "policy_expert")
    graph.add_edge("supervisor", "order_handler")

    # 3 个 Agent → synthesize（fan-in，等待全部完成）
    graph.add_edge("product_expert", "synthesize")
    graph.add_edge("policy_expert", "synthesize")
    graph.add_edge("order_handler", "synthesize")

    graph.add_edge("synthesize", "quality_check")
    graph.add_conditional_edges("quality_check", route_after_quality)
    graph.add_edge("respond", END)

    return graph.compile()


if __name__ == "__main__":
    app = build_multi_agent_graph()
    result = app.invoke(
        {
            "customer_question": "我的订单迟迟没有发货，而且产品规格和描述不一致，想了解退换货政策",
            "assigned_agents": [],
            "agent_responses": {},
            "final_answer": "",
            "quality_score": 0.0,
            "feedback": "",
            "iteration": 0,
            "error": None,
            "error_node": None,
        }
    )
    print(f"最终回答: {result['final_answer']}")
