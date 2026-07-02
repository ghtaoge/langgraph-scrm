"""售后工单模块 StateGraph — 长流程 + 双审批"""

from langgraph.graph import END, START, StateGraph

from src.core.checkpoint import get_checkpointer
from src.modules.after_sale.nodes import (
    analyze_node,
    approve_node,
    close_node,
    create_ticket_node,
    execute_node,
    route_after_approve,
    route_after_verify,
    verify_node,
)
from src.modules.after_sale.state import AfterSaleState


def build_after_sale_graph(checkpointer=None):
    """构建售后工单 StateGraph

    图结构:
    START → create_ticket → analyze → approve (interrupt) → (条件路由)
      ├── approved → execute → verify (interrupt) → (条件路由)
      │   ├── 满意 → close → END
      │   └── 不满意 → execute (重试)
      ├── rejected → analyze (重新分析)
      └── needs_info → analyze
    """
    graph = StateGraph(AfterSaleState)

    graph.add_node("create_ticket", create_ticket_node)
    graph.add_node("analyze", analyze_node)
    graph.add_node("approve", approve_node)
    graph.add_node("execute", execute_node)
    graph.add_node("verify", verify_node)
    graph.add_node("close", close_node)

    graph.add_edge(START, "create_ticket")
    graph.add_edge("create_ticket", "analyze")
    graph.add_edge("analyze", "approve")
    graph.add_conditional_edges("approve", route_after_approve)
    graph.add_edge("execute", "verify")
    graph.add_conditional_edges("verify", route_after_verify)
    graph.add_edge("close", END)

    cp = checkpointer or get_checkpointer(store="memory")
    return graph.compile(checkpointer=cp)


if __name__ == "__main__":
    app = build_after_sale_graph()
    config = {"configurable": {"thread_id": "after-sale-001"}}
    # 需要真实 LLM API Key 运行
