"""意图路由模块 StateGraph 定义

LangGraph 知识点:
- StateGraph(StateSchema) 创建图
- add_node(name, func) 添加节点
- add_edge(from, to) 添加固定边
- add_conditional_edges(from, condition_fn, map) 添加条件边
- compile() 编译为可运行图
"""

from langgraph.graph import END, START, StateGraph

from src.modules.intent_router.nodes import (
    classify_node,
    escalate_complaint,
    handle_after_sale,
    handle_other,
    respond_consult,
    route_by_intent,
)
from src.modules.intent_router.state import IntentRouterState


def build_intent_router_graph():
    """构建意图路由 StateGraph

    图结构:
    START → classify → (条件路由) → respond_consult / escalate_complaint / handle_after_sale / handle_other → END
    """
    graph = StateGraph(IntentRouterState)

    # 添加节点
    graph.add_node("classify", classify_node)
    graph.add_node("respond_consult", respond_consult)
    graph.add_node("escalate_complaint", escalate_complaint)
    graph.add_node("handle_after_sale", handle_after_sale)
    graph.add_node("handle_other", handle_other)

    # 添加边
    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", route_by_intent)
    graph.add_edge("respond_consult", END)
    graph.add_edge("escalate_complaint", END)
    graph.add_edge("handle_after_sale", END)
    graph.add_edge("handle_other", END)

    return graph.compile()


# 模块入口 — 直接运行此模块可测试意图路由
if __name__ == "__main__":
    app = build_intent_router_graph()
    result = app.invoke(
        {
            "message": "我想了解一下你们产品的价格",
            "intent": "",
            "confidence": 0.0,
            "skill_group": "",
            "response": "",
            "error": None,
            "error_node": None,
        }
    )
    print(f"意图: {result['intent']}")
    print(f"技能组: {result['skill_group']}")
    print(f"回复: {result['response']}")
