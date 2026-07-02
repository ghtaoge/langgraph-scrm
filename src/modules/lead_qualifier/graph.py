"""线索评级模块 StateGraph 定义

LangGraph 知识点:
- 循环图：evaluate → ask_question → evaluate（形成循环）
- interrupt()：human_review_node 内部调用 interrupt() 暂停
- Checkpoint：compile(checkpointer=...) 支持暂停恢复
- Command(resume=...)：恢复时传入人工决策
"""
from langgraph.graph import StateGraph, START, END

from src.modules.lead_qualifier.state import LeadQualifierState
from src.modules.lead_qualifier.nodes import (
    ask_question_node,
    evaluate_node,
    human_review_node,
    finalize_node,
    should_continue_evaluation,
)
from src.core.checkpoint import get_checkpointer


def build_lead_qualifier_graph(checkpointer=None):
    """构建线索评级 StateGraph

    图结构:
    START → ask_question → evaluate → (条件路由)
      ├── continue → ask_question (循环)
      └── score_lead → human_review (interrupt) → finalize → END
    """
    graph = StateGraph(LeadQualifierState)

    # 添加节点
    graph.add_node("ask_question", ask_question_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("finalize", finalize_node)

    # 添加边
    graph.add_edge(START, "ask_question")
    graph.add_edge("ask_question", "evaluate")

    # 循环条件路由：evaluate → ask_question 或 human_review
    graph.add_conditional_edges(
        "evaluate",
        should_continue_evaluation,
        {"ask_question": "ask_question", "score_lead": "human_review"},
    )

    graph.add_edge("human_review", "finalize")
    graph.add_edge("finalize", END)

    # 编译图（需要 checkpointer 支持 interrupt 暂停恢复）
    cp = checkpointer or get_checkpointer(store="memory")
    return graph.compile(checkpointer=cp)


if __name__ == "__main__":
    from langgraph.types import Command

    app = build_lead_qualifier_graph()
    thread_id = "lead-test-001"
    config = {"configurable": {"thread_id": thread_id}}

    # 第一轮运行 — 会触发 interrupt
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

    # 注意：实际运行需要真实 LLM API Key
    # result = app.invoke(initial_state, config=config)
    # print(f"中断 — 等待审核: {result}")
    # 恢复执行
    # resumed = app.invoke(Command(resume={"human_decision": "approve"}), config=config)
    # print(f"最终评级: {resumed['qualification']}")
