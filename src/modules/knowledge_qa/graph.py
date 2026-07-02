"""知识库问答模块 StateGraph — Corrective RAG 模式"""

from langgraph.graph import END, START, StateGraph

from src.modules.knowledge_qa.nodes import (
    generate_node,
    grade_docs_node,
    respond_node,
    retrieve_node,
    route_after_grade,
    route_after_verify,
    verify_node,
    web_search_node,
)
from src.modules.knowledge_qa.state import KnowledgeQAState


def build_knowledge_qa_graph():
    """构建知识库问答 StateGraph — Corrective RAG

    图结构:
    START → retrieve → grade_docs → (条件路由)
      ├── docs_irrelevant → web_search → retrieve (补充)
      └── docs_relevant → generate → verify → (条件路由)
          ├── failed → generate (重试)
          └── passed → respond → END
    """
    graph = StateGraph(KnowledgeQAState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade_docs", grade_docs_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("generate", generate_node)
    graph.add_node("verify", verify_node)
    graph.add_node("respond", respond_node)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade_docs")
    # 条件路由：grade_docs → web_search 或 generate（二选一，不再叠加固定边）
    graph.add_conditional_edges("grade_docs", route_after_grade)
    graph.add_edge("web_search", "retrieve")
    graph.add_edge("generate", "verify")
    graph.add_conditional_edges("verify", route_after_verify)
    graph.add_edge("respond", END)

    return graph.compile()


if __name__ == "__main__":
    app = build_knowledge_qa_graph()
    result = app.invoke(
        {
            "question": "SCRM 系统支持哪些微信功能？",
            "documents": [],
            "web_results": [],
            "answer": "",
            "citations": [],
            "verification": "",
            "retries": 0,
            "error": None,
            "error_node": None,
        }
    )
    print(f"回答: {result['answer']}")
