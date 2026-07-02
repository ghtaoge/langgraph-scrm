"""知识库问答 API 路由"""
from fastapi import APIRouter

from src.api.schemas import KnowledgeQARequest, GraphRunResponse
from src.modules.knowledge_qa.graph import build_knowledge_qa_graph

router = APIRouter(prefix="/knowledge-qa", tags=["知识库问答"])


@router.post("/", response_model=GraphRunResponse)
async def answer_question(request: KnowledgeQARequest) -> GraphRunResponse:
    """知识库问答 — Corrective RAG 流程"""
    graph = build_knowledge_qa_graph()
    result = graph.invoke({
        "question": request.question,
        "documents": [], "web_results": [], "answer": "",
        "citations": [], "verification": "", "retries": 0,
        "error": None, "error_node": None,
    })
    return GraphRunResponse(completed=True, data=result)
