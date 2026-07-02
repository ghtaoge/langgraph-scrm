"""线索评级 API 路由 — 含 interrupt 恢复"""
from fastapi import APIRouter

from src.api.deps import get_shared_checkpointer, new_thread_id, run_graph_with_interrupt, resume_graph
from src.api.schemas import LeadQualifierRequest, LeadQualifierResumeRequest, GraphRunResponse
from src.modules.lead_qualifier.graph import build_lead_qualifier_graph

router = APIRouter(prefix="/lead-qualifier", tags=["线索评级"])


@router.post("/", response_model=GraphRunResponse)
async def start_lead_qualifier(request: LeadQualifierRequest) -> GraphRunResponse:
    """启动线索评级 — 多轮评估后于人工审核处 interrupt"""
    graph = build_lead_qualifier_graph(checkpointer=get_shared_checkpointer())
    thread_id = new_thread_id()
    initial = {
        "lead_info": request.lead_info,
        "questions_asked": [], "answers_received": [],
        "score": 0.0, "score_history": [], "qualification": "",
        "human_decision": "", "error": None, "error_node": None,
    }
    completed, data = run_graph_with_interrupt(graph, initial, thread_id)
    return GraphRunResponse(thread_id=thread_id, completed=completed, data=data)


@router.post("/resume", response_model=GraphRunResponse)
async def resume_lead_qualifier(request: LeadQualifierResumeRequest) -> GraphRunResponse:
    """恢复线索评级 — 传入人工审核决策"""
    graph = build_lead_qualifier_graph(checkpointer=get_shared_checkpointer())
    completed, data = resume_graph(graph, {"human_decision": request.human_decision}, request.thread_id)
    return GraphRunResponse(thread_id=request.thread_id, completed=completed, data=data)
