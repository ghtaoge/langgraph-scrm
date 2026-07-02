"""多Agent客服 API 路由"""
from fastapi import APIRouter

from src.api.schemas import MultiAgentRequest, GraphRunResponse
from src.modules.multi_agent.graph import build_multi_agent_graph

router = APIRouter(prefix="/multi-agent", tags=["多Agent客服"])


@router.post("/", response_model=GraphRunResponse)
async def multi_agent_answer(request: MultiAgentRequest) -> GraphRunResponse:
    """多Agent客服 — Supervisor 调度 + 合成 + 质量检查"""
    graph = build_multi_agent_graph()
    result = graph.invoke({
        "customer_question": request.customer_question,
        "assigned_agents": [], "agent_responses": {},
        "final_answer": "", "quality_score": 0.0,
        "feedback": "", "iteration": 0,
        "error": None, "error_node": None,
    })
    return GraphRunResponse(completed=True, data=result)
