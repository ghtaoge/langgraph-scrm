"""微信风控 API 路由 — 含 interrupt 上报恢复"""

from fastapi import APIRouter

from src.api.deps import get_shared_checkpointer, new_thread_id, resume_graph, run_graph_with_interrupt
from src.api.schemas import GraphRunResponse, WeChatRiskRequest, WeChatRiskResumeRequest
from src.modules.wechat_risk.graph import build_wechat_risk_graph

router = APIRouter(prefix="/wechat-risk", tags=["微信风控"])


@router.post("/", response_model=GraphRunResponse)
async def start_wechat_risk(request: WeChatRiskRequest) -> GraphRunResponse:
    """启动微信风控 — 高风险消息于上报处 interrupt"""
    graph = build_wechat_risk_graph(checkpointer=get_shared_checkpointer())
    thread_id = new_thread_id()
    initial = {
        "sender": request.sender,
        "content": request.content,
        "message_id": "",
        "message_type": "",
        "risk_score": 0.0,
        "risk_category": "",
        "action": "",
        "escalation_decision": "",
        "log_entry": {},
        "error": None,
        "error_node": None,
    }
    completed, data = run_graph_with_interrupt(graph, initial, thread_id)
    return GraphRunResponse(thread_id=thread_id, completed=completed, data=data)


@router.post("/resume", response_model=GraphRunResponse)
async def resume_wechat_risk(request: WeChatRiskResumeRequest) -> GraphRunResponse:
    """恢复微信风控 — 传入主管决策"""
    graph = build_wechat_risk_graph(checkpointer=get_shared_checkpointer())
    completed, data = resume_graph(graph, {"decision": request.decision}, request.thread_id)
    return GraphRunResponse(thread_id=request.thread_id, completed=completed, data=data)
