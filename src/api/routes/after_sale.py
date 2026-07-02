"""售后工单 API 路由 — 含双 interrupt 恢复"""
from fastapi import APIRouter

from src.api.deps import get_shared_checkpointer, new_thread_id, run_graph_with_interrupt, resume_graph
from src.api.schemas import AfterSaleRequest, AfterSaleResumeRequest, GraphRunResponse
from src.modules.after_sale.graph import build_after_sale_graph

router = APIRouter(prefix="/after-sale", tags=["售后工单"])


@router.post("/", response_model=GraphRunResponse)
async def start_after_sale(request: AfterSaleRequest) -> GraphRunResponse:
    """启动售后工单 — 于主管审批处 interrupt"""
    graph = build_after_sale_graph(checkpointer=get_shared_checkpointer())
    thread_id = new_thread_id()
    initial = {
        "customer_request": request.customer_request,
        "ticket_id": "", "issue_type": "", "severity": "",
        "approval_status": "", "approver_comment": "", "resolution": "",
        "customer_feedback": "", "status": "", "error": None, "error_node": None,
    }
    completed, data = run_graph_with_interrupt(graph, initial, thread_id)
    return GraphRunResponse(thread_id=thread_id, completed=completed, data=data)


@router.post("/resume", response_model=GraphRunResponse)
async def resume_after_sale(request: AfterSaleResumeRequest) -> GraphRunResponse:
    """恢复售后工单 — 根据当前阶段传入审批决策或客户反馈"""
    graph = build_after_sale_graph(checkpointer=get_shared_checkpointer())
    # 主管审批阶段传 approval_status/comment；客户验证阶段传 feedback
    if request.feedback is not None:
        payload = {"feedback": request.feedback}
    else:
        payload = {"approval_status": request.approval_status or "approved", "comment": request.comment or ""}
    completed, data = resume_graph(graph, payload, request.thread_id)
    return GraphRunResponse(thread_id=request.thread_id, completed=completed, data=data)
