"""意图路由 API 路由"""

from fastapi import APIRouter

from src.api.schemas import IntentRouterRequest, IntentRouterResponse
from src.modules.intent_router.graph import build_intent_router_graph

router = APIRouter(prefix="/intent-router", tags=["意图路由"])


@router.post("/", response_model=IntentRouterResponse)
async def classify_intent(request: IntentRouterRequest) -> IntentRouterResponse:
    """分类客户消息意图 — 意图路由模块入口"""
    graph = build_intent_router_graph()
    result = graph.invoke(
        {
            "message": request.message,
            "intent": "",
            "confidence": 0.0,
            "skill_group": "",
            "response": "",
            "error": None,
            "error_node": None,
        }
    )
    return IntentRouterResponse(
        intent=result["intent"],
        confidence=result["confidence"],
        skill_group=result["skill_group"],
        response=result["response"],
        error=result.get("error"),
    )
