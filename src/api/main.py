"""FastAPI 应用入口 — 注册所有路由"""

from fastapi import FastAPI

from src.api.middleware import LoggingMiddleware
from src.api.routes import (
    after_sale,
    intent_router,
    knowledge_qa,
    lead_qualifier,
    multi_agent,
    wechat_risk,
)

app = FastAPI(
    title="LangGraph-SCRM",
    description="基于 LangGraph 的 SCRM 智能客服平台 API",
    version="0.1.0",
)

# 注册中间件
app.add_middleware(LoggingMiddleware)

# 注册路由
app.include_router(intent_router.router)
app.include_router(lead_qualifier.router)
app.include_router(knowledge_qa.router)
app.include_router(multi_agent.router)
app.include_router(after_sale.router)
app.include_router(wechat_risk.router)


@app.get("/")
async def root():
    """根路径 — 项目信息"""
    return {
        "name": "LangGraph-SCRM",
        "version": "0.1.0",
        "description": "基于 LangGraph 的 SCRM 智能客服平台",
        "modules": [
            "intent-router",
            "lead-qualifier",
            "knowledge-qa",
            "multi-agent",
            "after-sale",
            "wechat-risk",
        ],
        "docs": "/docs",
    }
