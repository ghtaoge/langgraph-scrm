"""API 请求/响应模型 — Pydantic 数据校验"""
from pydantic import BaseModel, Field
from typing import Optional


# ── 意图路由 ──
class IntentRouterRequest(BaseModel):
    """意图路由请求"""
    message: str = Field(..., description="客户消息", min_length=1)

class IntentRouterResponse(BaseModel):
    """意图路由响应"""
    intent: str
    confidence: float
    skill_group: str
    response: str
    error: Optional[str] = None


# ── 线索评级 ──
class LeadQualifierRequest(BaseModel):
    """线索评级请求"""
    lead_info: dict = Field(..., description="线索基本信息")

class LeadQualifierResumeRequest(BaseModel):
    """线索评级恢复请求（人工审核决策）"""
    thread_id: str = Field(..., description="Checkpoint thread ID")
    human_decision: str = Field(..., description="审核决策: approve/reject/needs_info")


# ── 知识库问答 ──
class KnowledgeQARequest(BaseModel):
    """知识库问答请求"""
    question: str = Field(..., description="用户问题", min_length=1)


# ── 多Agent客服 ──
class MultiAgentRequest(BaseModel):
    """多Agent客服请求"""
    customer_question: str = Field(..., description="客户问题", min_length=1)


# ── 售后工单 ──
class AfterSaleRequest(BaseModel):
    """售后工单请求"""
    customer_request: str = Field(..., description="客户诉求", min_length=1)

class AfterSaleResumeRequest(BaseModel):
    """售后工单恢复请求"""
    thread_id: str
    approval_status: Optional[str] = Field(None, description="审批决策: approved/rejected/needs_info（主管审批阶段）")
    comment: Optional[str] = None
    feedback: Optional[str] = Field(None, description="客户反馈（客户验证阶段）")


# ── 微信风控 ──
class WeChatRiskRequest(BaseModel):
    """微信风控请求"""
    sender: str = Field(..., description="发送者")
    content: str = Field(..., description="消息内容", min_length=1)

class WeChatRiskResumeRequest(BaseModel):
    """微信风控恢复请求"""
    thread_id: str
    decision: str = Field(..., description="主管决策: approve_block/dismiss")


# ── 通用响应 ──
class GraphRunResponse(BaseModel):
    """图运行响应（含 thread_id 用于 interrupt 恢复）"""
    thread_id: Optional[str] = None
    completed: bool = Field(..., description="是否已完成（未完成表示等待 interrupt 恢复）")
    data: dict = Field(default_factory=dict, description="当前 state 快照")
