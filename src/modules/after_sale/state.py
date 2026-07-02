"""售后工单模块 State 定义"""

from typing import Optional, TypedDict


class AfterSaleState(TypedDict):
    """售后工单状态 — 长流程 + 双审批节点"""

    ticket_id: str  # 工单 ID
    customer_request: str  # 客户诉求
    issue_type: str  # 问题分类: refund/exchange/repair/complaint
    severity: str  # 严重度: low/medium/high/critical
    approval_status: str  # 审批状态: pending/approved/rejected/needs_info
    approver_comment: str  # 审批人意见
    resolution: str  # 处理方案
    customer_feedback: str  # 客户反馈
    status: str  # 工单状态: created/analyzing/approved/executing/verifying/closed
    error: Optional[str]
    error_node: Optional[str]
