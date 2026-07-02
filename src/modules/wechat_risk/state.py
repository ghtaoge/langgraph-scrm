"""微信风控模块 State 定义"""
from typing import Optional, TypedDict


class WeChatRiskState(TypedDict):
    """微信风控状态 — 四路条件分支 + interrupt 上报"""
    message_id: str             # 消息 ID
    sender: str                 # 发送者
    content: str                # 消息内容
    message_type: str           # 分类: normal/business/sensitive/violation
    risk_score: float           # 风险评分 (0-100)
    risk_category: str          # 风险类别: info_leak/harassment/fraud/compliance/other
    action: str                 # 处理动作: allow/log_only/warn/escalate/block
    escalation_decision: str    # 主管决策: approve_block/dismiss
    log_entry: dict             # 风控日志记录
    error: Optional[str]
    error_node: Optional[str]
