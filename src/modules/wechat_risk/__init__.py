"""微信风控模块 — 四路条件分支 + interrupt 上报"""

from src.modules.wechat_risk.graph import build_wechat_risk_graph

__all__ = ["build_wechat_risk_graph"]
