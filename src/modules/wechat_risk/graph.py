"""微信风控模块 StateGraph — 四路条件分支"""
from langgraph.graph import StateGraph, START, END

from src.modules.wechat_risk.state import WeChatRiskState
from src.modules.wechat_risk.nodes import (
    receive_message_node,
    classify_node,
    allow_node,
    log_only_node,
    risk_assess_node,
    escalate_node,
    block_node,
    warn_node,
    log_node,
    route_after_classify,
    route_after_risk_assess,
)
from src.core.checkpoint import get_checkpointer


def build_wechat_risk_graph(checkpointer=None):
    """构建微信风控 StateGraph

    图结构:
    START → receive_message → classify → (四路路由)
      ├── normal → allow → END
      ├── business → log_only → END
      ├── sensitive/violation → risk_assess → (条件路由)
          ├── 高风险(≥80) → escalate (interrupt) → block → log → END
          └── 低风险 → warn → END
    """
    graph = StateGraph(WeChatRiskState)

    graph.add_node("receive_message", receive_message_node)
    graph.add_node("classify", classify_node)
    graph.add_node("allow", allow_node)
    graph.add_node("log_only", log_only_node)
    graph.add_node("risk_assess", risk_assess_node)
    graph.add_node("escalate", escalate_node)
    graph.add_node("block", block_node)
    graph.add_node("warn", warn_node)
    graph.add_node("log", log_node)

    graph.add_edge(START, "receive_message")
    graph.add_edge("receive_message", "classify")
    graph.add_conditional_edges("classify", route_after_classify)
    graph.add_conditional_edges("risk_assess", route_after_risk_assess)
    graph.add_edge("escalate", "block")
    graph.add_edge("block", "log")
    graph.add_edge("allow", END)
    graph.add_edge("log_only", END)
    graph.add_edge("warn", END)
    graph.add_edge("log", END)

    cp = checkpointer or get_checkpointer(store="memory")
    return graph.compile(checkpointer=cp)


if __name__ == "__main__":
    app = build_wechat_risk_graph()
    # 需要真实 LLM API Key 运行
