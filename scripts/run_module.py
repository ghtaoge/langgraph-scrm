"""单模块 CLI 运行器 — 独立运行任意模块进行学习"""
import argparse
import json

from src.modules.intent_router.graph import build_intent_router_graph
from src.modules.lead_qualifier.graph import build_lead_qualifier_graph
from src.modules.knowledge_qa.graph import build_knowledge_qa_graph
from src.modules.multi_agent.graph import build_multi_agent_graph
from src.modules.after_sale.graph import build_after_sale_graph
from src.modules.wechat_risk.graph import build_wechat_risk_graph


MODULE_MAP = {
    "intent-router": ("意图路由", build_intent_router_graph),
    "lead-qualifier": ("线索评级", build_lead_qualifier_graph),
    "knowledge-qa": ("知识库问答", build_knowledge_qa_graph),
    "multi-agent": ("多Agent客服", build_multi_agent_graph),
    "after-sale": ("售后工单", build_after_sale_graph),
    "wechat-risk": ("微信风控", build_wechat_risk_graph),
}

# 每个模块的默认输入 state
DEFAULT_INPUTS = {
    "intent-router": {"message": "我想咨询产品价格", "intent": "", "confidence": 0.0, "skill_group": "", "response": "", "error": None, "error_node": None},
    "lead-qualifier": {"lead_info": {"source": "官网", "company": "测试公司", "position": "CTO"}, "questions_asked": [], "answers_received": [], "score": 0.0, "score_history": [], "qualification": "", "human_decision": "", "error": None, "error_node": None},
    "knowledge-qa": {"question": "SCRM 系统支持哪些微信功能？", "documents": [], "web_results": [], "answer": "", "citations": [], "verification": "", "retries": 0, "error": None, "error_node": None},
    "multi-agent": {"customer_question": "我的订单迟迟没有发货", "assigned_agents": [], "agent_responses": {}, "final_answer": "", "quality_score": 0.0, "feedback": "", "iteration": 0, "error": None, "error_node": None},
    "after-sale": {"customer_request": "产品质量有问题，要求退款", "ticket_id": "", "issue_type": "", "severity": "", "approval_status": "", "approver_comment": "", "resolution": "", "customer_feedback": "", "status": "", "error": None, "error_node": None},
    "wechat-risk": {"sender": "员工A", "content": "我把客户名单发给你了", "message_id": "", "message_type": "", "risk_score": 0.0, "risk_category": "", "action": "", "escalation_decision": "", "log_entry": {}, "error": None, "error_node": None},
}

# 含 interrupt 的模块（需要 thread_id 才能运行/恢复）
INTERRUPT_MODULES = {"lead-qualifier", "after-sale", "wechat-risk"}


def main():
    parser = argparse.ArgumentParser(description="LangGraph-SCRM 单模块运行器")
    parser.add_argument("module", choices=MODULE_MAP.keys(), help="模块名")
    parser.add_argument("--input", help="JSON 格式的输入 state（可选，覆盖默认输入）")
    args = parser.parse_args()

    name, builder = MODULE_MAP[args.module]
    print(f">> 运行模块: {name} ({args.module})")

    graph = builder()

    input_state = DEFAULT_INPUTS.get(args.module, {})
    if args.input:
        input_state = json.loads(args.input)

    # interrupt 模块需要 thread_id；非 interrupt 模块传 config 也无害
    config = {"configurable": {"thread_id": "cli-run"}}
    graph.invoke(input_state, config=config)
    state = graph.get_state(config)

    if state.next:
        # 处于 interrupt 暂停
        print("[PAUSED] 图在 interrupt 处暂停，等待外部输入恢复：")
        print(json.dumps(state.values, ensure_ascii=False, indent=2, default=str))
        print(f"\n提示：该模块含人机协同节点，需通过 API /resume 或 Command(resume=...) 恢复（thread_id=cli-run）")
    else:
        print(json.dumps(state.values, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
