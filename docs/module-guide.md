# 模块详解与知识点

本文档逐模块讲解每个模块所覆盖的 LangGraph 知识点、图结构与设计要点。

## 模块 1：意图路由（intent_router）

**知识点**：`StateGraph`、`add_node`、`add_edge`、`add_conditional_edges`

**图结构**：
```
START → classify → (条件路由)
  ├── consult → respond_consult → END
  ├── complaint → escalate_complaint → END
  ├── after_sale → handle_after_sale → END
  └── other → handle_other → END
```

**要点**：
- `StateGraph(IntentRouterState)` 用 TypedDict 定义共享状态
- `add_conditional_edges("classify", route_by_intent)`：条件函数返回下一节点名实现分流
- 节点函数接收 state dict，返回部分更新 dict

## 模块 2：线索评级（lead_qualifier）

**知识点**：循环图、`interrupt()` + `Command(resume=...)`、Checkpoint、`Annotated` reducer

**图结构**：
```
START → ask_question → evaluate → (条件路由)
  ├── 继续 → ask_question (循环)
  └── 评分达标 → human_review (interrupt) → finalize → END
```

**要点**：
- 循环：`evaluate → ask_question` 形成环，条件函数控制退出
- `Annotated[list[str], operator.add]`：列表字段每轮追加而非覆盖
- `interrupt()` 暂停图执行，`Command(resume={"human_decision":"approve"})` 恢复
- 需 `checkpointer` 支持暂停/恢复状态持久化

## 模块 3：知识库问答（knowledge_qa）

**知识点**：Corrective RAG、ToolNode、Embedding 向量检索、自检重试循环

**图结构**：
```
START → retrieve → grade_docs → (条件路由)
  ├── docs_irrelevant → web_search → retrieve (补充检索)
  └── docs_relevant → generate → verify → (条件路由)
      ├── failed → generate (重试，上限 3 次)
      └── passed → respond → END
```

**要点**：
- Corrective RAG：检索不理想时自动转向网页搜索补充，再回到检索-评估
- `documents` / `web_results` 用 reducer 累积多轮检索结果
- `verify` 节点防幻觉自检，不通过则重试，受 `retries` 上限保护

## 模块 4：多Agent客服（multi_agent）

**知识点**：Supervisor 模式、fan-out/fan-in、dict reducer 合并并行结果

**图结构**：
```
START → supervisor → (fan-out 并行)
  ├── product_expert ─┐
  ├── policy_expert ──┼→ synthesize → quality_check → (条件路由)
  └── order_handler ─┘                  ├── quality<7 & iter<3 → supervisor
                                        └── 否则 → respond → END
```

**要点**：
- Supervisor 用 LLM 动态决定分派哪些 Agent
- fan-out：supervisor 连向多个 Agent 并行执行
- fan-in：多个 Agent 连向 synthesize，synthesize 等待全部完成
- `Annotated[dict, merge_dicts]` 自定义 reducer：并行 Agent 各自返回部分 dict，安全合并避免并发写冲突
- 质量检查不合格循环回到 supervisor 重新调度

## 模块 5：售后工单（after_sale）

**知识点**：长流程编排、双 `interrupt()` 审批门、条件回退

**图结构**：
```
START → create_ticket → analyze → approve (interrupt) → (条件路由)
  ├── approved → execute → verify (interrupt) → (条件路由)
  │   ├── 满意 → close → END
  │   └── 不满意 → execute (重试)
  ├── rejected → analyze (重新分析)
  └── needs_info → analyze
```

**要点**：
- 两个 `interrupt()` 暂停点：主管审批、客户验证
- 条件回退：拒绝/需补信息回到 analyze；客户不满意回到 execute
- 每个 interrupt 都需 checkpointer 持久化

## 模块 6：微信风控（wechat_risk）

**知识点**：四路条件分支、`interrupt()` 上报审批、风险阈值路由

**图结构**：
```
START → receive_message → classify → (四路路由)
  ├── normal → allow → END
  ├── business → log_only → END
  ├── sensitive/violation → risk_assess → (条件路由)
      ├── 高风险(≥80) → escalate (interrupt) → block → log → END
      └── 低风险 → warn → END
```

**要点**：
- 四路条件分支：根据消息类型分流到 4 条不同路径
- 二次风险阈值路由：`risk_score >= 80` 上报，否则直接提醒
- 高风险消息 `interrupt()` 暂停等待主管决策后继续拦截流程
