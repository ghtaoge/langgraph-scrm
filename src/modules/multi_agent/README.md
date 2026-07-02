# 模块 4：多Agent客服（Multi-Agent Service）

> 🎯 **LangGraph 知识点**：Supervisor 模式、fan-out / fan-in 并行、dict reducer 合并、质量检查循环

## 业务场景

客户提出复杂问题 → Supervisor 调度多个专业 Agent 并行回答 → 合成统一回复 → 质量检查不达标则重新调度

## 图结构

```
START → supervisor → (fan-out 并行)
  ├── product_expert ─┐
  ├── policy_expert ──┼→ synthesize → quality_check → (条件路由)
  └── order_handler ─┘                  ├── quality<7 & iter<3 → supervisor (重试)
                                        └── 否则 → respond → END
```

## LangGraph 核心知识点

### Supervisor 模式
Supervisor 节点用 LLM 动态决定分派哪些 Agent，而非硬编码分支。

### fan-out / fan-in
- fan-out：`supervisor` 同时连向 3 个 Agent，三者并行执行
- fan-in：3 个 Agent 均连向 `synthesize`，synthesize 等待全部完成后执行

### dict reducer 合并并行结果
并行 Agent 各自返回 `{"agent_responses": {自己: 回答}}`，通过 `Annotated[dict, merge_dicts]` reducer 安全合并，避免并发写冲突。

### 质量检查循环
`quality_check` 评分 <7 且重试未达上限时回到 `supervisor` 重新调度。

## 运行

```bash
python -m src.modules.multi_agent.graph
```
