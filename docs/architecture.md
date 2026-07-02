# 架构设计

## 总体架构

LangGraph-SCRM 采用「核心基础设施 + 6 个独立业务模块 + 统一 API 层」的三层架构。

```
┌─────────────────────────────────────────────────┐
│              FastAPI 层（src/api）                │
│   6 模块路由 + 中间件 + Pydantic schemas          │
└───────────────┬─────────────────────────────────┘
                │ 调用 build_xxx_graph()
┌───────────────▼─────────────────────────────────┐
│           业务模块层（src/modules）                │
│  intent_router / lead_qualifier / knowledge_qa  │
│  multi_agent / after_sale / wechat_risk          │
│  每个模块 = 一个独立 StateGraph                   │
└───────────────┬─────────────────────────────────┘
                │ 依赖
┌───────────────▼─────────────────────────────────┐
│           核心基础设施（src/core + src/config）    │
│  LLM 抽象层 / Checkpoint / State 基类 / 工具      │
│  Settings 单例 / Prompt 模板库                    │
└─────────────────────────────────────────────────┘
```

## 设计决策

### 1. 模块完全独立
每个模块有自己的 state、nodes、graph，**不跨模块 import**。这保证：
- 每个模块可独立学习、独立运行
- 修改一个模块不影响其他模块
- 便于按需复用到其他项目

### 2. 共享核心基础设施
LLM 抽象、Checkpoint、配置、Prompt 集中在 `src/core` 与 `src/config`，所有模块复用，避免重复。

### 3. LLM 双后端抽象
`get_llm()` 统一返回 `ChatOpenAI` 实例，OpenAI 与 Doubao 都走 OpenAI 兼容接口，通过 `LANGGRAPH_LLM_PROVIDER` 切换，业务代码无感知。

### 4. 错误处理统一模式
- 每个模块 State 都含 `error` / `error_node` 字段
- `@safe_llm_call` 装饰器捕获节点异常，返回错误 state 而非抛出
- 业务流程不会被单点 LLM 失败中断

### 5. interrupt 人机协同
需要人工介入的节点（审核、审批、上报）使用 `interrupt()` 暂停，配合 `checkpointer` 持久化，通过 `Command(resume=...)` 恢复。API 层通过共享 checkpointer 实现跨请求恢复。

### 6. 并行 Agent 安全合并
多 Agent 模块的并行节点通过自定义 dict reducer 合并结果，避免并发写同一 state 字段的冲突。

### 7. 测试不依赖真实 LLM
所有测试 mock LLM 调用，CI 可在无 API Key 环境运行。`@pytest.mark.integration` 标记真实 LLM 集成测试。

## 数据流

```
HTTP 请求
  → FastAPI 路由（Pydantic 校验）
  → 构建 StateGraph
  → graph.invoke(input_state, config)
  → 节点链执行（LLM 调用 / 工具调用 / 条件路由）
  → interrupt 暂停 → 返回 thread_id 等待恢复
  → /resume → Command(resume=...) → 继续执行
  → END → 返回最终 state
```

## 扩展指南

- **新增模块**：在 `src/modules/` 下新建目录，实现 state/nodes/graph，在 `src/api/routes/` 加路由，在 `scripts/run_module.py` 注册
- **替换 LLM**：在 `src/core/llm.py` 的 `get_llm()` 增加新 provider 分支
- **替换向量库**：在 `src/modules/knowledge_qa/tools.py` 替换 Chroma 为 FAISS 等
- **生产 Checkpoint**：将 `CHECKPOINT_STORE` 设为 redis，启用 `langgraph-scrm[redis]`
