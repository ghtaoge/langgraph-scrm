# LangGraph-SCRM 设计文档

> 日期：2026-07-02
> 状态：待用户审核

## 1. 项目定位与命名

**项目名**：`LangGraph-SCRM`（GitHub repo 名）

**一句话描述**：基于 LangGraph 的 SCRM 智能客服平台 — 6 个实战模块覆盖 LangGraph 所有核心模式，可学习、可实战、可二次开发。

**定位关键词**：
- 学习型：每个模块对应 1-2 个 LangGraph 核心概念，代码配有详细中文注释 + README 知识点索引
- 实战型：基于真实 SCRM 业务场景，非玩具 demo，公司可直接试用
- 开源型：MIT 协议，结构清晰，文档完善，方便社区贡献和二次开发

**GitHub 差异化卖点**：

| 对比项 | 现有热门项目 | 本项目 |
|--------|-------------|--------|
| 场景 | 通用 AI Agent / 客服 demo | SCRM 行业垂直（客户管理+微信风控） |
| 覆盖面 | 单一模式（只做 RAG / 只做多 Agent） | 6 模块全覆盖（路由/循环/RAG/多Agent/审批/风控） |
| 深度 | 简单示例，无持久化无检查点 | 生产级设计（Checkpoint/人机协同/状态持久化） |
| 语言 | 纯英文 | 中文注释+文档（国内开发者友好） |

**6 模块 → LangGraph 知识点映射**：

| # | 模块 | 业务场景 | LangGraph 知识点 |
|---|------|---------|-----------------|
| 1 | 意图路由 | 客户消息 → 分类 → 技能组分配 | `StateGraph` + 条件路由 + `add_conditional_edges` |
| 2 | 线索评级 | 多轮对话 → 自动评分 → 人工审核 | 循环图 + `interrupt_before` + Checkpoint |
| 3 | 知识库问答 | 企业文档 → 向量检索 → 回答生成 | RAG + 工具调用 + Embedding |
| 4 | 多Agent客服 | Supervisor 分派 → 3 个专业 Agent 协作回答 | 多 Agent Supervisor 模式 + Agent 间通信 |
| 5 | 售后工单 | 创建 → 分析 → 审批 → 执行 → 关闭 | 长流程 + 审批节点 + 状态持久化 |
| 6 | 微信风控 | 消息分类 → 风险评估 → 告警/拦截 | 条件分支 + 状态持久化 + 监控 |

## 2. 整体架构

### 分层架构

```
┌─────────────────────────────────────────────────┐
│                   Web UI 层                      │
│         FastAPI + Vue.js (可选轻量前端)            │
│  ┌──────┬──────┬──────┬──────┬──────┬──────┐     │
│  │意图  │线索  │知识  │多Agent│售后  │微信  │     │
│  │路由  │评级  │问答  │客服  │工单  │风控  │     │
│  └──┬───┴──┬───┴──┬───┴──┬───┴──┬───┴──┬──┘     │
├─────┼──────┼──────┼──────┼──────┼──────┼──────────┤
│     │  LangGraph Agent 层 (核心)                  │
│     │                                            │
│  ┌──▼──────────────────────────────────────────┐ │
│  │  StateGraph · 节点 · 边 · 条件路由 · 循环    │ │
│  │  Checkpoint · interrupt · 工具注册            │ │
│  └──────────────────────────────────────────────┘ │
├──────────────────────────────────────────────────┤
│               基础设施层                          │
│  ┌────────┬──────────┬──────────┬──────────┐     │
│  │ LLM    │ 向量存储  │ 消息队列  │ 数据存储  │     │
│  │(OpenAI/│(Chroma/  │(Redis/   │(SQLite/  │     │
│  │ Doubao)│ FAISS)   │ RabbitMQ)│ MySQL)   │     │
│  └────────┴──────────┴──────────┴──────────┘     │
└──────────────────────────────────────────────────┤
```

### 核心设计原则

1. 每个模块 = 一个独立 StateGraph：6 个模块各自是一个完整的 LangGraph 图，可独立运行、独立测试
2. 共享基础设施层：LLM 配置、向量存储、数据库连接抽象为公共模块，所有 Agent 共用
3. 渐进式学习路径：模块按复杂度排序 — 从条件路由（模块1）到多 Agent 协作（模块4）

### 项目结构

```
LangGraph-SCRM/
├── README.md                    # 项目总览 + 知识点索引表
├── LICENSE                      # MIT
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── pyproject.toml               # Python 项目配置 + 依赖
├── .env.example                 # 环境变量模板
│
├── docs/                        # 文档
│   ├── getting-started.md       # 快速上手指南
│   ├── module-guide.md          # 各模块知识点详解
│   └── architecture.md          # 架构设计说明
│
├── src/
│   ├── __init__.py
│   ├── config/                  # 全局配置
│   │   ├── settings.py          # LLM/向量库/DB 配置加载
│   │   └── prompts.py           # 所有 Prompt 模板集中管理
│   │
│   ├── core/                    # LangGraph 公共能力
│   │   ├── state.py             # 所有模块共享的 State 定义基类
│   │   ├── tools.py             # 通用工具注册（搜索/计算/数据库查询）
│   │   ├── llm.py               # LLM 抽象层（支持 OpenAI/Doubao 切换）
│   │   ├── checkpoint.py        # Checkpoint 配置（SQLite/Redis 持久化）
│   │   └── nodes.py             # 通用节点函数基类
│   │
│   ├── modules/                 # 6 个业务模块（每个 = 1 个独立 StateGraph）
│   │   ├── 01_intent_router/
│   │   │   ├── graph.py         # StateGraph 定义 + 边 + 路由
│   │   │   ├── state.py         # 模块专属 State
│   │   │   ├── nodes.py         # 分类节点/分配节点
│   │   │   ├── tools.py         # 模块专属工具
│   │   │   ├── __init__.py
│   │   │   └── README.md        # 模块独立说明 + 知识点标注
│   │   │
│   │   ├── 02_lead_qualifier/
│   │   ├── 03_knowledge_qa/
│   │   ├── 04_multi_agent/
│   │   ├── 05_after_sale/
│   │   ├── 06_wechat_risk/
│   │
│   ├── api/                     # FastAPI 接口层
│   │   ├── main.py              # FastAPI app + 路由注册
│   │   ├── routes/
│   │   │   ├── intent_router.py
│   │   │   ├── lead_qualifier.py
│   │   │   ├── ...
│   │   ├── middleware.py         # 认证/日志/异常处理
│   │   └── schemas.py           # API 请求/响应模型
│   │
│   └── data/                    # 示例数据
│       ├── sample_docs/         # 知识库示例文档（FAQ/产品手册）
│       ├── sample_messages/     # 微信消息样本
│       └── seed_db.py           # 初始化数据库脚本
│
├── tests/                       # 测试
│   ├── conftest.py              # 共享 fixtures（mock LLM、测试 State）
│   ├── unit/
│   │   ├── test_intent_nodes.py
│   │   ├── test_lead_nodes.py
│   │   ├── ...
│   ├── graph/
│   │   ├── test_intent_router_graph.py
│   │   ├── test_lead_qualifier_graph.py
│   │   ├── ...
│   ├── api/
│   │   ├── test_intent_router_api.py
│   │   ├── ...
│
├── scripts/                     # 运行脚本
│   ├── run_module.py            # 单模块 CLI 运行器
│   ├── run_all.py               # 启动完整 API 服务
│   └── seed_data.py             # 数据初始化
│
└── .github/
    └── workflows/
        └── ci.yml               # CI: lint + test
```

### 关键设计决策

| 决策点 | 选择 | 理由 |
|--------|------|------|
| LLM 抽象 | 支持 OpenAI + Doubao 双后端 | 国内用 Doubao 更实际，开源需兼容国际 |
| 向量存储 | Chroma（默认）+ FAISS（可选） | Chroma 轻量内嵌适合学习；FAISS 生产级 |
| Checkpoint 存储 | SQLite（默认）+ Redis（可选） | SQLite 零依赖适合学习；Redis 生产级持久化 |
| API 层 | FastAPI | Python 最主流、自带 OpenAPI 文档、async 原生 |
| 状态定义 | TypedDict + Annotated | LangGraph 官方推荐模式 |
| 模块间通信 | 独立运行，不耦合 | 学习者可单独跑任意模块，降低入门门槛 |

## 3. 六模块详细设计

### 模块 1：意图路由（Intent Router）

**业务**：客户发消息 → AI 分类意图 → 自动分配到对应技能组

**LangGraph 知识点**：`StateGraph`、`add_node`、`add_edge`、`add_conditional_edges`

**图结构**：

```
                    ┌─────────┐
  客户消息 ──→      │ classify │  (意图分类节点)
                    └───┬──┬──┘
                        │  │
          ┌─────────────┘  └───────────┐
          │                            │
    ┌─────▼─────┐                ┌─────▼─────┐
    │  咨询类    │                │  投诉类    │
    │ route_     │                │ route_     │
    │ consult    │                │ complaint  │
    └─────┬─────┘                └─────┬─────┘
          │                            │
    ┌─────▼─────┐                ┌─────▼─────┐
    │ respond_  │                │ escalate_  │
    │ consult   │                │ complaint  │
    └───────────┘                └───────────┘
          │                            │
          └──────→ END ←───────────────┘
```

**State 定义**：
```python
class IntentRouterState(TypedDict):
    message: str                # 客户原始消息
    intent: str                 # 分类结果: consult/complaint/after_sale/other
    confidence: float           # 分类置信度
    skill_group: str            # 分配的技能组
    response: str               # 最终回复
    error: Optional[str]        # 错误信息
    error_node: Optional[str]   # 发生错误的节点名
```

**关键节点**：
- `classify_node`：调用 LLM 做意图分类（4 类：咨询/投诉/售后/其他）
- `route_by_intent`：条件路由函数，按 `state["intent"]` 分流
- `respond_consult` / `escalate_complaint`：各技能组的处理节点

---

### 模块 2：线索评级（Lead Qualifier）

**业务**：销售人员接到新线索 → AI 多轮对话评估 → 自动评分 → 人工审核确认

**LangGraph 知识点**：循环图、`interrupt_before`、Checkpoint、人机协同

**图结构**：

```
  ┌──────────┐
  │  start    │
  └───┬──────┘
      │
  ┌───▼──────────┐     ┌─────────────┐
  │  ask_question │←──→│  evaluate    │  (循环：问→评→再问→再评)
  └───┬──────────┘     └──────┬──────┘
      │                        │ (评分达标)
  ┌───▼──────────┐              │
  │  score_lead  │◄─────────────┘
  └───┬──────────┘
      │
  ┌───▼──────────┐  ← interrupt_before（暂停等人工审核）
  │  human_review │
  └───┬──────────┘
      │ (审核通过/驳回)
  ┌───▼──────┐
  │  finalize │
  └────→ END
```

**State 定义**：
```python
class LeadQualifierState(TypedDict):
    lead_info: dict             # 线索基本信息（来源/公司/职位）
    questions_asked: list[str]  # 已提问列表
    answers_received: list[str] # 已回答列表
    score: float                # 当前评分 (0-100)
    score_history: list[float]  # 每轮评分记录
    qualification: str          # 最终评级: hot/warm/cold
    human_decision: str         # 人工审核结果: approve/reject/needs_info
    error: Optional[str]
    error_node: Optional[str]
```

**关键设计**：
- 循环终止条件：`score >= 60` 或 `questions_asked >= 5`
- `interrupt_before("human_review")`：暂停执行，等待人工输入
- Checkpoint：支持暂停/恢复，长流程不丢失状态

---

### 模块 3：知识库问答（Knowledge QA）

**业务**：客服查询企业产品文档/FAQ → 向量检索 + LLM 生成精准回答

**LangGraph 知识点**：RAG、工具调用（`ToolNode`）、Embedding + VectorStore

**图结构**（Corrective RAG 模式）：

```
  ┌──────────┐
  │  start    │  (用户提问)
  └───┬──────┘
      │
  ┌───▼──────────┐
  │  retrieve     │  (向量检索相关文档片段)
  └───┬──────────┘
      │
  ┌───▼──────────┐
  │  grade_docs   │  (评估检索文档相关性，过滤低质量片段)
  └───┬──────────┘
      │
      ├─ (文档不足) ──→ web_search → retrieve (补充检索)
      │
      │ (文档充足)
  ┌───▼──────────┐
  │  generate     │  (LLM 基于文档生成回答)
  └───┬──────────┘
      │
  ┌───▼──────────┐
  │  verify       │  (自检：回答是否基于文档？是否有幻觉？)
  └───┬──────────┘
      │
      ├─ (不合格) ──→ 回到 generate（重新生成）
      │ (合格)
  ┌───▼──────┐
  │  respond  │
  └────→ END
```

**State 定义**：
```python
class KnowledgeQAState(TypedDict):
    question: str               # 用户问题
    documents: list[str]        # 检索到的文档片段
    web_results: list[str]      # 补充的网页搜索结果
    answer: str                 # 生成的回答
    citations: list[dict]       # 引用来源
    verification: str           # 自检结果: passed/failed
    retries: int                # 生成重试次数
    error: Optional[str]
    error_node: Optional[str]
```

**关键设计**：
- Corrective RAG 模式（检索 → 评估 → 补充 → 生成 → 自检 → 重试）
- 向量存储用 Chroma 内嵌，开箱即用
- 示例文档：提供 5-10 份 SCRM FAQ/产品手册

---

### 模块 4：多Agent客服（Multi-Agent Service）

**业务**：复杂客户问题 → Supervisor 分派 → 3 个专业 Agent 协作 → 合成回答

**LangGraph 知识点**：多 Agent Supervisor 模式、Agent 间状态共享、子图嵌套

**图结构**：

```
                      ┌──────────┐
  客户问题 ──→         │ supervisor│  (调度中心)
                      └──┬─┬─┬──┘
                        │ │ │
          ┌─────────────┘ │ └─────────────┐
          │               │               │
    ┌─────▼─────┐   ┌────▼─────┐   ┌─────▼─────┐
    │  product   │   │  policy   │   │  order    │
    │  expert    │   │  expert   │   │  handler  │
    └─────┬─────┘   └────┬─────┘   └─────┬─────┘
          │               │               │
          └───────→ ┌─────▼─────┐ ←───────┘
                    │  synthesize│  (合成最终回答)
                    └─────┬─────┘
                          │
                    ┌─────▼─────┐
                    │  quality_  │  (质量检查)
                    │  check     │
                    └──┬──┬─────┘
                       │  │
                  (合格)  (不合格)
                       │  │
                  ┌────▼──▼────┐
                  │  respond    │  或回到 supervisor 重新分派
                  └────→ END
```

**State 定义**：
```python
class MultiAgentState(TypedDict):
    customer_question: str      # 客户问题
    assigned_agents: list[str]  # supervisor 分派的 Agent 列表
    agent_responses: dict       # {agent_name: response} 各 Agent 回答
    final_answer: str           # 合成后的最终回答
    quality_score: float        # 质量评分
    feedback: str               # 质量检查反馈
    iteration: int              # 重试轮次
    error: Optional[str]
    error_node: Optional[str]
```

**关键设计**：
- Supervisor 使用 LLM 决定分派哪些 Agent（非硬编码路由）
- 每个 Agent 是一个 LangGraph 子图（subgraph），可独立运行
- 质量检查不合格 → 回到 Supervisor 重新分配，形成循环

---

### 模块 5：售后工单（After-Sale Workflow）

**业务**：客户提交售后请求 → 创建工单 → 分析问题 → 审批 → 执行 → 验证 → 关闭

**LangGraph 知识点**：长流程、审批节点（`interrupt_before`）、状态持久化

**图结构**：

```
  ┌──────────┐
  │  create   │  (创建工单)
  │  ticket   │
  └───┬──────┘
      │
  ┌───▼──────────┐
  │  analyze      │  (AI 分析问题类型/严重度)
  └───┬──────────┘
      │
  ┌───▼──────────┐  ← interrupt_before
  │  approve      │  (主管审批：同意/拒绝/需更多信息)
  └───┬──────────┘
      │ (同意)
  ┌───▼──────────┐
  │  execute      │  (执行退款/换货/维修)
  └───┬──────────┘
      │
  ┌───▼──────────┐  ← interrupt_before
  │  verify       │  (客服验证客户满意度)
  └───┬──────────┘
      │ (满意)
  ┌───▼──────┐
  │  close    │
  └────→ END
```

**State 定义**：
```python
class AfterSaleState(TypedDict):
    ticket_id: str              # 工单 ID
    customer_request: str       # 客户诉求
    issue_type: str             # 问题分类: refund/exchange/repair/complaint
    severity: str               # 严重度: low/medium/high/critical
    approval_status: str        # 审批状态: pending/approved/rejected
    approver_comment: str       # 审批人意见
    resolution: str             # 处理方案
    customer_feedback: str      # 客户反馈
    status: str                 # 工单状态: created/analyzing/approved/executing/verifying/closed
    error: Optional[str]
    error_node: Optional[str]
```

**关键设计**：
- 2 个审批节点（主管审批 + 客户验证），2 次 `interrupt_before`
- 每个审批节点支持 3 种决策：同意/拒绝/需更多信息
- 拒绝 → 回到 analyze 重新分析；需更多信息 → 回到上一节点补充

---

### 模块 6：微信风控（WeChat Risk Control）

**业务**：员工微信聊天消息 → 实时分类 → 风险评估 → 告警/拦截/放行

**LangGraph 知识点**：条件分支、状态持久化、实时监控

**图结构**：

```
  ┌──────────┐
  │  receive   │  (接收微信消息)
  │  message   │
  └───┬──────┘
      │
  ┌───▼──────────┐
  │  classify     │  (消息分类: normal/business/sensitive/violation)
  └───┬──┬──┬──┘
      │  │  │
      │  │  └─────→ allow (正常，直接放行) → END
      │  │
      │  └─────→ log_only (业务相关，记录+放行) → END
      │
  ┌───▼──────────┐
  │  risk_assess  │  (敏感/违规消息进入风险评估)
  └───┬──┬──┘
      │  │
      │  └─────→ warn (低风险，记录+提醒) → END
      │
  ┌───▼──────────┐  ← interrupt_before
  │  escalate     │  (高风险，上报主管)
  └───┬──────────┘
      │
  ┌───▼──────────┐
  │  block        │  (拦截消息 + 通知发送者)
  └───┬──────────┘
      │
  ┌───▼──────┐
  │  log      │  (记录风控日志)
  └────→ END
```

**State 定义**：
```python
class WeChatRiskState(TypedDict):
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
```

**关键设计**：
- 四路条件分支：normal → 直接放行，business → 记录，sensitive → 风险评估，violation → 上报
- 高风险 `interrupt_before("escalate")`：暂停等待主管决策
- 风控日志持久化：每条消息的处理结果都写入数据库，支持事后审计

---

### 模块复杂度排序（学习路径）

```
模块1 (意图路由) ──→ 模块3 (知识问答) ──→ 模块2 (线索评级)
    ↑ 简单                  ↑ 中等                ↑ 中等+人机协同

模块5 (售后工单) ──→ 模块6 (微信风控) ──→ 模块4 (多Agent)
    ↑ 长流程                ↑ 条件分支密集          ↑ 最复杂
```

## 4. 数据流、错误处理、测试与部署

### 数据流与状态管理

**核心原则**：每个模块独立 State，通过 FastAPI 接口层串联

```
客户端请求
    │
    ▼
FastAPI Route (api/routes/xxx.py)
    │  ← 请求校验 + 参数注入
    ▼
StateGraph.invoke(state_dict)
    │  ← LangGraph 运行图
    │  ← Checkpoint 自动保存每步状态
    ▼
返回最终 State → API Response
```

**跨模块协作场景**（不耦合，通过 API 调用）：

| 场景 | 流程 |
|------|------|
| 意图路由 → 售后工单 | 意图路由识别为"售后" → 调用售后工单模块 API 创建工单 |
| 微信风控 → 知识问答 | 风控识别为"正常业务咨询" → 调用知识问答模块提供标准回复 |
| 线索评级 → 多Agent客服 | 线索评级完成 → 调用多Agent模块生成客户跟进策略 |

### LLM 抽象层

```python
# src/core/llm.py — 统一接口，支持多后端切换

class LLMProvider(Enum):
    OPENAI = "openai"
    DOUBAO = "doubao"

def get_llm(provider: LLMProvider = None) -> ChatLangChain:
    provider = provider or Settings.LLM_PROVIDER
    if provider == LLMProvider.OPENAI:
        return ChatOpenAI(model="gpt-4o-mini", ...)
    elif provider == LLMProvider.DOUBAO:
        return ChatDoubao(model="doubao-pro-32k", ...)
```

**环境变量**（`.env.example`）：

```env
# LLM 配置
LANGGRAPH_LLM_PROVIDER=openai    # 或 doubao
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=https://api.openai.com/v1
DOUBAO_API_KEY=xxx
DOUBAO_ENDPOINT=https://ark.cn-beijing.volces.com/api/v3

# 向量存储
VECTOR_STORE=chroma              # 或 faiss
CHROMA_PERSIST_DIR=./data/chroma

# Checkpoint
CHECKPOINT_STORE=sqlite          # 或 redis
REDIS_URL=redis://localhost:6379

# 数据库
DATABASE_URL=sqlite:///./data/scrm.db   # 或 mysql://...
```

### 错误处理策略

| 场景 | 处理方式 |
|------|---------|
| LLM 调用失败 | 重试 3 次（指数退避），失败后返回 `error` 状态字段 |
| 向量检索无结果 | 降级到网页搜索（模块3），或返回"未找到相关信息" |
| 人工审批超时（24h 无响应） | 自动提醒 + 默认策略（售后默认同意低金额，风控默认拦截高风险） |
| 模块内部异常 | 捕获 → 写入 state `error` 字段 → API 返回 500 + 详细错误信息 |
| 输入校验失败 | FastAPI 层拦截 → 返回 400 + 校验错误详情 |

**统一错误 State 字段**：每个模块的 State 都包含 `error: Optional[str]` 和 `error_node: Optional[str]`。

### 测试策略

| 类型 | 覆盖范围 | 工具 |
|------|---------|------|
| 单元测试 | 每个节点函数独立测试 | pytest + mock LLM |
| 图测试 | 整个 StateGraph 流程测试 | pytest + LangGraph 测试模式 |
| API 测试 | FastAPI 端点测试 | pytest + httpx |
| 集成测试 | 真实 LLM 调用（可选） | pytest + .env 配置 |

**Mock LLM 策略**：
- 单元测试：mock 节点函数的 LLM 调用，返回固定结果
- 图测试：使用 LangGraph 的 `fake_llm` 模式，验证路由逻辑和循环终止条件
- 集成测试：真实 LLM 调用，标记 `@pytest.mark.integration`，CI 默认 skip

### 部署方案

**本地开发**：
```bash
pip install -e ".[dev]"
cp .env.example .env           # 编辑填入 API Key
python scripts/seed_data.py    # 初始化数据
python scripts/run_module.py intent-router --message "我想咨询产品价格"  # 单模块 CLI
python scripts/run_all.py      # 完整 API 服务 → http://localhost:8000
```

**生产部署**：Docker Compose（API + Redis + Chroma）或 `uvicorn src.api.main:app`

**CI/CD**：
```yaml
# .github/workflows/ci.yml
on: [push, pull_request]
jobs:
  test:
    - pip install -e ".[dev]"
    - pytest tests/unit tests/graph   # 不依赖真实 LLM
    - ruff check src/
```

### 开源社区文档

| 文档 | 内容 |
|------|------|
| `README.md` | 项目介绍 + badges + 知识点索引表 + 快速开始 |
| `docs/getting-started.md` | 详细安装配置指南（含 Doubao 配置） |
| `docs/module-guide.md` | 每模块知识点详解 |
| `docs/architecture.md` | 架构图 + 设计决策说明 |
| 各模块 `README.md` | 模块独立说明 + 代码知识点标注 |
| `CONTRIBUTING.md` | 贡献流程 + Conventional Commits |
| `CODE_OF_CONDUCT.md` | Contributor Covenant v2.1 中文版 |
