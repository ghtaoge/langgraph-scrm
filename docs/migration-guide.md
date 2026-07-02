# 场景迁移指南

> 本项目 6 个模块虽基于 SCRM（社交客户关系管理）设计，但其底层 LangGraph 模式是**通用可复用**的。
> 下文逐模块说明如何将同一套图结构迁移到其他行业场景，帮你快速在自身业务中落地。

---

## 模块 1：意图路由 → 任意分类-分流场景

**LangGraph 核心**：`StateGraph` + `add_conditional_edges` 条件路由

### 迁移路径

| 目标场景 | 输入 | 分类维度 | 分流目标 |
|----------|------|----------|----------|
| **医疗分诊** | 患者症状描述 | 急诊/内科/外科/心理 | 对应科室号源 |
| **政务热线** | 市民诉求文本 | 城管/交通/社保/环保 | 对应职能部门 |
| **电商客服** | 买家消息 | 退货/催发/改地址/问规格 | 对应售后/物流/运营组 |
| **SaaS 支持** | 用户工单描述 | Bug/配置/计费/需求 | 技术组/运营组/产品组 |
| **教育咨询** | 学生提问 | 选课/就业/奖学金/心理 | 教务/就业中心/学工 |

### 迁移要点

1. **替换分类 Prompt**：修改 `INTENT_ROUTER_CLASSIFY_PROMPT` 中的意图类别和描述
2. **替换技能组映射**：修改 `skill_group_map` dict，映射到你业务的对应组
3. **替换各意图处理 Prompt**：如 `respond_consult` → `respond_medical_referral`
4. **State 定义不变**：`message` / `intent` / `confidence` / `skill_group` / `response` 通用

### 示例：迁移到医疗分诊

```python
# 只需修改 3 处：

# 1. 分类 Prompt
MEDICAL_CLASSIFY_PROMPT = """\
你是一个医疗分诊助手。请将患者症状描述分类为：
- emergency: 急诊（胸痛、呼吸困难、大出血）
- internal: 内科（感冒、发烧、消化不良）
- surgery: 外科（骨折、外伤、运动损伤）
- mental: 心理咨询（焦虑、失眠、情绪低落）
患者描述：{message}
"""

# 2. 技能组映射
skill_group_map = {
    "emergency": "急诊科",
    "internal": "内科门诊",
    "surgery": "外科门诊",
    "mental": "心理科",
}

# 3. 各意图处理节点名称和 Prompt（与原结构一致）
```

---

## 模块 2：线索评级 → 任意多轮评估-审核场景

**LangGraph 核心**：循环图 + `interrupt()` 人机协同 + `Annotated` reducer 累积

### 迁移路径

| 目标场景 | 输入 | 循环评估内容 | 审核角色 |
|----------|------|------------|----------|
| **HR 招聘筛选** | 候选人简历 | 技术/沟通/文化匹配度 | HR 主管确认 |
| **保险核保** | 投保人信息 | 健康/财务/职业风险 | 核保师审批 |
| **贷款审批** | 借款人资料 | 信用/收入/负债评估 | 信贷员复核 |
| **学术论文评审** | 论文初稿 | 方法/创新/实验质量 | 领域专家终审 |
| **房产经纪评估** | 客户需求 | 预算/偏好/紧迫度 | 经理分配房源 |

### 迁移要点

1. **替换 `lead_info` 字段**：改为你的业务信息 dict（如 `candidate_info` / `applicant_info`）
2. **替换提问和评估 Prompt**：围绕你的评估维度提问、评分
3. **调整终止条件**：`LEAD_QUALIFIER_MIN_SCORE` 和 `MAX_QUESTIONS` 按业务调整
4. **替换审核 Prompt**：`interrupt()` payload 中的审核问题改为你的业务描述

### 示例：迁移到 HR 招聘筛选

```python
# State 字段替换
class CandidateScreeningState(TypedDict):
    candidate_info: dict              # 姓名/学历/经验/技能
    questions_asked: Annotated[list[str], operator.add]
    answers_received: Annotated[list[str], operator.add]
    score: float                     # 综合匹配度 0-100
    score_history: Annotated[list[float], operator.add]
    qualification: str               # strong_match / potential / no_match
    human_decision: str              # approve / reject / schedule_interview
    error: Optional[str]
    error_node: Optional[str]

# 终止条件调整
MIN_SCORE = 70  # 招聘门槛更高
MAX_QUESTIONS = 7  # 深度评估更多轮
```

---

## 模块 3：知识库问答 → 任意 RAG 场景

**LangGraph 核心**：Corrective RAG + ToolNode + 自检重试循环

### 迁移路径

| 目标场景 | 知识源 | 检索方式 | 补充检索 |
|----------|--------|---------|---------|
| **法律咨询** | 法律条文/判例库 | 向量检索法规 | 法条更新搜索 |
| **技术支持** | API 文档/StackOverflow | 向量检索文档 | 互联网搜索 |
| **医疗知识** | 临床指南/药物手册 | 向量检索指南 | 最新文献搜索 |
| **金融合规** | 监管政策/合规手册 | 向量检索政策 | 监管动态搜索 |
| **教育答疑** | 教材/课件/题库 | 向量检索教材 | 学术搜索 |

### 迁移要点

1. **替换知识文档**：`src/data/sample_docs/` 放入你的领域文档
2. **重建向量索引**：`python scripts/seed_data.py` 自动重建 Chroma
3. **替换检索工具**：`knowledge_qa/tools.py` 中的 `get_retriever_tool()` 集合名和描述改为你的
4. **替换生成/评估/验证 Prompt**：改为你的领域术语
5. **替换 web_search**：接入真实搜索 API（Tavily/SerpAPI），目前是占位实现

### 示例：迁移到法律咨询

```python
# 只需 3 步：

# 1. 放入法律文档
# src/data/sample_docs/06_contract_law.md
# src/data/sample_docs/07_labor_law.md
# ...

# 2. 重建索引
# python scripts/seed_data.py  # 自动加载新文档

# 3. 修改 Prompt 术语
LEGAL_GENERATE_PROMPT = """\
你是一个法律咨询助手。请基于提供的法律条文和判例回答用户问题。
如法律条文未覆盖，请明确说明，不要自行推断。
用户问题：{question}
法律依据：{documents}
"""
```

---

## 模块 4：多Agent客服 → 任意专业协作场景

**LangGraph 核心**：Supervisor 模式 + fan-out/fan-in + dict reducer 合并

### 迁移路径

| 目标场景 | Supervisor 角色 | 专业 Agent | 合成方式 |
|----------|----------------|-----------|---------|
| **医疗会诊** | 分诊调度 | 心内科/影像科/检验科 Agent | 综合诊断意见 |
| **投资顾问** | 理财规划 | 宏观经济/行业分析/风控 Agent | 投资组合建议 |
| **供应链诊断** | 供应链经理 | 采购/物流/质检 Agent | 供应链优化方案 |
| **代码审查** | 项目负责人 | 安全/性能/可维护性 Agent | 代码改进方案 |
| **城市规划** | 规划主任 | 交通/环保/经济 Agent | 综合规划建议 |

### 迁移要点

1. **替换 Supervisor Prompt**：列出你的专业 Agent 名称和能力描述
2. **替换各 Agent Prompt**：每个 Agent 节点改为你的领域专家视角
3. **替换合成 Prompt**：合成器需知道如何整合不同领域回答
4. **替换质量检查标准**：评分维度改为你的业务质量标准

### 示例：迁移到代码审查

```python
# Supervisor Prompt
CODE_REVIEW_SUPERVISOR_PROMPT = """\
你是代码审查调度中心。根据代码变更，决定需要哪些审查视角。
可选 Agent：
- security_expert: 安全审查（漏洞、注入、权限）
- performance_expert: 性能审查（算法、内存、IO）
- maintainability_expert: 可维护性审查（命名、结构、测试）
代码变更：{customer_question}
"""

# 质量：分数 < 7 → 回到 supervisor 补充审查视角
# 完全复用现有图结构和 fan-in/fan-out 逻辑
```

---

## 模块 5：售后工单 → 任意长流程审批场景

**LangGraph 核心**：长流程编排 + 多 `interrupt()` 审批门 + 条件回退

### 迁移路径

| 目标场景 | 流程节点 | 审批门 1 | 审批门 2 | 回退路径 |
|----------|---------|---------|---------|---------|
| **保险理赔** | 报案→定损→核保→赔付→确认 | 核保师审批 | 赔付确认 | 核保拒绝→补充材料 |
| **贷款审批** | 申请→初审→终审→放款→还款确认 | 信贷员初审 | 风控终审 | 终审拒绝→重新申请 |
| **HR 入职** | Offer→背调→审批→入职→试用期确认 | HR 主管审批 | 部门负责人确认 | 背调异常→重新评估 |
| **政府采购** | 需求→招标→评标→审批→签约→验收 | 财务审批 | 领导审批 | 评标不符→重新招标 |
| **内容发布** | 撰写→审核→法务→发布→效果评估 | 编辑审核 | 法务合规 | 法务驳回→修改内容 |

### 迁移要点

1. **替换工单字段**：`ticket_id` → `claim_id` / `application_id` 等
2. **替换流程节点**：每个节点函数改为你的业务步骤
3. **调整审批逻辑**：审批门数量可增减，每门 `interrupt()` 独立
4. **调整回退条件**：`route_after_approve` 和 `route_after_verify` 改为你的回退逻辑

### 示例：迁移到保险理赔

```python
# 流程：报案 → 定损 → 核保审批(interrupt) → 赔付执行 → 客户确认(interrupt) → 结案

class InsuranceClaimState(TypedDict):
    claim_id: str              # 理赔编号
    incident_description: str  # 事故描述
    damage_type: str           # 定损类型: vehicle/property/health
    damage_amount: float       # 定损金额
    underwriting_status: str   # 核保: pending/approved/rejected
    payout_amount: float       # 赔付金额
    customer_confirmation: str # 客户确认: accepted/disputed
    status: str                # 状态链: filed→assessed→underwritten→paid→confirmed→closed
    error: Optional[str]
    error_node: Optional[str]

# 回退：核保拒绝 → 补充材料（回到定损）
# 客户争议 → 重新核保
```

---

## 模块 6：微信风控 → 任意内容/行为风险识别场景

**LangGraph 核心**：多路条件分支 + `interrupt()` 上报 + 阈值路由

### 迁移路径

| 目标场景 | 输入 | 分类维度 | 风险阈值路由 | 上报角色 |
|----------|------|----------|-------------|---------|
| **金融交易风控** | 交易行为 | 正常/异常/可疑/违规 | 金额/频率阈值 | 合规官 |
| **社区内容审核** | 用户帖子 | 正常/争议/敏感/违规 | 举报数/关键词阈值 | 内容运营 |
| **医疗数据合规** | 数据访问请求 | 合规/内部/敏感/违规 | 数据级别阈值 | 安全官 |
| **游戏行为风控** | 玩家行为 | 正常/可疑/作弊/封号 | 行为偏离阈值 | 运维 |
| **供应链合规** | 供应商行为 | 合规/关注/警告/违规 | 信用评分阈值 | 采购合规 |

### 迁移要点

1. **替换分类维度**：4 路分支改为你的风险分级
2. **替换风险评估 Prompt**：围绕你的风险类别（欺诈/作弊/泄露等）
3. **调整风险阈值**：`risk_score >= 80` 改为你的业务阈值
4. **替换上报审批**：`interrupt()` payload 改为你的审批场景

### 示例：迁移到金融交易风控

```python
# 分类维度
TRANSACTION_CLASSIFY_PROMPT = """\
你是一个交易行为分类器。请将交易分类为：
- normal: 正常消费
- unusual: 异常模式（大额/异地/深夜）
- suspicious: 可疑欺诈（多笔小额/新卡/频繁转账）
- violation: 明确违规（洗钱特征/黑名单匹配）
交易详情：{content}
"""

# 阈值：suspicious 评分 ≥ 70 → 上报合规官
# 与微信风控完全同构，只需替换 Prompt 和字段名
```

---

## 通用迁移 Checklist

无论迁移到哪个场景，以下步骤通用：

| 步骤 | 操作 | 涉及文件 |
|------|------|---------|
| 1 | 复制目标模块目录 | `src/modules/{module}/` → 新目录 |
| 2 | 重命名 State 字段 | `state.py` — 替换业务字段名 |
| 3 | 替换 Prompt 模板 | `src/config/prompts.py` — 新增领域 Prompt |
| 4 | 修改节点函数逻辑 | `nodes.py` — 替换业务术语和处理逻辑 |
| 5 | 修改条件路由函数 | `nodes.py` — 替换分支名和映射 |
| 6 | 图结构通常不变 | `graph.py` — 大多数场景边关系可直接复用 |
| 7 | 注册新 API 路由 | `src/api/routes/` — 新增路由文件 |
| 8 | 注册 CLI 入口 | `scripts/run_module.py` — 新增模块映射 |

**核心原则**：LangGraph 的图结构（节点-边-条件路由-循环-interrupt）是**模式层**，与业务无关。迁移只需替换**数据层**（State 字段）和**知识层**（Prompt 模板），图编排逻辑几乎不变。

---

## 选型决策树

```
你的场景是什么？
│
├─ 分类 → 分流 ────────────→ 模块1 意图路由
│
├─ 多轮评估 → 人工确认 ────→ 模块2 线索评级
│
├─ 文档检索 → 问答 ────────→ 模块3 知识库问答
│
├─ 多视角协作 → 合成 ──────→ 模块4 多Agent客服
│
├─ 长流程 → 审批 → 执行 ──→ 模块5 售后工单
│
└─ 风险识别 → 分级处置 ───→ 模块6 微信风控
```

如果你的场景涉及多个模式，可以在 FastAPI 层组合多个模块（通过 API 调用串联），每个模块保持独立运行。
