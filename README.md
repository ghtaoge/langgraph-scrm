# LangGraph-SCRM

> 基于 LangGraph 的 SCRM 智能客服平台 — 6 个实战模块覆盖 LangGraph 所有核心模式

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-1.0+-green)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688)
![License](https://img.shields.io/badge/License-MIT-yellow)

本项目通过 6 个真实 SCRM（社交客户关系管理）业务场景，系统讲解 LangGraph 的全部核心模式：StateGraph、条件路由、循环图、interrupt 人机协同、Checkpoint、Corrective RAG、多 Agent Supervisor、fan-in/fan-out 等。每个模块可独立运行、独立学习，同时通过 FastAPI 统一对外提供服务。

## 模块 → LangGraph 知识点映射

| # | 模块 | 业务场景 | LangGraph 核心知识点 |
|---|------|---------|---------------------|
| 1 | [意图路由](src/modules/intent_router/README.md) | 客户消息分类 → 分配技能组 | `StateGraph`、`add_node`、`add_conditional_edges` |
| 2 | [线索评级](src/modules/lead_qualifier/README.md) | 多轮对话评估线索 → 人工审核 | 循环图、`interrupt()` + `Command(resume=...)`、Checkpoint、`Annotated` reducer |
| 3 | [知识库问答](src/modules/knowledge_qa/README.md) | 知识库 RAG 问答 | Corrective RAG、ToolNode、Embedding 向量检索、自检重试循环 |
| 4 | [多Agent客服](src/modules/multi_agent/README.md) | 多专业 Agent 协作回答 | Supervisor 模式、fan-out/fan-out、dict reducer 合并并行结果 |
| 5 | [售后工单](src/modules/after_sale/README.md) | 售后工单长流程审批 | 长流程编排、双 `interrupt()` 审批门、条件回退 |
| 6 | [微信风控](src/modules/wechat_risk/README.md) | 微信消息风险识别上报 | 四路条件分支、`interrupt()` 上报审批、风险阈值路由 |

## 快速开始

```bash
# 1. 克隆并安装
git clone <repo-url> langgraph-scrm
cd langgraph-scrm
pip install -e ".[dev]"

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 填入 OPENAI_API_KEY（或使用 Doubao，见 docs/getting-started.md）

# 3. 初始化数据（向量库 + SQLite）
python scripts/seed_data.py

# 4. 启动 API 服务
python scripts/run_all.py
# 访问 http://localhost:8000/docs

# 5. 或单独运行某个模块学习
python scripts/run_module.py intent-router
```

## 项目结构

```
langgraph-scrm/
├── src/
│   ├── config/          # Settings 单例 + Prompt 模板库
│   ├── core/            # LLM 抽象层、Checkpoint、State 基类、通用工具
│   ├── modules/         # 6 个业务模块（每个独立 StateGraph）
│   │   ├── intent_router/
│   │   ├── lead_qualifier/
│   │   ├── knowledge_qa/
│   │   ├── multi_agent/
│   │   ├── after_sale/
│   │   └── wechat_risk/
│   ├── api/             # FastAPI 应用、路由、中间件、Pydantic schemas
│   └── data/            # 示例文档、示例消息、数据库初始化
├── tests/               # 单元测试 / 图集成测试 / API 测试
├── scripts/             # CLI 运行脚本
├── docs/                # 详细文档
└── pyproject.toml
```

## 技术栈

| 层 | 技术 |
|----|------|
| 图编排 | LangGraph 1.0+ |
| LLM 框架 | LangChain 0.3+（OpenAI / Doubao 双后端） |
| 向量存储 | Chroma |
| 持久化 | SQLite（Checkpoint + 业务数据） |
| API | FastAPI + Uvicorn |
| 校验 | Pydantic 2 |
| 测试 | pytest + pytest-asyncio |
| 代码规范 | ruff |

## 测试

```bash
# 单元 + 图集成测试（mock LLM，CI 可跑）
pytest tests/unit tests/graph -v

# API 测试
pytest tests/api -v

# 全部
pytest -v
```

## 文档

- [快速开始（含 Doubao 配置）](docs/getting-started.md)
- [模块详解与知识点](docs/module-guide.md)
- [架构设计](docs/architecture.md)
- [场景迁移指南](docs/migration-guide.md) — 如何将 6 模块复用到医疗/金融/HR/电商等行业
- [贡献指南](CONTRIBUTING.md)

## License

[MIT](LICENSE)
