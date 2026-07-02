# 快速开始

本指南帮助你从零搭建 LangGraph-SCRM 开发环境，包括 OpenAI 与 Doubao（火山方舟）两种 LLM 后端配置。

## 环境要求

- Python >= 3.11
- pip / venv
- （可选）Git

## 1. 安装

```bash
git clone <repo-url> langgraph-scrm
cd langgraph-scrm

# 建议使用虚拟环境
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 安装项目及开发依赖
pip install -e ".[dev]"
```

如需 Redis Checkpoint 或 Doubao SDK，按需安装可选依赖：

```bash
pip install -e ".[redis]"      # Redis checkpoint
pip install -e ".[doubao]"     # Doubao SDK（可选，Doubao 走 OpenAI 兼容接口可不装）
```

## 2. 配置 LLM

复制环境变量模板：

```bash
cp .env.example .env
```

### 方式 A：OpenAI

```env
LANGGRAPH_LLM_PROVIDER=openai
OPENAI_API_KEY=sk-你的密钥
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
```

如使用代理/兼容网关，修改 `OPENAI_BASE_URL` 指向代理地址即可。

### 方式 B：Doubao（火山方舟）

Doubao 提供 OpenAI 兼容接口，无需额外 SDK：

```env
LANGGRAPH_LLM_PROVIDER=doubao
DOUBAO_API_KEY=你的火山方舟 API Key
DOUBAO_ENDPOINT=https://ark.cn-beijing.volces.com/api/v3
DOUBAO_MODEL=doubao-pro-32k
```

> 在火山方舟控制台创建推理接入点，获取 API Key 与 endpoint，`DOUBAO_MODEL` 填接入点对应的模型名。

## 3. 其他配置

```env
# 向量存储
VECTOR_STORE=chroma
CHROMA_PERSIST_DIR=./data/chroma

# Checkpoint（memory/sqlite/redis）
CHECKPOINT_STORE=sqlite
CHECKPOINT_DB_PATH=./data/checkpoints.db

# 业务数据库
DATABASE_URL=sqlite:///./data/scrm.db
```

## 4. 初始化数据

```bash
python scripts/seed_data.py
```

该脚本会：
- 将 `src/data/sample_docs/` 下的 FAQ 文档向量化写入 Chroma
- 创建 SQLite 表结构（售后工单表、风控日志表）

## 5. 运行

### 启动 API 服务

```bash
python scripts/run_all.py
```

访问 http://localhost:8000/docs 查看交互式 API 文档。

### 单模块运行（学习用）

```bash
python scripts/run_module.py intent-router
python scripts/run_module.py lead-qualifier
# 含 interrupt 的模块会提示等待恢复
```

## 6. 运行测试

```bash
pytest -v
```

测试全部 mock LLM 调用，无需真实 API Key 即可通过。
