# LangGraph-SCRM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete LangGraph-based SCRM intelligent customer service platform with 6 modules covering all LangGraph core patterns, deployable as an open-source learning + production reference project.

**Architecture:** Each module is an independent StateGraph with its own state, nodes, and routing logic. Modules share a core infrastructure layer (LLM abstraction, checkpointing, config). A FastAPI layer exposes all modules as API endpoints. CLI scripts allow running individual modules for learning.

**Tech Stack:** Python 3.11+, LangGraph 1.0+, LangChain, FastAPI, Chroma (vector store), SQLite (checkpoint + data), pytest, ruff

## Global Constraints

- Python >= 3.11
- LangGraph >= 1.0 (use `interrupt()` + `Command(resume=...)` pattern, not deprecated `interrupt_before` compile param)
- LangChain >= 0.3
- State definitions use `TypedDict` + `Annotated` (LangGraph official pattern)
- All code comments and docstrings in Chinese (中文)
- Every module must run independently — no cross-module imports in module code
- Every module State includes `error: Optional[str]` and `error_node: Optional[str]` fields
- LLM calls retry 3x with exponential backoff on failure
- Tests mock LLM calls — no real LLM in CI
- MIT license
- Conventional Commits format

---

### Task 1: Project Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `.gitignore`
- Create: `src/__init__.py`
- Create: `src/config/__init__.py`
- Create: `src/core/__init__.py`
- Create: `src/modules/__init__.py`
- Create: `src/api/__init__.py`
- Create: `src/api/routes/__init__.py`
- Create: `src/data/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/unit/__init__.py`
- Create: `tests/graph/__init__.py`
- Create: `tests/api/__init__.py`
- Create: `tests/conftest.py`

**Interfaces:**
- Consumes: None (first task)
- Produces: Installable Python package `langgraph-scrm`, directory structure for all subsequent tasks

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "langgraph-scrm"
version = "0.1.0"
description = "基于 LangGraph 的 SCRM 智能客服平台 — 6 个实战模块覆盖所有核心模式"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.11"
dependencies = [
    "langgraph>=1.0",
    "langchain>=0.3",
    "langchain-openai>=0.3",
    "langchain-community>=0.3",
    "langchain-chroma>=0.2",
    "chromadb>=0.5",
    "fastapi>=0.115",
    "uvicorn>=0.34",
    "pydantic>=2.10",
    "python-dotenv>=1.0",
    "httpx>=0.28",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
    "ruff>=0.9",
    "httpx>=0.28",
]
doubao = [
    "volcengine-python-sdk>=1.0",
]
redis = [
    "redis>=5.0",
    "langgraph-checkpoint-redis>=0.1",
]

[tool.ruff]
target-version = "py311"
line-length = 120

[tool.ruff.lint]
select = ["E", "F", "I", "W"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "integration: 真实 LLM 调用测试（CI 中 skip）",
]
asyncio_mode = "auto"

[build-system]
requires = ["setuptools>=75"]
build-backend = "setuptools.backends._legacy:_Backend"
```

- [ ] **Step 2: Create .env.example**

```env
# LLM 配置（选择一个即可）
LANGGRAPH_LLM_PROVIDER=openai    # 或 doubao
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini

# Doubao 配置（当 LANGGRAPH_LLM_PROVIDER=doubao 时使用）
DOUBAO_API_KEY=xxx
DOUBAO_ENDPOINT=https://ark.cn-beijing.volces.com/api/v3
DOUBAO_MODEL=doubao-pro-32k

# 向量存储
VECTOR_STORE=chroma              # 或 faiss
CHROMA_PERSIST_DIR=./data/chroma

# Checkpoint
CHECKPOINT_STORE=sqlite          # 或 redis
CHECKPOINT_DB_PATH=./data/checkpoints.db
REDIS_URL=redis://localhost:6379

# 数据库（售后工单/风控日志持久化）
DATABASE_URL=sqlite:///./data/scrm.db
```

- [ ] **Step 3: Create .gitignore**

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
*.egg

# Environment
.env
.env.local

# Data (本地运行生成的数据)
data/
*.db

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# Claude
.claude/

# Testing
.pytest_cache/
.coverage
htmlcov/
```

- [ ] **Step 4: Create all __init__.py files and directory structure**

```bash
mkdir -p src/config src/core src/modules src/api/routes src/data
mkdir -p tests/unit tests/graph tests/api
mkdir -p data
```

Create each `__init__.py` with a one-line docstring:

```python
# src/__init__.py
"""LangGraph-SCRM: 基于 LangGraph 的 SCRM 智能客服平台"""
```

(Repeat for each sub-package: `src/config`, `src/core`, `src/modules`, `src/api`, `src/api/routes`, `src/data`, `tests`, `tests/unit`, `tests/graph`, `tests/api`)

- [ ] **Step 5: Create tests/conftest.py with shared fixtures**

```python
"""共享测试 fixtures — mock LLM、测试 State、临时目录"""
import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.messages import AIMessage


@pytest.fixture
def mock_llm():
    """Mock LLM，返回固定 AIMessage"""
    llm = MagicMock()
    llm.invoke = MagicMock(return_value=AIMessage(content="mock response"))
    llm.ainvoke = AsyncMock(return_value=AIMessage(content="mock response"))
    return llm


@pytest.fixture
def mock_llm_with_tool_calls():
    """Mock LLM，返回带 tool_calls 的 AIMessage"""
    llm = MagicMock()
    msg = AIMessage(
        content="",
        tool_calls=[{"name": "retriever", "args": {"query": "test"}, "id": "tc1"}],
    )
    llm.invoke = MagicMock(return_value=msg)
    llm.ainvoke = AsyncMock(return_value=msg)
    return llm


@pytest.fixture
def tmp_data_dir(tmp_path):
    """临时数据目录（用于 SQLite/Chroma 测试）"""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir
```

- [ ] **Step 6: Verify project is installable**

Run: `pip install -e ".[dev]"`
Expected: Installation succeeds, `langgraph-scrm` package available

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat: 项目骨架 — pyproject.toml, 目录结构, 测试 fixtures"
```

---

### Task 2: Core Config

**Files:**
- Create: `src/config/settings.py`
- Create: `src/config/prompts.py`

**Interfaces:**
- Consumes: `.env` file (environment variables)
- Produces: `Settings` singleton (all config values), `INTENT_ROUTER_PROMPT`, `LEAD_QUALIFIER_PROMPT`, `KNOWLEDGE_QA_PROMPT`, `MULTI_AGENT_PROMPT`, `AFTER_SALE_PROMPT`, `WECHAT_RISK_PROMPT`

- [ ] **Step 1: Write test for Settings loading**

```python
# tests/unit/test_settings.py
"""测试 Settings 配置加载"""
import pytest
from src.config.settings import Settings, LLMProvider


def test_default_provider_is_openai():
    """默认 LLM 提供者应为 OpenAI"""
    s = Settings()
    assert s.LLM_PROVIDER == LLMProvider.OPENAI


def test_provider_enum_values():
    """LLMProvider 枚举值正确"""
    assert LLMProvider.OPENAI.value == "openai"
    assert LLMProvider.DOUBAO.value == "doubao"


def test_default_checkpoint_is_sqlite():
    """默认 Checkpoint 存储应为 SQLite"""
    s = Settings()
    assert s.CHECKPOINT_STORE == "sqlite"


def test_default_vector_store_is_chroma():
    """默认向量存储应为 Chroma"""
    s = Settings()
    assert s.VECTOR_STORE == "chroma"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_settings.py -v`
Expected: FAIL — `Settings` not defined

- [ ] **Step 3: Implement src/config/settings.py**

```python
"""全局配置加载 — 从 .env 读取所有环境变量，提供 Settings 单例"""
import os
from enum import Enum
from dotenv import load_dotenv

load_dotenv()


class LLMProvider(Enum):
    """LLM 提供者枚举"""
    OPENAI = "openai"
    DOUBAO = "doubao"


class Settings:
    """全局配置单例 — 所有模块共享"""

    # LLM 配置
    LLM_PROVIDER: LLMProvider = LLMProvider(os.getenv("LANGGRAPH_LLM_PROVIDER", "openai"))
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    DOUBAO_API_KEY: str = os.getenv("DOUBAO_API_KEY", "")
    DOUBAO_ENDPOINT: str = os.getenv("DOUBAO_ENDPOINT", "https://ark.cn-beijing.volces.com/api/v3")
    DOUBAO_MODEL: str = os.getenv("DOUBAO_MODEL", "doubao-pro-32k")

    # 向量存储配置
    VECTOR_STORE: str = os.getenv("VECTOR_STORE", "chroma")
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")

    # Checkpoint 配置
    CHECKPOINT_STORE: str = os.getenv("CHECKPOINT_STORE", "sqlite")
    CHECKPOINT_DB_PATH: str = os.getenv("CHECKPOINT_DB_PATH", "./data/checkpoints.db")
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")

    # 数据库配置
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./data/scrm.db")

    # LLM 重试配置
    LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "3"))
    LLM_RETRY_DELAY: float = float(os.getenv("LLM_RETRY_DELAY", "1.0"))

    # 业务配置
    LEAD_QUALIFIER_MIN_SCORE: float = float(os.getenv("LEAD_QUALIFIER_MIN_SCORE", "60.0"))
    LEAD_QUALIFIER_MAX_QUESTIONS: int = int(os.getenv("LEAD_QUALIFIER_MAX_QUESTIONS", "5"))
    AFTER_SALE_APPROVAL_TIMEOUT_HOURS: int = int(os.getenv("AFTER_SALE_APPROVAL_TIMEOUT_HOURS", "24"))


# 全局单例
settings = Settings()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_settings.py -v`
Expected: All PASS

- [ ] **Step 5: Implement src/config/prompts.py**

```python
"""所有 Prompt 模板集中管理 — 每个模块的 prompt 模板定义在此"""

# ── 模块 1：意图路由 ──
INTENT_ROUTER_CLASSIFY_PROMPT = """\
你是一个客户消息意图分类器。请将客户消息分类为以下四种意图之一：

- consult: 咨询类（产品咨询、价格询问、功能了解）
- complaint: 投诉类（服务不满、质量投诉、态度投诉）
- after_sale: 售后类（退款申请、换货需求、维修请求）
- other: 其他（无法明确分类的消息）

请以 JSON 格式返回分类结果，包含 intent（意图类别）和 confidence（置信度 0-1）。

客户消息：{message}
"""

INTENT_ROUTER_RESPOND_CONSULT_PROMPT = """\
你是一个产品咨询客服。请根据客户消息给出专业的咨询回复。

客户消息：{message}
意图：{intent}
"""

INTENT_ROUTER_ESCALATE_COMPLAINT_PROMPT = """\
你是一个投诉处理客服。客户提交了投诉，请给出安抚性回复并告知将升级处理。

客户消息：{message}
"""

INTENT_ROUTER_HANDLE_AFTER_SALE_PROMPT = """\
你是一个售后客服。客户提交了售后请求，请给出引导性回复帮助客户描述具体问题。

客户消息：{message}
"""

INTENT_ROUTER_OTHER_PROMPT = """\
你是一个通用客服。客户的请求无法明确分类，请给出友好的通用回复并引导客户进一步描述需求。

客户消息：{message}
"""

# ── 模块 2：线索评级 ──
LEAD_QUALIFIER_QUESTION_PROMPT = """\
你是一个线索评级助手。根据线索信息和已收集的问答，提出下一个评估问题。

线索信息：{lead_info}
已提问：{questions_asked}
已回答：{answers_received}
当前评分：{score}

请提出一个有助于评估线索质量的问题。
"""

LEAD_QUALIFIER_EVALUATE_PROMPT = """\
你是一个线索评级评估器。根据线索信息和收集的问答，给出线索评分（0-100）和评级。

线索信息：{lead_info}
问题和回答：
{qa_pairs}

评分标准：
- 60+ 分为 hot（高价值线索）
- 30-60 分为 warm（中等线索）
- 0-30 分为 cold（低价值线索）

请以 JSON 返回 score（分数）和 qualification（评级：hot/warm/cold）。
"""

# ── 模块 3：知识库问答 ──
KNOWLEDGE_QA_RETRIEVE_PROMPT = """\
你是一个知识库问答助手。请根据用户问题，判断是否需要检索知识库来回答。

用户问题：{question}
"""

KNOWLEDGE_QA_GRADE_PROMPT = """\
你是一个文档相关性评估器。请评估检索到的文档片段与用户问题的相关性。

用户问题：{question}
文档片段：{documents}

请判断文档是否足以回答问题。返回 "relevant" 或 "irrelevant"。
"""

KNOWLEDGE_QA_GENERATE_PROMPT = """\
你是一个知识库问答助手。请基于提供的文档片段回答用户问题。如果文档中没有相关信息，请明确说明。

用户问题：{question}
文档片段：{documents}

请在回答末尾标注引用来源。
"""

KNOWLEDGE_QA_VERIFY_PROMPT = """\
你是一个回答质量检查器。请检查以下回答是否：
1. 基于提供的文档片段（非幻觉）
2. 直接回答了用户问题
3. 包含引用来源

用户问题：{question}
回答：{answer}
文档片段：{documents}

返回 "passed" 或 "failed"，并说明原因。
"""

# ── 模块 4：多Agent客服 ──
MULTI_AGENT_SUPERVISOR_PROMPT = """\
你是一个客服调度中心。根据客户问题，决定需要哪些专业 Agent 来协作回答。

可选 Agent：
- product_expert: 产品专家（了解产品功能、规格、对比）
- policy_expert: 政策专家（了解退换货政策、保修条款、公司规则）
- order_handler: 订单处理员（了解订单状态、物流、支付问题）

客户问题：{customer_question}

请以 JSON 返回需要分派的 Agent 列表（assigned_agents）。
"""

MULTI_AGENT_PRODUCT_PROMPT = """\
你是产品专家 Agent。请从产品角度回答客户问题。

客户问题：{customer_question}
"""

MULTI_AGENT_POLICY_PROMPT = """\
你是政策专家 Agent。请从公司政策角度回答客户问题。

客户问题：{customer_question}
"""

MULTI_AGENT_ORDER_PROMPT = """\
你是订单处理 Agent。请从订单和物流角度回答客户问题。

客户问题：{customer_question}
"""

MULTI_AGENT_SYNTHESIZE_PROMPT = """\
你是一个回答合成器。请将多个专业 Agent 的回答合成为一份完整、连贯的客户回复。

客户问题：{customer_question}
各 Agent 回答：{agent_responses}
"""

MULTI_AGENT_QUALITY_PROMPT = """\
你是一个回答质量检查器。请检查合成的回答质量。

客户问题：{customer_question}
合成回答：{final_answer}

返回质量评分（0-10）和改进建议。低于 7 分需要重新生成。
"""

# ── 模块 5：售后工单 ──
AFTER_SALE_ANALYZE_PROMPT = """\
你是一个售后问题分析器。请分析客户售后请求，确定问题类型和严重度。

问题类型：refund（退款）/ exchange（换货）/ repair（维修）/ complaint（投诉）
严重度：low / medium / high / critical

客户诉求：{customer_request}

请以 JSON 返回 issue_type 和 severity。
"""

AFTER_SALE_EXECUTE_PROMPT = """\
你是一个售后执行助手。请根据工单信息生成处理方案。

工单 ID：{ticket_id}
问题类型：{issue_type}
严重度：{severity}
客户诉求：{customer_request}

请给出具体的处理方案。
"""

# ── 模块 6：微信风控 ──
WECHAT_RISK_CLASSIFY_PROMPT = """\
你是一个微信消息分类器。请将消息分类为以下四种类型之一：

- normal: 正常闲聊（无关业务，无风险）
- business: 业务相关（客户沟通、工作讨论，需记录但不风险）
- sensitive: 敏感信息（涉及价格泄露、客户数据、内部消息，需风险评估）
- violation: 明确违规（辱骂客户、泄密、欺诈证据，需立即上报）

消息内容：{content}
发送者：{sender}

请以 JSON 返回 message_type 和分类理由。
"""

WECHAT_RISK_ASSESS_PROMPT = """\
你是一个风险评估器。请评估敏感/违规消息的风险等级。

消息内容：{content}
消息类型：{message_type}

风险类别：
- info_leak: 信息泄露
- harassment: 骚扰辱骂
- fraud: 欺诈嫌疑
- compliance: 合规风险
- other: 其他风险

请以 JSON 返回 risk_score（0-100）、risk_category 和处理建议。
"""
```

- [ ] **Step 6: Run all tests**

Run: `pytest tests/unit/test_settings.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add src/config/ tests/unit/test_settings.py
git commit -m "feat: 核心配置 — Settings 单例 + Prompt 模板库"
```

---

### Task 3: Core Infrastructure

**Files:**
- Create: `src/core/llm.py`
- Create: `src/core/checkpoint.py`
- Create: `src/core/state.py`
- Create: `src/core/tools.py`
- Create: `src/core/nodes.py`
- Create: `tests/unit/test_llm.py`
- Create: `tests/unit/test_checkpoint.py`

**Interfaces:**
- Consumes: `Settings` from Task 2
- Produces: `get_llm()` function, `get_checkpointer()` function, `BaseState` TypedDict, `safe_llm_call()` helper, `create_error_state()` helper

- [ ] **Step 1: Write tests for core infrastructure**

```python
# tests/unit/test_llm.py
"""测试 LLM 抽象层"""
from src.config.settings import LLMProvider
from src.core.llm import get_llm


def test_get_llm_returns_openai_by_default():
    """默认返回 OpenAI LLM"""
    llm = get_llm()
    assert llm is not None


def test_get_llm_openai_provider():
    """OpenAI 提供者返回 ChatOpenAI 实例"""
    llm = get_llm(LLMProvider.OPENAI)
    assert llm is not None


def test_get_llm_with_custom_base_url():
    """支持自定义 base_url（用于代理）"""
    llm = get_llm(base_url="https://custom-proxy.com/v1")
    assert llm is not None
```

```python
# tests/unit/test_checkpoint.py
"""测试 Checkpoint 配置"""
import tempfile
from src.core.checkpoint import get_checkpointer


def test_get_sqlite_checkpointer():
    """SQLite checkpointer 正常创建"""
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        cp = get_checkpointer(store="sqlite", db_path=f.name)
        assert cp is not None


def test_get_memory_checkpointer():
    """InMemory checkpointer 正常创建（测试用）"""
    cp = get_checkpointer(store="memory")
        assert cp is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_llm.py tests/unit/test_checkpoint.py -v`
Expected: FAIL — modules not defined

- [ ] **Step 3: Implement src/core/llm.py**

```python
"""LLM 抽象层 — 统一接口，支持 OpenAI + Doubao 双后端切换"""
import asyncio
from functools import wraps
from typing import Callable, Optional

from langchain_openai import ChatOpenAI

from src.config.settings import Settings, LLMProvider, settings


def get_llm(
    provider: Optional[LLMProvider] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.0,
) -> ChatOpenAI:
    """获取 LLM 实例

    Args:
        provider: LLM 提供者，默认从 Settings 读取
        model: 模型名，默认从 Settings 读取
        base_url: API base URL，默认从 Settings 读取
        temperature: 温度参数，默认 0（确定性输出）

    Returns:
        ChatOpenAI 实例（Doubao 也通过 OpenAI 兼容接口接入）
    """
    provider = provider or settings.LLM_PROVIDER
    if provider == LLMProvider.OPENAI:
        return ChatOpenAI(
            model=model or settings.OPENAI_MODEL,
            api_key=settings.OPENAI_API_KEY,
            base_url=base_url or settings.OPENAI_BASE_URL,
            temperature=temperature,
            max_retries=settings.LLM_MAX_RETRIES,
        )
    elif provider == LLMProvider.DOUBAO:
        # Doubao 使用 OpenAI 兼容接口接入
        return ChatOpenAI(
            model=model or settings.DOUBAO_MODEL,
            api_key=settings.DOUBAO_API_KEY,
            base_url=base_url or settings.DOUBAO_ENDPOINT,
            temperature=temperature,
            max_retries=settings.LLM_MAX_RETRIES,
        )
    else:
        raise ValueError(f"不支持的 LLM 提供者: {provider}")


def safe_llm_call(func: Callable) -> Callable:
    """LLM 调用安全装饰器 — 捕获异常并返回错误 state 字段

    用法：
        @safe_llm_call
        def my_node(state):
            result = llm.invoke(...)
            return {"response": result.content}
    """
    @wraps(func)
    def wrapper(state):
        try:
            return func(state)
        except Exception as e:
            node_name = func.__name__
            return {
                "error": f"节点 {node_name} 执行失败: {str(e)}",
                "error_node": node_name,
            }

    @wraps(func)
    async def async_wrapper(state):
        try:
            return await func(state)
        except Exception as e:
            node_name = func.__name__
            return {
                "error": f"节点 {node_name} 执行失败: {str(e)}",
                "error_node": node_name,
            }

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return wrapper


def create_error_state(error_msg: str, node_name: str) -> dict:
    """创建错误 state 字段（用于节点内部异常处理）"""
    return {"error": error_msg, "error_node": node_name}
```

- [ ] **Step 4: Implement src/core/checkpoint.py**

```python
"""Checkpoint 配置 — 支持 SQLite/Redis/InMemory 三种持久化"""
import sqlite3
from typing import Optional

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from src.config.settings import settings


def get_checkpointer(
    store: Optional[str] = None,
    db_path: Optional[str] = None,
) -> SqliteSaver | InMemorySaver:
    """获取 Checkpointer 实例

    Args:
        store: 存储类型 "sqlite"/"memory"/"redis"，默认从 Settings 读取
        db_path: SQLite 数据库路径，默认从 Settings 读取

    Returns:
        Checkpointer 实例
    """
    store = store or settings.CHECKPOINT_STORE

    if store == "sqlite":
        path = db_path or settings.CHECKPOINT_DB_PATH
        conn = sqlite3.connect(path, check_same_thread=False)
        return SqliteSaver(conn)
    elif store == "memory":
        return InMemorySaver()
    elif store == "redis":
        try:
            from langgraph.checkpoint.redis import RedisSaver
            return RedisSaver.from_conn_string(settings.REDIS_URL)
        except ImportError:
            raise ImportError("Redis checkpoint 需要安装: pip install langgraph-scrm[redis]")
    else:
        raise ValueError(f"不支持的 checkpoint 存储: {store}")
```

- [ ] **Step 5: Implement src/core/state.py**

```python
"""State 定义基类 — 所有模块共享的错误字段"""
from typing import Optional, TypedDict


class BaseState(TypedDict):
    """所有模块 State 的基类字段

    每个模块的 State 定义应继承或包含这些字段。
    LangGraph 的 TypedDict 不能直接继承（需要全部字段在同一个 TypedDict 中），
    所以这个基类只作为文档参考 — 各模块需显式包含 error 和 error_node 字段。
    """
    error: Optional[str]        # 错误信息（null = 无错误）
    error_node: Optional[str]   # 发生错误的节点名
```

- [ ] **Step 6: Implement src/core/tools.py**

```python
"""通用工具注册 — 搜索/计算等基础工具"""
from langchain_core.tools import tool


@tool
def web_search(query: str) -> str:
    """网页搜索工具（用于知识问答模块的补充检索）

    Args:
        query: 搜索关键词

    Returns:
        搜索结果摘要
    """
    # 生产环境应接入真实搜索 API（如 Tavily/SerpAPI）
    # 这里返回占位结果，方便学习和测试
    return f"[搜索结果] 关键词 '{query}' 的搜索结果（生产环境请接入真实搜索 API）"


@tool
def calculate(expression: str) -> str:
    """简单计算工具

    Args:
        expression: 数学表达式

    Returns:
        计算结果
    """
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception:
        return f"计算错误: 无法计算 '{expression}'"


# 工具名称映射（供 ToolNode 使用）
COMMON_TOOLS = [web_search, calculate]
```

- [ ] **Step 7: Implement src/core/nodes.py**

```python
"""通用节点函数基类 — 提供错误处理和日志记录的标准模式"""
import logging
from typing import Any, TypedDict

from src.core.llm import create_error_state

logger = logging.getLogger("langgraph-scrm")


def create_error_node(error_msg: str, node_name: str) -> dict:
    """创建错误节点返回值（各模块节点内部使用）

    Args:
        error_msg: 错误信息
        node_name: 节点名

    Returns:
        包含 error 和 error_node 的 dict
    """
    logger.error(f"节点 {node_name} 错误: {error_msg}")
    return create_error_state(error_msg, node_name)
```

- [ ] **Step 8: Run all core tests**

Run: `pytest tests/unit/test_llm.py tests/unit/test_checkpoint.py -v`
Expected: All PASS

- [ ] **Step 9: Commit**

```bash
git add src/core/ tests/unit/test_llm.py tests/unit/test_checkpoint.py
git commit -m "feat: 核心基础设施 — LLM 抽象层, Checkpoint, State 基类, 工具注册"
```

---

### Task 4: Module 1 — Intent Router

**Files:**
- Create: `src/modules/01_intent_router/__init__.py`
- Create: `src/modules/01_intent_router/state.py`
- Create: `src/modules/01_intent_router/nodes.py`
- Create: `src/modules/01_intent_router/graph.py`
- Create: `src/modules/01_intent_router/README.md`
- Create: `tests/unit/test_intent_nodes.py`
- Create: `tests/graph/test_intent_router_graph.py`

**Interfaces:**
- Consumes: `get_llm()` from Task 3, `Settings` from Task 2, prompts from Task 2
- Produces: `IntentRouterState`, `build_intent_router_graph()`, runnable StateGraph for intent classification

- [ ] **Step 1: Write state.py**

```python
"""意图路由模块 State 定义"""
from typing import Optional, TypedDict


class IntentRouterState(TypedDict):
    """意图路由状态

    LangGraph 知识点: StateGraph 使用 TypedDict 定义状态，
    所有节点函数接收 state 并返回 state 更新 dict。
    """
    message: str                # 客户原始消息
    intent: str                 # 分类结果: consult/complaint/after_sale/other
    confidence: float           # 分类置信度 (0-1)
    skill_group: str            # 分配的技能组
    response: str               # 最终回复
    error: Optional[str]        # 错误信息
    error_node: Optional[str]   # 发生错误的节点名
```

- [ ] **Step 2: Write nodes.py**

```python
"""意图路由模块节点函数

LangGraph 知识点:
- 每个节点是一个函数，接收 state dict，返回 state 更新 dict
- 节点函数只更新需要修改的字段，其他字段保持不变
- 使用 @safe_llm_call 装饰器处理 LLM 调用异常
"""
import json
import logging

from langchain_core.messages import HumanMessage

from src.config.prompts import (
    INTENT_ROUTER_CLASSIFY_PROMPT,
    INTENT_ROUTER_RESPOND_CONSULT_PROMPT,
    INTENT_ROUTER_ESCALATE_COMPLAINT_PROMPT,
    INTENT_ROUTER_HANDLE_AFTER_SALE_PROMPT,
    INTENT_ROUTER_OTHER_PROMPT,
)
from src.core.llm import get_llm, safe_llm_call

logger = logging.getLogger("langgraph-scrm.intent_router")


# ── 意图分类节点 ──
@safe_llm_call
def classify_node(state: dict) -> dict:
    """意图分类节点 — 调用 LLM 将客户消息分为 4 类

    LangGraph 知识点: add_node("classify", classify_node)
    节点函数接收完整 state，返回需要更新的字段。
    """
    llm = get_llm()
    prompt = INTENT_ROUTER_CLASSIFY_PROMPT.format(message=state["message"])
    response = llm.invoke([HumanMessage(content=prompt)])

    # 解析 LLM 返回的 JSON
    try:
        result = json.loads(response.content)
        intent = result.get("intent", "other")
        confidence = result.get("confidence", 0.5)
    except (json.JSONDecodeError, KeyError):
        intent = "other"
        confidence = 0.0
        logger.warning(f"意图分类 JSON 解析失败，降级为 other: {response.content}")

    # 根据意图分配技能组
    skill_group_map = {
        "consult": "产品咨询组",
        "complaint": "投诉处理组",
        "after_sale": "售后服务组",
        "other": "通用客服组",
    }

    return {
        "intent": intent,
        "confidence": confidence,
        "skill_group": skill_group_map.get(intent, "通用客服组"),
    }


# ── 各意图处理节点 ──
@safe_llm_call
def respond_consult(state: dict) -> dict:
    """咨询类处理节点"""
    llm = get_llm()
    prompt = INTENT_ROUTER_RESPOND_CONSULT_PROMPT.format(
        message=state["message"], intent=state["intent"]
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"response": response.content}


@safe_llm_call
def escalate_complaint(state: dict) -> dict:
    """投诉类处理节点"""
    llm = get_llm()
    prompt = INTENT_ROUTER_ESCALATE_COMPLAINT_PROMPT.format(message=state["message"])
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"response": response.content}


@safe_llm_call
def handle_after_sale(state: dict) -> dict:
    """售后类处理节点"""
    llm = get_llm()
    prompt = INTENT_ROUTER_HANDLE_AFTER_SALE_PROMPT.format(message=state["message"])
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"response": response.content}


@safe_llm_call
def handle_other(state: dict) -> dict:
    """其他类处理节点"""
    llm = get_llm()
    prompt = INTENT_ROUTER_OTHER_PROMPT.format(message=state["message"])
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"response": response.content}


# ── 条件路由函数 ──
def route_by_intent(state: dict) -> str:
    """条件路由函数 — 根据意图分流到不同处理节点

    LangGraph 知识点: add_conditional_edges("classify", route_by_intent)
    条件路由函数返回下一个节点名（字符串）。
    """
    intent = state.get("intent", "other")
    route_map = {
        "consult": "respond_consult",
        "complaint": "escalate_complaint",
        "after_sale": "handle_after_sale",
        "other": "handle_other",
    }
    return route_map.get(intent, "handle_other")
```

- [ ] **Step 3: Write graph.py**

```python
"""意图路由模块 StateGraph 定义

LangGraph 知识点:
- StateGraph(StateSchema) 创建图
- add_node(name, func) 添加节点
- add_edge(from, to) 添加固定边
- add_conditional_edges(from, condition_fn, map) 添加条件边
- compile() 编译为可运行图
"""
from langgraph.graph import StateGraph, START, END

from src.modules.01_intent_router.state import IntentRouterState
from src.modules.01_intent_router.nodes import (
    classify_node,
    respond_consult,
    escalate_complaint,
    handle_after_sale,
    handle_other,
    route_by_intent,
)


def build_intent_router_graph():
    """构建意图路由 StateGraph

    图结构:
    START → classify → (条件路由) → respond_consult / escalate_complaint / handle_after_sale / handle_other → END
    """
    graph = StateGraph(IntentRouterState)

    # 添加节点
    graph.add_node("classify", classify_node)
    graph.add_node("respond_consult", respond_consult)
    graph.add_node("escalate_complaint", escalate_complaint)
    graph.add_node("handle_after_sale", handle_after_sale)
    graph.add_node("handle_other", handle_other)

    # 添加边
    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", route_by_intent)
    graph.add_edge("respond_consult", END)
    graph.add_edge("escalate_complaint", END)
    graph.add_edge("handle_after_sale", END)
    graph.add_edge("handle_other", END)

    return graph.compile()


# 模块入口 — 直接运行此模块可测试意图路由
if __name__ == "__main__":
    app = build_intent_router_graph()
    result = app.invoke({
        "message": "我想了解一下你们产品的价格",
        "intent": "",
        "confidence": 0.0,
        "skill_group": "",
        "response": "",
        "error": None,
        "error_node": None,
    })
    print(f"意图: {result['intent']}")
    print(f"技能组: {result['skill_group']}")
    print(f"回复: {result['response']}")
```

- [ ] **Step 4: Write __init__.py**

```python
"""意图路由模块 — 客户消息意图分类与技能组分配"""
from src.modules.01_intent_router.graph import build_intent_router_graph

__all__ = ["build_intent_router_graph"]
```

- [ ] **Step 5: Write module README.md**

```markdown
# 模块 1：意图路由（Intent Router）

> 🎯 **LangGraph 知识点**：`StateGraph`、`add_node`、`add_edge`、`add_conditional_edges`

## 业务场景

客户发消息 → AI 分类意图 → 自动分配到对应技能组

## 图结构

```
START → classify → (条件路由)
  ├── consult → respond_consult → END
  ├── complaint → escalate_complaint → END
  ├── after_sale → handle_after_sale → END
  └── other → handle_other → END
```

## 知识点详解

### StateGraph

LangGraph 的核心概念。用 `StateGraph(StateSchema)` 创建一个状态图，图的每个节点共享同一个 State（TypedDict）。

### add_node

添加节点：`graph.add_node("name", func)`。节点函数接收 state dict，返回 state 更新 dict。

### add_conditional_edges

条件路由：`graph.add_conditional_edges("from_node", condition_fn)`。条件函数根据 state 返回下一个节点名。

## 运行

```bash
python -m src.modules.01_intent_router.graph
```
```

- [ ] **Step 6: Write unit tests**

```python
# tests/unit/test_intent_nodes.py
"""意图路由节点单元测试 — mock LLM，验证节点逻辑"""
import json
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

from src.modules.01_intent_router.nodes import (
    classify_node,
    respond_consult,
    route_by_intent,
)


@patch("src.modules.01_intent_router.nodes.get_llm")
def test_classify_node_returns_intent(mock_get_llm):
    """classify_node 应返回意图和置信度"""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(
        content=json.dumps({"intent": "consult", "confidence": 0.95})
    )
    mock_get_llm.return_value = mock_llm

    state = {"message": "我想咨询产品价格", "intent": "", "confidence": 0.0, "skill_group": "", "response": "", "error": None, "error_node": None}
    result = classify_node(state)

    assert result["intent"] == "consult"
    assert result["confidence"] == 0.95
    assert result["skill_group"] == "产品咨询组"


@patch("src.modules.01_intent_router.nodes.get_llm")
def test_classify_node_handles_json_error(mock_get_llm):
    """classify_node 应处理 JSON 解析失败"""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="not json")
    mock_get_llm.return_value = mock_llm

    state = {"message": "test", "intent": "", "confidence": 0.0, "skill_group": "", "response": "", "error": None, "error_node": None}
    result = classify_node(state)

    assert result["intent"] == "other"
    assert result["confidence"] == 0.0


def test_route_by_intent_returns_correct_node():
    """route_by_intent 应正确路由到对应节点"""
    assert route_by_intent({"intent": "consult"}) == "respond_consult"
    assert route_by_intent({"intent": "complaint"}) == "escalate_complaint"
    assert route_by_intent({"intent": "after_sale"}) == "handle_after_sale"
    assert route_by_intent({"intent": "other"}) == "handle_other"
    assert route_by_intent({"intent": "unknown"}) == "handle_other"


@patch("src.modules.01_intent_router.nodes.get_llm")
def test_respond_consult_returns_response(mock_get_llm):
    """respond_consult 应返回咨询回复"""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="我们的产品价格如下...")
    mock_get_llm.return_value = mock_llm

    state = {"message": "价格咨询", "intent": "consult", "confidence": 0.9, "skill_group": "产品咨询组", "response": "", "error": None, "error_node": None}
    result = respond_consult(state)

    assert "response" in result
```

- [ ] **Step 7: Write graph integration test**

```python
# tests/graph/test_intent_router_graph.py
"""意图路由图集成测试 — 使用 fake LLM 验证完整图流程"""
import json
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

from src.modules.01_intent_router.graph import build_intent_router_graph


@patch("src.modules.01_intent_router.nodes.get_llm")
def test_intent_router_full_flow_consult(mock_get_llm):
    """意图路由完整流程 — 咨询类"""
    mock_llm = MagicMock()
    # classify 节点返回咨询意图
    mock_llm.invoke.side_effect = [
        AIMessage(content=json.dumps({"intent": "consult", "confidence": 0.9})),
        AIMessage(content="我们的产品价格如下..."),
    ]
    mock_get_llm.return_value = mock_llm

    graph = build_intent_router_graph()
    result = graph.invoke({
        "message": "我想咨询产品价格",
        "intent": "",
        "confidence": 0.0,
        "skill_group": "",
        "response": "",
        "error": None,
        "error_node": None,
    })

    assert result["intent"] == "consult"
    assert result["skill_group"] == "产品咨询组"
    assert result["response"] != ""


@patch("src.modules.01_intent_router.nodes.get_llm")
def test_intent_router_full_flow_complaint(mock_get_llm):
    """意图路由完整流程 — 投诉类"""
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = [
        AIMessage(content=json.dumps({"intent": "complaint", "confidence": 0.85})),
        AIMessage(content="非常抱歉给您带来不好的体验..."),
    ]
    mock_get_llm.return_value = mock_llm

    graph = build_intent_router_graph()
    result = graph.invoke({
        "message": "服务太差了我要投诉",
        "intent": "",
        "confidence": 0.0,
        "skill_group": "",
        "response": "",
        "error": None,
        "error_node": None,
    })

    assert result["intent"] == "complaint"
    assert result["skill_group"] == "投诉处理组"
```

- [ ] **Step 8: Run all module 1 tests**

Run: `pytest tests/unit/test_intent_nodes.py tests/graph/test_intent_router_graph.py -v`
Expected: All PASS

- [ ] **Step 9: Commit**

```bash
git add src/modules/01_intent_router/ tests/unit/test_intent_nodes.py tests/graph/test_intent_router_graph.py
git commit -m "feat: 模块1 意图路由 — StateGraph + 条件路由"
```

---

### Task 5: Module 2 — Lead Qualifier

**Files:**
- Create: `src/modules/02_lead_qualifier/__init__.py`
- Create: `src/modules/02_lead_qualifier/state.py`
- Create: `src/modules/02_lead_qualifier/nodes.py`
- Create: `src/modules/02_lead_qualifier/graph.py`
- Create: `src/modules/02_lead_qualifier/README.md`
- Create: `tests/unit/test_lead_nodes.py`
- Create: `tests/graph/test_lead_qualifier_graph.py`

**Interfaces:**
- Consumes: `get_llm()` from Task 3, prompts from Task 2, `get_checkpointer()` from Task 3
- Produces: `LeadQualifierState`, `build_lead_qualifier_graph()`, runnable StateGraph with interrupt + checkpoint

- [ ] **Step 1: Write state.py**

```python
"""线索评级模块 State 定义"""
from typing import Annotated, Optional, TypedDict, operator


class LeadQualifierState(TypedDict):
    """线索评级状态

    LangGraph 知识点:
    - Annotated[list, operator.add] 用于列表追加（reducer）
    - 循环图中的状态需要在每轮迭代中累积
    """
    lead_info: dict                     # 线索基本信息（来源/公司/职位）
    questions_asked: Annotated[list[str], operator.add]    # 已提问列表（追加模式）
    answers_received: Annotated[list[str], operator.add]   # 已回答列表（追加模式）
    score: float                        # 当前评分 (0-100)
    score_history: Annotated[list[float], operator.add]    # 每轮评分记录（追加模式）
    qualification: str                  # 最终评级: hot/warm/cold
    human_decision: str                 # 人工审核结果: approve/reject/needs_info
    error: Optional[str]
    error_node: Optional[str]
```

- [ ] **Step 2: Write nodes.py**

```python
"""线索评级模块节点函数

LangGraph 知识点:
- 循环图：节点之间形成循环（ask_question → evaluate → ask_question）
- interrupt()：在节点内暂停执行，等待外部输入
- Command(resume=...)：恢复执行时传入人工决策
"""
import json
import logging

from langchain_core.messages import HumanMessage
from langgraph.types import interrupt

from src.config.prompts import LEAD_QUALIFIER_QUESTION_PROMPT, LEAD_QUALIFIER_EVALUATE_PROMPT
from src.config.settings import settings
from src.core.llm import get_llm, safe_llm_call

logger = logging.getLogger("langgraph-scrm.lead_qualifier")


@safe_llm_call
def ask_question_node(state: dict) -> dict:
    """提问节点 — 根据线索信息提出评估问题

    LangGraph 知识点: 循环图中的节点，每次迭代生成新问题。
    questions_asked 使用 Annotated[list, operator.add] reducer，
    返回 {"questions_asked": [new_question]} 会追加到现有列表。
    """
    llm = get_llm()
    qa_pairs = "\n".join(
        f"Q: {q}\nA: {a}"
        for q, a in zip(state.get("questions_asked", []), state.get("answers_received", []))
    )
    prompt = LEAD_QUALIFIER_QUESTION_PROMPT.format(
        lead_info=json.dumps(state["lead_info"], ensure_ascii=False),
        questions_asked=json.dumps(state.get("questions_asked", []), ensure_ascii=False),
        answers_received=json.dumps(state.get("answers_received", []), ensure_ascii=False),
        score=state.get("score", 0),
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"questions_asked": [response.content]}


@safe_llm_call
def evaluate_node(state: dict) -> dict:
    """评估节点 — 根据问答给出评分和评级

    LangGraph 知识点: 循环终止条件判断。
    """
    llm = get_llm()
    qa_pairs = "\n".join(
        f"Q: {q}\nA: {a}"
        for q, a in zip(state.get("questions_asked", []), state.get("answers_received", []))
    )
    prompt = LEAD_QUALIFIER_EVALUATE_PROMPT.format(
        lead_info=json.dumps(state["lead_info"], ensure_ascii=False),
        qa_pairs=qa_pairs,
    )
    response = llm.invoke([HumanMessage(content=prompt)])

    try:
        result = json.loads(response.content)
        score = float(result.get("score", 0))
        qualification = result.get("qualification", "cold")
    except (json.JSONDecodeError, KeyError, ValueError):
        score = 0.0
        qualification = "cold"
        logger.warning(f"评估结果 JSON 解析失败: {response.content}")

    return {
        "score": score,
        "score_history": [score],
        "qualification": qualification,
    }


def human_review_node(state: dict) -> dict:
    """人工审核节点 — 使用 interrupt() 暂停等待人工决策

    LangGraph 知识点:
    - interrupt() 在节点内部调用，暂停图执行
    - interrupt() 的返回值是 resume 时传入的值
    - 使用 Command(resume={"human_decision": "approve"}) 恢复
    """
    decision = interrupt({
        "question": "请审核线索评级结果",
        "score": state.get("score", 0),
        "qualification": state.get("qualification", "cold"),
        "lead_info": state.get("lead_info", {}),
    })

    # decision 是 Command(resume=...) 传入的值
    human_decision = decision.get("human_decision", "approve")
    return {"human_decision": human_decision}


def finalize_node(state: dict) -> dict:
    """最终节点 — 根据人工决策确定最终评级"""
    if state.get("human_decision") == "reject":
        return {"qualification": "cold", "score": 0.0}
    elif state.get("human_decision") == "needs_info":
        return {"qualification": "warm"}
    # approve — 保持原有评分和评级
    return {}


# ── 循环终止条件 ──
def should_continue_evaluation(state: dict) -> str:
    """循环路由函数 — 判断是否继续提问评估

    LangGraph 知识点: add_conditional_edges 中的循环终止条件。
    当评分达标或提问次数达到上限时，退出循环进入评分汇总。
    """
    score = state.get("score", 0)
    questions_count = len(state.get("questions_asked", []))
    min_score = settings.LEAD_QUALIFIER_MIN_SCORE
    max_questions = settings.LEAD_QUALIFIER_MAX_QUESTIONS

    if score >= min_score or questions_count >= max_questions:
        return "score_lead"

    # 评分未达标且提问次数未达上限，继续提问
    return "ask_question"
```

- [ ] **Step 3: Write graph.py**

```python
"""线索评级模块 StateGraph 定义

LangGraph 知识点:
- 循环图：evaluate → ask_question → evaluate（形成循环）
- interrupt()：human_review_node 内部调用 interrupt() 暂停
- Checkpoint：compile(checkpointer=...) 支持暂停恢复
- Command(resume=...)：恢复时传入人工决策
"""
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from src.modules.02_lead_qualifier.state import LeadQualifierState
from src.modules.02_lead_qualifier.nodes import (
    ask_question_node,
    evaluate_node,
    human_review_node,
    finalize_node,
    should_continue_evaluation,
)
from src.core.checkpoint import get_checkpointer


def build_lead_qualifier_graph(checkpointer=None):
    """构建线索评级 StateGraph

    图结构:
    START → ask_question → evaluate → (条件路由)
      ├── continue → ask_question (循环)
      └── score_lead → human_review (interrupt) → finalize → END
    """
    graph = StateGraph(LeadQualifierState)

    # 添加节点
    graph.add_node("ask_question", ask_question_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("finalize", finalize_node)

    # 添加边
    graph.add_edge(START, "ask_question")
    graph.add_edge("ask_question", "evaluate")

    # 循环条件路由：evaluate → ask_question 或 score_lead
    graph.add_conditional_edges(
        "evaluate",
        should_continue_evaluation,
        {"ask_question": "ask_question", "score_lead": "human_review"},
    )

    graph.add_edge("human_review", "finalize")
    graph.add_edge("finalize", END)

    # 编译图（需要 checkpointer 支持 interrupt 暂停恢复）
    cp = checkpointer or get_checkpointer(store="memory")
    return graph.compile(checkpointer=cp)


if __name__ == "__main__":
    from langgraph.types import Command

    app = build_lead_qualifier_graph()
    thread_id = "lead-test-001"
    config = {"configurable": {"thread_id": thread_id}}

    # 第一轮运行 — 会触发 interrupt
    initial_state = {
        "lead_info": {"source": "官网", "company": "测试公司", "position": "CTO"},
        "questions_asked": [],
        "answers_received": [],
        "score": 0.0,
        "score_history": [],
        "qualification": "",
        "human_decision": "",
        "error": None,
        "error_node": None,
    }

    # 注意：实际运行需要真实 LLM API Key
    # result = app.invoke(initial_state, config=config)
    # print(f"中断 — 等待审核: {result}")
    # 恢复执行
    # resumed = app.invoke(Command(resume={"human_decision": "approve"}), config=config)
    # print(f"最终评级: {resumed['qualification']}")
```

- [ ] **Step 4: Write __init__.py, README.md, tests**

```python
# src/modules/02_lead_qualifier/__init__.py
"""线索评级模块 — 多轮对话评估 + 人工审核"""
from src.modules.02_lead_qualifier.graph import build_lead_qualifier_graph
__all__ = ["build_lead_qualifier_graph"]
```

```markdown
# 模块 2：线索评级（Lead Qualifier）

> 🎯 **LangGraph 知识点**：循环图、`interrupt()` + `Command(resume=...)`、Checkpoint、人机协同

## 业务场景

销售人员接到新线索 → AI 多轮对话评估 → 自动评分 → 人工审核确认

## LangGraph 核心知识点

### 循环图
节点之间形成循环：`ask_question → evaluate → ask_question`。
使用 `add_conditional_edges` + 条件函数控制循环终止。

### interrupt() + Command(resume=...)
- `interrupt()` 在节点内部调用，暂停图执行，返回 payload 给调用方
- 恢复时使用 `Command(resume={"human_decision": "approve"})`
- 需要 `checkpointer` 支持暂停/恢复

### Annotated reducer
- `Annotated[list[str], operator.add]` 使列表字段自动追加而非覆盖
- 适合循环中累积的列表（如 questions_asked）

## 运行

```bash
python -m src.modules.02_lead_qualifier.graph
```
```

```python
# tests/unit/test_lead_nodes.py
"""线索评级节点单元测试"""
import json
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

from src.modules.02_lead_qualifier.nodes import (
    ask_question_node,
    evaluate_node,
    should_continue_evaluation,
)


@patch("src.modules.02_lead_qualifier.nodes.get_llm")
def test_ask_question_node_appends_question(mock_get_llm):
    """ask_question_node 应追加新问题"""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="贵公司目前使用什么 CRM 系统？")
    mock_get_llm.return_value = mock_llm

    state = {
        "lead_info": {"source": "官网", "company": "测试公司"},
        "questions_asked": ["你们团队规模多大？"],
        "answers_received": ["约50人"],
        "score": 30.0,
        "score_history": [30.0],
        "qualification": "warm",
        "human_decision": "",
        "error": None,
        "error_node": None,
    }
    result = ask_question_node(state)
    assert "questions_asked" in result
    assert len(result["questions_asked"]) == 1


@patch("src.modules.02_lead_qualifier.nodes.get_llm")
def test_evaluate_node_returns_score(mock_get_llm):
    """evaluate_node 应返回评分和评级"""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(
        content=json.dumps({"score": 75.0, "qualification": "hot"})
    )
    mock_get_llm.return_value = mock_llm

    state = {
        "lead_info": {"source": "官网"},
        "questions_asked": ["Q1"],
        "answers_received": ["A1"],
        "score": 0.0,
        "score_history": [],
        "qualification": "",
        "human_decision": "",
        "error": None,
        "error_node": None,
    }
    result = evaluate_node(state)
    assert result["score"] == 75.0
    assert result["qualification"] == "hot"


def test_should_continue_evaluation_loop():
    """评分未达标时应继续循环"""
    state = {"score": 30.0, "questions_asked": ["Q1", "Q2"]}
    assert should_continue_evaluation(state) == "ask_question"


def test_should_continue_evaluation_exit():
    """评分达标时应退出循环"""
    state = {"score": 75.0, "questions_asked": ["Q1", "Q2"]}
    assert should_continue_evaluation(state) == "score_lead"


def test_should_continue_max_questions():
    """提问次数达到上限时应退出循环"""
    state = {"score": 30.0, "questions_asked": ["Q1", "Q2", "Q3", "Q4", "Q5"]}
    assert should_continue_evaluation(state) == "score_lead"
```

```python
# tests/graph/test_lead_qualifier_graph.py
"""线索评级图集成测试"""
import json
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from src.modules.02_lead_qualifier.graph import build_lead_qualifier_graph


@patch("src.modules.02_lead_qualifier.nodes.get_llm")
def test_lead_qualifier_full_flow_with_interrupt(mock_get_llm):
    """线索评级完整流程 — 包含 interrupt 和 resume"""
    mock_llm = MagicMock()
    # 第一轮 ask_question
    mock_llm.invoke.side_effect = [
        AIMessage(content="贵公司目前使用什么 CRM 系统？"),  # ask_question
        AIMessage(content=json.dumps({"score": 75.0, "qualification": "hot"})),  # evaluate
    ]
    mock_get_llm.return_value = mock_llm

    checkpointer = InMemorySaver()
    graph = build_lead_qualifier_graph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "test-lead-001"}}

    initial_state = {
        "lead_info": {"source": "官网", "company": "测试公司", "position": "CTO"},
        "questions_asked": [],
        "answers_received": [],
        "score": 0.0,
        "score_history": [],
        "qualification": "",
        "human_decision": "",
        "error": None,
        "error_node": None,
    }

    # 第一轮 — 应在 human_review 处 interrupt
    result = graph.invoke(initial_state, config=config)

    # 评分已达标（75.0），应到达 human_review 并 interrupt
    assert result.get("score") == 75.0 or result.get("qualification") != ""

    # 恢复执行 — 人工审核通过
    resumed = graph.invoke(Command(resume={"human_decision": "approve"}), config=config)
    assert resumed["qualification"] == "hot"
```

- [ ] **Step 5: Run all module 2 tests**

Run: `pytest tests/unit/test_lead_nodes.py tests/graph/test_lead_qualifier_graph.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/modules/02_lead_qualifier/ tests/unit/test_lead_nodes.py tests/graph/test_lead_qualifier_graph.py
git commit -m "feat: 模块2 线索评级 — 循环图 + interrupt + Checkpoint + 人机协同"
```

---

### Task 6: Module 3 — Knowledge QA

**Files:**
- Create: `src/modules/03_knowledge_qa/__init__.py`
- Create: `src/modules/03_knowledge_qa/state.py`
- Create: `src/modules/03_knowledge_qa/nodes.py`
- Create: `src/modules/03_knowledge_qa/graph.py`
- Create: `src/modules/03_knowledge_qa/tools.py`
- Create: `src/modules/03_knowledge_qa/README.md`
- Create: `src/data/sample_docs/` (5-10 FAQ/产品文档)
- Create: `tests/unit/test_knowledge_nodes.py`
- Create: `tests/graph/test_knowledge_qa_graph.py`

**Interfaces:**
- Consumes: `get_llm()`, `COMMON_TOOLS`, Chroma vector store, prompts
- Produces: `KnowledgeQAState`, `build_knowledge_qa_graph()`, Corrective RAG pipeline with document grading and self-verification

- [ ] **Step 1: Write state.py**

```python
"""知识库问答模块 State 定义"""
from typing import Annotated, Optional, TypedDict, operator


class KnowledgeQAState(TypedDict):
    """知识库问答状态 — Corrective RAG 模式

    LangGraph 知识点:
    - RAG 流程的 state 需要追踪文档、答案和验证结果
    - documents 和 web_results 使用 reducer 追加模式
    """
    question: str                           # 用户问题
    documents: Annotated[list[str], operator.add]   # 检索到的文档片段
    web_results: Annotated[list[str], operator.add] # 补充网页搜索结果
    answer: str                             # 生成的回答
    citations: Annotated[list[dict], operator.add]  # 引用来源
    verification: str                       # 自检结果: passed/failed
    retries: int                            # 生成重试次数
    error: Optional[str]
    error_node: Optional[str]
```

- [ ] **Step 2: Write tools.py with retriever tool**

```python
"""知识库问答模块工具 — 向量检索工具"""
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from src.config.settings import settings


def get_retriever_tool(collection_name: str = "scrm_knowledge"):
    """创建向量检索工具

    Args:
        collection_name: Chroma 集合名

    Returns:
        LangChain tool 对象
    """
    embeddings = OpenAIEmbeddings(
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL,
    )
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=settings.CHROMA_PERSIST_DIR,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    @tool
    def retriever(query: str) -> str:
        """知识库检索 — 搜索 SCRM 产品文档和 FAQ

        Args:
            query: 搜索关键词

        Returns:
            相关文档片段
        """
        docs = retriever.invoke(query)
        return "\n\n".join(doc.page_content for doc in docs)

    return retriever
```

- [ ] **Step 3: Write nodes.py**

```python
"""知识库问答模块节点函数 — Corrective RAG 模式

LangGraph 知识点:
- ToolNode：LangGraph 内置的工具执行节点
- RAG 流程：retrieve → grade → generate → verify → (循环重试)
"""
import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import ToolNode

from src.config.prompts import (
    KNOWLEDGE_QA_GENERATE_PROMPT,
    KNOWLEDGE_QA_GRADE_PROMPT,
    KNOWLEDGE_QA_VERIFY_PROMPT,
)
from src.core.llm import get_llm, safe_llm_call
from src.modules.03_knowledge_qa.tools import get_retriever_tool

logger = logging.getLogger("langgraph-scrm.knowledge_qa")


@safe_llm_call
def retrieve_node(state: dict) -> dict:
    """检索节点 — 向量检索相关文档"""
    retriever_tool = get_retriever_tool()
    docs = retriever_tool.invoke(state["question"])
    return {"documents": [docs]}


@safe_llm_call
def grade_docs_node(state: dict) -> dict:
    """文档评估节点 — 评估检索文档相关性"""
    llm = get_llm()
    prompt = KNOWLEDGE_QA_GRADE_PROMPT.format(
        question=state["question"],
        documents="\n".join(state.get("documents", [])),
    )
    response = llm.invoke([HumanMessage(content=prompt)])

    if "irrelevant" in response.content.lower():
        logger.info("文档相关性不足，将补充网页搜索")
        return {"verification": "docs_irrelevant"}
    return {"verification": "docs_relevant"}


@safe_llm_call
def web_search_node(state: dict) -> dict:
    """网页搜索节点 — 文档不足时的补充检索"""
    from src.core.tools import web_search
    result = web_search.invoke(state["question"])
    return {"web_results": [result]}


@safe_llm_call
def generate_node(state: dict) -> dict:
    """回答生成节点 — 基于文档和搜索结果生成回答"""
    llm = get_llm()
    all_docs = "\n".join(state.get("documents", []) + state.get("web_results", []))
    prompt = KNOWLEDGE_QA_GENERATE_PROMPT.format(
        question=state["question"],
        documents=all_docs,
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"answer": response.content, "retries": state.get("retries", 0) + 1}


@safe_llm_call
def verify_node(state: dict) -> dict:
    """自检节点 — 检查回答是否有幻觉"""
    llm = get_llm()
    all_docs = "\n".join(state.get("documents", []) + state.get("web_results", []))
    prompt = KNOWLEDGE_QA_VERIFY_PROMPT.format(
        question=state["question"],
        answer=state["answer"],
        documents=all_docs,
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    verification = "passed" if "passed" in response.content.lower() else "failed"
    return {"verification": verification}


def respond_node(state: dict) -> dict:
    """最终回复节点"""
    return {}


# ── 条件路由函数 ──
def route_after_grade(state: dict) -> str:
    """文档评估后路由"""
    if state.get("verification") == "docs_irrelevant":
        return "web_search"
    return "generate"


def route_after_verify(state: dict) -> str:
    """自检后路由 — 重试上限 3 次"""
    if state.get("verification") == "failed" and state.get("retries", 0) < 3:
        return "generate"
    return "respond"
```

- [ ] **Step 4: Write graph.py**

```python
"""知识库问答模块 StateGraph — Corrective RAG 模式"""
from langgraph.graph import StateGraph, START, END

from src.modules.03_knowledge_qa.state import KnowledgeQAState
from src.modules.03_knowledge_qa.nodes import (
    retrieve_node,
    grade_docs_node,
    web_search_node,
    generate_node,
    verify_node,
    respond_node,
    route_after_grade,
    route_after_verify,
)


def build_knowledge_qa_graph():
    """构建知识库问答 StateGraph — Corrective RAG

    图结构:
    START → retrieve → grade_docs → (条件路由)
      ├── docs_irrelevant → web_search → retrieve (补充)
      └── docs_relevant → generate → verify → (条件路由)
          ├── failed → generate (重试)
          └── passed → respond → END
    """
    graph = StateGraph(KnowledgeQAState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade_docs", grade_docs_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("generate", generate_node)
    graph.add_node("verify", verify_node)
    graph.add_node("respond", respond_node)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade_docs")
    graph.add_conditional_edges("grade_docs", route_after_grade)
    graph.add_edge("web_search", "retrieve")
    graph.add_edge("grade_docs", "generate")
    graph.add_edge("generate", "verify")
    graph.add_conditional_edges("verify", route_after_verify)
    graph.add_edge("respond", END)

    return graph.compile()


if __name__ == "__main__":
    app = build_knowledge_qa_graph()
    result = app.invoke({
        "question": "SCRM 系统支持哪些微信功能？",
        "documents": [],
        "web_results": [],
        "answer": "",
        "citations": [],
        "verification": "",
        "retries": 0,
        "error": None,
        "error_node": None,
    })
    print(f"回答: {result['answer']}")
```

- [ ] **Step 5: Create sample documents in src/data/sample_docs/**

Create 5 sample FAQ documents (markdown files) covering SCRM topics:
- `01_product_overview.md` — SCRM 产品功能概述
- `02_wechat_features.md` — 微信功能说明
- `03_customer_management.md` — 客户管理功能
- `04_after_sale_policy.md` — 售后政策
- `05_order_tracking.md` — 订单追踪说明

Each file: ~200-300 words of realistic SCRM FAQ content in Chinese.

- [ ] **Step 6: Write __init__.py, README.md, tests (same pattern as Modules 1-2)**

- [ ] **Step 7: Run all module 3 tests**

Run: `pytest tests/unit/test_knowledge_nodes.py tests/graph/test_knowledge_qa_graph.py -v`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add src/modules/03_knowledge_qa/ src/data/ tests/
git commit -m "feat: 模块3 知识库问答 — Corrective RAG + 工具调用 + Embedding"
```

---

### Task 7: Module 4 — Multi-Agent Service

**Files:**
- Create: `src/modules/04_multi_agent/__init__.py`
- Create: `src/modules/04_multi_agent/state.py`
- Create: `src/modules/04_multi_agent/nodes.py`
- Create: `src/modules/04_multi_agent/graph.py`
- Create: `src/modules/04_multi_agent/README.md`
- Create: `tests/unit/test_multi_agent_nodes.py`
- Create: `tests/graph/test_multi_agent_graph.py`

**Interfaces:**
- Consumes: `get_llm()`, prompts, checkpointer
- Produces: `MultiAgentState`, `build_multi_agent_graph()`, Supervisor + 3 specialist agents + quality check loop

- [ ] **Step 1: Write state.py**

```python
"""多Agent客服模块 State 定义"""
from typing import Annotated, Optional, TypedDict, operator


class MultiAgentState(TypedDict):
    """多Agent客服状态 — Supervisor 模式

    LangGraph 知识点:
    - agent_responses 使用 reducer dict 合并各 Agent 回答
    - assigned_agents 使用 reducer list 追加分派记录
    """
    customer_question: str                              # 客户问题
    assigned_agents: Annotated[list[str], operator.add] # supervisor 分派的 Agent 列表
    agent_responses: dict                               # {agent_name: response}
    final_answer: str                                   # 合成后的最终回答
    quality_score: float                                # 质量评分 (0-10)
    feedback: str                                       # 质量检查反馈
    iteration: int                                      # 重试轮次
    error: Optional[str]
    error_node: Optional[str]
```

- [ ] **Step 2: Write nodes.py**

```python
"""多Agent客服模块节点函数 — Supervisor 模式

LangGraph 知识点:
- Supervisor 节点使用 LLM 决定分派哪些 Agent
- 每个 Agent 是独立节点（模拟 subgraph）
- 质量检查不合格时循环回到 Supervisor
"""
import json
import logging

from langchain_core.messages import HumanMessage

from src.config.prompts import (
    MULTI_AGENT_SUPERVISOR_PROMPT,
    MULTI_AGENT_PRODUCT_PROMPT,
    MULTI_AGENT_POLICY_PROMPT,
    MULTI_AGENT_ORDER_PROMPT,
    MULTI_AGENT_SYNTHESIZE_PROMPT,
    MULTI_AGENT_QUALITY_PROMPT,
)
from src.core.llm import get_llm, safe_llm_call

logger = logging.getLogger("langgraph-scrm.multi_agent")


@safe_llm_call
def supervisor_node(state: dict) -> dict:
    """Supervisor 调度节点 — 使用 LLM 决定分派哪些 Agent

    LangGraph 知识点: Supervisor 模式的核心 — LLM 动态决定路由，
    而非硬编码条件分支。
    """
    llm = get_llm()
    prompt = MULTI_AGENT_SUPERVISOR_PROMPT.format(
        customer_question=state["customer_question"]
    )
    response = llm.invoke([HumanMessage(content=prompt)])

    try:
        result = json.loads(response.content)
        agents = result.get("assigned_agents", ["product_expert"])
    except (json.JSONDecodeError, KeyError):
        agents = ["product_expert"]
        logger.warning(f"Supervisor JSON 解析失败: {response.content}")

    return {"assigned_agents": agents}


@safe_llm_call
def product_expert_node(state: dict) -> dict:
    """产品专家 Agent 节点"""
    llm = get_llm()
    prompt = MULTI_AGENT_PRODUCT_PROMPT.format(
        customer_question=state["customer_question"]
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    existing = state.get("agent_responses", {})
    return {"agent_responses": {**existing, "product_expert": response.content}}


@safe_llm_call
def policy_expert_node(state: dict) -> dict:
    """政策专家 Agent 节点"""
    llm = get_llm()
    prompt = MULTI_AGENT_POLICY_PROMPT.format(
        customer_question=state["customer_question"]
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    existing = state.get("agent_responses", {})
    return {"agent_responses": {**existing, "policy_expert": response.content}}


@safe_llm_call
def order_handler_node(state: dict) -> dict:
    """订单处理 Agent 节点"""
    llm = get_llm()
    prompt = MULTI_AGENT_ORDER_PROMPT.format(
        customer_question=state["customer_question"]
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    existing = state.get("agent_responses", {})
    return {"agent_responses": {**existing, "order_handler": response.content}}


@safe_llm_call
def synthesize_node(state: dict) -> dict:
    """合成节点 — 将各 Agent 回答合成为最终回答"""
    llm = get_llm()
    responses_str = json.dumps(state.get("agent_responses", {}), ensure_ascii=False)
    prompt = MULTI_AGENT_SYNTHESIZE_PROMPT.format(
        customer_question=state["customer_question"],
        agent_responses=responses_str,
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"final_answer": response.content}


@safe_llm_call
def quality_check_node(state: dict) -> dict:
    """质量检查节点 — 评估合成回答质量"""
    llm = get_llm()
    prompt = MULTI_AGENT_QUALITY_PROMPT.format(
        customer_question=state["customer_question"],
        final_answer=state.get("final_answer", ""),
    )
    response = llm.invoke([HumanMessage(content=prompt)])

    try:
        result = json.loads(response.content)
        score = float(result.get("score", 0))
        feedback = result.get("feedback", "")
    except (json.JSONDecodeError, KeyError, ValueError):
        score = 5.0
        feedback = "质量检查 JSON 解析失败"

    return {"quality_score": score, "feedback": feedback, "iteration": state.get("iteration", 0) + 1}


# ── 条件路由函数 ──
def route_from_supervisor(state: dict) -> list[str]:
    """Supervisor 路由函数 — 根据分派的 Agent 列表确定后续节点

    LangGraph 知识点: 多路条件路由 — 返回节点名列表，
    图会依次执行这些节点（通过 Send 或 conditional edges）。
    简化实现：固定执行所有 3 个 Agent，Supervisor 决定哪些参与回答。
    """
    agents = state.get("assigned_agents", [])
    # 确保至少有 1 个 Agent
    if not agents:
        agents = ["product_expert"]
    return agents[0] if len(agents) == 1 else "run_all_agents"


def route_after_quality(state: dict) -> str:
    """质量检查后路由 — 不合格时回到 Supervisor"""
    if state.get("quality_score", 0) < 7.0 and state.get("iteration", 0) < 3:
        return "supervisor"
    return "respond"


def respond_node(state: dict) -> dict:
    """最终回复节点"""
    return {}
```

- [ ] **Step 3: Write graph.py**

```python
"""多Agent客服模块 StateGraph — Supervisor 模式"""
from langgraph.graph import StateGraph, START, END

from src.modules.04_multi_agent.state import MultiAgentState
from src.modules.04_multi_agent.nodes import (
    supervisor_node,
    product_expert_node,
    policy_expert_node,
    order_handler_node,
    synthesize_node,
    quality_check_node,
    respond_node,
    route_after_quality,
)


def build_multi_agent_graph():
    """构建多Agent客服 StateGraph

    图结构:
    START → supervisor → (run_all_agents) → synthesize → quality_check → (条件路由)
      ├── quality < 7 → supervisor (循环重试)
      └── quality >= 7 → respond → END

    Agent 执行策略：
    简化实现中，supervisor 分派后执行所有 3 个 Agent 节点，
    然后进入 synthesize 合成。
    """
    graph = StateGraph(MultiAgentState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("product_expert", product_expert_node)
    graph.add_node("policy_expert", policy_expert_node)
    graph.add_node("order_handler", order_handler_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("quality_check", quality_check_node)
    graph.add_node("respond", respond_node)

    graph.add_edge(START, "supervisor")

    # Supervisor → 所有 3 个 Agent（简化：全部执行）
    graph.add_edge("supervisor", "product_expert")
    graph.add_edge("supervisor", "policy_expert")
    graph.add_edge("supervisor", "order_handler")

    # 3 个 Agent → synthesize（等待全部完成后合成）
    # LangGraph fan-in：所有 Agent 边指向 synthesize
    graph.add_edge("product_expert", "synthesize")
    graph.add_edge("policy_expert", "synthesize")
    graph.add_edge("order_handler", "synthesize")

    graph.add_edge("synthesize", "quality_check")
    graph.add_conditional_edges("quality_check", route_after_quality)
    graph.add_edge("supervisor", "synthesize")
    graph.add_edge("respond", END)

    return graph.compile()


if __name__ == "__main__":
    app = build_multi_agent_graph()
    result = app.invoke({
        "customer_question": "我的订单迟迟没有发货，而且产品规格和描述不一致，想了解退换货政策",
        "assigned_agents": [],
        "agent_responses": {},
        "final_answer": "",
        "quality_score": 0.0,
        "feedback": "",
        "iteration": 0,
        "error": None,
        "error_node": None,
    })
    print(f"最终回答: {result['final_answer']}")
```

- [ ] **Step 4: Write __init__.py, README.md, tests (same pattern as prior modules)**

- [ ] **Step 5: Run all module 4 tests**

Run: `pytest tests/unit/test_multi_agent_nodes.py tests/graph/test_multi_agent_graph.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/modules/04_multi_agent/ tests/
git commit -m "feat: 模块4 多Agent客服 — Supervisor 模式 + fan-in/fan-out"
```

---

### Task 8: Module 5 — After-Sale Workflow

**Files:**
- Create: `src/modules/05_after_sale/__init__.py`
- Create: `src/modules/05_after_sale/state.py`
- Create: `src/modules/05_after_sale/nodes.py`
- Create: `src/modules/05_after_sale/graph.py`
- Create: `src/modules/05_after_sale/README.md`
- Create: `tests/unit/test_after_sale_nodes.py`
- Create: `tests/graph/test_after_sale_graph.py`

**Interfaces:**
- Consumes: `get_llm()`, `get_checkpointer()`, prompts
- Produces: `AfterSaleState`, `build_after_sale_graph()`, long workflow with 2 interrupt approval gates

- [ ] **Step 1: Write state.py**

```python
"""售后工单模块 State 定义"""
from typing import Optional, TypedDict


class AfterSaleState(TypedDict):
    """售后工单状态 — 长流程 + 双审批节点"""
    ticket_id: str              # 工单 ID
    customer_request: str       # 客户诉求
    issue_type: str             # 问题分类: refund/exchange/repair/complaint
    severity: str               # 严重度: low/medium/high/critical
    approval_status: str        # 审批状态: pending/approved/rejected/needs_info
    approver_comment: str       # 审批人意见
    resolution: str             # 处理方案
    customer_feedback: str      # 客户反馈
    status: str                 # 工单状态: created/analyzing/approved/executing/verifying/closed
    error: Optional[str]
    error_node: Optional[str]
```

- [ ] **Step 2: Write nodes.py**

```python
"""售后工单模块节点函数 — 长流程 + 双审批"""
import json
import logging
import uuid

from langchain_core.messages import HumanMessage
from langgraph.types import interrupt

from src.config.prompts import AFTER_SALE_ANALYZE_PROMPT, AFTER_SALE_EXECUTE_PROMPT
from src.core.llm import get_llm, safe_llm_call

logger = logging.getLogger("langgraph-scrm.after_sale")


def create_ticket_node(state: dict) -> dict:
    """创建工单节点 — 生成工单 ID，设置初始状态"""
    return {
        "ticket_id": str(uuid.uuid4())[:8],
        "status": "created",
        "approval_status": "pending",
    }


@safe_llm_call
def analyze_node(state: dict) -> dict:
    """分析节点 — AI 分析问题类型和严重度"""
    llm = get_llm()
    prompt = AFTER_SALE_ANALYZE_PROMPT.format(
        customer_request=state["customer_request"]
    )
    response = llm.invoke([HumanMessage(content=prompt)])

    try:
        result = json.loads(response.content)
        issue_type = result.get("issue_type", "complaint")
        severity = result.get("severity", "medium")
    except (json.JSONDecodeError, KeyError):
        issue_type = "complaint"
        severity = "medium"

    return {"issue_type": issue_type, "severity": severity, "status": "analyzing"}


def approve_node(state: dict) -> dict:
    """主管审批节点 — interrupt() 暂停等待人工决策"""
    decision = interrupt({
        "question": "请审批售后工单",
        "ticket_id": state.get("ticket_id", ""),
        "issue_type": state.get("issue_type", ""),
        "severity": state.get("severity", ""),
        "customer_request": state.get("customer_request", ""),
    })

    approval_status = decision.get("approval_status", "approved")
    approver_comment = decision.get("comment", "")

    return {"approval_status": approval_status, "approver_comment": approver_comment}


@safe_llm_call
def execute_node(state: dict) -> dict:
    """执行节点 — 根据审批结果执行处理方案"""
    llm = get_llm()
    prompt = AFTER_SALE_EXECUTE_PROMPT.format(
        ticket_id=state["ticket_id"],
        issue_type=state["issue_type"],
        severity=state["severity"],
        customer_request=state["customer_request"],
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"resolution": response.content, "status": "executing"}


def verify_node(state: dict) -> dict:
    """客户验证节点 — interrupt() 暂停等待客户反馈"""
    feedback = interrupt({
        "question": "请确认客户满意度",
        "ticket_id": state.get("ticket_id", ""),
        "resolution": state.get("resolution", ""),
    })

    return {"customer_feedback": feedback.get("feedback", "满意"), "status": "verifying"}


def close_node(state: dict) -> dict:
    """关闭工单节点"""
    return {"status": "closed"}


# ── 条件路由 ──
def route_after_approve(state: dict) -> str:
    """审批后路由"""
    if state.get("approval_status") == "approved":
        return "execute"
    elif state.get("approval_status") == "rejected":
        return "analyze"
    else:  # needs_info
        return "analyze"


def route_after_verify(state: dict) -> str:
    """验证后路由"""
    feedback = state.get("customer_feedback", "")
    if "不满意" in feedback or "不满" in feedback:
        return "execute"
    return "close"
```

- [ ] **Step 3: Write graph.py**

```python
"""售后工单模块 StateGraph — 长流程 + 双审批"""
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from src.modules.05_after_sale.state import AfterSaleState
from src.modules.05_after_sale.nodes import (
    create_ticket_node,
    analyze_node,
    approve_node,
    execute_node,
    verify_node,
    close_node,
    route_after_approve,
    route_after_verify,
)
from src.core.checkpoint import get_checkpointer


def build_after_sale_graph(checkpointer=None):
    """构建售后工单 StateGraph

    图结构:
    START → create_ticket → analyze → approve (interrupt) → (条件路由)
      ├── approved → execute → verify (interrupt) → (条件路由)
      │   ├── 满意 → close → END
      │   └── 不满意 → execute (重试)
      ├── rejected → analyze (重新分析)
      └── needs_info → analyze
    """
    graph = StateGraph(AfterSaleState)

    graph.add_node("create_ticket", create_ticket_node)
    graph.add_node("analyze", analyze_node)
    graph.add_node("approve", approve_node)
    graph.add_node("execute", execute_node)
    graph.add_node("verify", verify_node)
    graph.add_node("close", close_node)

    graph.add_edge(START, "create_ticket")
    graph.add_edge("create_ticket", "analyze")
    graph.add_edge("analyze", "approve")
    graph.add_conditional_edges("approve", route_after_approve)
    graph.add_edge("execute", "verify")
    graph.add_conditional_edges("verify", route_after_verify)
    graph.add_edge("close", END)

    cp = checkpointer or get_checkpointer(store="memory")
    return graph.compile(checkpointer=cp)


if __name__ == "__main__":
    from langgraph.types import Command
    app = build_after_sale_graph()
    config = {"configurable": {"thread_id": "after-sale-001"}}
    # 需要真实 LLM API Key 运行
```

- [ ] **Step 4: Write __init__.py, README.md, tests**

- [ ] **Step 5: Run all module 5 tests**

Run: `pytest tests/unit/test_after_sale_nodes.py tests/graph/test_after_sale_graph.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/modules/05_after_sale/ tests/
git commit -m "feat: 模块5 售后工单 — 长流程 + 双审批节点 + interrupt"
```

---

### Task 9: Module 6 — WeChat Risk Control

**Files:**
- Create: `src/modules/06_wechat_risk/__init__.py`
- Create: `src/modules/06_wechat_risk/state.py`
- Create: `src/modules/06_wechat_risk/nodes.py`
- Create: `src/modules/06_wechat_risk/graph.py`
- Create: `src/modules/06_wechat_risk/README.md`
- Create: `tests/unit/test_wechat_risk_nodes.py`
- Create: `tests/graph/test_wechat_risk_graph.py`

**Interfaces:**
- Consumes: `get_llm()`, `get_checkpointer()`, prompts
- Produces: `WeChatRiskState`, `build_wechat_risk_graph()`, four-way conditional branching with interrupt escalation

- [ ] **Step 1: Write state.py**

```python
"""微信风控模块 State 定义"""
from typing import Optional, TypedDict


class WeChatRiskState(TypedDict):
    """微信风控状态 — 四路条件分支 + interrupt 上报"""
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

- [ ] **Step 2: Write nodes.py**

```python
"""微信风控模块节点函数 — 四路条件分支 + interrupt 上报"""
import json
import logging
import uuid

from langchain_core.messages import HumanMessage
from langgraph.types import interrupt

from src.config.prompts import WECHAT_RISK_CLASSIFY_PROMPT, WECHAT_RISK_ASSESS_PROMPT
from src.core.llm import get_llm, safe_llm_call

logger = logging.getLogger("langgraph-scrm.wechat_risk")


def receive_message_node(state: dict) -> dict:
    """接收消息节点 — 设置初始状态"""
    return {"message_id": str(uuid.uuid4())[:8]}


@safe_llm_call
def classify_node(state: dict) -> dict:
    """消息分类节点 — 将消息分为 4 类"""
    llm = get_llm()
    prompt = WECHAT_RISK_CLASSIFY_PROMPT.format(
        content=state["content"], sender=state["sender"]
    )
    response = llm.invoke([HumanMessage(content=prompt)])

    try:
        result = json.loads(response.content)
        message_type = result.get("message_type", "normal")
    except (json.JSONDecodeError, KeyError):
        message_type = "normal"
        logger.warning(f"消息分类 JSON 解析失败: {response.content}")

    return {"message_type": message_type}


def allow_node(state: dict) -> dict:
    """放行节点 — 正常消息直接放行"""
    return {"action": "allow", "log_entry": {"action": "allow", "message_id": state.get("message_id")}}


def log_only_node(state: dict) -> dict:
    """记录节点 — 业务消息记录后放行"""
    return {"action": "log_only", "log_entry": {"action": "log_only", "message_id": state.get("message_id")}}


@safe_llm_call
def risk_assess_node(state: dict) -> dict:
    """风险评估节点 — 评估敏感/违规消息的风险"""
    llm = get_llm()
    prompt = WECHAT_RISK_ASSESS_PROMPT.format(
        content=state["content"], message_type=state.get("message_type", "sensitive")
    )
    response = llm.invoke([HumanMessage(content=prompt)])

    try:
        result = json.loads(response.content)
        risk_score = float(result.get("risk_score", 50))
        risk_category = result.get("risk_category", "other")
    except (json.JSONDecodeError, KeyError, ValueError):
        risk_score = 50.0
        risk_category = "other"

    return {"risk_score": risk_score, "risk_category": risk_category}


def escalate_node(state: dict) -> dict:
    """上报节点 — interrupt() 暂停等待主管决策"""
    decision = interrupt({
        "question": "高风险消息需主管审核",
        "message_id": state.get("message_id"),
        "content": state.get("content"),
        "risk_score": state.get("risk_score"),
        "risk_category": state.get("risk_category"),
    })
    return {"escalation_decision": decision.get("decision", "approve_block")}


def block_node(state: dict) -> dict:
    """拦截节点 — 拦截消息 + 通知发送者"""
    return {"action": "block", "log_entry": {"action": "block", "message_id": state.get("message_id")}}


def warn_node(state: dict) -> dict:
    """提醒节点 — 低风险消息提醒发送者"""
    return {"action": "warn", "log_entry": {"action": "warn", "message_id": state.get("message_id")}}


def log_node(state: dict) -> dict:
    """日志记录节点 — 写入风控审计日志"""
    return {}


# ── 条件路由 ──
def route_after_classify(state: dict) -> str:
    """分类后四路路由"""
    message_type = state.get("message_type", "normal")
    route_map = {
        "normal": "allow",
        "business": "log_only",
        "sensitive": "risk_assess",
        "violation": "risk_assess",
    }
    return route_map.get(message_type, "allow")


def route_after_risk_assess(state: dict) -> str:
    """风险评估后路由"""
    risk_score = state.get("risk_score", 0)
    if risk_score >= 80:
        return "escalate"
    return "warn"
```

- [ ] **Step 3: Write graph.py**

```python
"""微信风控模块 StateGraph — 四路条件分支"""
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from src.modules.06_wechat_risk.state import WeChatRiskState
from src.modules.06_wechat_risk.nodes import (
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
```

- [ ] **Step 4: Write __init__.py, README.md, tests**

- [ ] **Step 5: Run all module 6 tests**

Run: `pytest tests/unit/test_wechat_risk_nodes.py tests/graph/test_wechat_risk_graph.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/modules/06_wechat_risk/ tests/
git commit -m "feat: 模块6 微信风控 — 四路条件分支 + interrupt 上报"
```

---

### Task 10: FastAPI API Layer

**Files:**
- Create: `src/api/main.py`
- Create: `src/api/schemas.py`
- Create: `src/api/middleware.py`
- Create: `src/api/routes/intent_router.py`
- Create: `src/api/routes/lead_qualifier.py`
- Create: `src/api/routes/knowledge_qa.py`
- Create: `src/api/routes/multi_agent.py`
- Create: `src/api/routes/after_sale.py`
- Create: `src/api/routes/wechat_risk.py`
- Create: `tests/api/test_routes.py`

**Interfaces:**
- Consumes: All 6 module `build_xxx_graph()` functions, `get_checkpointer()`
- Produces: FastAPI app with 6 route groups, OpenAPI docs at `/docs`, running at `http://localhost:8000`

- [ ] **Step 1: Write src/api/schemas.py**

```python
"""API 请求/响应模型 — Pydantic 数据校验"""
from pydantic import BaseModel, Field
from typing import Optional


# ── 意图路由 ──
class IntentRouterRequest(BaseModel):
    """意图路由请求"""
    message: str = Field(..., description="客户消息", min_length=1)

class IntentRouterResponse(BaseModel):
    """意图路由响应"""
    intent: str
    confidence: float
    skill_group: str
    response: str
    error: Optional[str] = None


# ── 线索评级 ──
class LeadQualifierRequest(BaseModel):
    """线索评级请求"""
    lead_info: dict = Field(..., description="线索基本信息")

class LeadQualifierResumeRequest(BaseModel):
    """线索评级恢复请求（人工审核决策）"""
    thread_id: str = Field(..., description="Checkpoint thread ID")
    human_decision: str = Field(..., description="审核决策: approve/reject/needs_info")


# ── 知识库问答 ──
class KnowledgeQARequest(BaseModel):
    """知识库问答请求"""
    question: str = Field(..., description="用户问题", min_length=1)


# ── 多Agent客服 ──
class MultiAgentRequest(BaseModel):
    """多Agent客服请求"""
    customer_question: str = Field(..., description="客户问题", min_length=1)


# ── 售后工单 ──
class AfterSaleRequest(BaseModel):
    """售后工单请求"""
    customer_request: str = Field(..., description="客户诉求", min_length=1)

class AfterSaleResumeRequest(BaseModel):
    """售后工单恢复请求"""
    thread_id: str
    approval_status: str = Field(..., description="审批决策: approved/rejected/needs_info")
    comment: Optional[str] = None


# ── 微信风控 ──
class WeChatRiskRequest(BaseModel):
    """微信风控请求"""
    sender: str = Field(..., description="发送者")
    content: str = Field(..., description="消息内容", min_length=1)

class WeChatRiskResumeRequest(BaseModel):
    """微信风控恢复请求"""
    thread_id: str
    decision: str = Field(..., description="主管决策: approve_block/dismiss")
```

- [ ] **Step 2: Write src/api/middleware.py**

```python
"""API 中间件 — 请求日志 + 异常处理"""
import logging
import time

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("langgraph-scrm.api")


class LoggingMiddleware(BaseHTTPMiddleware):
    """请求日志中间件 — 记录每个 API 请求的方法、路径、耗时"""

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response: Response = await call_next(request)
        duration = time.time() - start_time
        logger.info(f"{request.method} {request.url.path} — {duration:.3f}s — {response.status_code}")
        return response
```

- [ ] **Step 3: Write src/api/routes/intent_router.py (reference pattern, other routes follow same structure)**

```python
"""意图路由 API 路由"""
from fastapi import APIRouter

from src.api.schemas import IntentRouterRequest, IntentRouterResponse
from src.modules.01_intent_router.graph import build_intent_router_graph

router = APIRouter(prefix="/intent-router", tags=["意图路由"])


@router.post("/", response_model=IntentRouterResponse)
async def classify_intent(request: IntentRouterRequest) -> IntentRouterResponse:
    """分类客户消息意图 — 意图路由模块入口"""
    graph = build_intent_router_graph()
    result = graph.invoke({
        "message": request.message,
        "intent": "",
        "confidence": 0.0,
        "skill_group": "",
        "response": "",
        "error": None,
        "error_node": None,
    })
    return IntentRouterResponse(
        intent=result["intent"],
        confidence=result["confidence"],
        skill_group=result["skill_group"],
        response=result["response"],
        error=result.get("error"),
    )
```

- [ ] **Step 4: Write remaining 5 route files (lead_qualifier, knowledge_qa, multi_agent, after_sale, wechat_risk — same pattern, each calls its module's build_xxx_graph)**

- [ ] **Step 5: Write src/api/main.py**

```python
"""FastAPI 应用入口 — 注册所有路由"""
from fastapi import FastAPI

from src.api.middleware import LoggingMiddleware
from src.api.routes import (
    intent_router,
    lead_qualifier,
    knowledge_qa,
    multi_agent,
    after_sale,
    wechat_risk,
)

app = FastAPI(
    title="LangGraph-SCRM",
    description="基于 LangGraph 的 SCRM 智能客服平台 API",
    version="0.1.0",
)

# 注册中间件
app.add_middleware(LoggingMiddleware)

# 注册路由
app.include_router(intent_router.router)
app.include_router(lead_qualifier.router)
app.include_router(knowledge_qa.router)
app.include_router(multi_agent.router)
app.include_router(after_sale.router)
app.include_router(wechat_risk.router)


@app.get("/")
async def root():
    """根路径 — 项目信息"""
    return {
        "name": "LangGraph-SCRM",
        "version": "0.1.0",
        "description": "基于 LangGraph 的 SCRM 智能客服平台",
        "modules": [
            "intent-router",
            "lead-qualifier",
            "knowledge-qa",
            "multi-agent",
            "after-sale",
            "wechat-risk",
        ],
        "docs": "/docs",
    }
```

- [ ] **Step 6: Write API test**

```python
# tests/api/test_routes.py
"""API 路由测试 — 使用 httpx AsyncClient"""
import pytest
from httpx import AsyncClient, ASGITransport
from src.api.main import app


@pytest.fixture
async def client():
    """异步 HTTP 测试客户端"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_root_endpoint(client):
    """根路径应返回项目信息"""
    response = await client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "LangGraph-SCRM"
    assert len(data["modules"]) == 6
```

- [ ] **Step 7: Run API tests**

Run: `pytest tests/api/test_routes.py -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add src/api/ tests/api/
git commit -m "feat: FastAPI 层 — 6 模块路由 + 中间件 + Pydantic schemas"
```

---

### Task 11: CLI Scripts + Sample Data

**Files:**
- Create: `scripts/run_module.py`
- Create: `scripts/run_all.py`
- Create: `scripts/seed_data.py`
- Create: `src/data/sample_docs/` (5 FAQ documents from Task 6)
- Create: `src/data/sample_messages/` (10 WeChat message samples)
- Create: `src/data/seed_db.py`

**Interfaces:**
- Consumes: All module `build_xxx_graph()` functions, `get_checkpointer()`, Chroma vectorstore
- Produces: CLI entry points for single-module and full-service running, seeded vector DB and SQLite DB

- [ ] **Step 1: Write scripts/run_module.py**

```python
"""单模块 CLI 运行器 — 独立运行任意模块进行学习"""
import argparse
import json
import sys

from src.modules.01_intent_router.graph import build_intent_router_graph
from src.modules.02_lead_qualifier.graph import build_lead_qualifier_graph
from src.modules.03_knowledge_qa.graph import build_knowledge_qa_graph
from src.modules.04_multi_agent.graph import build_multi_agent_graph
from src.modules.05_after_sale.graph import build_after_sale_graph
from src.modules.06_wechat_risk.graph import build_wechat_risk_graph


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


def main():
    parser = argparse.ArgumentParser(description="LangGraph-SCRM 单模块运行器")
    parser.add_argument("module", choices=MODULE_MAP.keys(), help="模块名")
    parser.add_argument("--input", help="JSON 格式的输入 state（可选）")
    args = parser.parse_args()

    name, builder = MODULE_MAP[args.module]
    print(f"▶ 运行模块: {name} ({args.module})")

    graph = builder()

    input_state = DEFAULT_INPUTS.get(args.module, {})
    if args.input:
        input_state = json.loads(args.input)

    result = graph.invoke(input_state)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write scripts/run_all.py**

```python
"""启动完整 API 服务"""
import uvicorn


def main():
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Write scripts/seed_data.py**

```python
"""数据初始化脚本 — 向量索引 + SQLite 数据库"""
import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from src.config.settings import settings


def seed_vector_store():
    """初始化 Chroma 向量存储 — 加载示例文档"""
    embeddings = OpenAIEmbeddings(
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL,
    )

    # 示例 FAQ 文档
    docs_dir = os.path.join(os.path.dirname(__file__), "..", "src", "data", "sample_docs")
    documents = []

    if os.path.exists(docs_dir):
        for filename in os.listdir(docs_dir):
            if filename.endswith(".md"):
                path = os.path.join(docs_dir, filename)
                content = open(path, encoding="utf-8").read()
                documents.append(Document(
                    page_content=content,
                    metadata={"source": filename, "type": "faq"},
                ))

    if not documents:
        # 如果没有文件，使用内置示例
        documents = [
            Document(page_content="SCRM 系统支持微信客户管理、朋友圈营销、社群运营等功能。", metadata={"source": "builtin", "type": "faq"}),
            Document(page_content="退换货政策：7天无理由退货，15天质量问题换货，需保留原始包装。", metadata={"source": "builtin", "type": "policy"}),
            Document(page_content="订单追踪功能支持物流信息实时查询，异常订单自动告警。", metadata={"source": "builtin", "type": "faq"}),
        ]

    Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="scrm_knowledge",
        persist_directory=settings.CHROMA_PERSIST_DIR,
    )
    print(f"✅ 向量存储初始化完成 — {len(documents)} 份文档")


def main():
    """运行所有数据初始化"""
    os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)
    seed_vector_store()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Create sample WeChat messages in src/data/sample_messages/**

```json
// src/data/sample_messages/normal_messages.json
[
  {"sender": "员工A", "content": "今天天气不错，周末打算去爬山"},
  {"sender": "员工B", "content": "中午一起吃饭吧"}
]
```

```json
// src/data/sample_messages/business_messages.json
[
  {"sender": "员工C", "content": "客户王先生约了明天下午2点见面"},
  {"sender": "员工D", "content": "这个月的销售报表已经发到邮箱了"}
]
```

```json
// src/data/sample_messages/sensitive_messages.json
[
  {"sender": "员工E", "content": "我把客户名单发给你了，注意保密"},
  {"sender": "员工F", "content": "内部价格调整通知：下月产品降价20%"}
]
```

```json
// src/data/sample_messages/violation_messages.json
[
  {"sender": "员工G", "content": "客户就是个傻X，不想理会他"},
  {"sender": "员工H", "content": "我帮你把公司的客户数据导出来了，方便你跳槽"}
]
```

- [ ] **Step 5: Write src/data/seed_db.py**

```python
"""数据库初始化 — 创建 SQLite 表结构"""
import sqlite3
from src.config.settings import settings


def init_db():
    """创建售后工单和风控日志表"""
    db_path = settings.DATABASE_URL.replace("sqlite:///./", "")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 售后工单表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS after_sale_tickets (
            ticket_id TEXT PRIMARY KEY,
            customer_request TEXT NOT NULL,
            issue_type TEXT,
            severity TEXT,
            approval_status TEXT,
            approver_comment TEXT,
            resolution TEXT,
            customer_feedback TEXT,
            status TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # 风控日志表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS wechat_risk_logs (
            message_id TEXT PRIMARY KEY,
            sender TEXT NOT NULL,
            content TEXT NOT NULL,
            message_type TEXT,
            risk_score REAL,
            risk_category TEXT,
            action TEXT,
            escalation_decision TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()
    print(f"✅ 数据库初始化完成 — {db_path}")


if __name__ == "__main__":
    init_db()
```

- [ ] **Step 6: Verify CLI works**

Run: `python scripts/run_module.py intent-router --input '{"message": "测试消息", "intent": "", "confidence": 0.0, "skill_group": "", "response": "", "error": null, "error_node": null}'`
Expected: Output JSON with classified intent (requires LLM API key)

- [ ] **Step 7: Commit**

```bash
git add scripts/ src/data/
git commit -m "feat: CLI 运行脚本 + 示例数据 + 数据库初始化"
```

---

### Task 12: Documentation + CI/CD

**Files:**
- Create: `README.md`
- Create: `docs/getting-started.md`
- Create: `docs/module-guide.md`
- Create: `docs/architecture.md`
- Create: `CONTRIBUTING.md`
- Create: `CODE_OF_CONDUCT.md`
- Create: `LICENSE`
- Create: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: All prior tasks
- Produces: Complete open-source documentation, CI pipeline, MIT license

- [ ] **Step 1: Write README.md (comprehensive, with badges, knowledge index, quick start)**

Key sections:
- Project title + description + badges (Python, LangGraph, FastAPI, License)
- 6 modules → LangGraph knowledge mapping table
- Quick start (install, configure, run)
- Project structure tree
- Module links to their individual READMEs
- Tech stack table
- License

- [ ] **Step 2: Write docs/getting-started.md (detailed setup guide including Doubao config)**

- [ ] **Step 3: Write docs/module-guide.md (per-module LangGraph knowledge point explanations)**

- [ ] **Step 4: Write docs/architecture.md (architecture diagram + design decisions)**

- [ ] **Step 5: Write CONTRIBUTING.md (Conventional Commits, dev flow)**

- [ ] **Step 6: Write CODE_OF_CONDUCT.md (Contributor Covenant v2.1 中文版)**

- [ ] **Step 7: Write LICENSE (MIT, Copyright 2026 DongWeitao 董伟涛)**

- [ ] **Step 8: Write .github/workflows/ci.yml**

```yaml
name: CI
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e ".[dev]"
      - run: ruff check src/
      - run: pytest tests/unit tests/graph -v --ignore=tests/api
```

- [ ] **Step 9: Final integration test — run all tests together**

Run: `pytest -v --ignore=tests/api`
Expected: All unit + graph tests PASS (API tests need running server)

- [ ] **Step 10: Commit**

```bash
git add README.md docs/ CONTRIBUTING.md CODE_OF_CONDUCT.md LICENSE .github/
git commit -m "feat: 开源文档 + CI — README, docs, CONTRIBUTING, CODE_OF_CONDUCT, LICENSE, CI"
```

- [ ] **Step 11: Create GitHub repo and push**

```bash
# 在 GitHub 上创建 langgraph-scrm 仓库后：
git remote add origin https://github.com/ghtaoge/langgraph-scrm.git
git push -u origin main
```
