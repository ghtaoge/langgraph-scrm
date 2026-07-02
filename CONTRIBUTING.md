# 贡献指南

感谢你对 LangGraph-SCRM 的关注！欢迎提交 Issue 和 Pull Request。

## 开发流程

1. Fork 仓库并克隆到本地
2. 创建功能分支：`git checkout -b feat/your-feature`
3. 安装开发依赖：`pip install -e ".[dev]"`
4. 编写代码与测试
5. 确保通过所有检查（见下）
6. 提交并推送，发起 Pull Request

## 提交规范（Conventional Commits）

使用约定式提交格式：

```
<type>: <description>
```

常用 type：
- `feat`：新功能
- `fix`：缺陷修复
- `docs`：文档变更
- `refactor`：重构（不改功能）
- `test`：测试相关
- `chore`：构建/工具/杂项

示例：
```
feat: 模块3 知识库问答 — Corrective RAG + 工具调用 + Embedding
fix: 修正 multi_agent 并行 Agent 并发写 agent_responses 冲突
docs: 补充架构设计文档
```

## 代码规范

- Python >= 3.11
- 使用 ruff 格式化与检查：`ruff check src/ tests/`
- 所有代码注释与 docstring 使用中文
- 每个模块 State 必须包含 `error: Optional[str]` 与 `error_node: Optional[str]` 字段
- LLM 调用使用 `@safe_llm_call` 装饰器或 try/except 处理异常
- 测试 mock LLM 调用，CI 中不依赖真实 LLM

## 测试要求

新增功能需配套测试：

```bash
# 运行全部测试
pytest -v

# 仅单元 + 图集成测试（CI 范围）
pytest tests/unit tests/graph -v
```

- 节点逻辑：`tests/unit/`
- 图流程：`tests/graph/`
- API 路由：`tests/api/`
- 真实 LLM 测试用 `@pytest.mark.integration` 标记，CI 中跳过

## 模块开发模板

新增模块需包含：
- `state.py`：TypedDict State 定义（含 error/error_node）
- `nodes.py`：节点函数（用 `@safe_llm_call`）
- `graph.py`：`build_xxx_graph()` + `__main__` 入口
- `__init__.py`：导出 build 函数
- `README.md`：知识点说明
- `tests/unit/test_xxx_nodes.py` + `tests/graph/test_xxx_graph.py`

## 报告问题

提交 Issue 时请包含：
- 问题描述与复现步骤
- Python / LangGraph 版本
- 期望行为与实际行为
- （如有）错误日志
