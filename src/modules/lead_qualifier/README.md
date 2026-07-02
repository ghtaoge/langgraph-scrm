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
python -m src.modules.lead_qualifier.graph
```
