# 模块 5：售后工单（After-Sale Workflow）

> 🎯 **LangGraph 知识点**：长流程编排、双 `interrupt()` 审批门、Checkpoint 暂停/恢复、条件回退

## 业务场景

客户提交售后请求 → AI 分析问题类型与严重度 → 主管审批（暂停） → 执行处理方案 → 客户确认（暂停） → 满意则关闭，不满意则重新执行

## 图结构

```
START → create_ticket → analyze → approve (interrupt) → (条件路由)
  ├── approved → execute → verify (interrupt) → (条件路由)
  │   ├── 满意 → close → END
  │   └── 不满意 → execute (重试)
  ├── rejected → analyze (重新分析)
  └── needs_info → analyze
```

## LangGraph 核心知识点

### 双 interrupt 审批门
- `approve` 节点：主管审批，`interrupt()` 暂停等待 `Command(resume={"approval_status":"approved"})`
- `verify` 节点：客户确认，`interrupt()` 暂停等待 `Command(resume={"feedback":"满意"})`
- 两个暂停点都需要 checkpointer 支持状态持久化

### 条件回退
审批拒绝/需补充信息时回到 `analyze` 重新分析；客户不满意时回到 `execute` 重新处理，形成受控循环。

## 运行

```bash
python -m src.modules.after_sale.graph
```
