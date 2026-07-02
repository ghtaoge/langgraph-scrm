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

请以 JSON 返回 score（质量评分 0-10）和 feedback（改进建议）。低于 7 分需要重新生成。
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
