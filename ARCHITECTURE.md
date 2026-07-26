# 代码架构

项目采用“传输入口、组合根、应用用例、核心契约、适配器与基础设施”分层。具体实现只在 `bootstrap.py` 中装配；应用层只依赖 `ports.py` 中的协议，不读取环境变量，也不构造具体存储或模型客户端。

```text
server.py + transport/              HTTP、SSE、输入边界与资源限制
        │
        ▼
application/agent.py               唯一应用入口、模型选择与用例协调
        │
        ▼
application/workflow.py             Plan-Execute-Evaluate-Repair 编排
        │
        ├── agent_pipeline.py        单任务三阶段流程
        ├── dag_scheduler.py         DAG 调度核心
        ├── evidence/evaluator       证据注册与确定性策略
        ├── report/graph             结果投影
        ├── prompting.py             全部模型提示词
        └── contracts.py / ports.py  数据契约与依赖端口

bootstrap.py                         唯一组合根
        ├── adapters/                OpenAI 兼容模型适配
        ├── infrastructure/          HTTP 客户端与配置读取
        ├── retrieval/               知识库、患者检索与评分
        ├── run_archive/audit_log    归档与安全审计实现
        └── web/ + resources/        随包发布的页面与演示知识库
```

## 目录职责

- `contracts.py`：维护实际使用的 Plan、Claim、RunResult 和模型返回结构。
- `ports.py`：集中定义模型、知识库、运行归档和审计端口。
- `application/agent.py`：唯一应用入口，负责模型档案选择、知识库操作、聊天结果和运行查询；不提供历史兼容门面。
- `application/workflow.py`：只负责单次运行的计划、执行、评估、修复和结果组装。
- `application/workflow.py`：可将 `verifier_model` 与生成模型分离；默认未配置时回退到选中的模型以保持兼容。
- `bootstrap.py`：读取配置并装配模型、知识库、归档和审计实现，是唯一允许同时依赖应用层与具体实现的组合根。
- `prompting.py`：集中维护规划、查询、抽取、总结、评估、对话和修复 Prompt，避免为短函数建立过多文件。
- `adapters/`：实现 `ModelAdapter`，将核心调用转换为 Prompt 和外部客户端调用，不负责 DAG、证据 ID 或报告。
- `infrastructure/`：处理网络和运行时配置。模型密钥只在配置对象到客户端构造过程短暂传递，不进入健康检查、日志或结果。
- `observability/`：对白名单运行事件进行裁剪、脱敏和并发排序；`model_metrics.py` 汇总 provider usage、延迟和调用阶段，不接触提示词或模型原文。
- `retrieval/`：患者病历检索、知识库导入/持久化、词法评分、RRF 融合、可选向量检索和外部重排；`state.py` 维护有界检索轮数、候选预算和停止原因，`fusion.py` 统一不同后端的多查询融合，`vector.py` 通过 `FaissKnowledgeBase` 装饰器隔离 FAISS/embedding 依赖，`reranker.py` 只负责结构化 HTTP 传输。
- `server.py`：本机 HTTP 入口。Agent 实例由 `MedicalAgentHTTPServer` 持有，限制高成本运行并发，只允许绑定回环地址；导入模块不会读取配置。
- `transport/`：校验并限制请求、病历和历史字段，未经验证的数据不会进入应用层。
- `web/`：页面采用 ES Modules；公共 DOM、SVG、引用解析和图布局位于 `shared.js`，任务进度状态机及渲染位于 `execution-view.js`，页面入口只负责用例交互。
- 模型实现统一位于 `adapters/`；HTTP 客户端、URL 规范化和配置解析统一位于 `infrastructure/`。

## 扩展方式

新增模型供应商时，优先复用 `OpenAIChatClient`；若协议不同，在 `infrastructure/` 新增客户端，并按 `ports.ModelAdapter` 实现适配器。新增推理阶段时直接扩展 `prompting.py`，再由应用编排层接入。不要让模型生成证据 ID、依赖状态、引用图或报告布局。

检索后端通过组合根选择：应用层只依赖 `KnowledgeBasePort`，词法后端和 FAISS 后端都输出同一份带来源的文档契约。FAISS 未安装或 embedding 响应不符合契约时应给出可诊断错误，索引缓存写入失败则不影响当前请求。外部重排器和路由策略同样只能在组合根装配，不能在任务管线中读取环境变量或依赖具体 HTTP 客户端。

Prompt 的输出结构应继续保持扁平、字段少、长度有上限。修改 Prompt 时应补充 Prompt 构造器测试和完整端到端测试，避免弱模型输出协议发生无意漂移。

## 模型调用预算

- 有显式计划时不调用 Planner；模型规划默认要求使用最少的 2–4 个任务。
- 每个任务保留查询生成和事实抽取；没有检索证据时跳过抽取与总结，没有有效事实时跳过总结。
- “提取/检索”类来源任务直接把已验证的原子事实投影成带引用摘要，不再调用模型重复改写。
- 语义证据核验只处理已通过引用完整性硬校验的结论，每批最多 8 条；修复轮复用未变化结论的核验结果。
- 总结 Prompt 仅接收经过校验的 `{text, ref}` 事实，不重复发送完整证据原文。
- 证据注册表为每个证据生成内容哈希和默认 chunk span；评估器将每条有效引用投影为 `SupportEdge`，图层保留旧边 ID并增加 relation/span/verifier 字段。
