# 代码架构

项目采用“传输入口、组合根、应用用例、核心契约、适配器与基础设施”分层。具体实现只在 `bootstrap.py` 中装配；应用服务依赖 `ports.py` 和 `model_adapter.py` 中的协议，不读取环境变量，也不构造具体存储或模型客户端。

```text
server.py + transport/              HTTP、SSE、输入边界与资源限制
        │
        ▼
service.py                          稳定应用门面、模型选择与用例入口
        │
        ▼
application/workflow.py             Plan-Execute-Evaluate-Repair 编排
        │
        ├── agent_pipeline.py        单任务三阶段流程
        ├── dag_scheduler.py         DAG 调度核心
        ├── evidence/evaluator       证据注册与确定性策略
        ├── report/graph             结果投影
        └── contracts.py / ports.py  结构契约与依赖端口

bootstrap.py                         唯一组合根
        ├── adapters/                模型协议适配与工厂
        ├── infrastructure/          HTTP 客户端与配置读取
        ├── retrieval/               知识库、患者检索与评分
        ├── run_archive/audit_log    归档与安全审计实现
        └── web/ + resources/        随包发布的页面与演示知识库
```

## 目录职责

- `contracts.py`：集中维护 Plan、Claim、Evidence、RunResult 和模型小协议的 `TypedDict`，避免跨层字段靠隐式约定传播。
- `ports.py`：定义知识库、运行归档和审计输出端口；应用层只依赖这些协议。
- `application/workflow.py`：只负责单次运行的计划、执行、评估、修复和结果组装。
- `bootstrap.py`：读取配置并装配模型、知识库、归档和审计实现，是唯一允许同时依赖应用层与具体实现的组合根。
- `prompts/`：唯一允许维护模型指令文本的位置。按规划、查询、抽取、总结、评估、对话和修复拆分；构造器返回极简不可变对象。
- `adapters/`：实现 `ModelAdapter`，将核心调用转换为 Prompt 和外部客户端调用，不负责 DAG、证据 ID 或报告。
- `infrastructure/`：处理网络和运行时配置。模型密钥只在配置对象到客户端构造过程短暂传递，不进入健康检查、日志或结果。
- `observability/`：对白名单运行事件进行裁剪、脱敏和并发排序；应用服务只负责发送领域事件。
- `utils/`：不包含医疗业务的纯函数，如弱模型 JSON 提取和有界文本处理。
- `retrieval/`：保持统一公共导入路径，并将患者病历检索、知识库导入/持久化和词法评分拆成独立实现。
- `service.py`：稳定应用门面，负责模型档案选择、知识库用例和运行查询；复杂编排委托给 `MedicalWorkflow`。
- `server.py`：本机 HTTP 入口。服务实例由 `MedicalAgentHTTPServer` 持有，限制高成本运行并发，只允许绑定回环地址；导入模块不会读取配置。
- `transport/`：校验并限制请求、病历和历史字段，未经验证的数据不会进入应用层。
- `web/`：页面采用 ES Modules；公共 DOM、SVG、引用解析和图布局位于 `shared.js`，任务进度状态机及渲染位于 `execution-view.js`，页面入口只负责用例交互。
- 模型实现统一位于 `adapters/`；HTTP 客户端、URL 规范化和配置解析统一位于 `infrastructure/`。

## 扩展方式

新增模型供应商时，优先复用 `OpenAIChatClient`；若协议不同，在 `infrastructure/` 新增客户端，并在 `adapters/` 实现 `ModelAdapter`。新增推理阶段时，在 `prompts/` 新建对应 Prompt 文件，在适配器中增加最小调用，再由应用编排层接入。不要让模型生成证据 ID、依赖状态、引用图或报告布局。

Prompt 的输出结构应继续保持扁平、字段少、长度有上限。修改 Prompt 时应补充 Prompt 构造器测试和完整端到端测试，避免弱模型输出协议发生无意漂移。

## 模型调用预算

- 有显式计划时不调用 Planner；模型规划默认要求使用最少的 2–4 个任务。
- 每个任务保留查询生成和事实抽取；没有检索证据时跳过抽取与总结，没有有效事实时跳过总结。
- “提取/检索”类来源任务直接把已验证的原子事实投影成带引用摘要，不再调用模型重复改写。
- 语义证据核验只处理已通过引用完整性硬校验的结论，每批最多 8 条；修复轮复用未变化结论的核验结果。
- 总结 Prompt 仅接收经过校验的 `{text, ref}` 事实，不重复发送完整证据原文。
