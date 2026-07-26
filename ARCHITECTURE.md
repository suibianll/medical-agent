# 代码架构

项目采用“核心协议与策略、应用编排、Prompt、适配器、基础设施、通用工具”分层。依赖方向保持由外向内：外部服务可以依赖核心协议，核心代码不依赖具体 API、配置文件或网页。

```text
server.py / public                 HTTP 与页面入口
        │
        ▼
service.py                        Plan-Execute-Evaluate 应用编排
        │
        ├── agent_pipeline.py      单任务三阶段流程
        ├── dag_scheduler.py       DAG 调度核心
        ├── evidence/evaluator     证据注册与确定性策略
        ├── report/graph           服务端结果投影
        │
        └── model_adapter.py       模型能力接口
                ▲
                │
adapters/                          协议适配、模型工厂
        ├── openai_compatible.py
        └── model_factory.py
                │
                ├── prompts/       所有模型提示词与输入模板
                ├── infrastructure/ HTTP 客户端与配置读取
                ├── observability/ 安全进度与审计投影
                └── utils/         无状态文本、JSON 工具
```

## 目录职责

- `prompts/`：唯一允许维护模型指令文本的位置。按规划、查询、抽取、总结、评估、对话和修复拆分；构造器返回极简不可变对象。
- `adapters/`：实现 `ModelAdapter`，将核心调用转换为 Prompt 和外部客户端调用，不负责 DAG、证据 ID 或报告。
- `infrastructure/`：处理网络和运行时配置。模型密钥只在配置对象到客户端构造过程短暂传递，不进入健康检查、日志或结果。
- `observability/`：对白名单运行事件进行裁剪、脱敏和并发排序；应用服务只负责发送领域事件。
- `utils/`：不包含医疗业务的纯函数，如弱模型 JSON 提取和有界文本处理。
- `service.py`：应用用例入口，协调计划、执行、评估、修复、报告和归档，不维护具体 Prompt 或 HTTP 调用。
- 根目录的 `aliyun_model.py` 是兼容层。新代码应从 `adapters` 或 `infrastructure` 导入；兼容层可在下一个破坏性版本移除。

## 扩展方式

新增模型供应商时，优先复用 `OpenAIChatClient`；若协议不同，在 `infrastructure/` 新增客户端，并在 `adapters/` 实现 `ModelAdapter`。新增推理阶段时，在 `prompts/` 新建对应 Prompt 文件，在适配器中增加最小调用，再由应用编排层接入。不要让模型生成证据 ID、依赖状态、引用图或报告布局。

Prompt 的输出结构应继续保持扁平、字段少、长度有上限。修改 Prompt 时应补充 Prompt 构造器测试和完整端到端测试，避免弱模型输出协议发生无意漂移。
