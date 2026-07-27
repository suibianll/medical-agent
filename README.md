# 医疗 Agent MVP

这是一个可本地运行的、以证据为中心的医疗 Agent 原型。它实现了：

- 保留任务依赖关系的 Plan-Execute DAG；
- 每个任务固定执行“检索 → 关键信息抽取 → 基于证据总结”三阶段；
- 面向弱模型的极简协议；
- 患者事实（`P#`）与知识库事实（`K#`）的可追溯引用；
- 多查询 RRF 融合、候选预算与可审计检索停止状态；
- 结论级证据校验、定向修复及人工审核降级；
- `SupportEdge`/span 证据关系和模型调用 token/延迟观测；
- 可插拔的安全路由：默认只依据证据链状态回答/澄清/延迟；配置规则或外部 API 可额外提供急症升级信号；
- Markdown 报告和浏览器中的任务/证据图可视化。

> **重要：** 此项目是技术原型，默认知识库内容为合成演示数据。它不是医疗器械、诊断工具或处方系统，不能替代有资质的医疗专业人员。

## 快速运行

默认路径没有第三方 Python 依赖，需要 Python 3.11 以上；启用 FAISS 向量检索时安装 `pip install -e .[vector]`。

```powershell
python -m pip install -e .
medical-agent
```

浏览器打开 <http://127.0.0.1:8000>，点击“载入演示病例”后运行即可查看完整链路。

如果系统未将 Python 加入 `PATH`，可使用 Codex 工作区提供的运行时：

```powershell
$python = 'C:\Users\chuzhaole\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $python -m pip install -e .
& $python -m medical_agent.server
```

执行测试：

```powershell
python -m unittest discover -s tests -v
```

服务刻意限制为本机原型，只允许绑定 `127.0.0.1`、`::1` 或 `localhost`。对外部署必须使用具备 TLS、认证、授权、限流和并发治理的正式 Web/API 运行时。

## 用户验收样例

可直接使用 [examples](examples/README.md) 中的合成知识库、病历和三轮对话脚本验证工作台。导入 `medication_review_knowledge.md`、粘贴 `medication_review_patient_record.txt`，再按 `medication_review_dialogue.json` 依次提问，即可检查逐句引用、任务进程、文本报告和独立证据页。

## 最小模型协议

Planner 只需输出任务 ID、目标和依赖：

```json
{
  "tasks": [
    {"id": 1, "goal": "提取患者关键事实", "deps": []},
    {"id": 2, "goal": "检索相关医学依据", "deps": [1]},
    {"id": 3, "goal": "综合分析", "deps": [1, 2]}
  ]
}
```

子 Agent 的结论协议同样扁平：

```json
{
  "claims": [
    {"text": "一条原子结论", "refs": ["P1", "K1"]}
  ],
  "unknowns": []
}
```

模型不生成 URL、页码、文档版本、证据 ID、任务状态或图关系；这些均由服务端创建和验证。

## 工作流

```text
病历 + 请求
  → Planner（id / goal / deps）
  → DAG 调度器
  → 每个任务：检索 → 抽取 → 总结
  → 多查询融合检索与候选预算
  → 证据注册表（P# / K# / span）
  → 结论级评估
  → 定向修复失败任务及其后代
  → 报告 + 证据图 / 人工审核
```

依赖只能指向较小的任务 ID，因此弱模型不会生成环形依赖。调度器会并行执行就绪任务；上游失败时，下游会被标记为 `blocked`，不会带着不完整上下文运行。

## 代码分层

所有模型提示词集中在 `src/medical_agent/prompting.py`。`contracts.py` 和 `ports.py` 定义数据结构及模型、知识库、归档和审计协议，`application/agent.py` 是唯一应用入口，`application/workflow.py` 负责完整运行编排。具体实现只在 `bootstrap.py` 中装配；项目不提供历史接口别名或兼容转发。

任务管线会主动控制真实模型调用：无证据时不调用抽取和总结，无有效事实时不调用总结；纯提取/检索任务由代码直接把已验证事实生成引用摘要；语义评估按最多 8 条结论批量调用，并在修复轮复用未变化结论的核验结果。核验 verdict 细分为 `SUPPORTED`、`PARTIALLY_SUPPORTED`、`CONTRADICTED` 和 `INSUFFICIENT`，部分支持、冲突或证据不足都会进入修复/人工复核路径；旧 provider 的 `NOT_SUPPORTED`/`UNCERTAIN` 仍兼容。总结阶段只发送已验证事实，不重复发送完整证据原文。

知识库在保留 `search()` 兼容接口的同时提供 `search_many()`：对最多 3 条查询做 RRF 融合并限制同一文档占比。默认使用无依赖词法后端；在 `model.local.json` 的 `retrieval.backend` 设置为 `faiss` 后，组合根会装配 FAISS 装饰器，并由 `retrieval.embedding` 选择本地哈希向量或 OpenAI 兼容 embedding API。FAISS 与 NumPy 是可选依赖，可用 `pip install -e .[vector]` 安装。`retrieval.embedding.cache_size`/`cache_ttl_seconds` 对重复查询 embedding 做有界缓存；索引仍按文档指纹复用。每个任务结果包含检索轮数、候选数和停止原因，`run.retrieval_usage.embedding` 记录 batch、provider 调用、缓存命中和延迟。真实模型调用会记录阶段、延迟和 provider usage；`run.model_usage` 只含聚合统计，不含提示词、病历或模型原文。每轮还会在 `run.quality` 给出不含正文的证据链质量摘要；离线回归集可调用 `medical_agent.quality.evaluate_retrieval_cases` 计算 Recall@K、MRR 和 nDCG@K，不需要额外模型调用。

可在同一配置文件中启用外部 reranker：`reranker.enabled`、`endpoint`、`provider`、`model` 和 `api_key_env` 决定请求；适配器只接受带候选 `index` 与分数的结构化结果，并在失败时保留原检索排序，同时在任务检索摘要中标注 `failed_fallback`。`max_calls_per_run` 限制单轮（包括并行任务与自动修复）外部调用次数，`min_candidates` 避免只有一个候选时无收益地调用，`cache_size`/`cache_ttl_seconds` 对同一轮重复查询做短期去重。执行结果的 `run.retrieval_usage` 只记录调用数、缓存命中、跳过和延迟等安全统计，不含查询、病历或候选正文；执行流会输出经过白名单裁剪的 `rerank` 事件，不包含候选原文或密钥。

可在 `retrieval.source_policy` 启用来源治理：`allowed_source_types`、`blocked_statuses`、`min_priority`、`require_version`、`allow_synthetic`、`max_age_days` 和 `reject_unknown_date` 只读取来源元数据，不扫描正文或写死关键词。被撤回、不安全、过期或不满足版本/优先级要求的资料会在词法排序、FAISS 建索引和外部 reranker 之前被排除；`documents` 库存接口仍保留完整导入清单。运行健康信息只显示策略摘要，证据引用会保留有限的来源类型、版本和时效诊断，便于人工复核。

风险路由默认使用 `routing.mode = "evidence"`，不扫描请求或结论文本关键词。需要本地透明规则时使用 `routing.mode = "rules"` 并在配置中提供正则；需要模型分类时使用 `routing.mode = "api"`，外部响应必须返回 `outcome`、`risk_level` 等结构化字段。计划任务可用 `evidence_scope`（`patient`/`knowledge`/`both`/`none`）和 `analysis_mode` 声明路由意图，避免从 `goal` 文字猜测来源。

从配置启动时会先执行 `validate_model_configuration`：FAISS + OpenAI 兼容 embedding、启用的 reranker、API 路由缺少必要字段或使用不安全 URL 会在建客户端前以结构化错误返回；仅包含字段路径和修复提示，不回显密钥。`/api/health` 的 `models.runtime` 只展示检索后端、provider、模型/路由模式等安全身份，不展示 endpoint、索引路径或认证信息。

完整的依赖方向、目录职责和新增供应商/推理阶段的方法见 [ARCHITECTURE.md](ARCHITECTURE.md)。模型适配器统一从 `medical_agent.adapters.openai_compatible` 导入。

### 自适应检索预算

检索默认最多运行 `retrieval.max_rounds` 轮；当候选数低于 `retrieval.refine_min_candidates` 且 `refine_on_empty=true` 时，模型只根据结构化缺口生成补充查询。代码负责去重查询、限制候选数并在预算耗尽时停止，补充轮次会计入 `run.retrieval_usage`，不会扫描关键词或绕过来源治理。

### 冲突关系展示

证据图会把 `supports`、`qualifies`、`contradicts` 和 `uncertain` 作为独立关系保存；同一结论同时存在支持与冲突依据时，图层额外生成 `conflict` 证据间关系，并在 `graph.summary` 提供冲突结论数、关系计数和缺少支持的结论 ID。页面用颜色/虚线区分冲突与不确定关系，选中结论可直接看到需要人工复核的提示。

### 反事实/临床风险回归

离线回归可调用 `medical_agent.quality.evaluate_counterfactual_cases`（或别名 `evaluate_counterfactual_pairs`）比较同一病例的 baseline 与 counterfactual 运行。每个 fixture 只需提供结构化 `decision`、证据 ID、结论引用和 `expected` 规则；输出 `decision_responsiveness`、`evidence_responsiveness`、`citation_integrity`、`safe_abstention_rate` 与 `regression_rate`，不输出查询、病历、结论或证据正文，也不会产生模型调用。将 `must_abstain`、`safe_outcomes` 和 `expected_outcome` 纳入 CI 门禁即可捕获“反事实变化后仍沿用旧结论”与不安全继续回答。

## 对话工作台与本地知识库

启动 `medical-agent`（或 `python -m medical_agent.server`）后访问 <http://127.0.0.1:8000>。页面分为三个区域：

- 左侧可选择 `.txt`、`.md`、`.csv` 或 `.json` 文本文件，也可直接粘贴脱敏资料；导入后会显示资料名称、分块数和版本信息。
- 中间是多轮对话区。患者上下文为可选项：留空时只能得到带 `K#` 知识库引用的一般信息；填写脱敏病历时，患者特异性结论须同时带 `P#` 病历事实和 `K#` 知识库引用。
- 页头可以选择本轮使用的模型配置档案，并显示它是“真实模型”还是“本地演示模型”。模型选择随单次请求提交，不修改全局状态，因此并发请求可以安全使用不同模型。右侧会实时展示任务计划、运行状态和可审计执行摘要。
- 发起对话前可选择“证据核验摘要”“任务与证据链追踪”或“人工复核交接单”模板，并可提供一个简短的报告标题。模板只改变服务器生成的文本报告结构，不会改变证据校验规则。
- 右侧即时展示本轮的证据链图、引用清单和选中节点详情。图中保留任务、证据、结论和最终回答之间的关系；回答中的每个引用可悬浮查看来源、定位与原文摘要，也可以点击打开独立的证据页。
- 独立证据页使用运行编号加载同一轮的文本报告、任务 DAG、证据明细和审计时间线，便于链接、复核和演示。运行结果仅在本机服务进程内短暂归档以支持该页面，不会被写入审计日志。

执行摘要只包含计划、检索查询、证据 ID、关键事实摘要、结论引用和评估状态；它不是、也不会尝试展示模型的原始内部思维链。

服务端还会通过标准 Python 日志输出 `medical_agent_audit` JSON 行，记录运行编号、阶段、状态、引用计数、评估摘要和任务计数。默认日志策略不写入病历、请求、查询/事实/结论正文、模型原始响应、密钥或隐藏推理；部署时可将该安全日志接入现有的日志采集系统。

报告协议刻意保持简短，弱模型无需生成复杂的文档结构。请求可传入模板名称，或一个只包含名称、标题和固定章节标记的小对象：

```json
{
  "reportTemplate": {
    "name": "handoff",
    "title": "用药安全复核交接",
    "sections": ["summary", "conclusions", "unknowns", "evaluation", "evidence"]
  }
}
```

可用章节标记为 `summary`、`conclusions`、`unknowns`、`tasks`、`evaluation`、`evidence`。未知模板或标记会安全地回退到默认的证据核验摘要。最终报告的每条结论会按句切分，并在每一句之后追加其 `P#`/`K#` 引用。

导入内容会被分块并保存在当前运行目录的 `data/imported_knowledge.json`，服务重启后仍可检索。可通过 `MEDICAL_AGENT_DATA_DIR` 指定其他可写目录。该文件已被 Git 忽略；请只导入已获授权、完成脱敏的资料，生产环境应替换为满足访问控制、审计、留存和删除策略的知识库服务。

## API

### `POST /api/runs`

请求：

```json
{
  "patientRecord": "患者病历原文",
  "request": "用户问题",
  "reportTemplate": "task_trace",
  "plan": {
    "tasks": [
      {"id": 1, "goal": "提取关键事实", "deps": []}
    ]
  }
}
```

`plan` 为可选字段。缺省时使用本地 `DemoModelAdapter` 生成一个五步依赖计划。返回结果包含：

- `run`：计划、任务状态、执行波次、评估和修复记录；
- `claims`：带 `P#`/`K#` 引用的原子结论；
- `evidence`：完整来源与定位信息；
- `report.text`：按模板生成的纯文本可审计报告；
- `report.markdown`：同一报告的 Markdown 版本；
- `graph`：可直接渲染的节点和边。
- `run.model_calls` / `run.model_usage`：本轮安全的调用明细与 token/延迟汇总；
- `run.decision`：规则化的风险等级、路由结果、理由和缺失信息；
- 每条 claim 的 `support_edges`：证据、span、relation、verifier 和状态。

其他接口：

- `GET /api/health`
- `GET /api/sample`
- `GET /api/runs/{run_id}`：读取当前服务进程内短暂保留的一轮完整结果，供独立证据页加载。
- `GET /api/runs/{run_id}/events`：读取该轮的安全审计事件。事件不含请求正文、病历、密钥、模型原始响应或隐藏思维链。

- `GET /api/knowledge`：返回知识库资料的元数据，不返回全文。
- `POST /api/knowledge/import`：导入本地文本资料。

  ```json
  {"name": "肾功能用药说明.md", "content": "已脱敏的资料正文"}
  ```

- `POST /api/chat/stream`：执行一轮直接对话，并以 Server-Sent Events 返回实时进度和最终结果。

  ```json
  {
    "message": "这份资料对肾功能复核有哪些一般提示？",
    "patientRecord": "可选的脱敏病历",
    "modelProfile": "openrouter-deepseek",
    "history": [{"role": "user", "content": "上一轮问题"}]
  }
  ```

  响应中的 `answer` 可直接显示；`claims` 保留每条结论及其 `refs`，`evidence` 给出来源和定位信息，`graph` 可用于渲染证据链。对话历史仅用于理解上下文，不会被当作医学证据。

  事件包括：

  - `progress`：安全的规划、任务状态、检索、事实提取、结论生成、评估与修复摘要；
  - `result`：包含回答、结论、证据、DAG 和证据图的完整结果；
  - `error`：不含病历或模型原始响应的失败信息。

## 替换为真实模型和知识库

`src/medical_agent/ports.py` 定义当前模型协议，OpenAI 兼容实现在 `src/medical_agent/adapters/openai_compatible.py`。真实模型只需实现：

- `plan()`：返回最小任务 DAG；
- `make_queries()`：返回查询字符串；
- `extract_facts()`：返回 `{text, ref}`；
- `synthesize()`：返回 `{text, refs}`；
- `judge_claims()`：批量返回每个结论的 `SUPPORTED`、`NOT_SUPPORTED` 或 `UNCERTAIN`。
- 可选 `drain_call_metrics()`：返回当前线程已完成的安全调用指标；不应返回提示词、病历或模型原文。

应用装配还支持可选的 `verifier_model`。传入后，生成模型只负责计划、检索查询、事实抽取和总结，结论核验改由独立模型执行；不传时保持现有单模型兼容行为。

知识库协议位于 `ports.py`，当前 JSON 实现在 `retrieval/knowledge.py`。生产接入时应替换演示 JSON，实现来源准入、版本管理、准确定位、脱敏、访问控制、审计、数据留存策略和提示注入防护。

### 模型配置与切换

项目内置了标准库实现的 OpenAI 兼容适配器，可同时配置阿里云百炼、OpenRouter 或其他兼容服务。单模型环境可继续使用环境变量：

```powershell
$env:MEDICAL_AGENT_API_KEY = 'sk-...'
$env:MEDICAL_AGENT_BASE_URL = 'https://{workspace_id}.cn-beijing.maas.aliyuncs.com'
$env:MEDICAL_AGENT_MODEL = 'qwen3.7-plus'
$env:MEDICAL_AGENT_PROVIDER = 'aliyun-model-studio'
medical-agent
```

需要在页面切换多个模型时，将 `config/model.example.json` 复制为 `config/model.local.json`，使用配置档案数组：

```json
{
  "default_profile": "aliyun-qwen",
  "profiles": [
    {
      "id": "aliyun-qwen",
      "label": "Qwen 3.7 Plus",
      "provider": "aliyun-model-studio",
      "api_key": "sk-...",
      "base_url": "https://{workspace_id}.cn-beijing.maas.aliyuncs.com",
      "model": "qwen3.7-plus"
    },
    {
      "id": "openrouter-deepseek",
      "label": "DeepSeek V4 Flash",
      "provider": "openrouter",
      "api_key": "sk-or-v1-...",
      "base_url": "https://openrouter.ai/api/v1",
      "model": "deepseek/deepseek-v4-flash"
    }
  ]
}
```

环境变量或外部密钥管理器应作为首选。`config/model.local.json` 仅适合受控的本机开发环境：它虽然已被 Git 忽略，但仍是明文凭据文件，不应进入同步盘、备份包、日志或共享目录。页面和 `GET /api/health` 只返回配置档案 ID、标签、供应商和模型名称，不返回 API key 或 base URL。环境变量配置完整时会作为 `environment` 档案并成为默认选项。

适配器只会为阿里云工作区地址补全 `/compatible-mode/v1`；已带 `/v1` 的 OpenRouter 地址会保持不变。重新启动服务后，页面顶部的“本轮模型”选择器会列出所有完整配置以及本地演示模型。

## 关键安全边界

- 不记录或展示模型完整思维链；仅记录可审计的检索查询、证据 ID、任务状态和结构化结果。
- 病历事实与外部知识分开编号；患者特异性分析结论必须同时具有 `P#` 和 `K#` 支持。
- 引用不存在、来源不匹配或证据不支持结论时触发修复。
- 证据边保留 `span_id`、relation 和 verifier 状态；默认 span 是 chunk 级，接入 PDF/OCR 后可替换为页码、字符区间、bbox 或表格单元格。
- 每个任务的检索受轮数和候选数预算限制，并输出停止原因，禁止无界检索循环。
- 自动修复最多两轮；仍失败或高风险时返回 `needs_human_review`。
- HTTP 服务不在日志中写入请求体或病历内容。
- 高成本运行具有全局并发上限；请求、患者上下文、历史记录和模型响应均有独立大小限制。
- 本地 HTTP 入口拒绝绑定非回环地址；它不是生产 Web 服务器。
- 运行归档的完整结果仅保存在当前本机进程的受限内存中（默认最多 24 条、30 分钟）；它用于独立证据页，服务重启或过期后即不可读取。
