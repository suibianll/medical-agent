# 医疗 Agent MVP

这是一个可本地运行的、以证据为中心的医疗 Agent 原型。它实现了：

- 保留任务依赖关系的 Plan-Execute DAG；
- 每个任务固定执行“检索 → 关键信息抽取 → 基于证据总结”三阶段；
- 面向弱模型的极简协议；
- 患者事实（`P#`）与知识库事实（`K#`）的可追溯引用；
- 结论级证据校验、定向修复及人工审核降级；
- Markdown 报告和浏览器中的任务/证据图可视化。

> **重要：** 此项目是技术原型，默认知识库内容为合成演示数据。它不是医疗器械、诊断工具或处方系统，不能替代有资质的医疗专业人员。

## 快速运行

项目没有第三方 Python 依赖，需要 Python 3.11 以上。

```powershell
python run.py
```

浏览器打开 <http://127.0.0.1:8000>，点击“载入演示病例”后运行即可查看完整链路。

如果系统未将 Python 加入 `PATH`，可使用 Codex 工作区提供的运行时：

```powershell
& 'C:\Users\chuzhaole\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' run.py
```

执行测试：

```powershell
$env:PYTHONPATH = 'src'
python -m unittest discover -s tests -v
```

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
  → 证据注册表（P# / K#）
  → 结论级评估
  → 定向修复失败任务及其后代
  → 报告 + 证据图 / 人工审核
```

依赖只能指向较小的任务 ID，因此弱模型不会生成环形依赖。调度器会并行执行就绪任务；上游失败时，下游会被标记为 `blocked`，不会带着不完整上下文运行。

## 对话工作台与本地知识库

启动 `python run.py` 后访问 <http://127.0.0.1:8000>。页面分为三个区域：

- 左侧可选择 `.txt`、`.md`、`.csv` 或 `.json` 文本文件，也可直接粘贴脱敏资料；导入后会显示资料名称、分块数和版本信息。
- 中间是多轮对话区。患者上下文为可选项：留空时只能得到带 `K#` 知识库引用的一般信息；填写脱敏病历时，患者特异性结论须同时带 `P#` 病历事实和 `K#` 知识库引用。
- 右侧即时展示本轮的证据链图、引用清单和选中节点详情。图中保留任务、证据、结论和最终回答之间的关系；点击回答中的引用或图节点可查看来源定位信息。

导入内容会被分块并保存在本机的 `data/imported_knowledge.json`，服务重启后仍可检索。该文件已被 Git 忽略；请只导入已获授权、完成脱敏的资料，生产环境应替换为满足访问控制、审计、留存和删除策略的知识库服务。

## API

### `POST /api/runs`

请求：

```json
{
  "patientRecord": "患者病历原文",
  "request": "用户问题",
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
- `report.markdown`：可审计报告；
- `graph`：可直接渲染的节点和边。

其他接口：

- `GET /api/health`
- `GET /api/sample`

- `GET /api/knowledge`：返回知识库资料的元数据，不返回全文。
- `POST /api/knowledge/import`：导入本地文本资料。

  ```json
  {"name": "肾功能用药说明.md", "content": "已脱敏的资料正文"}
  ```

- `POST /api/chat`：执行一轮直接对话，并返回回答、结论、证据、DAG 和证据图。

  ```json
  {
    "message": "这份资料对肾功能复核有哪些一般提示？",
    "patientRecord": "可选的脱敏病历",
    "history": [{"role": "user", "content": "上一轮问题"}]
  }
  ```

  响应中的 `answer` 可直接显示；`claims` 保留每条结论及其 `refs`，`evidence` 给出来源和定位信息，`graph` 可用于渲染证据链。对话历史仅用于理解上下文，不会被当作医学证据。

## 替换为真实模型和知识库

`src/medical_agent/model_adapter.py` 定义了模型接口。真实模型只需实现：

- `plan()`：返回最小任务 DAG；
- `make_queries()`：返回查询字符串；
- `extract_facts()`：返回 `{text, ref}`；
- `synthesize()`：返回 `{text, refs}`；
- 可选 `judge_claim()`：返回 `SUPPORTED`、`NOT_SUPPORTED` 或 `UNCERTAIN`。

知识库接口位于 `src/medical_agent/retrieval.py`。生产接入时应替换演示 JSON，实现来源准入、版本管理、准确定位、脱敏、访问控制、审计、数据留存策略和提示注入防护。

### 阿里云百炼 / Model Studio

项目内置了标准库实现的 OpenAI 兼容适配器。**不要把密钥写入文件或提交到仓库**；仅在运行进程中设置环境变量：

```powershell
$env:MEDICAL_AGENT_API_KEY = 'sk-...'
$env:MEDICAL_AGENT_BASE_URL = 'https://{workspace_id}.cn-beijing.maas.aliyuncs.com'
$env:MEDICAL_AGENT_MODEL = 'qwen3.7-plus'
python run.py
```

适配器会自动补全 `/compatible-mode/v1`，并调用 Chat Completions 接口。阿里云官方文档说明北京地域的兼容接口为 `POST /compatible-mode/v1/chat/completions`。[官方说明](https://help.aliyun.com/en/model-studio/qwen-api-via-openai-chat-completions)

## 关键安全边界

- 不记录或展示模型完整思维链；仅记录检索查询、证据 ID、任务状态和结构化结果。
- 病历事实与外部知识分开编号；患者特异性分析结论必须同时具有 `P#` 和 `K#` 支持。
- 引用不存在、来源不匹配或证据不支持结论时触发修复。
- 自动修复最多两轮；仍失败或高风险时返回 `needs_human_review`。
- HTTP 服务不在日志中写入请求体或病历内容。
