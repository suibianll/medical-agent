# 医疗 Agent 完整实验报告（OpenRouter DeepSeek + FAISS）

- 实验日期：2026-07-30（Asia/Shanghai）
- 仓库提交基线：`1f4373c fix: enforce workflow evidence and output boundaries`
- 分支：`agent/real-model-progress-config`
- 生成模型：`openrouter-deepseek` / `deepseek/deepseek-v4-flash`
- 推理配置：核心阶段启用 thinking，预算 2048；流式响应；单请求超时 600 秒
- 输出配置：`max_output_tokens=16384`
- 检索配置：FAISS 1.14.3、BGE-M3 embedding、BGE reranker、Top-K=5
- 数据 split：`test`

## 1. 执行摘要

本次实验完成了单元测试、仓库审计、合成冒烟、公开数据集 FAISS 检索、真实 OpenRouter DeepSeek 端到端 agent、答案契约修复验证和 Faithfulness-QA 反事实验证。

主要结论：

1. 工程基础回归稳定：最终 145 个单元测试全部通过；合成冒烟的 Recall/MRR/nDCG、证据链覆盖和反事实指标均为 1.0。
2. FAISS 确实被使用。128 案例/数据集正式检索中，共完成 512 个带文档级 gold 的案例；916 次 embedding provider 调用、22,811 条文本、0 次 embedding 失败、0 个检索失败案例。
3. 检索表现高度依赖数据集：Evidence Inference 与 Faithfulness-QA 的 Recall@5 均为 0.9922，PubMedQA 为 0.8349；EvidenceBench 只有 0.0893，是最明确的检索短板。
4. OpenRouter 解决了 SiliconFlow 4B 高 thinking 的单调用超时。正式 8 案例/数据集端到端运行 6 个可运行数据集全部完成，48 个主案例没有 `error`；20 个 passed、27 个 needs_human_review、1 个 plan_rejected。
5. 真实端到端吞吐较低。正式 agent 运行耗时 4,023.9 秒（67.1 分钟，包含 Faithfulness 反事实变体）；48 个主案例的平均墙钟延迟约 68.5 秒/案例，Evidence Inference 平均 161.5 秒/案例、最高 316.5 秒。
6. 正式主案例共观察到 266 次模型调用，供应商报告 288,104 input tokens、79,987 output tokens、368,091 total tokens。该基线生成于反事实成本补记修复之前，因此 token/call 汇总不含 16 条 Faithfulness 变体轨迹；总墙钟时间包含它们。
7. 证据链在可回答数据集上总体较强：PubMedQA、EvidenceBench、Faithfulness-QA 的 citation coverage / precision / support-edge coverage 均为 1.0；Evidence Inference 为 0.5，且全部 8 个案例进入人工审核。
8. 选择题 QA-only 镜像没有教材语料，MedMCQA 与 MedQA 的 16/16 案例均安全退让，没有凭模型记忆猜答案；因此不能从本轮得出选择题准确率。
9. 修复答案输出契约后，PubMedQA 2 案例中 1 个可解析且命中 gold，另 1 个安全退让；样本太小，不能把 100% 当作总体准确率。
10. Faithfulness-QA 的稳定证据响应为 1.0、引用完整性为 1.0，但答案决策响应为 0.0、回归率为 1.0（2 案例验证）。系统能换证据，却没有稳定地让最终答案随反事实证据变化，这是上线前必须解决的质量风险。

## 2. 数据与覆盖范围

本地下载清单包含 8 条成功记录，共 573,032,871 字节（约 546.5 MiB）：PubMedQA PQA-L/PQA-U、EvidenceBench、Evidence Inference 2.0、Faithfulness-QA、MedMCQA 镜像、MedQA-USMLE QA-only 镜像和 RAGChecker 资产。

统一 runner 登记了 13 个数据集：

- 完成：PubMedQA、MedMCQA、MedQA-USMLE、Evidence Inference 2.0、EvidenceBench、Faithfulness-QA。
- 按契约跳过：RAGChecker（当前资产缺统一输入语料）、BioASQ（需 challenge 语料/权限）、MedNLI 与 MIMIC-IV（credential/DUA）、HealthSearchQA（需人工 rubric）、BRIDGE 与 MedQA-CS-2026（无已核验本地输入文件）。

跳过项未伪造分数，也未绕过授权边界。

## 3. 基础验证

### 3.1 单元测试

- 最终结果：145/145 通过，耗时约 4.5 秒。
- 覆盖：DAG、检索、FAISS、reranker、模型协议、thinking、流式超时、引用核验、归档、SSE/API、安全边界、数据集适配与反事实评分。
- 一次使用随附 Python 直接运行 `unittest discover` 时出现 `ModuleNotFoundError: medical_agent`，原因是该解释器未安装项目且未设置 `src` 导入路径；改用已安装项目的系统 Python 后全量通过。数据集脚本自身会注入 `src`，正式实验不受影响。

### 3.2 合成冒烟

- 3 个检索案例：Recall@1/3/5、MRR、nDCG@1/3/5 全部 1.0。
- 证据链：citation coverage、citation precision、support-edge coverage、dual-support coverage 全部 1.0。
- 反事实：decision/evidence responsiveness、citation integrity、safe abstention 全部 1.0，regression rate 0。

合成冒烟只验证结构回归，不代表真实临床效果。

### 3.3 仓库审计

- 下载清单有效：8/8 成功，0 失败。
- 必需路径无缺失。
- 高优先级发现：被 Git 忽略的 `config/model.local.json` 包含 5 个 API-key-shaped 值。它们未被跟踪，但仍应迁移到环境变量/密钥管理器，并避免同步、备份和日志泄露。
- 中优先级发现：没有 dependency lock/constraints 文件，环境复现存在版本漂移风险。
- 低优先级发现：缺浏览器级 E2E 覆盖。

## 4. FAISS 检索实验

正式命令使用 `--dataset all --mode retrieval --max-cases 128 --retrieval-backend faiss --top-k 5`。MedMCQA/MedQA 是 QA-only 文件，没有文档级 gold，因此完成适配但不伪造检索指标。

| 数据集 | Gold 案例 | Recall@1 | Recall@3 | Recall@5 | MRR | nDCG@5 |
|---|---:|---:|---:|---:|---:|---:|
| PubMedQA | 128 | 0.3073 | 0.7609 | 0.8349 | 0.9922 | 0.8839 |
| Evidence Inference 2.0 | 128 | 0.8359 | 0.9766 | 0.9922 | 0.9072 | 0.9290 |
| EvidenceBench | 128 | 0.0267 | 0.0649 | 0.0893 | 0.2447 | 0.1551 |
| Faithfulness-QA | 128 | 0.9531 | 0.9922 | 0.9922 | 0.9714 | 0.9767 |

运行统计：

- 总耗时：370.3 秒。
- 512 个 gold 案例，0 invalid，0 failed。
- embedding：916 provider calls、22,811 provider texts、0 failures；供应商延迟累计 330,545 ms。
- EvidenceBench 单独消耗 470 次 embedding 调用和 21,699 条文本，Recall@5 仍仅 0.0893。当前通用 BGE-M3 + Top-5 对该大候选句检索任务明显不足，应优先做 query/pooling、索引粒度、候选 Top-K 和 reranker 的针对性消融。

## 5. 真实 OpenRouter Agent 实验

正式性能基线使用 `--dataset all --mode agent --max-cases 8 --max-repair-rounds 1 --retrieval-backend faiss`。

### 5.1 主案例状态与质量

| 数据集 | 状态 | 引用覆盖 | 支持边覆盖 | 未解决率 | 平均延迟 |
|---|---|---:|---:|---:|---:|
| PubMedQA | passed 7 / review 1 | 1.000 | 1.000 | 0.0625 | 58.4 s |
| MedMCQA | review 8 | 0.000 | 0.000 | 0.0000 | 28.3 s |
| MedQA-USMLE | review 8 | 0.000 | 0.000 | 0.0000 | 67.6 s |
| Evidence Inference 2.0 | review 8 | 0.500 | 0.500 | 0.4792 | 161.5 s |
| EvidenceBench | passed 7 / review 1 | 1.000 | 1.000 | 0.0250 | 62.9 s |
| Faithfulness-QA | passed 6 / review 1 / plan_rejected 1 | 1.000 | 1.000 | 0.1429 | 32.7 s |

总体：

- 48 个主案例：20 passed（41.7%）、27 needs_human_review（56.3%）、1 plan_rejected（2.1%）、0 error。
- 主案例模型调用：266。
- 主案例 token：288,104 input、79,987 output、368,091 total。
- 供应商报告的主案例调用延迟累计：3,279,528 ms。
- 端到端正式运行总耗时：4,023.9 秒。该值包含 Faithfulness 的 16 条反事实变体轨迹。
- 主案例平均延迟约 68.5 秒；最慢单案例为 Evidence Inference 316.5 秒。
- embedding 与 reranker 在正式 agent 运行中均有真实调用，记录的 failure 为 0。
- OpenRouter 接受了统一 `reasoning` 请求，但返回 usage 中 `completion_tokens_details.reasoning_tokens` 为 0/缺失，不能仅凭 telemetry 证明实际消耗的 reasoning token 数；可以证明的是请求参数已发送且模型调用成功。

### 5.2 答案准确率可测性修复

最初 8 案例基线的 48 个案例全部显示 unparseable。根因是 benchmark adapter 没有：

- 把 MedMCQA/MedQA 选项加入请求；
- 要求标签题/文本题输出机器可解析的 `Final answer:` 原子 claim；
- 区分“该数据集无答案 gold”与“有 gold 但未解析”。

修复后，对全部可运行数据集各 2 案例复跑：

| 数据集 | 可解析 / 案例 | 准确率 | 解释 |
|---|---:|---:|---|
| PubMedQA | 1 / 2 | 1.000 | 1 个正确，1 个安全退让 |
| MedMCQA | 0 / 2 | 不适用 | 无教材证据，2 个均退让 |
| MedQA-USMLE | 0 / 2 | 不适用 | 无教材证据，2 个均退让 |
| Evidence Inference 2.0 | 0 / 2 | 不适用 | 2 个均需人工审核 |
| EvidenceBench | 不计分 | 不适用 | 无 answer gold；修复后不再误计 unparseable |
| Faithfulness-QA | 0 / 2 | 不适用 | 未稳定生成约定文本答案 |

PubMedQA 的 1/1 正确只能说明契约修复有效，样本过小，不能当作模型总体准确率。

## 6. Faithfulness-QA 反事实实验

评测器原先用每次运行都会重置的 `K1/K2` 注册表 ID 比较变体，且没有显式传入 `evidence_should_change`，会误导 evidence responsiveness。现已改用稳定 `document_id/content_hash` 并明确要求证据变化。

最终 2 案例验证：

| 指标 | 结果 |
|---|---:|
| Decision responsiveness | 0.000 |
| Evidence responsiveness | 1.000 |
| Citation integrity | 1.000 |
| Safe abstention rate | 0.000（本组未配置必须退让目标） |
| Regression rate | 1.000 |

解释：检索证据确实随 original/modified 文档改变，引用仍完整，但模型没有稳定产出随证据改变的最终答案。因此主要问题不再是证据 ID 统计，而是最终决策/答案对反事实证据不敏感。

## 7. 实验中修复的工程问题

1. OpenRouter thinking 兼容：把本地 `enable_thinking/thinking_budget` 转换为官方统一 `reasoning.enabled/reasoning.max_tokens`，并设置 `exclude=true`，不接收或保存隐藏思考正文。
2. reasoning token 遥测：安全聚合 `completion_tokens_details.reasoning_tokens`，不记录 prompt、原始响应或推理内容。
3. 答案契约：选择题注入选项，标签/文本题加入不含 gold 的 `Final answer` 输出约束。
4. 可解析性统计：无答案 gold 的数据集不再计入 unparseable。
5. 反事实证据身份：跨隔离运行使用稳定 document key，不再比较会重置的 `K1/K2`。
6. 反事实期望：显式传入 `evidence_should_change=true`。
7. 成本完整性：新增 `counterfactual_cost`，单独统计变体案例的调用、token、阶段、检索与墙钟；本次正式 8 案例基线生成在该修复之前，故只把主案例精确 token 作为正式数值。
8. telemetry 完整性：只有 provider reported calls 与 observed calls 相等时才标记完整，并新增 `unreported_calls`。

## 8. 风险与改进优先级

### P0：Faithfulness 决策不响应

- 证据变了、引用也完整，但最终答案不稳定变化。
- 建议把“最终短答案/标签”升级为一等结构字段，不再依赖普通 claim 文本；verifier 应同时核验答案字段与证据跨度。
- 将 Faithfulness 反事实集设为发布门禁：decision responsiveness < 1 或 regression > 0 时阻止上线。

### P0：EvidenceBench 检索召回过低

- Recall@5=0.0893，明显低于其他数据集。
- 建议做 Top-K（5/20/50）、chunk/sentence 粒度、query expansion、pool-aware rerank 和 embedding 模型消融。
- 当前 MRR 0.2447 表明首个相关证据偶尔靠前，但多证据覆盖严重不足。

### P1：Evidence Inference 端到端退让与延迟

- 8/8 人工审核，引用/支持覆盖 0.5，未解决率 0.4792。
- 平均 161.5 秒、最大 316.5 秒，是最慢数据集。
- 建议加入 PICO 专用查询模板、证据跨度检索与三分类结构化答案，减少通用规划/修复成本。

### P1：答案结构应成为生产契约

- 目前 benchmark 通过 claim 文本提取答案，只是兼容修复。
- 建议在 `SynthesisPayload` 增加可选 `final_answer` / `answer_type` / `confidence`，由 verifier 直接核验，避免自然语言解析。

### P1：吞吐与成本

- 48 主案例 + 16 反事实变体耗时 67.1 分钟。
- 建议只在 plan/synthesize/judge 的高风险路径开启 thinking；query/extract 保持非 thinking。
- 可按数据集/案例安全并行，但应保留全局 provider 并发预算并报告 p50/p95，而不只均值。

### P2：可复现环境与密钥

- 增加 constraints/lock 文件；统一项目 Python 启动方式。
- 把 `model.local.json` 中的实际密钥迁移到环境变量或外部密钥管理器。
- 增加浏览器 E2E 和真实 SSE 长响应测试。

## 9. 复现命令

以下命令从仓库根目录执行；数据集实验使用随附 Python（包含 FAISS、NumPy、pyarrow）。

```powershell
$py = 'C:\Users\chuzhaole\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'

# 单元测试（系统 Python 已安装本项目）
python -m unittest discover -s tests -v

# 审计与合成冒烟
& $py scripts/audit_repository.py --out data/evaluation/reports/2026-07-30-repository-audit.json
& $py scripts/run_evaluation.py --out data/evaluation/reports/2026-07-30-smoke-evaluation.json

# 128 案例/数据集 FAISS 检索
& $py scripts/run_dataset_evaluation.py --dataset all --split test --max-cases 128 `
  --top-k 5 --mode retrieval --model-source environment `
  --model-profile openrouter-deepseek --retrieval-backend faiss `
  --out data/evaluation/reports/2026-07-30-openrouter-deepseek-faiss-retrieval-all-128.json

# 8 案例/数据集真实 agent 性能基线
& $py scripts/run_dataset_evaluation.py --dataset all --split test --max-cases 8 `
  --top-k 5 --mode agent --model-source environment `
  --model-profile openrouter-deepseek --retrieval-backend faiss `
  --max-repair-rounds 1 `
  --out data/evaluation/reports/2026-07-30-openrouter-deepseek-thinking2048-faiss-agent-all-8.json
```

## 10. 结果文件

- `data/evaluation/reports/2026-07-30-repository-audit.json`
- `data/evaluation/reports/2026-07-30-smoke-evaluation.json`
- `data/evaluation/reports/2026-07-30-openrouter-deepseek-faiss-retrieval-all-128.json`
- `data/evaluation/reports/2026-07-30-openrouter-deepseek-thinking2048-faiss-agent-all-8.json`
- `data/evaluation/reports/2026-07-30-openrouter-deepseek-thinking2048-faiss-agent-all-2-answer-contract.json`
- `data/evaluation/reports/2026-07-30-openrouter-deepseek-faithfulness-2-counterfactual-fixed.json`
- 本报告：`evaluation/reports/2026-07-30-openrouter-deepseek-full-experiment.md`

原始数据和 JSON 结果位于 Git 忽略目录；Markdown 报告只包含安全聚合指标。
