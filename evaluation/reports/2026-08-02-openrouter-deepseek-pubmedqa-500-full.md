# PubMedQA 官方 test 全量实验报告

## 结论

本次实验覆盖官方 PubMedQA `test` 的 **500/500 条 QA**，共 100 个 5-case 窗口，全部生成当前版本的 case-level 结果；FAISS 检索与 agent 推理均实际执行。

模型答案效果偏低：500 条可评分样本中，380 条得到可解析标签，187 条正确。整体准确率为 **37.40%**，已解析样本的选择性准确率为 **49.21%**，答案覆盖率为 **76.00%**。

## 实验配置

- 数据集：PubMedQA 官方 `test`，500 条。
- 模型：`openrouter-deepseek` profile，即 `deepseek/deepseek-v4-flash`；这是根据“换成 openrouter-deepseek”后的实际配置，不是 SiliconFlow 的 Qwen 3.5 4B profile。
- Thinking：开启；thinking budget 2048；模型输出上限 16384；stream 开启。
- 检索：FAISS，`top_k=5`；embedding 使用 BGE-M3，启用 reranker。
- Agent：`max_repair_rounds=1`；benchmark 末端增加结构化 `yes/no/maybe` synthesis 阶段；不使用 gold label 回填。
- 执行：先四路并发跑 5-case 分片，随后以两路低并发补齐旧版本/缺失窗口；失败窗口保留为错误状态，不伪造成功结果。

主要执行命令（分片的 `--case-offset` 为 `0,5,...,495`）：

```powershell
python scripts/run_dataset_evaluation.py `
  --dataset pubmedqa --split test --max-cases 500 `
  --case-offset <offset> --case-limit 5 `
  --top-k 5 --mode agent --model-source environment `
  --model-profile openrouter-deepseek --retrieval-backend faiss `
  --max-repair-rounds 1 --out <chunk-json> --quiet
```
## 结果

### Agent 状态与答案

| 指标 | 结果 |
| --- | ---: |
| cases attempted | 500 |
| passed | 331 |
| needs human review | 54 |
| plan rejected | 5 |
| provider/error | 110 |
| scorable cases | 500 |
| evaluated/parsible labels | 380 |
| unparseable / unavailable | 120 |
| correct | 187 |
| coverage | 76.00% |
| selective accuracy | 49.21% |
| overall accuracy | 37.40% |

按 gold label 的整体准确率：`yes` 50.00%（101/202），`no` 41.04%（55/134），`maybe` 57.41%（31/54）。主要混淆是模型过度输出 `maybe`：`yes→maybe` 91 条、`no→maybe` 63 条。

### FAISS 检索（500/500）

检索单独评测无失败、无 invalid case：

| 指标 | 结果 |
| --- | ---: |
| Recall@1 | 0.311341 |
| Recall@3 | 0.729338 |
| Recall@5 | 0.792789 |
| MRR | 0.987667 |
| nDCG@5 | 0.838822 |
| embedding failures | 0 |

Recall@5 约 0.793，意味着约 20.7% 的 gold 相关片段没有进入前 5 候选；这是答案质量的主要上游风险之一。

### 证据与引用质量

在有 workflow 质量记录的 390 条 case 上：citation coverage、citation precision、support-edge coverage 的 case 平均均为 **0.992208**；claim-level citation coverage 为 **1.0**。2660 个 claims 中 2607 个 `SUPPORTED`、14 个 `PARTIALLY_SUPPORTED`、39 个 `INSUFFICIENT`；claim-level unresolved rate 为 **2.594%**，contradiction rate 为 **0**。

### 延迟与模型用量

case-level 延迟（不把 FAISS 建库等待重复计入 case latency）：

- 全部 500 条：平均 102.06 s，P50 69.76 s，P95 301.18 s，P99 704.93 s，最大 798.32 s。
- 非 error 的 390 条：平均 124.46 s，P50 88.55 s，P95 345.53 s，P99 712.58 s，最大 798.32 s。

模型 telemetry 汇总：观察到 3066 个逻辑调用，provider 报告 3051 个调用；总 token 3,696,325（输入 2,304,640，输出 1,391,685，reasoning 125,311）。75/100 个窗口 telemetry 完整；其余窗口因 provider 中断产生 114 个未报告调用。

错误类型为 108 个 `ModelProviderError`、1 个 `SSLError`、1 个 `RemoteDisconnected`。错误主要集中在四路并发启动后的早期窗口（010–034、130–159、255–284、385–414），这个分布与 provider 限流/连接不稳定相符，但仅凭脱敏结果不能断言具体 HTTP 状态码。

## 效果不佳时的解决方案

1. **先解决可用性（P0）**：对 429、5xx、连接断开和请求超时增加带上限的指数退避重试；保留 case checkpoint，只重跑失败 case；默认并发降到 1–2。当前 110 个 error 会直接压低整体准确率和覆盖率。
2. **再解决标签决策（P0）**：把“证据抽取”和“最终标签判定”拆成独立短分类器，严格只输出 `yes/no/maybe`，并用一小段校准集调节 `maybe` 阈值。当前最明显的问题是 `yes/no` 被过多判成 `maybe`。
3. **提升检索召回（P1）**：将候选从 top-5 提到 10–20，使用 lexical+FAISS hybrid，按文章聚合片段后再 rerank；目标是先把 Recall@5 从 0.793 提高，再观察答案准确率变化。
4. **降低长尾延迟（P1）**：持久化 FAISS/embedding cache，减少重复建库；对 plan/query/extract/judge/synthesis 分别设置预算，只有需要复杂推理的阶段开启 thinking；为单 case 设置总 deadline。
5. **建立发布门槛（P2）**：同时要求 overall accuracy、coverage、provider error rate 和 citation/support 质量，不能只看已解析样本的 selective accuracy。

## 复现与产物

- Agent 聚合结果：`data/evaluation/reports/2026-08-02-openrouter-deepseek-pubmedqa-test-500-agent-v2-full.json`
- FAISS 检索结果：`data/evaluation/reports/2026-07-30-openrouter-deepseek-pubmedqa-test-500-faiss-retrieval.json`
- 分片目录：`data/evaluation/reports/pubmedqa-test-500-agent-v2-chunks/`
- 聚合脚本：`scripts/aggregate_pubmedqa_chunks.py`
- 主要修复：`src/medical_agent/dataset_evaluation.py`、`src/medical_agent/prompting.py`，以及对应的 `tests/test_dataset_evaluation.py`。

聚合命令：

```powershell
python scripts/aggregate_pubmedqa_chunks.py `
  --chunk-root data/evaluation/reports/pubmedqa-test-500-agent-v2-chunks `
  --out data/evaluation/reports/2026-08-02-openrouter-deepseek-pubmedqa-test-500-agent-v2-full.json `
  --expected-cases 500
```
