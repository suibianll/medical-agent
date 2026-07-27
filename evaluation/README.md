# 评测数据集与运行说明

本目录把“数据集来源、访问条件、可评指标”和“可离线运行的最小回归集”分开管理：

- `datasets.json` 是不含数据正文的元数据清单。它只记录官方入口、论文、规模提示、访问级别和指标映射；不会自动下载数据。
- `smoke_cases.json` 是合成案例，不能代表真实临床效果，也不含真实患者信息。
- `scripts/run_evaluation.py` 通过项目现有的 `search_many`、证据链校验和质量指标运行冒烟评测。
- `scripts/audit_repository.py` 是只读的仓库健康/安全审计入口。
- `download_sources.json` 和 `scripts/download_evaluation_datasets.py` 负责可复现地下载公开评测资产；原始数据只写入被 Git 忽略的 `data/evaluation/`。

## 快速开始

在仓库根目录执行：

```powershell
python scripts/run_evaluation.py
python scripts/run_evaluation.py --out evaluation/reports/smoke-latest.json
python scripts/audit_repository.py --out evaluation/reports/repository-audit.json
```

## 下载评测数据

下载器不会把原始数据提交到仓库，也不会绕过注册、credential、培训或 DUA。每个文件会记录大小和 SHA-256 到本地 `data/evaluation/download-manifest.json`；重复执行会复用已有文件。

```powershell
$python = 'C:\Users\chuzhaole\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $python scripts/download_evaluation_datasets.py `
  --dataset pubmedqa `
  --dataset evidencebench `
  --dataset ragchecker `
  --dataset faithfulness-qa-2026 `
  --dataset evidence-inference-2 `
  --dataset pubmedqa-hf-mirror `
  --dataset pubmedqa-hf-a-mirror `
  --dataset medmcqa-hf-mirror `
  --dataset medqa-usmle-hf-qa-mirror `
  --allow-terms-check
```

当前本地已下载：PubMedQA PQA-L、PQA-U、EvidenceBench、RAGChecker、Faithfulness-QA、Evidence Inference 2.0，以及 MedMCQA 和 MedQA 的 Hugging Face 非官方镜像。PQA-A 镜像和 MedQA textbook 全量镜像仍需在可访问 Hugging Face 网络中补齐；镜像只作为评测便利，不等同于官方发布。`download_sources.json` 保留官方仓库、镜像地址和条款提示。MedMCQA/MedQA 官方 Google Drive 入口若在当前网络不可达，应改在可访问网络中运行对应的 `medmcqa`/`medqa-usmle` 下载项，并再次核对题库与教材的再分发许可。

建议把各 split 分开挂载：EvidenceBench 的 train/dev/test、Faithfulness-QA 的 train/dev/test 和 MedMCQA 的 train/validation/test 不得混入同一个测试知识库。镜像文件在接入自动评测前还应检查字段映射和版本漂移。

冒烟评测不调用模型、不联网、不读取真实病历，输出只包含计数和指标。若使用工作区自带 Python：

```powershell
$python = 'C:\Users\chuzhaole\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $python scripts/run_evaluation.py
```

## 首批推荐接入顺序

1. **PubMedQA**：公开仓库包含 JSON 和评测脚本，可先验证回答准确率，再把摘要/候选文档接入 Recall@K、MRR 和引用覆盖率。
2. **BioASQ**：同时有 articles、snippets、RDF triples 和 exact/ideal answers，适合验证“检索 → 证据跨度 → 答案”的完整链路；训练数据按官方页面注册获取。
3. **Evidence Inference 2.0**：把 PICO 研究问题和论文证据跨度映射到支持/矛盾/证据不足分类，直接覆盖 claim verifier。
4. **EvidenceBench 原始 test**：作为大规模生物医学证据检索的外部回归集；训练规模版本不能直接当测试集。
5. **RAGChecker benchmark + Faithfulness-QA**：前者覆盖 RAG 相关性/忠实性，后者覆盖实体替换反事实，分别对应引用质量和“证据变化后结论是否响应”。
6. **MedNLI / MIMIC-IV**：只在完成 PhysioNet credential、培训和 DUA 后接入授权环境；仓库禁止提交原始文本。
7. **HealthSearchQA**：公开问题没有稳定的 gold answer/reference，必须采用双人专家 rubric，不能用 exact match 冒充自动评测。

## 数据集清单字段

每条记录至少包含 `id`、`task`、`role`、`access`、`official_url`、`citation_url`、`metrics` 和 `notes`。`access` 的含义为：

| 级别 | 含义 | 仓库策略 |
| --- | --- | --- |
| `downloadable` | 可从公开官方入口下载 | 只在本地缓存，提交版本号/哈希与指标 |
| `public_dev_registration_train` | 开发集公开，训练集需注册 | 先用 dev 做回归，记录 challenge/year |
| `registration` / `credentialed` | 需要注册、培训、DUA 或 credential | 不入库、不进 CI，授权环境运行 |
| `human_only` | 缺少稳定 gold answer/reference | 使用专家 rubric，保留盲评协议与一致性 |
| `paper_access` / `terms_check` | 需要按论文/仓库条款确认 | 接入前核对许可证、可再分发范围和版本 |

## 统一评测契约

接入真实数据集时，不要修改应用工作流来适配某个数据集。先写一个 dataset adapter，把样本转换成以下内部结构：

```json
{
  "id": "stable-case-id",
  "query": "question or task text",
  "relevant_ids": ["corpus-document-id"],
  "gold_answer": "optional answer label",
  "gold_spans": ["optional span id"],
  "metadata": {
    "dataset": "pubmedqa",
    "split": "test",
    "version": "2026-..."
  }
}
```

检索评测使用 `medical_agent.quality.evaluate_retrieval_cases`，报告 Recall@K、MRR、nDCG@K；证据链使用 `evaluate_evidence_chain`，报告 citation coverage/precision、support-edge coverage、dual-support、矛盾率和证据不足率；反事实回归使用 `evaluate_counterfactual_cases`，报告 decision/evidence responsiveness、citation integrity、safe abstention 和 regression rate。

## 防止数据泄漏与错误结论

- 固定每个数据集的 `dataset_id + version + split + checksum`；只把 metadata 和结果摘要提交到仓库。
- 训练/调参/测试语料分离，特别是不要把 BioASQ train、MedMCQA train 或 EvidenceBench-100k 混入测试知识库。
- 每个结果必须带检索后端、reranker 配置、模型身份、温度/seed、候选预算和时间戳；模型调用统计不记录查询、病历或证据正文。
- 把 `HealthSearchQA`、MIMIC、MedNLI 的人工/受限结果与公开自动分数分开，不用单一总分掩盖数据访问差异。
- 任何安全门禁（急症升级、无法判断时拒答、引用完整性）都应以失败率和案例清单为主，不以平均答案分数替代。

## 官方入口（清单摘要）

- [PubMedQA](https://github.com/pubmedqa/pubmedqa)；[MedMCQA](https://github.com/medmcqa/medmcqa)；[MedQA](https://github.com/jind11/MedQA)
- [BioASQ 数据页面](https://participants-area.bioasq.org/datasets/)；[Evidence Inference 下载页](https://evidence-inference.ebm-nlp.com/download/)
- [EvidenceBench](https://github.com/EvidenceBench/EvidenceBench)；[RAGChecker](https://github.com/amazon-science/RAGChecker)
- [Faithfulness-QA](https://github.com/qzhangFDU/faithfulness-qa-dataset)；[MedNLI](https://physionet.org/content/mednli/1.0.0/)
- [MIMIC-IV](https://www.physionet.org/content/mimiciv/1.0/)；[HealthSearchQA 论文](https://arxiv.org/abs/2212.13138)；[BRIDGE](https://arxiv.org/abs/2504.19467)
