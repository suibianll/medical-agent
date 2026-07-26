# 2026 Evidence-Chain and Medical RAG Intelligence Summary

## Scope
- Domain: 2026 Evidence-Chain and Medical RAG
- Generated on: 2026-07-26

## Executive Summary
- Collected papers: 21
- Collected news items: 7
- Paper download success: 14
- Paper download failed: 7
- Top venues/sources: arXiv (5), AAAI 2026 (2), Cell Reports Medicine (1), ACL 2026 (1), ACL CustomNLP4U 2026 (1)

## New Papers
| Date | Title | Venue | Why it matters | Link |
|---|---|---|---|---|
| 2026-07-20 | Retrieval-augmented generation in medicine: A scoping review of technical implementations, clinical applications, and ethical considerations | Cell Reports Medicine | 最新医学 RAG 范围综述指出临床验证、跨语言适配、低资源环境、安全和偏差评估仍明显不足。 | https://pubmed.ncbi.nlm.nih.gov/42476143/ |
| 2026-07 | LLM-Generated Text May Harm Your Retrieval! A Robust Detection Strategy for Retrieval-Augmented Generation | ACL 2026 | 说明外部语料被模型生成内容污染会持续降低检索可靠性，来源治理需要增加内容来源和污染检测。 | https://aclanthology.org/2026.acl-long.1475/ |
| 2026-07-10 | Evaluating and Guarding Citation Faithfulness in Agentic Scientific Synthesis | arXiv | 同一输出的不支持引用率会因验证器严格度从约 3% 变化到约 18%，提出金标准校准和 conformal guard。 | https://arxiv.org/abs/2607.20527 |
| 2026-07-03 | Efficiency vs. Verifiability in Evidence-Aware RAG: Does Prompt Compression Preserve Citation Grounding? | ACL CustomNLP4U 2026 | 表明压缩上下文可能保持答案质量却显著破坏引用 grounding，成本优化必须加入引用保持指标。 | https://aclanthology.org/2026.customnlp4u-1.19/ |
| 2026-07 | Medical multi-recall embedding: Adaptive retrieval for diverse evidence in medical RAG systems | Information Processing & Management | 一对多医学检索在五个专科平均提升 Recall@10 超过 10 个百分点，并使下游 QA 至少提升 4 个百分点。 | https://www.sciencedirect.com/science/article/pii/S0306457326000865 |
| 2026-06-17 | BRIDGE: benchmarking large language models for understanding real-world clinical practice texts | Nature Biomedical Engineering | 以 59 个真实临床数据源、9 种语言、87 项任务评估 95 个模型，显示模型、语言、专科和任务间差异巨大。 | https://www.nature.com/articles/s41551-026-01719-2 |
| 2026-06-16 | When silence is safer: a review and decision-theoretic framework for LLM abstention in healthcare | npj Digital Medicine | 提出 MedSAFE 和基于潜在伤害的决策论拒答框架，强调回答、澄清、转诊和拒答不应合并成一个置信度。 | https://www.nature.com/articles/s41746-026-02882-1 |
| 2026-06-12 | General-purpose large language models outperform specialized clinical AI tools on medical benchmarks | Nature Medicine | 独立盲评显示专用临床工具并不天然优于前沿通用模型，证明真实临床问题和独立评估比产品标签更重要。 | https://www.nature.com/articles/s41591-026-04431-5 |
| 2026-06 | MR-RAG: Multimodal Relevance-Aware Retrieval-Augmented Generation for Medical Visual Question Answering | CVPR 2026 | 在检索与生成两阶段同时建模跨模态相关性，在三个医学视觉 QA 数据集上最高提升 6.4%。 | https://openaccess.thecvf.com/content/CVPR2026/html/Li_MR-RAG_Multimodal_Relevance-Aware_Retrieval-Augmented_Generation_for_Medical_Visual_Question_Answering_CVPR_2026_paper.html |
| 2026-05-07 | AgenticRAG: Agentic Retrieval for Enterprise Knowledge Bases | arXiv | 将 search/find/open/summarize 变成受控迭代工具，报告 BRIGHT Recall@1 比最佳 embedding 基线高 21.8 个百分点。 | https://arxiv.org/abs/2605.05538 |
| 2026-05-28 | CRITIC-R1: Learning Structured Critics for Retrieval-Augmented Generation | arXiv | 把 RAG 批评建模为显式错误诊断并以过程级监督学习，可用于定位检索、归因和推理故障。 | https://arxiv.org/abs/2605.29886 |
| 2026-05-15 | CRITIC-RAG: Knowledge-Augmented Large Language Models With Verified Retrieval for Improved Medical Reasoning | IEEE Journal of Biomedical and Health Informatics | 小型独立 verifier 贯穿选择性检索、过滤、自一致推理和 groundedness 核验；LLaMA-3 的 MedQA 从 41.3% 升至 48.8%。 | https://doi.org/10.1109/JBHI.2026.3687666 |
| 2026-05-25 | Reason and Verify: A Framework for Faithful Retrieval-Augmented Generation | Canadian AI / PMLR 318 | 结合查询改写、交叉编码重排、子结论到精确片段绑定和八类忠实度诊断，在 BioASQ/PubMedQA 验证。 | https://proceedings.mlr.press/v318/khan26a.html |
| 2026-03-30 | Evaluating large language models for evidence-based clinical question answering | Patterns | 两万多问答显示结构化指南准确率 90%，系统综述仅 50%–60%；模型对模糊和弱证据普遍过度自信。 | https://pmc.ncbi.nlm.nih.gov/articles/PMC13161685/ |
| 2026-03-14 | Look as You Think: Unifying Reasoning and Visual Evidence Attribution for Verifiable Document RAG via Reinforcement Learning | AAAI 2026 | 用页码和 bounding box 把推理步骤绑定到视觉证据区域，平均提升 soft EM 8.23% 和 IoU@0.5 47.0%。 | https://ojs.aaai.org/index.php/AAAI/article/view/40488 |
| 2026-03-14 | Faithful in Steps: Improving Generalization and Citation in RAG via Query Decomposition | AAAI 2026 | 将复杂问题拆成原子子问题并重排去除干扰，减少多跳与多模态任务中的过度引用和隐式实体遗漏。 | https://ojs.aaai.org/index.php/AAAI/article/view/40879 |
| 2026-03-01 | Tiny-Critic RAG: Empowering Agentic Fallback with Parameter-Efficient Small Language Models | arXiv | 以小模型做确定性检索门控，报告路由准确率接近 GPT-4o-mini、延迟低一个数量级，适合控制 verifier 成本。 | https://arxiv.org/abs/2603.00846 |
| 2026-02-07 | Agentic memory-augmented retrieval and evidence grounding for medical question-answering tasks | International Journal of Medical Informatics | 把检索、重排、证据 grounding、诊断工具和 cache-and-prune memory 合并为动态多步医学 Agent。 | https://pubmed.ncbi.nlm.nih.gov/41713127/ |
| 2026-01-26 | Multi-Evidence Clinical Reasoning With Retrieval-Augmented Generation for Emergency Triage: Retrospective Evaluation Study | JMIR Medical Informatics | 指南与 3000 个历史病例双源检索使 236 个真实分诊病例准确率从 0.542 升至 0.802，并将过度分诊从 28.8% 降至 12.7%。 | https://medinform.jmir.org/2026/1/e82026/ |
| 2026-01-10 | MedRAGChecker: Claim-Level Verification for Biomedical Retrieval-Augmented Generation | arXiv | 以原子 claim、医学 NLI 和知识图谱一致性区分忠实度不足、证据不足、矛盾和安全关键错误。 | https://arxiv.org/abs/2601.06519 |
| 2026 | Web Search Is Not Enough: LLMs Fail to Account for Paper Retraction Status | OpenReview / EMNLP 2026 submission | 显示仅进行网页检索仍可能把撤稿论文当作有效证据，来源生命周期必须独立建模。 | https://openreview.net/forum?id=yvivK8FdVI |

## Industry / Policy News
| Event date | Organization | Update | Impact | Link |
|---|---|---|---|---|
| 2026-07-15 | WHO Europe | 37 国讨论健康 AI 治理；WHO 调查显示近三分之二国家已在诊断中部署 AI，但只有 8% 有健康专用 AI 战略和责任标准。 | 证据链产品需要责任归属、人工门控、互操作和部署后监测，而不仅是模型准确率。 | https://www.who.int/europe/news/item/15-07-2026-who-brings-37-countries-together-in-lisbon-to-get-ai-governance-right-and-make-it-work-for-every-patient |
| 2026-07-14 | FDA | FDA 就非医疗器械软件功能及患者安全征求意见，覆盖有限临床决策支持。 | 需要把有限 CDS 的使用边界、用户能力要求、风险和教育材料写进产品与审计设计。 | https://www.fda.gov/about-fda/cdrh-reports/reports-non-device-software-functions |
| 2026-06-03 | WHO | WHO 东南亚区域与 Jhpiego 合作建立数字健康和健康 AI 的证据、实施科学、评估和政策转化能力。 | 真实世界实施研究和可测量临床结果成为从试验走向部署的必要条件。 | https://www.who.int/southeastasia/news/detail/03-06-2026-who-searo-and-jhpiego-collaborate-on-evidence-and-implementation-of-digital-health-and-ai-in-health-systems |
| 2026-06-02 | WHO | WHO 发布 AI 与循证健康政策讨论文件，要求 living evidence、自动检索加人工核验和多学科监督。 | 支持把证据版本、人工决策点和持续更新建成一等数据结构。 | https://www.who.int/news/item/02-06-2026-new-who-discussion-paper-sets-out-opportunities-and-risks-of-ai-in-evidence-informed-health-policy |
| 2026-04-07 | NIST | NIST 建设嵌入 Agent 工作流的评估 probes，对每个文档片段和引用执行严格 rubric 核验并生成机器可读审计轨迹。 | 直接支持逐 claim/逐 citation 的外部验证器和可复现实验记录。 | https://www.nist.gov/programs-projects/building-evaluation-probes-agentic-ai |
| 2026-01 | FDA | FDA 发布 Clinical Decision Support Software 最终指南，澄清 Non-Device CDS 与设备软件边界。 | 系统必须让医疗专业人员能够独立审查建议依据，并明确产品用途与用户。 | https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software |
| 2026-07 | ACL | 首届 RAG4Reports 研讨会集中讨论多语言报告生成、引用和 RAG 评估。 | 多语言、报告级引用覆盖和任务特异评估已成为独立研究方向。 | https://aclanthology.org/volumes/2026.rag4reports-1/ |

## Download Status
- Success: 14
- Failed: 7
- Failed items:
  - Medical multi-recall embedding: Adaptive retrieval for diverse evidence in medical RAG systems: no public PDF URL recorded (https://www.sciencedirect.com/science/article/pii/S0306457326000865)
  - CRITIC-RAG: Knowledge-Augmented Large Language Models With Verified Retrieval for Improved Medical Reasoning: no public PDF URL recorded (https://doi.org/10.1109/JBHI.2026.3687666)
  - Faithful in Steps: Improving Generalization and Citation in RAG via Query Decomposition: publisher access, timeout, or unavailable public PDF (https://ojs.aaai.org/index.php/AAAI/article/view/40879)
  - Agentic memory-augmented retrieval and evidence grounding for medical question-answering tasks: publisher access, timeout, or unavailable public PDF (https://pubmed.ncbi.nlm.nih.gov/41713127/)
  - Multi-Evidence Clinical Reasoning With Retrieval-Augmented Generation for Emergency Triage: Retrospective Evaluation Study: publisher access, timeout, or unavailable public PDF (https://medinform.jmir.org/2026/1/e82026/)
  - MedRAGChecker: Claim-Level Verification for Biomedical Retrieval-Augmented Generation: publisher access, timeout, or unavailable public PDF (https://arxiv.org/abs/2601.06519)
  - Web Search Is Not Enough: LLMs Fail to Account for Paper Retraction Status: publisher access, timeout, or unavailable public PDF (https://openreview.net/forum?id=yvivK8FdVI)

## Signals to Watch Next
- New benchmark papers that compare multimodal and hybrid retrieval routing.
- Product releases exposing reranker and routing controls for multi-route retrieval.
- Regulation and compliance updates affecting retrieval data provenance.
