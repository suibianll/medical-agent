# 证据中心医疗 Agent / 医疗 RAG Intelligence Summary

## Scope
- Domain: 证据中心医疗 Agent / 医疗 RAG
- Generated on: 2026-08-03

## Executive Summary
- Collected papers: 20
- Collected news items: 8
- Paper download success: 0
- Paper download failed: 0
- Top venues/sources: arXiv (7), Findings of ACL 2026 (4), arXiv / AAAI FrontierIR 2026 (1), arXiv scoping review (1), npj Digital Medicine (1)

## New Papers
| Date | Title | Venue | Why it matters | Link |
|---|---|---|---|---|
| 2026-07-28 | RAGuard: A Layered Defense Framework for Retrieval-Augmented Generation Systems Against Data Poisoning | arXiv / AAAI FrontierIR 2026 | 当前系统允许导入外部文本但只有元数据准入；该工作支持增加语料投毒红队、异常检测和高风险片段的反事实影响测试。 | https://arxiv.org/abs/2607.26339 |
| 2026-07-28 | PatientAgentBench: A Benchmark Framework for Evaluating Patient-Facing Health AI Agents | arXiv | 1200 个持续对话工具场景显示静态 QA 无法暴露分诊、工具结果核验、危机资源遗漏和虚构动作等故障。 | https://arxiv.org/abs/2607.25485 |
| 2026-07-28 | Agentic AI in medicine: architectures, applications, evaluation, and challenges for clinical translation | arXiv scoping review | 系统映射 557 项研究，指出真实工作流、过程可靠性、证据追踪、安全、外部有效性和前瞻验证仍不足。 | https://arxiv.org/abs/2607.25489 |
| 2026-07-23 | Meaningful oversight of medical AI beyond human in the loop | npj Digital Medicine | 人工在环只有在具备知识、时间、决策权和真正可暂停/撤销的技术控制时才是有效安全机制。 | https://www.nature.com/articles/s41746-026-02971-1 |
| 2026-07-20 | Retrieval-augmented generation in medicine: A scoping review of technical implementations, clinical applications, and ethical considerations | Cell Reports Medicine | 医学 RAG 仍偏英文、公有数据和静态任务，对临床验证、跨语言、低资源、安全与偏差评测投入不足。 | https://pubmed.ncbi.nlm.nih.gov/42476143/ |
| 2026-07-15 | Addressing benchmarking gaps in large language models for health and medicine with dynamic red-teaming | Nature Health | 动态红队发现 94% 的原本正确答案在鲁棒性变体下失败，支持把隐私、偏差、幻觉和扰动变成持续门禁。 | https://www.nature.com/articles/s44360-026-00152-8 |
| 2026-07-15 | DS@GT ARC at LongEval: Citation Integrity and Factual Grounding in Scientific QA | arXiv / CLEF 2026 LongEval | 答案相关性和流畅性可与实际使用证据脱钩；生成前过滤和生成后严格蕴含核验更接近可信 RAG 目标。 | https://arxiv.org/abs/2607.14400 |
| 2026-07-10 | Evaluating and Guarding Citation Faithfulness in Agentic Scientific Synthesis | arXiv | 同一输出的不支持引用率可因 verifier 严格度从约 3% 变为约 18%，要求金标准校准、命名协议和置信界。 | https://arxiv.org/abs/2607.20527 |
| 2026-07-05 | Uncertainty-Aware Abstention in Large Language Models with Provable Alignment Guarantees | arXiv | 将任意不确定性分数转换为有限样本风险受控的选择性回答规则，适合替代当前无校准的证据状态路由。 | https://arxiv.org/abs/2607.04430 |
| 2026-07-03 | Efficiency vs. Verifiability in Evidence-Aware RAG: Does Prompt Compression Preserve Citation Grounding? | ACL CustomNLP4U 2026 | 上下文压缩可能保住答案质量却损坏引用 grounding，成本优化必须同时守住证据指标。 | https://aclanthology.org/2026.customnlp4u-1.19/ |
| 2026-06-30 | HealthAgentBench: A Unified Benchmark Suite of Realistic Agentic Healthcare Environments for Challenging Frontier AI Agents | arXiv | 54 个端到端医疗环境任务中最强系统成功率仍约 42%，支持环境级、执行级而非只看文本答案的评测。 | https://arxiv.org/abs/2606.31179 |
| 2026-07 | SEMA-RAG: A Self-Evolving Multi-Agent Retrieval-Augmented Generation Framework for Medical Reasoning | Findings of ACL 2026 | 将临床语义解释、充分性驱动检索和证据裁决解耦，在五个 benchmark 和五个模型上平均提升 6.46 个准确率点。 | https://aclanthology.org/2026.findings-acl.917/ |
| 2026-07 | AgenticRAGTracer: A Hop-Aware Benchmark for Diagnosing Multi-Step Retrieval Reasoning in Agentic RAG | Findings of ACL 2026 | 逐跳诊断揭示过早收敛和过度延展；当前项目只有最终任务状态，缺少查询/证据/推理 hop 的正确性标注。 | https://aclanthology.org/2026.findings-acl.66/ |
| 2026-07 | Agentic-R: Learning to Retrieve for Agentic Search | Findings of ACL 2026 | 多轮检索器应同时优化局部相关性和最终答案正确性，而非只优化相似度；直接对应当前检索与答案脱钩。 | https://aclanthology.org/2026.findings-acl.785/ |
| 2026-07 | LLM-Generated Text May Harm Your Retrieval! A Robust Detection Strategy for Retrieval-Augmented Generation | ACL 2026 | 模型生成文本污染外部语料会降低长期检索质量，来源治理需要内容来源标记和污染检测。 | https://aclanthology.org/2026.acl-long.1475/ |
| 2026-07 | RIPRAG: Hack a Black-box Retrieval-Augmented Generation Question-Answering System with Reinforcement Learning | Findings of ACL 2026 | 黑盒反馈即可优化投毒文档并显著提高攻击成功率，说明只做来源元数据治理不足以覆盖导入知识风险。 | https://aclanthology.org/2026.findings-acl.833/ |
| 2026-05-04 | PhysicianBench: Evaluating LLM Agents in Real-World EHR Environments | arXiv | 100 个真实咨询改编的长程 EHR 任务平均需 27 次工具调用，最强模型 pass@1 仅 46%，强调执行检查点和结果状态验证。 | https://arxiv.org/abs/2605.02240 |
| 2026-05-13 | Reinforcement Learning for Tool-Calling Agents in Fast Healthcare Interoperability Resources (FHIR) | arXiv | FHIR 图上的执行约束和奖励训练将正确率从 50% 提至 77%，支持增加结构化 EHR 端口和确定性工具状态校验。 | https://arxiv.org/abs/2605.14126 |
| 2026-01-30 | MedMCP-Calc: Benchmarking LLMs for Realistic Medical Calculator Scenarios via MCP Integration | arXiv | 118 个真实计算场景显示模型在计算器选择、EHR 取数和 SQL 迭代上仍弱，临床数值任务需要代码工具而非文本推理。 | https://arxiv.org/abs/2601.23049 |
| 2026-06-01 | LabSage: Structural-Semantic Decoupling for Enhanced Retrieval-Augmented Generation in Clinical Laboratories | AMIA Joint Summits 2026 | 检索小单元、推理扩展上下文的分层架构比固定切块提高答案准确率 8.3%、上下文召回 5.7%。 | https://pubmed.ncbi.nlm.nih.gov/42317863/ |

## Industry / Policy News
| Event date | Organization | Update | Impact | Link |
|---|---|---|---|---|
| 2026-07-27 | European Commission | 更新 AI Act 实施指引；医疗软件等高风险系统的分类、透明度、严重事件、质量管理和部署后监测将由后续指南细化。 | 现在就应建立 intended purpose、风险分类、版本、严重事件、监测与停用记录，而非等正式临床部署后补文档。 | https://digital-strategy.ec.europa.eu/en/news/supporting-implementation-ai-act-clear-guidelines |
| 2026-07-22 | WHO SEARO | 数字健康与 AI 基层医疗准备度活动强调安全、可用性、真实实施、治理、数据标准和人员能力。 | 评测应覆盖工作流适配、可用性和组织准备度，不只覆盖模型指标。 | https://www.who.int/southeastasia/news/events/detail/2026/07/22/south-east-asia-events/digital-health-ai-readiness-in-primary-care-systems |
| 2026-07-15 | WHO Europe | 37 国讨论健康 AI 治理；近三分之二国家已在诊断中部署 AI，但仅 8% 有健康专用 AI 战略和责任标准。 | 项目需要明确责任、人工控制点、互操作、人员培训与部署后监测。 | https://www.who.int/europe/news/item/15-07-2026-who-brings-37-countries-together-in-lisbon-to-get-ai-governance-right-and-make-it-work-for-every-patient |
| 2026-07-14 | FDA | FDA 就非医疗器械软件功能及有限临床决策支持的患者安全、教育和能力要求征求意见，截止 2026-08-13。 | 产品边界、目标用户、独立审查能力、安全培训和风险说明需要成为运行配置和审计的一部分。 | https://www.fda.gov/about-fda/cdrh-reports/reports-non-device-software-functions |
| 2026-07-09 | WHO | WHO 就健康数字公共基础设施参考架构征求意见，覆盖两层架构、互操作、语义治理、符合性和测试。 | 知识库与病历输入应逐步从自由文本文件升级为可验证的标准化互操作层。 | https://www.who.int/news-room/articles-detail/call-for-public-comments-and-invitation-to-review |
| 2025-11-04 | National Health Commission of China | 关于促进和规范人工智能+医疗卫生应用发展的实施意见提出高质量数据集、可信数据空间、专科模型与智能体、安全可控和标准体系目标。 | 本项目面向中国应用时要补齐数据分级、质量、评测验证、应用边界和安全监测。 | https://www.nhc.gov.cn/guihuaxxs/c100133/202511/d1a42ae835c743b9b3e83ac0253c3e9f.shtml |
| 2026-05-19 | National Cybersecurity Standardization Technical Committee of China | 发布人工智能应用伦理安全指引 1.0，覆盖开发、服务提供和应用使用各方。 | 需要把伦理、安全、公平、用户权益与责任分工纳入产品生命周期检查表。 | https://www.cac.gov.cn/2026-05/22/c_1781191242686496.htm |
| 2026-01-30 | NIST CAISI | NIST AI 800-2 草案提出语言模型和 Agent 自动 benchmark 的有效性、透明度、可复现性、预算与显式停止条件实践。 | 每次评测需要固定模型、脚手架、预算、停止条件、重试策略、数据版本和统计协议。 | https://www.nist.gov/news-events/news/2026/01/towards-best-practices-automated-benchmark-evaluations |

## Download Status
- Success: 0
- Failed: 0

## Signals to Watch Next
- New benchmark papers that compare multimodal and hybrid retrieval routing.
- Product releases exposing reranker and routing controls for multi-route retrieval.
- Regulation and compliance updates affecting retrieval data provenance.
