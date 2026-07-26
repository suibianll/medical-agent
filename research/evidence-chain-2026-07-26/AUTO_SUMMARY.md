# 医疗 Agent 证据链准确性、推理可靠性与可视化 Intelligence Summary

## Scope
- Domain: 医疗 Agent 证据链准确性、推理可靠性与可视化
- Generated on: 2026-07-26

## Executive Summary
- Collected papers: 20
- Collected news items: 10
- Paper download success: 20
- Paper download failed: 0
- Top venues/sources: arXiv preprint (4), npj Digital Medicine (3), ICLR (3), EMNLP (2), NAACL (2)

## New Papers
| Date | Title | Venue | Why it matters | Link |
|---|---|---|---|---|
| 2026-02-04 | Synthesizing scientific literature with retrieval-augmented language models | Nature | OpenScholar combines a 45-million-paper store, bi-encoder retrieval, cross-encoder reranking, iterative self-feedback and citation verification; it is the strongest peer-reviewed blueprint found for long-form scientific synthesis. | https://www.nature.com/articles/s41586-025-10072-4 |
| 2026-07-10 | Evaluating and Guarding Citation Faithfulness in Agentic Scientific Synthesis | arXiv preprint | Shows citation-verifier conclusions vary materially with verifier strictness; proposes gold-anchored calibration and split-conformal bounds for unsupported citations. | https://arxiv.org/abs/2607.20527 |
| 2025-11-10 | Rethinking Retrieval-Augmented Generation for Medicine: A Large-Scale, Systematic Expert Evaluation and Practical Insights | arXiv preprint | 18 medical experts and 80,502 annotations show naive RAG can degrade medical answers; evidence filtering and query reformulation are high-value interventions. | https://arxiv.org/abs/2511.06738 |
| 2025-11 | HydraRAG: Structured Cross-Source Enhanced Large Language Model Reasoning | EMNLP | Combines graph topology, document semantics and source reliability with cross-source corroboration and entity-path alignment for multi-hop verification. | https://aclanthology.org/2025.emnlp-main.730/ |
| 2025-07 | Medical Graph RAG: Evidence-based Medical Large Language Model via Graph Retrieval-Augmented Generation | ACL | Medical GraphRAG architecture links private user documents to credible medical sources and balances top-down precise retrieval with bottom-up refinement. | https://aclanthology.org/2025.acl-long.1381/ |
| 2025-04 | Rationale-Guided Retrieval Augmented Generation for Medical Question Answering | NAACL | RAG-squared uses rationale-guided query reformulation, distractor filtering and balanced retrieval across four biomedical corpora. | https://aclanthology.org/2025.naacl-long.635/ |
| 2025-11-18 | Uncertainty-aware large language models for explainable disease diagnosis | npj Digital Medicine | ConfiDx aligns models to diagnostic criteria and explicitly identifies and explains missing-evidence uncertainty; supports criterion-level abstention. | https://www.nature.com/articles/s41746-025-02071-6 |
| 2025-01-14 | Large Language Models lack essential metacognition for reliable medical reasoning | Nature Communications | MetaMedQA shows strong medical-answer accuracy does not imply reliable self-assessment; models remain overconfident on unanswerable or malformed questions. | https://www.nature.com/articles/s41467-024-55628-6 |
| 2025-03-17 | Large language model agents can use tools to perform clinical calculations | npj Digital Medicine | Task-specific deterministic clinical calculators sharply reduce calculation errors compared with unaided generation, RAG or generic code execution. | https://www.nature.com/articles/s41746-025-01475-8 |
| 2025-03-04 | Evaluating base and retrieval augmented LLMs with document or online support for evidence based neurology | npj Digital Medicine | Guideline-grounded document RAG improved correctness and citations, but case-based questions remained harder and harmful answers were not eliminated. | https://www.nature.com/articles/s41746-025-01536-y |
| 2025 | Scaling LLM Test-Time Compute Optimally Can be More Effective than Scaling Parameters for Reasoning | ICLR | Adaptive allocation of inference compute and process-verifier-guided search can outperform uniform best-of-N, especially when matched to task difficulty. | https://proceedings.iclr.cc/paper_files/paper/2025/hash/1b623663fd9b874366f3ce019fdfdd44-Abstract-Conference.html |
| 2024-09-10 | Language agents achieve superhuman synthesis of scientific knowledge | arXiv preprint | PaperQA2 provides an agentic blueprint for full-text literature search, cited synthesis and contradiction detection with expert evaluation. | https://arxiv.org/abs/2409.13740 |
| 2024 | RAGChecker: A Fine-grained Framework for Diagnosing Retrieval-Augmented Generation | NeurIPS Datasets and Benchmarks | Claim-level entailment diagnostics separate retrieval failures from generation failures and correlate better with human judgement than coarse scores. | https://proceedings.neurips.cc/paper_files/paper/2024/hash/27245589131d17368cccdfa990cbf16e-Abstract-Datasets_and_Benchmarks_Track.html |
| 2024-06 | ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems | NAACL | Evaluates context relevance, answer faithfulness and answer relevance using lightweight judges calibrated with a small human-labelled set. | https://aclanthology.org/2024.naacl-long.20/ |
| 2024-11 | Model Internals-based Answer Attribution for Trustworthy Retrieval-Augmented Generation | EMNLP | MIRAGE attributes context-sensitive answer tokens to contributing documents using model-internal saliency rather than trusting self-generated citations alone. | https://aclanthology.org/2024.emnlp-main.347/ |
| 2024 | Chain-of-Verification Reduces Hallucination in Large Language Models | Findings of ACL | Draft, plan independent verification questions, answer them without draft anchoring, then regenerate a verified response. | https://aclanthology.org/2024.findings-acl.212/ |
| 2024 | Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection | ICLR | Learns retrieval-on-demand and explicit relevance, support and utility reflection signals, improving factuality and citation accuracy. | https://openreview.net/forum?id=hSyW5go0v8 |
| 2024-01-29 | Corrective Retrieval Augmented Generation | arXiv preprint | Retrieval quality gating, fallback search and decompose-then-recompose filtering improve robustness when first-pass retrieval is poor. | https://arxiv.org/abs/2401.15884 |
| 2024 | Large Language Models Cannot Self-Correct Reasoning Yet | ICLR | Intrinsic self-correction without external feedback is unreliable and can degrade reasoning, supporting independent verifier and tool-based checks. | https://proceedings.iclr.cc/paper_files/paper/2024/hash/8b4add8b0aa8749d80a34ca5d941c355-Abstract-Conference.html |
| 2023-05-31 | Let's Verify Step by Step | Technical report / PRM800K | Process supervision and step-level reward models outperform outcome-only supervision in the reported math setting and motivate step-verifier architectures. | https://arxiv.org/abs/2305.20050 |

## Industry / Policy News
| Event date | Organization | Update | Impact | Link |
|---|---|---|---|---|
| 2026-05-01 | NIST | Building Evaluation Probes into Agentic AI | Official deep-research testbed evaluates every source chunk and citation against strict rubrics and stores a structured audit trail; directly applicable to evidence-chain agents. | https://www.nist.gov/programs-projects/building-evaluation-probes-agentic-ai |
| 2025-03-28 | HL7 International | Evidence Based Medicine on FHIR Implementation Guide ballot 2 | Defines machine-interpretable Evidence, Citation and ArtifactAssessment profiles, including certainty of evidence, risk of bias and recommendation justification. | https://www.hl7.org/fhir/uv/ebm/2025May/ |
| 2025 | NIST TREC | TREC 2025 Retrieval-Augmented Generation Track proceedings | Uses long narrative queries and evaluates relevance, completeness, attribution verification and agreement; participating systems expose sentence-support metrics and hierarchical citations. | https://pages.nist.gov/trec-browser/trec34/rag/proceedings/ |
| 2025-01-06 | FDA | Draft lifecycle guidance for AI-enabled medical devices | Emphasizes transparency, bias controls, documentation and performance monitoring across the total product lifecycle; useful as a governance baseline if the prototype approaches clinical use. | https://www.fda.gov/news-events/press-announcements/fda-issues-comprehensive-draft-guidance-developers-artificial-intelligence-enabled-medical-devices |
| 2024-01-18 | WHO | Ethics and governance guidance for large multi-modal models in health | Calls for well-defined tasks, transparent design, stakeholder participation, independent post-release auditing and published impact assessments. | https://www.who.int/news/item/18-01-2024-who-releases-ai-ethics-and-governance-guidance-for-large-multi-modal-models |
| 2024-07-26 | NIST | Generative AI Profile NIST AI 600-1 | Treats confabulated facts, logic and citations as consequential risks and provides lifecycle risk-management actions. | https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-generative-artificial-intelligence |
| 2024-06 | FDA, Health Canada and MHRA | Transparency principles for machine-learning-enabled medical devices | Frames transparency around audience, purpose, content, placement, timing and medium, with emphasis on human-AI team performance and known information gaps. | https://www.fda.gov/medical-devices/software-medical-device-samd/transparency-machine-learning-enabled-medical-devices-guiding-principles |
| 2026 | GRADE Working Group | Living GRADE Book and decision-threshold guidance | Operationalizes certainty assessment across risk of bias, inconsistency, indirectness, imprecision and dissemination bias, linked to decision thresholds. | https://book.gradepro.org/guideline/overview-of-the-grade-approach |
| 2020 | PRISMA | PRISMA 2020 statement and flow diagrams | Provides a transparent search-screen-include-exclude flow model that can be adapted to show how an agent assembled an evidence set. | https://www.prisma-statement.org/prisma-2020 |
| 2013-04-30 | W3C | PROV-O Recommendation | Defines interoperable Entity, Activity and Agent provenance chains with derivation, usage, generation and attribution relations. | https://www.w3.org/TR/prov-o/ |

## Download Status
- Success: 20
- Failed: 0

## Signals to Watch Next
- New benchmark papers that compare multimodal and hybrid retrieval routing.
- Product releases exposing reranker and routing controls for multi-route retrieval.
- Regulation and compliance updates affecting retrieval data provenance.
