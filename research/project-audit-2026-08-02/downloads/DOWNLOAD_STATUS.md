# 论文下载状态

- 有效 PDF：19/20。
- 在线全文：20/20；LabSage 可通过 `papers.csv` 中的 PubMed/PMC 页面阅读。
- 未取得有效 PDF：LabSage。PMC 的 PDF 端点返回下载挑战页，备用文件名端点返回 404。
- `paper_downloads.csv`、`paper_downloads_retry.csv`、`paper_downloads_remaining.csv` 和 `labsage_download.csv` 是逐次重试的原始日志；其中下载器按 HTTP 成功记录的 LabSage 项后来经 `%PDF` 文件签名复核判定为无效。
- `papers/` 已去除重复文件和伪装为 `.pdf` 的 HTML 下载挑战页，现有 19 个文件均通过 `%PDF` 文件签名检查。
