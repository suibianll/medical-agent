# 用户验收样例：用药安全复核

本目录中的内容全部是合成、脱敏的测试资料，仅用于验证系统的 Plan-Execute、引用、报告和证据页功能，不能用于诊断、处方或真实患者决策。

## 文件说明

- `medication_review_knowledge.md`：可在工作台左侧“导入参考资料”中直接导入的知识库文档。
- `medication_review_patient_record.txt`：粘贴到“患者上下文（可选）”中的合成病历。
- `medication_review_dialogue.json`：三轮对话脚本、建议报告模板和人工验收点。

## 使用步骤

1. 启动服务：`python run.py`，打开 <http://127.0.0.1:8000>。
2. 在“导入参考资料”中选择并导入 `medication_review_knowledge.md`。
3. 打开“患者上下文（可选）”，粘贴 `medication_review_patient_record.txt` 的全文。
4. 选择“任务与证据链追踪”模板，依次发送 `medication_review_dialogue.json` 中 `turns` 的 `message`。
5. 每轮检查：
   - 回答的每个结论句末都有 `[P#]`、`[K#]` 引用；
   - 悬浮引用可显示来源、定位和证据摘要；
   - 右侧出现任务状态、审计时间线和任务依赖关系；
   - 点击“打开证据页”后，可见文本报告、任务 DAG、证据明细和安全审计日志。

模型回答会随知识库和模型版本而变化。请按 `acceptance_checks` 判断证据链是否完整，而不要比对固定措辞。
