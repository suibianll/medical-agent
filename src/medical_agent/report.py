"""Render an auditable Markdown report from server-owned claims and evidence."""

from __future__ import annotations

from typing import Any


def _citation_text(refs: list[str]) -> str:
    return "".join(f"[{ref}]" for ref in refs) if refs else "[无引用]"


def render_report(
    *,
    request: str,
    task_states: dict[int, dict[str, Any]],
    claims: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    evaluation: dict[str, Any],
    status: str,
) -> dict[str, str]:
    """Render every conclusion with its evidence IDs and a source appendix."""

    lines = [
        "# 医疗 Agent 证据报告",
        "",
        "> **安全提示：** 本系统为演示性的临床决策支持原型，不可用于自动诊断、处方或替代专业医疗意见。",
        "",
        "## 用户请求",
        request.strip(),
        "",
        "## 结论",
    ]

    if claims:
        for claim in claims:
            lines.append(
                f"- **{claim['id']}**：{claim['text']} {_citation_text(claim.get('refs', []))}"
            )
    else:
        lines.append("- 未形成可引用的结论；请补充病历或转人工审核。")

    lines.extend(["", "## 任务执行"])
    for task_id, state in sorted(task_states.items()):
        task = state["task"]
        lines.append(f"- T{task_id}（{state['status']}）：{task['goal']}")
        for unknown in (state.get("result") or {}).get("unknowns", []):
            lines.append(f"  - 待补充：{unknown}")
        if state.get("error"):
            lines.append(f"  - 错误：{state['error']}")

    lines.extend(["", "## 评估结果"])
    lines.append(f"- 总体状态：{status}")
    if evaluation.get("issues"):
        for issue in evaluation["issues"]:
            lines.append(f"- {issue['claim']}：{issue['code']}")
    else:
        lines.append("- 所有已生成结论均通过当前引用完整性检查。")

    lines.extend(["", "## 证据明细"])
    for item in evidence:
        lines.extend(
            [
                f"### [{item['id']}] {item['source']}",
                f"- 类型：{'患者事实' if item['kind'] == 'patient' else '知识库证据'}",
                f"- 定位：{item['locator']}",
                f"- 原文：{item['text']}",
            ]
        )
        version = item.get("metadata", {}).get("version")
        if version:
            lines.append(f"- 版本：{version}")

    markdown = "\n".join(lines)
    return {"markdown": markdown, "text": markdown}
