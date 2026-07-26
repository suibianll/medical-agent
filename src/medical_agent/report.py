"""Render deterministic, citation-first text reports from server-owned data.

The report protocol intentionally stays small: callers select a named template
or provide ``{"name": "...", "title": "..."}``.  The model never has to
produce a rich document schema or free-form report layout.
"""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Any


DEFAULT_TEMPLATE_NAME = "evidence_summary"
MAX_TEMPLATE_TITLE_CHARS = 80
_SECTION_ORDER = (
    "summary",
    "conclusions",
    "unknowns",
    "tasks",
    "evaluation",
    "evidence",
)

# These are server-authored layouts.  A caller may safely override just the
# title or choose a subset of the fixed section tokens; arbitrary Markdown or
# nested template expressions are deliberately not accepted.
_TEMPLATES: dict[str, dict[str, Any]] = {
    "evidence_summary": {
        "name": "evidence_summary",
        "title": "医疗 Agent 证据报告",
        "sections": list(_SECTION_ORDER),
    },
    "task_trace": {
        "name": "task_trace",
        "title": "医疗 Agent 任务追踪报告",
        "sections": ["summary", "tasks", "conclusions", "evaluation", "evidence"],
    },
    "handoff": {
        "name": "handoff",
        "title": "医疗 Agent 交接摘要",
        "sections": ["summary", "conclusions", "unknowns", "evaluation", "evidence"],
    },
}


def _plain_text(value: Any, limit: int = MAX_TEMPLATE_TITLE_CHARS) -> str:
    """Keep a user-provided template title as inert, compact plain text."""

    text = " ".join(str(value or "").split())
    text = re.sub(r"<[^>]*>", "", text)
    # Template titles are rendered in Markdown and plain text.  Strip the
    # small set of characters that could turn a title into Markdown/HTML.
    text = re.sub(r"[`*_#\[\]<>]", "", text)
    return text[:limit].strip()


def normalize_report_template(template: Any = None) -> dict[str, Any]:
    """Normalize a weak-model-friendly report template request.

    Accepted forms are a template name or a small object with ``name``, an
    optional plain-text ``title``, and (optionally) a list of fixed ``sections``
    tokens.  Unknown values fall back to the evidence-first default instead of
    causing a model/report failure.
    """

    raw_name = ""
    raw_title = ""
    raw_sections: Any = None
    if isinstance(template, str):
        raw_name = template
    elif isinstance(template, dict):
        raw_name = str(template.get("name", ""))
        raw_title = template.get("title", "")
        raw_sections = template.get("sections")

    name = raw_name.strip().lower().replace("-", "_")
    if name not in _TEMPLATES:
        name = DEFAULT_TEMPLATE_NAME
    normalized = deepcopy(_TEMPLATES[name])

    title = _plain_text(raw_title)
    if title:
        normalized["title"] = title

    if isinstance(raw_sections, list):
        sections: list[str] = []
        for item in raw_sections[: len(_SECTION_ORDER)]:
            token = str(item).strip().lower()
            if token in _SECTION_ORDER and token not in sections:
                sections.append(token)
        if sections:
            normalized["sections"] = sections
    return normalized


def _citation_text(refs: list[str]) -> str:
    valid_refs = [ref for ref in refs if isinstance(ref, str) and ref.strip()]
    return "".join(f"[{ref}]" for ref in valid_refs) if valid_refs else "[无引用]"


def _split_sentences(value: Any) -> list[str]:
    """Split conservatively while retaining Chinese and English punctuation."""

    text = " ".join(str(value or "").split())
    if not text:
        return []
    sentences: list[str] = []
    buffer: list[str] = []
    for index, character in enumerate(text):
        buffer.append(character)
        is_chinese_terminal = character in "。！？"
        if character in ".!?":
            # English punctuation inside a quoted phrase (for example a user
            # question repeated by the model) is not a conclusion boundary.
            remaining = text[index + 1 :].lstrip("”’\"')]}）")
            is_english_terminal = not remaining or remaining[0].isspace()
        else:
            is_english_terminal = False
        if is_chinese_terminal or is_english_terminal:
            sentence = "".join(buffer).strip()
            if sentence:
                sentences.append(sentence)
            buffer = []
    remainder = "".join(buffer).strip()
    if remainder:
        sentences.append(remainder)
    return sentences or [text]


def render_cited_claim(claim: dict[str, Any]) -> str:
    """Repeat the claim's original refs at the end of every visible sentence."""

    citations = _citation_text(claim.get("refs", []))
    rendered: list[str] = []
    for sentence in _split_sentences(claim.get("text", "")):
        # A generated fragment without terminal punctuation is still shown as
        # one complete report sentence before the unchanged citation IDs.
        if sentence[-1:] not in "。！？.!?":
            sentence = f"{sentence}。"
        rendered.append(f"{sentence} {citations}")
    return " ".join(rendered) or citations


def _unknowns(task_states: dict[int, dict[str, Any]]) -> list[str]:
    values: list[str] = []
    for _task_id, state in sorted(task_states.items()):
        for unknown in (state.get("result") or {}).get("unknowns", []):
            compact = " ".join(str(unknown or "").split())
            if compact and compact not in values:
                values.append(compact)
    return values


def _heading(title: str, *, markdown: bool, level: int = 2) -> str:
    if markdown:
        return f"{'#' * level} {title}"
    return title if level == 1 else f"{title}\n{'-' * max(4, len(title))}"


def _render_lines(
    *,
    markdown: bool,
    template: dict[str, Any],
    request: str,
    task_states: dict[int, dict[str, Any]],
    claims: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    evaluation: dict[str, Any],
    status: str,
) -> list[str]:
    title = template["title"]
    lines = [_heading(title, markdown=markdown, level=1), ""]
    lines.append(
        "> **安全提示：** 本系统为演示性的临床决策支持原型，不可用于自动诊断、处方或替代专业医疗意见。"
        if markdown
        else "安全提示：本系统为演示性的临床决策支持原型，不可用于自动诊断、处方或替代专业医疗意见。"
    )

    for section in template["sections"]:
        lines.append("")
        if section == "summary":
            lines.extend(
                [
                    _heading("任务概览", markdown=markdown),
                    f"- 总体状态：{status}",
                    f"- 已完成任务：{sum(state.get('status') == 'completed' for state in task_states.values())}/{len(task_states)}",
                    f"- 已生成结论：{len(claims)} 条；证据：{len(evidence)} 条。",
                    "- 用户请求：",
                    request.strip() or "（未提供）",
                ]
            )
        elif section == "conclusions":
            lines.append(_heading("结论（逐句引用）", markdown=markdown))
            if claims:
                for claim in claims:
                    claim_id = str(claim.get("id", "结论"))
                    label = f"**{claim_id}**" if markdown else claim_id
                    lines.append(f"- {label}：{render_cited_claim(claim)}")
            else:
                lines.append("- 未形成可引用的结论；请补充病历或转人工审核。")
        elif section == "unknowns":
            lines.append(_heading("待补充信息", markdown=markdown))
            unknowns = _unknowns(task_states)
            if unknowns:
                lines.extend(f"- {item}" for item in unknowns)
            else:
                lines.append("- 当前未报告额外待补充信息。")
        elif section == "tasks":
            lines.append(_heading("任务执行", markdown=markdown))
            for task_id, state in sorted(task_states.items()):
                task = state["task"]
                lines.append(f"- T{task_id}（{state['status']}）：{task['goal']}")
                if state.get("error"):
                    lines.append("  - 错误：该子任务未能完成，详见评估结果。")
        elif section == "evaluation":
            lines.append(_heading("评估结果", markdown=markdown))
            lines.append(f"- 总体状态：{status}")
            if evaluation.get("issues"):
                for issue in evaluation["issues"]:
                    lines.append(f"- {issue['claim']}：{issue['code']}")
            else:
                lines.append("- 所有已生成结论均通过当前引用完整性检查。")
        elif section == "evidence":
            lines.append(_heading("证据明细", markdown=markdown))
            if not evidence:
                lines.append("- 未检索到可展示的证据。")
            for item in evidence:
                source = str(item.get("source", "未知来源"))
                if markdown:
                    lines.append(f"### [{item['id']}] {source}")
                else:
                    lines.append(f"[{item['id']}] {source}")
                lines.extend(
                    [
                        f"- 类型：{'患者事实' if item.get('kind') == 'patient' else '知识库证据'}",
                        f"- 定位：{item.get('locator', '未标注')}",
                        f"- 原文：{item.get('text', '')}",
                    ]
                )
                version = (item.get("metadata") or {}).get("version")
                if version:
                    lines.append(f"- 版本：{version}")
    return lines


def render_report(
    *,
    request: str,
    task_states: dict[int, dict[str, Any]],
    claims: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    evaluation: dict[str, Any],
    status: str,
    template: Any = None,
) -> dict[str, Any]:
    """Render a complete report, with sentence-level source citations."""

    normalized_template = normalize_report_template(template)
    markdown = "\n".join(
        _render_lines(
            markdown=True,
            template=normalized_template,
            request=request,
            task_states=task_states,
            claims=claims,
            evidence=evidence,
            evaluation=evaluation,
            status=status,
        )
    )
    text = "\n".join(
        _render_lines(
            markdown=False,
            template=normalized_template,
            request=request,
            task_states=task_states,
            claims=claims,
            evidence=evidence,
            evaluation=evaluation,
            status=status,
        )
    )
    return {
        "markdown": markdown,
        "text": text,
        "template": normalized_template,
    }
