"""Read-only repository health and security audit.

The audit intentionally reports paths and counters rather than file contents.
It is safe to run before every evaluation or release and does not modify the
working tree, call external services, or execute application code.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
from typing import Any


TEXT_SUFFIXES = frozenset(
    {".json", ".md", ".py", ".js", ".css", ".html", ".toml", ".ini", ".yml", ".yaml"}
)
SKIP_PARTS = frozenset({".git", "__pycache__", ".pytest_cache", "build", "dist", ".venv", "venv"})
SECRET_PATTERNS = (
    (re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"), "api_key_shaped_value"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "private_key_marker"),
)


def _run_git(root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError:
        return ""
    return result.stdout.strip()


def _iter_text_files(root: Path):
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if any(part in SKIP_PARTS for part in path.relative_to(root).parts):
            continue
        yield path


def _secret_findings(root: Path, tracked: set[str]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for path in _iter_text_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for pattern, category in SECRET_PATTERNS:
            count = len(pattern.findall(text))
            if count:
                relative = str(path.relative_to(root)).replace("\\", "/")
                findings.append(
                    {
                        "path": relative,
                        "category": category,
                        "count": count,
                        "tracked": relative in tracked,
                    }
                )
    return findings


def _tracked_files(root: Path) -> set[str]:
    output = _run_git(root, "ls-files")
    return {line.replace("\\", "/") for line in output.splitlines() if line.strip()}


def _count_matches(root: Path, pattern: str) -> int:
    compiled = re.compile(pattern, re.IGNORECASE)
    count = 0
    for path in _iter_text_files(root):
        try:
            count += len(compiled.findall(path.read_text(encoding="utf-8")))
        except (OSError, UnicodeDecodeError):
            continue
    return count


def run_repository_audit(root: str | Path | None = None) -> dict[str, Any]:
    """Return a JSON-safe, read-only health snapshot for ``root``."""

    repo_root = Path(root or Path(__file__).resolve().parents[1]).resolve()
    tracked = _tracked_files(repo_root)
    status_lines = _run_git(repo_root, "status", "--short").splitlines()
    branch = _run_git(repo_root, "branch", "--show-current")
    head = _run_git(repo_root, "rev-parse", "--short", "HEAD")
    upstream = _run_git(
        repo_root,
        "rev-parse",
        "--abbrev-ref",
        "--symbolic-full-name",
        "@{upstream}",
    )

    required_paths = [
        "README.md",
        "ARCHITECTURE.md",
        "pyproject.toml",
        "src/medical_agent/quality.py",
        "src/medical_agent/evaluator.py",
        "evaluation/datasets.json",
        "evaluation/smoke_cases.json",
        "scripts/run_evaluation.py",
    ]
    missing = [item for item in required_paths if not (repo_root / item).is_file()]
    source_files = (
        list((repo_root / "src").rglob("*.py"))
        if (repo_root / "src").is_dir()
        else []
    )
    test_files = (
        list((repo_root / "tests").glob("test_*.py"))
        if (repo_root / "tests").is_dir()
        else []
    )
    lockfiles = [
        str(path.relative_to(repo_root)).replace("\\", "/")
        for name in ("uv.lock", "poetry.lock", "Pipfile.lock", "requirements.lock")
        for path in [repo_root / name]
        if path.is_file()
    ]
    workflows = (
        list((repo_root / ".github" / "workflows").glob("*"))
        if (repo_root / ".github" / "workflows").is_dir()
        else []
    )
    secret_findings = _secret_findings(repo_root, tracked)
    findings: list[dict[str, Any]] = []
    if secret_findings:
        findings.append(
            {
                "severity": "high",
                "code": "SECRET_SHAPED_VALUE",
                "message": "发现疑似密钥或私钥标记；不要提交，已忽略具体值。",
                "count": len(secret_findings),
            }
        )
    if missing:
        findings.append(
            {
                "severity": "high",
                "code": "MISSING_REQUIRED_PATH",
                "message": "评测/架构关键文件缺失。",
                "paths": missing,
            }
        )
    if not workflows:
        findings.append(
            {
                "severity": "medium",
                "code": "NO_CI_WORKFLOW",
                "message": "未发现 GitHub Actions 工作流；建议将冒烟评测、单测和敏感信息扫描纳入 CI。",
            }
        )
    if not lockfiles:
        findings.append(
            {
                "severity": "medium",
                "code": "NO_DEPENDENCY_LOCK",
                "message": "未发现依赖锁文件；可在部署环境生成 constraints/lock 以固定可复现版本。",
            }
        )
    if not any(path.name in {"test_web.py", "test_browser.py"} for path in test_files):
        findings.append(
            {
                "severity": "low",
                "code": "NO_BROWSER_E2E",
                "message": "当前测试清单未包含浏览器端到端用例；JavaScript 语法检查不能覆盖渲染交互。",
            }
        )
    findings.append(
        {
            "severity": "info",
            "code": "RESTRICTED_DATA_BOUNDARY",
            "message": "MedNLI/MIMIC 等受限数据只允许在授权环境运行，仓库仅保存 manifest 与汇总指标。",
        }
    )

    return {
        "repo": {
            "root": str(repo_root),
            "branch": branch,
            "head": head,
            "upstream": upstream,
            "working_tree_clean": not status_lines,
            "status_entries": len(status_lines),
        },
        "inventory": {
            "tracked_files": len(tracked),
            "source_python_files": len(source_files),
            "test_files": len(test_files),
            "lockfiles": lockfiles,
            "ci_workflows": len(workflows),
            "required_paths_missing": missing,
        },
        "static_checks": {
            "secret_findings": secret_findings,
            "todo_count": _count_matches(repo_root, r"\b(?:TODO|FIXME)\b"),
            "broad_exception_count": _count_matches(repo_root, r"except\s+Exception"),
            "evaluation_manifest_present": (repo_root / "evaluation" / "datasets.json").is_file(),
            "synthetic_smoke_fixture_present": (
                repo_root / "evaluation" / "smoke_cases.json"
            ).is_file(),
        },
        "findings": findings,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="只读审计当前代码仓库")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, help="将 JSON 审计结果写入文件")
    parser.add_argument(
        "--fail-on-high",
        action="store_true",
        help="发现高危项时返回非零，适合 CI 门禁",
    )
    args = parser.parse_args(argv)
    result = run_repository_audit(args.root)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if args.fail_on_high and any(
        finding.get("severity") == "high"
        for finding in result.get("findings", [])
        if isinstance(finding, dict)
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
