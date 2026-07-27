"""Run the repository's dependency-free evaluation smoke suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_agent.evaluation import run_local_smoke_evaluation  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="运行合成冒烟评测；不会下载数据集或调用模型。"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "evaluation" / "datasets.json",
        help="数据集元数据清单路径",
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        default=ROOT / "evaluation" / "smoke_cases.json",
        help="合成冒烟案例路径",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="可选：将不含正文的指标 JSON 写入文件",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result: dict[str, Any] = run_local_smoke_evaluation(args.manifest, args.fixture)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
