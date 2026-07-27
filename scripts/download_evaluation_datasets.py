"""Download reproducible evaluation assets into the ignored local data area.

The source mapping is data-driven and deliberately separate from application
routing.  By default only public HTTP/repository sources are attempted.  The
``--allow-terms-check`` switch is required for sources whose repository is
public but whose dataset/re-distribution terms still need a local review.
Credentialed and manual sources are reported, never bypassed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any
import urllib.parse
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_MANIFEST = ROOT / "evaluation" / "download_sources.json"
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "evaluation"
MAX_MANIFEST_ITEMS = 64
MAX_DOWNLOAD_BYTES = 512 * 1024 * 1024
SAFE_DATASET_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,80}$")


class DownloadError(RuntimeError):
    """Raised when an evaluation asset cannot be downloaded safely."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DownloadError(f"无法读取下载清单: {path}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise DownloadError("下载清单 schema_version 不受支持")
    records = payload.get("downloads")
    if not isinstance(records, list) or not records or len(records) > MAX_MANIFEST_ITEMS:
        raise DownloadError("下载清单 downloads 必须是有界非空数组")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise DownloadError("下载清单中的记录必须是对象")
        dataset_id = record.get("dataset_id")
        if not isinstance(dataset_id, str) or not SAFE_DATASET_ID.fullmatch(dataset_id):
            raise DownloadError(f"非法 dataset_id: {dataset_id!r}")
        if dataset_id in seen:
            raise DownloadError(f"重复 dataset_id: {dataset_id}")
        seen.add(dataset_id)
        if record.get("method") not in {"http_files", "git_sparse", "gdrive", "manual"}:
            raise DownloadError(f"{dataset_id} 的 method 不受支持")
        normalized.append(record)
    return normalized


def _sha256_and_size(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def _download_stream(url: str, target: Path, *, max_bytes: int) -> tuple[int, str]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "medical-agent-evaluation-downloader/1.0"},
    )
    digest = hashlib.sha256()
    size = 0
    temporary = target.with_suffix(target.suffix + ".part")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            content_length = response.headers.get("Content-Length")
            if content_length and content_length.isdigit() and int(content_length) > max_bytes:
                raise DownloadError(f"远端文件超过上限 {max_bytes:,} bytes")
            with temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    size += len(chunk)
                    if size > max_bytes:
                        raise DownloadError(f"下载文件超过上限 {max_bytes:,} bytes")
                    digest.update(chunk)
                    output.write(chunk)
        temporary.replace(target)
    except DownloadError:
        temporary.unlink(missing_ok=True)
        raise
    except Exception as exc:  # noqa: BLE001 - converted to a safe download error
        temporary.unlink(missing_ok=True)
        raise DownloadError(f"网络请求失败: {type(exc).__name__}") from exc
    return size, digest.hexdigest()


def _gdrive_url(file_id: str) -> str:
    query = urllib.parse.urlencode(
        {"export": "download", "id": file_id},
    )
    return f"https://drive.google.com/uc?{query}"


def _download_gdrive(
    file_id: str,
    target: Path,
    *,
    max_bytes: int,
) -> tuple[int, str]:
    """Handle the public Google Drive virus-scan confirmation page."""

    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor())
    request = urllib.request.Request(
        _gdrive_url(file_id),
        headers={"User-Agent": "medical-agent-evaluation-downloader/1.0"},
    )
    temporary = target.with_suffix(target.suffix + ".part")
    try:
        with opener.open(request, timeout=60) as response:
            first = response.read(4096)
            content_type = response.headers.get("Content-Type", "")
            if "text/html" not in content_type.lower():
                digest = hashlib.sha256()
                size = 0
                with temporary.open("wb") as output:
                    output.write(first)
                    size += len(first)
                    digest.update(first)
                    while chunk := response.read(1024 * 1024):
                        size += len(chunk)
                        if size > max_bytes:
                            raise DownloadError(f"下载文件超过上限 {max_bytes:,} bytes")
                        digest.update(chunk)
                        output.write(chunk)
                temporary.replace(target)
                return size, digest.hexdigest()
            html_payload = first + response.read()
    except DownloadError:
        temporary.unlink(missing_ok=True)
        raise
    except Exception as exc:  # noqa: BLE001 - converted to a safe download error
        temporary.unlink(missing_ok=True)
        raise DownloadError(f"Google Drive 请求失败: {type(exc).__name__}") from exc
    page = html_payload.decode("utf-8", "replace")
    form = re.search(
        r'<form[^>]+action="([^"]+)"[^>]*>.*?</form>',
        page,
        re.IGNORECASE | re.DOTALL,
    )
    if not form:
        raise DownloadError("Google Drive 返回确认页，但未找到下载表单")
    action = html.unescape(form.group(1))
    fields = dict(
        re.findall(
            r'<input[^>]+name="([^"]+)"[^>]+value="([^"]*)"',
            form.group(0),
            re.IGNORECASE,
        )
    )
    fields.setdefault("id", file_id)
    fields.setdefault("export", "download")
    fields.setdefault("confirm", "t")
    download_url = f"{action}?{urllib.parse.urlencode(fields)}"
    return _download_stream(download_url, target, max_bytes=max_bytes)


def _run_git_sparse(
    record: dict[str, Any],
    target: Path,
    *,
    max_bytes: int,
) -> tuple[int, str]:
    url = str(record.get("url", ""))
    ref = str(record.get("ref", "main"))
    sparse_paths = record.get("sparse_paths")
    if not url or not isinstance(sparse_paths, list) or not sparse_paths:
        raise DownloadError("git_sparse 缺少 url 或 sparse_paths")
    temporary_root = Path(tempfile.mkdtemp(prefix=f"{record['dataset_id']}-", dir=target.parent))
    temporary_repo = temporary_root / "repo"
    try:
        command = [
            "git",
            "clone",
            "--depth",
            "1",
            "--filter=blob:none",
            "--sparse",
            "--branch",
            ref,
            url,
            str(temporary_repo),
        ]
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=900,
        )
        if completed.returncode:
            raise DownloadError(f"git clone 失败 exit={completed.returncode}")
        sparse = subprocess.run(
            ["git", "-C", str(temporary_repo), "sparse-checkout", "set", "--skip-checks", *sparse_paths],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=900,
        )
        if sparse.returncode:
            raise DownloadError(f"git sparse-checkout 失败 exit={sparse.returncode}")
        size = sum(path.stat().st_size for path in temporary_repo.rglob("*") if path.is_file())
        if size > max_bytes:
            raise DownloadError(f"稀疏检出超过上限 {max_bytes:,} bytes")
        if target.exists():
            raise DownloadError(f"目标目录已存在: {target}")
        temporary_repo.replace(target)
        temporary_root.rmdir()
        return _sha256_and_size_tree(target)
    except subprocess.TimeoutExpired as exc:
        raise DownloadError("git 操作超时") from exc
    finally:
        if temporary_root.exists():
            shutil.rmtree(temporary_root, ignore_errors=True)


def _sha256_and_size_tree(root: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    for path in sorted(path for path in root.rglob("*") if path.is_file() and ".git" not in path.parts):
        file_size, file_hash = _sha256_and_size(path)
        relative = str(path.relative_to(root)).replace("\\", "/")
        digest.update(relative.encode("utf-8"))
        digest.update(file_hash.encode("ascii"))
        size += file_size
    return size, digest.hexdigest()


def _download_record(
    record: dict[str, Any],
    output_root: Path,
    *,
    allow_terms_check: bool,
    max_bytes: int,
) -> dict[str, Any]:
    dataset_id = str(record["dataset_id"])
    access = str(record["access"])
    method = str(record["method"])
    result: dict[str, Any] = {
        "dataset_id": dataset_id,
        "access": access,
        "method": method,
        "source_url": record.get("source_url", ""),
        "started_at": _utc_now(),
    }
    for provenance_key in ("mirror_url", "terms_note"):
        if provenance_key in record:
            result[provenance_key] = record[provenance_key]
    if access == "credentialed":
        result.update(
            {
                "status": "blocked_access",
                "message": "需要 credential/DUA，下载器不会绕过访问控制。",
                "finished_at": _utc_now(),
            }
        )
        return result
    if method == "manual":
        result.update(
            {
                "status": "manual_required",
                "message": str(record.get("terms_note", "需要按官方页面操作")),
                "finished_at": _utc_now(),
            }
        )
        return result
    if access == "terms_check" and not allow_terms_check:
        result.update(
            {
                "status": "terms_confirmation_required",
                "message": str(record.get("terms_note", "请先确认数据条款")),
                "finished_at": _utc_now(),
            }
        )
        return result

    dataset_root = output_root / dataset_id
    if method == "git_sparse" and dataset_root.is_dir() and any(dataset_root.iterdir()):
        try:
            file_size, file_hash = _sha256_and_size_tree(dataset_root)
            result.update(
                {
                    "status": "already_present",
                    "files": [{"path": ".", "bytes": file_size, "sha256": file_hash}],
                    "bytes": file_size,
                    "message": "目标目录已存在，未重复下载。",
                    "finished_at": _utc_now(),
                }
            )
            return result
        except OSError as exc:
            result.update(
                {
                    "status": "failed",
                    "message": f"无法读取已存在目录: {type(exc).__name__}",
                    "finished_at": _utc_now(),
                }
            )
            return result
    dataset_root.mkdir(parents=True, exist_ok=True)
    try:
        if method == "http_files":
            files = record.get("files")
            if not isinstance(files, list) or not files:
                raise DownloadError("http_files 缺少 files")
            file_results: list[dict[str, Any]] = []
            total_bytes = 0
            for item in files:
                if not isinstance(item, dict) or not isinstance(item.get("url"), str):
                    raise DownloadError("http_files 包含无效文件记录")
                relative = Path(str(item.get("path", "")))
                if relative.is_absolute() or ".." in relative.parts or not relative.name:
                    raise DownloadError("下载路径必须位于数据集目录内")
                target = dataset_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.is_file() and target.stat().st_size > 0:
                    file_size, file_hash = _sha256_and_size(target)
                    source_status = "already_present"
                else:
                    file_size, file_hash = _download_stream(
                        str(item["url"]), target, max_bytes=max_bytes
                    )
                    source_status = "downloaded"
                total_bytes += file_size
                file_results.append(
                    {
                        "path": str(relative).replace("\\", "/"),
                        "bytes": file_size,
                        "sha256": file_hash,
                        "url": item["url"],
                        "status": source_status,
                    }
                )
            result.update({"status": "downloaded", "files": file_results, "bytes": total_bytes})
        elif method == "gdrive":
            file_id = str(record.get("file_id", ""))
            filename = Path(str(record.get("filename", "dataset.bin"))).name
            if not file_id or not filename:
                raise DownloadError("gdrive 缺少 file_id/filename")
            target = dataset_root / filename
            file_size, file_hash = _download_gdrive(file_id, target, max_bytes=max_bytes)
            result.update(
                {
                    "status": "downloaded",
                    "files": [{"path": filename, "bytes": file_size, "sha256": file_hash}],
                    "bytes": file_size,
                }
            )
        elif method == "git_sparse":
            if dataset_root.exists() and any(dataset_root.iterdir()):
                raise DownloadError(f"目标目录已存在且非空: {dataset_root}")
            if dataset_root.exists():
                dataset_root.rmdir()
            file_size, file_hash = _run_git_sparse(
                record, dataset_root, max_bytes=max_bytes
            )
            result.update(
                {
                    "status": "downloaded",
                    "files": [{"path": ".", "bytes": file_size, "sha256": file_hash}],
                    "bytes": file_size,
                    "revision": record.get("ref", "main"),
                }
            )
        else:
            raise DownloadError(f"未知下载方式: {method}")
    except DownloadError as exc:
        result.update({"status": "failed", "message": str(exc)})
    result["finished_at"] = _utc_now()
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="下载公开评测数据到本地忽略目录")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset", action="append", dest="datasets")
    parser.add_argument("--allow-terms-check", action="store_true")
    parser.add_argument("--max-bytes", type=int, default=MAX_DOWNLOAD_BYTES)
    parser.add_argument("--metadata-out", type=Path)
    args = parser.parse_args(argv)
    if args.max_bytes < 1 or args.max_bytes > MAX_DOWNLOAD_BYTES:
        parser.error(f"--max-bytes 必须在 1 和 {MAX_DOWNLOAD_BYTES:,} 之间")
    records = _load_manifest(args.manifest)
    requested = set(args.datasets or [])
    if requested:
        known = {str(record["dataset_id"]) for record in records}
        unknown = requested - known
        if unknown:
            parser.error(f"未知 dataset_id: {', '.join(sorted(unknown))}")
        records = [record for record in records if record["dataset_id"] in requested]
    args.output_root.mkdir(parents=True, exist_ok=True)
    new_results = [
        _download_record(
            record,
            args.output_root,
            allow_terms_check=args.allow_terms_check,
            max_bytes=args.max_bytes,
        )
        for record in records
    ]
    metadata_out = args.metadata_out or args.output_root / "download-manifest.json"
    merged_results: dict[str, dict[str, Any]] = {}
    if metadata_out.is_file():
        try:
            previous = json.loads(metadata_out.read_text(encoding="utf-8"))
            previous_results = previous.get("results", [])
            if isinstance(previous_results, list):
                merged_results.update(
                    {
                        str(item["dataset_id"]): item
                        for item in previous_results
                        if isinstance(item, dict) and isinstance(item.get("dataset_id"), str)
                    }
                )
        except (OSError, json.JSONDecodeError):
            # A corrupt/old local manifest must not block a fresh download.
            merged_results = {}
    merged_results.update({str(item["dataset_id"]): item for item in new_results})
    results = list(merged_results.values())
    payload = {
        "schema_version": 1,
        "generated_at": _utc_now(),
        "output_root": str(args.output_root.resolve()),
        "results": results,
    }
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    metadata_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if all(result["status"] not in {"failed"} for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
