#!/usr/bin/env python3
# Derived from PredictiveScienceLab/paper-replication-paper (Apache-2.0).
# Modified on 2026-08-19 to execute tracked commands through PowerShell on Windows.
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import urllib.request
import zipfile
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
SCHEMA_VERSION = 1
REQUIRED_MATRIX_COLUMNS = [
    "target_id",
    "kind",
    "paper_locator",
    "source_locator",
    "runner",
    "config",
    "output_path",
    "acceptance_mode",
    "comparison_metric",
    "tolerance",
    "status",
    "report_anchor",
]
REQUIRED_TODO_HEADERS = [
    "Current phase",
    "Active target",
    "Acceptance gates",
    "Open unknowns",
    "Completed",
]
REQUIRED_SPEC_DOCS = [
    "spec/targets.md",
    "spec/math_audit.md",
    "spec/implementation_plan.md",
    "spec/assumptions_and_unknowns.md",
    "spec/paper_figure_notes.md",
]
MATCHED_STATUSES = {"MATCHED", "GREEN", "COMPLETE", "REPRODUCED", "REPLICATED"}
TERMINAL_STATUSES = MATCHED_STATUSES | {"SKIPPED"}
PLACEHOLDER_TOKENS = ("[TODO", "TODO:", "TODO ", "<fill", "REPLACE_ME")
RASTER_EXTENSIONS = {".png", ".jpg", ".jpeg"}
VECTOR_EXTENSIONS = {".pdf", ".eps", ".svg"}
CODE_EXTENSIONS = {".py", ".ipynb", ".m", ".jl", ".r", ".cpp", ".cc", ".c", ".h", ".hpp"}
SUSPECT_TEX_KEYWORDS = ("supp", "supplement", "appendix", "methods", "method", "proof", "additional", "si")
RUN_WRAPPER_NAME = "paper_harness.track-run"
REGISTER_WRAPPER_NAME = "paper_harness.register-target-artifact"
COMPARISON_WRAPPER_NAME = "paper_harness.record-comparison"
PLACEHOLDER_COMPARISON_NOTE = "Replace this note with the visual inspection summary before marking the target MATCHED."
ACCEPTANCE_MODES = {
    "exact-visual",
    "numeric-equivalence",
    "distributional-equivalence",
    "qualitative-structural",
}
VISUAL_METRIC_MARKERS = ("ssim", "pixel", "image", "lpips", "byte", "mean_abs_diff", "mse")
COMPARISON_METADATA_KEYS = {
    "reference_path",
    "candidate_path",
    "reference_sha256",
    "candidate_sha256",
    "reference_size_bytes",
    "candidate_size_bytes",
    "reference_dimensions",
    "candidate_dimensions",
    "bytes_identical",
    "mean_abs_diff",
}
CLUSTER_EXEC_ENV_VAR = "PAPER_REPLICATION_RUNNING_IN_CLUSTER_EXEC"
CLUSTER_TRANSPORT_STATUS_PATH = Path("artifacts") / "status" / "cluster_transport.json"
SUSPICIOUS_BASELINE_MARKERS = (
    "artifact generator",
    "pattern generator",
    "matches reported figure/table patterns",
    "match reported figure/table patterns",
    "matches reported figure patterns",
    "reported figure/table patterns",
    "reported figure patterns",
    "paper pattern",
    "fit to paper",
    "fitted to paper",
    "curve-fit to paper",
    "curve fit to paper",
    "hard-coded paper",
    "hard coded paper",
)


class PaperReplicationError(RuntimeError):
    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    return re.sub(r"-{2,}", "-", slug) or "paper-replication"


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def project_root_from(path: str | Path | None) -> Path:
    if path is None:
        return Path.cwd()
    return Path(path).expanduser().resolve()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, content: str, *, mode: int | None = None) -> None:
    ensure_parent(path)
    path.write_text(content, encoding="utf-8")
    if mode is not None:
        path.chmod(mode)


def write_text_if_missing(path: Path, content: str, *, mode: int | None = None) -> None:
    if path.exists():
        if mode is not None:
            path.chmod(mode)
        return
    write_text(path, content, mode=mode)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    return rows


def cluster_transport_status(project_dir: Path) -> dict[str, Any]:
    path = project_dir / CLUSTER_TRANSPORT_STATUS_PATH
    if not path.exists():
        return {}
    payload = read_json(path)
    payload["status_path"] = str(path)
    return payload


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def deep_merge(defaults: dict[str, Any], existing: dict[str, Any]) -> dict[str, Any]:
    merged = dict(defaults)
    for key, value in existing.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_project_path(project_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (project_dir / path).resolve()
    return path.resolve()


def relative_project_path(project_dir: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(project_dir.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def run_index_path(project_dir: Path) -> Path:
    return project_dir / "artifacts" / "runs" / "run_index.jsonl"


def run_record_path(project_dir: Path, run_id: str) -> Path:
    return project_dir / "artifacts" / "runs" / run_id / "record.json"


def provenance_path(project_dir: Path, target_id: str) -> Path:
    return project_dir / "artifacts" / "provenance" / f"{target_id}.json"


def comparison_paths(project_dir: Path, target_id: str, kind: str) -> tuple[Path, Path]:
    normalized_kind = kind.strip().lower()
    if normalized_kind == "table":
        root = project_dir / "artifacts" / "tables" / "comparison"
    else:
        root = project_dir / "artifacts" / "figures" / "comparison"
    return root / f"{target_id}.json", root / f"{target_id}.md"


def next_run_id(project_dir: Path, prefix: str = "run") -> str:
    existing = read_jsonl(run_index_path(project_dir))
    next_index = len(existing) + 1
    return f"{prefix}-{next_index:04d}"


def load_manifest(project_dir: Path) -> dict[str, Any]:
    manifest_path = project_dir / "paper_manifest.json"
    if not manifest_path.exists():
        raise PaperReplicationError(
            f"Missing manifest: {manifest_path}",
            details={"manifest_path": str(manifest_path)},
        )
    data = read_json(manifest_path)
    if not isinstance(data, dict):
        raise PaperReplicationError("paper_manifest.json must be a JSON object.")
    return data


def save_manifest(project_dir: Path, manifest: dict[str, Any]) -> None:
    write_json(project_dir / "paper_manifest.json", manifest)


def file_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def directory_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(str(child.relative_to(path)).encode("utf-8"))
        digest.update(file_checksum(child).encode("utf-8"))
    return digest.hexdigest()


def source_checksum(path: Path) -> str:
    if path.is_dir():
        return directory_checksum(path)
    return file_checksum(path)


def read_small_text(path: Path, max_bytes: int = 512 * 1024) -> str:
    try:
        if not path.exists() or not path.is_file():
            return ""
        if path.stat().st_size > max_bytes:
            return ""
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def suspicious_baseline_hits(*texts: str) -> list[str]:
    haystack = "\n".join(texts).lower()
    return list(dict.fromkeys(marker for marker in SUSPICIOUS_BASELINE_MARKERS if marker in haystack))


def normalized_acceptance_mode(value: str) -> str:
    mode = value.strip().lower()
    return mode if mode in ACCEPTANCE_MODES else ""


def is_visual_metric_name(name: str) -> bool:
    metric_name = name.strip().lower()
    return any(marker in metric_name for marker in VISUAL_METRIC_MARKERS)


def primary_metric_name(row: dict[str, str]) -> str:
    return str(row.get("comparison_metric", "") or "").strip()


def inferred_acceptance_mode(row: dict[str, str], comparison_payload: dict[str, Any] | None = None) -> str:
    explicit_mode = normalized_acceptance_mode(str(row.get("acceptance_mode", "") or ""))
    if explicit_mode:
        return explicit_mode
    if comparison_payload:
        payload_mode = normalized_acceptance_mode(str(comparison_payload.get("acceptance_mode", "") or ""))
        if payload_mode:
            return payload_mode
    metric_name = primary_metric_name(row)
    if metric_name and is_visual_metric_name(metric_name):
        return "exact-visual"
    return "numeric-equivalence"


def comparison_evidence_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = payload.get("metrics", {})
    if not isinstance(metrics, dict):
        return {}
    return {str(key): value for key, value in metrics.items()}


def non_metadata_metric_keys(metrics: dict[str, Any]) -> list[str]:
    return [key for key in metrics if key not in COMPARISON_METADATA_KEYS]


def visual_metric_keys(metrics: dict[str, Any]) -> list[str]:
    return [key for key in non_metadata_metric_keys(metrics) if is_visual_metric_name(key)]


def actual_source_path(project_dir: Path, manifest: dict[str, Any]) -> Path | None:
    source = manifest.get("source", {})
    location = str(source.get("location", "")).strip()
    extracted = str(source.get("extracted_path", "")).strip()
    if extracted:
        extracted_path = (project_dir / extracted).resolve()
        if extracted_path.exists():
            return extracted_path
    if not location:
        return None
    location_path = Path(location).expanduser()
    if not location_path.is_absolute():
        location_path = (project_dir / location_path).resolve()
    if location_path.exists():
        return location_path.resolve()
    return None


def enforce_author_code_policy(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    policies = manifest.get("policies", {})
    mode = str(policies.get("author_code_policy", "forbid_by_default")).strip()
    author_paths = policies.get("author_code_paths", [])
    if not isinstance(author_paths, list):
        errors.append("policies.author_code_paths must be a JSON array.")
        return errors
    if mode == "forbid_by_default" and any(str(path).strip() for path in author_paths):
        errors.append("author_code_policy=forbid_by_default but author_code_paths is not empty.")
    return errors


def detect_main_tex(source_root: Path) -> Path:
    source_root = source_root.resolve()
    candidates = [
        source_root / "main.tex",
        source_root / "paper.tex",
        source_root / "manuscript.tex",
        source_root / "ms.tex",
        source_root / "arxiv.tex",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    tex_files = sorted(source_root.rglob("*.tex"))
    for tex_file in tex_files:
        text = tex_file.read_text(encoding="utf-8", errors="ignore")
        if "\\documentclass" in text:
            return tex_file
    raise PaperReplicationError(
        f"Could not detect a main TeX file under {source_root}",
        details={"source_root": str(source_root)},
    )


def strip_latex_comments(text: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", text)


def find_latex_commands(text: str, command: str) -> list[str]:
    pattern = re.compile(rf"\\{command}(?:\[[^\]]*\])?\{{([^}}]+)\}}")
    return pattern.findall(text)


def resolve_tex_reference(base_dir: Path, value: str, *, exts: list[str]) -> Path | None:
    candidate = (base_dir / value).resolve()
    if candidate.exists():
        return candidate
    for ext in exts:
        with_ext = (base_dir / f"{value}{ext}").resolve()
        if with_ext.exists():
            return with_ext
    return None


def inventory_latex_tree(source_root: Path, entrypoint: Path) -> dict[str, Any]:
    source_root = source_root.resolve()
    queue = [entrypoint.resolve()]
    seen: set[Path] = set()
    included_files: list[str] = []
    figures: list[dict[str, Any]] = []
    bibliographies: list[str] = []
    appendices: list[str] = []
    data_refs: list[str] = []
    all_tex_files = sorted(path.resolve() for path in source_root.rglob("*.tex"))

    while queue:
        tex_path = queue.pop(0)
        if tex_path in seen or not tex_path.exists():
            continue
        seen.add(tex_path)
        text = strip_latex_comments(tex_path.read_text(encoding="utf-8", errors="ignore"))
        included_files.append(str(tex_path.relative_to(source_root)))
        if "\\appendix" in text or "appendix" in tex_path.stem.lower():
            appendices.append(str(tex_path.relative_to(source_root)))
        for command in ("input", "include"):
            for raw in find_latex_commands(text, command):
                resolved = resolve_tex_reference(tex_path.parent, raw, exts=[".tex"])
                if resolved is not None:
                    queue.append(resolved)
        for raw in find_latex_commands(text, "includegraphics"):
            resolved = resolve_tex_reference(
                tex_path.parent,
                raw,
                exts=["", ".pdf", ".png", ".jpg", ".jpeg", ".eps", ".svg"],
            )
            figures.append(
                {
                    "referenced_from": str(tex_path.relative_to(source_root)),
                    "raw_reference": raw,
                    "resolved_path": (
                        str(resolved.relative_to(source_root))
                        if resolved is not None and resolved.is_relative_to(source_root)
                        else (str(resolved) if resolved is not None else "")
                    ),
                    "exists": bool(resolved and resolved.exists()),
                }
            )
        for command in ("bibliography", "addbibresource"):
            bibliographies.extend(find_latex_commands(text, command))
        data_refs.extend(
            sorted(
                {
                    match
                    for match in re.findall(
                        r"([A-Za-z0-9_./-]+\.(?:csv|tsv|txt|json|npy|npz|mat|h5|hdf5))",
                        text,
                    )
                }
            )
        )

    unreferenced_tex = [path for path in all_tex_files if path not in seen]
    suspect_unreferenced = [
        str(path.relative_to(source_root))
        for path in unreferenced_tex
        if any(keyword in path.stem.lower() for keyword in SUSPECT_TEX_KEYWORDS)
    ]

    return {
        "source_root": str(source_root),
        "entrypoint": str(entrypoint.relative_to(source_root)),
        "tex_files": sorted(included_files),
        "all_tex_files": [str(path.relative_to(source_root)) for path in all_tex_files],
        "unreferenced_tex_files": [str(path.relative_to(source_root)) for path in unreferenced_tex],
        "suspect_unreferenced_tex_files": sorted(suspect_unreferenced),
        "figure_references": figures,
        "bibliography_references": sorted(set(bibliographies)),
        "appendix_files": sorted(set(appendices)),
        "data_references": sorted(set(data_refs)),
    }


def ensure_source_ready(project_dir: Path, manifest: dict[str, Any]) -> Path:
    source = manifest.get("source", {})
    location = str(source.get("location", "")).strip()
    extracted_path = str(source.get("extracted_path", "")).strip()
    if extracted_path:
        materialized = (project_dir / extracted_path).resolve()
        if materialized.exists():
            return materialized
    if not location:
        raise PaperReplicationError("Manifest source.location is empty.")
    location_path = Path(location).expanduser()
    if not location_path.is_absolute():
        location_path = (project_dir / location_path).resolve()
    if not location_path.exists():
        raise PaperReplicationError(
            f"Configured source path does not exist: {location_path}",
            details={"source_location": str(location_path)},
        )
    expected_checksum = str(source.get("checksum", "")).strip()
    if expected_checksum:
        actual_checksum = source_checksum(location_path)
        if actual_checksum != expected_checksum:
            raise PaperReplicationError(
                "Paper source checksum mismatch.",
                details={
                    "expected_checksum": expected_checksum,
                    "actual_checksum": actual_checksum,
                    "source_location": str(location_path),
                },
            )
    if location_path.is_dir():
        return location_path.resolve()
    if location_path.suffix.lower() == ".zip":
        extract_root = project_dir / "paper" / "source_tree"
        if extract_root.exists():
            shutil.rmtree(extract_root)
        extract_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(location_path, "r") as archive:
            archive.extractall(extract_root)
        manifest.setdefault("source", {})["extracted_path"] = str(extract_root.relative_to(project_dir))
        save_manifest(project_dir, manifest)
        roots = [child for child in extract_root.iterdir() if child.is_dir()]
        if len(roots) == 1 and not any(extract_root.glob("*.tex")):
            return roots[0].resolve()
        return extract_root.resolve()
    return location_path.parent.resolve()


def matrix_rows(project_dir: Path) -> list[dict[str, str]]:
    matrix_path = project_dir / "spec" / "reproduction_matrix.csv"
    if not matrix_path.exists():
        raise PaperReplicationError(
            f"Missing reproduction matrix: {matrix_path}",
            details={"matrix_path": str(matrix_path)},
        )
    with matrix_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != REQUIRED_MATRIX_COLUMNS:
            raise PaperReplicationError(
                "reproduction_matrix.csv has unexpected columns.",
                details={
                    "expected": REQUIRED_MATRIX_COLUMNS,
                    "actual": reader.fieldnames or [],
                },
            )
        return [row for row in reader if any((value or "").strip() for value in row.values())]


def row_by_target_id(rows: list[dict[str, str]], target_id: str) -> dict[str, str]:
    for row in rows:
        if row.get("target_id", "").strip() == target_id:
            return row
    raise PaperReplicationError(f"Unknown target_id: {target_id}")


def run_records(project_dir: Path) -> list[dict[str, Any]]:
    return read_jsonl(run_index_path(project_dir))


def run_record_by_id(project_dir: Path, run_id: str) -> dict[str, Any]:
    for record in run_records(project_dir):
        if str(record.get("run_id", "")).strip() == run_id:
            return record
    record_path = run_record_path(project_dir, run_id)
    if record_path.exists():
        return read_json(record_path)
    raise PaperReplicationError(f"Unknown run id: {run_id}")


def target_provenance(project_dir: Path, target_id: str) -> dict[str, Any] | None:
    path = provenance_path(project_dir, target_id)
    if not path.exists():
        return None
    data = read_json(path)
    if not isinstance(data, dict):
        raise PaperReplicationError(f"Invalid provenance JSON at {path}")
    return data


def hash_registry(project_dir: Path) -> dict[str, list[str]]:
    registry: dict[str, list[str]] = {}
    asset_index_path = project_dir / "artifacts" / "paper_figures" / "asset_index.json"
    if asset_index_path.exists():
        asset_index = read_json(asset_index_path)
        for asset in asset_index.get("figure_assets", []):
            sha256 = str(asset.get("sha256", "")).strip()
            if not sha256:
                continue
            registry.setdefault(sha256, []).append(str(asset.get("resolved_path", "")))
    page_index_path = project_dir / "artifacts" / "paper_figures" / "page_index.json"
    if page_index_path.exists():
        page_index = read_json(page_index_path)
        for page in page_index.get("pages", []):
            sha256 = str(page.get("sha256", "")).strip()
            if not sha256:
                continue
            registry.setdefault(sha256, []).append(str(page.get("path", "")))
    source_hash_path = project_dir / "artifacts" / "paper_figures" / "source_hashes.json"
    if source_hash_path.exists():
        source_hashes = read_json(source_hash_path)
        for item in source_hashes.get("entries", []):
            sha256 = str(item.get("sha256", "")).strip()
            if not sha256:
                continue
            registry.setdefault(sha256, []).append(str(item.get("path", "")))
    return registry


def source_code_hash_registry(project_dir: Path) -> dict[str, list[str]]:
    path = project_dir / "artifacts" / "paper_figures" / "source_hashes.json"
    if not path.exists():
        return {}
    source_hashes = read_json(path)
    registry: dict[str, list[str]] = {}
    for entry in source_hashes.get("entries", []):
        if str(entry.get("entry_kind", "")).strip() != "source_code":
            continue
        sha256 = str(entry.get("sha256", "")).strip()
        if not sha256:
            continue
        registry.setdefault(sha256, []).append(str(entry.get("path", "")))
    return registry


def artifact_matches_reference(project_dir: Path, artifact_path: Path) -> list[str]:
    sha256 = file_checksum(artifact_path)
    return hash_registry(project_dir).get(sha256, [])


def project_code_matches_source(project_dir: Path) -> list[str]:
    source_registry = source_code_hash_registry(project_dir)
    if not source_registry:
        return []
    errors: list[str] = []
    for relative_root in ("src", "scripts", "configs"):
        root = project_dir / relative_root
        if not root.exists():
            continue
        for file_path in root.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in CODE_EXTENSIONS:
                continue
            sha256 = file_checksum(file_path)
            if sha256 in source_registry:
                errors.append(
                    f"{relative_project_path(project_dir, file_path)} matches source-tree code hash from {', '.join(source_registry[sha256])}."
                )
    return errors


def scan_source_hashes(project_dir: Path, source_root: Path) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for file_path in sorted(path for path in source_root.rglob("*") if path.is_file()):
        suffix = file_path.suffix.lower()
        entry_kind = "other"
        if suffix in CODE_EXTENSIONS:
            entry_kind = "source_code"
        elif suffix in RASTER_EXTENSIONS | VECTOR_EXTENSIONS:
            entry_kind = "reference_asset"
        elif suffix in {".csv", ".tsv", ".json", ".txt"}:
            entry_kind = "tabular_reference"
        entries.append(
            {
                "path": str(file_path.relative_to(source_root)),
                "sha256": file_checksum(file_path),
                "entry_kind": entry_kind,
            }
        )
    payload = {"source_root": str(source_root), "entries": entries}
    output_path = project_dir / "artifacts" / "paper_figures" / "source_hashes.json"
    write_json(output_path, payload)
    return payload


def normalized_status(value: str) -> str:
    return value.strip().upper()


def read_todo_sections(project_dir: Path) -> dict[str, str]:
    todo_path = project_dir / "todo.md"
    if not todo_path.exists():
        raise PaperReplicationError(f"Missing todo.md at {todo_path}")
    text = todo_path.read_text(encoding="utf-8")
    sections: dict[str, str] = {}
    parts = re.split(r"^##\s+", text, flags=re.MULTILINE)
    for part in parts[1:]:
        header, _, body = part.partition("\n")
        sections[header.strip()] = body.strip()
    return sections


def placeholder_file(path: Path) -> bool:
    if not path.exists():
        return True
    text = path.read_text(encoding="utf-8", errors="ignore")
    stripped = text.strip()
    if not stripped:
        return True
    return any(token in text for token in PLACEHOLDER_TOKENS)


def find_active_targets(rows: list[dict[str, str]]) -> list[str]:
    active: list[str] = []
    for row in rows:
        if normalized_status(row.get("status", "")) == "ACTIVE":
            active.append(row.get("target_id", "").strip())
    return active


def count_by_status(rows: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = normalized_status(row.get("status", "")) or "UNSET"
        counts[status] = counts.get(status, 0) + 1
    return counts


def matched_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if normalized_status(row.get("status", "")) in MATCHED_STATUSES]


def incomplete_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if normalized_status(row.get("status", "")) not in MATCHED_STATUSES]


def required_report_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    required: list[dict[str, str]] = []
    for row in matched_rows(rows):
        kind = row.get("kind", "").strip().lower()
        if kind in {"figure", "table"}:
            required.append(row)
    return required


def report_text(project_dir: Path) -> str:
    report_path = project_dir / "report" / "main.tex"
    if not report_path.exists():
        raise PaperReplicationError(f"Missing report template: {report_path}")
    return report_path.read_text(encoding="utf-8", errors="ignore")


def report_errors(project_dir: Path, rows: list[dict[str, str]]) -> list[str]:
    text = report_text(project_dir)
    errors: list[str] = []
    for row in required_report_rows(rows):
        target_id = row.get("target_id", "").strip() or "<missing-target-id>"
        output_path = row.get("output_path", "").strip()
        report_anchor = row.get("report_anchor", "").strip()
        if not output_path:
            errors.append(f"{target_id}: MATCHED rows must declare output_path.")
        elif output_path not in text:
            errors.append(f"{target_id}: report/main.tex does not reference {output_path}.")
        if not report_anchor:
            errors.append(f"{target_id}: MATCHED rows must declare report_anchor.")
        elif report_anchor not in text:
            errors.append(f"{target_id}: report/main.tex does not include report anchor {report_anchor}.")
    return errors


def cluster_delegate_script() -> Path | None:
    env_override = os.environ.get("PAPER_REPLICATION_CLUSTER_SLURM")
    if env_override:
        path = Path(env_override).expanduser().resolve()
        return path if path.exists() else None
    default = Path("~/.codex/skills/cluster-slurm/scripts/cluster_slurm.py").expanduser()
    return default if default.exists() else None


def cluster_delegate_runner_source() -> Path | None:
    runner_path = Path(__file__).resolve().parent / "cluster_reproduce.py"
    return runner_path if runner_path.exists() else None


def running_in_cluster_exec() -> bool:
    value = str(os.environ.get(CLUSTER_EXEC_ENV_VAR, "")).strip().lower()
    return value in {"1", "true", "yes", "on"}


def cluster_delegate_info(project_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    compute = manifest.get("compute", {})
    mode = str(compute.get("mode", "auto")).strip().lower() or "auto"
    script_path = cluster_delegate_script()
    available = script_path is not None
    slug = manifest.get("paper_slug", "paper-replication")
    powershell_reproduce = project_dir / "scripts" / "reproduce_all.ps1"
    reproduce_script = (
        powershell_reproduce
        if os.name == "nt" and powershell_reproduce.exists()
        else project_dir / "scripts" / "reproduce_all.sh"
    )
    command = []
    if available and mode in {"cluster", "auto"}:
        wrapper_path = project_dir / "scripts" / "cluster_reproduce.py"
        if wrapper_path.exists():
            command = [
                sys.executable,
                str(wrapper_path),
                "--project-dir",
                str(project_dir),
            ]
        else:
            command = [
                sys.executable,
                str(script_path),
                "run-workload",
                "--prefix",
                slug,
                "--workload",
                f"paper replication: {slug}",
                "--command",
                f"cd {shell_quote(str(project_dir))} && bash scripts/reproduce_all.sh --local-exec",
                "--wait",
                "--fetch-logs",
                "--tail",
                "200",
            ]
        profile_hint = str(compute.get("cluster_profile_hint", "")).strip()
        if profile_hint:
            command.extend(["--profile", profile_hint])
    return {
        "mode": mode,
        "available": available,
        "script_path": str(script_path) if script_path else "",
        "project_reproduce_script": str(reproduce_script),
        "command": command,
        "command_string": " ".join(shell_quote(part) for part in command) if command else "",
    }


def enforce_policy_contract(manifest: dict[str, Any]) -> list[str]:
    errors = enforce_author_code_policy(manifest)
    policies = manifest.get("policies", {})
    if not bool(policies.get("paper_assets_reference_only", False)):
        errors.append("policies.paper_assets_reference_only must be true.")
    if not bool(policies.get("artifact_provenance_required", False)):
        errors.append("policies.artifact_provenance_required must be true.")
    if not bool(policies.get("comparison_evidence_required", False)):
        errors.append("policies.comparison_evidence_required must be true.")
    if not bool(policies.get("manual_artifact_claims_forbidden", False)):
        errors.append("policies.manual_artifact_claims_forbidden must be true.")
    if str(policies.get("baseline_method_policy", "")).strip() != "paper-faithful":
        errors.append("policies.baseline_method_policy must be 'paper-faithful'.")
    if not bool(policies.get("baseline_method_evidence_required", False)):
        errors.append("policies.baseline_method_evidence_required must be true.")
    if not bool(policies.get("baseline_pattern_matching_forbidden", False)):
        errors.append("policies.baseline_pattern_matching_forbidden must be true.")
    return errors


def comparison_note_placeholder(path: Path) -> bool:
    if not path.exists():
        return True
    text = path.read_text(encoding="utf-8", errors="ignore")
    return PLACEHOLDER_COMPARISON_NOTE in text or text.strip().endswith(f"- {PLACEHOLDER_COMPARISON_NOTE}")


def matched_row_errors(project_dir: Path, row: dict[str, str]) -> list[str]:
    errors: list[str] = []
    target_id = row.get("target_id", "").strip() or "<missing-target-id>"
    output_path = row.get("output_path", "").strip()
    if not output_path:
        return [f"{target_id}: MATCHED rows must define output_path."]

    artifact_path = resolve_project_path(project_dir, output_path)
    if not artifact_path.exists():
        errors.append(f"{target_id}: expected matched artifact is missing at {artifact_path}.")
        return errors

    if not artifact_path.resolve().is_relative_to(project_dir.resolve()):
        errors.append(f"{target_id}: matched artifact must live inside the case-study project.")

    paper_figures_root = (project_dir / "artifacts" / "paper_figures").resolve()
    if artifact_path.resolve().is_relative_to(paper_figures_root):
        errors.append(f"{target_id}: matched artifact may not live under artifacts/paper_figures.")

    source_root = actual_source_path(project_dir, load_manifest(project_dir))
    if source_root is not None and artifact_path.resolve().is_relative_to(source_root.resolve()):
        errors.append(f"{target_id}: matched artifact may not resolve inside the paper source tree.")

    matching_references = artifact_matches_reference(project_dir, artifact_path)
    if matching_references:
        errors.append(
            f"{target_id}: artifact hash matches paper-provided reference content ({', '.join(matching_references)})."
        )

    provenance = target_provenance(project_dir, target_id)
    if provenance is None:
        errors.append(f"{target_id}: MATCHED targets require artifacts/provenance/{target_id}.json.")
    else:
        if str(provenance.get("generated_by", "")).strip() != REGISTER_WRAPPER_NAME:
            errors.append(f"{target_id}: provenance must be generated by {REGISTER_WRAPPER_NAME}.")
        if str(provenance.get("output_path", "")).strip() != output_path:
            errors.append(f"{target_id}: provenance output_path does not match reproduction_matrix.csv.")
        if str(provenance.get("artifact_sha256", "")).strip() != file_checksum(artifact_path):
            errors.append(f"{target_id}: provenance artifact hash does not match the current artifact file.")

        code_path_value = str(provenance.get("code_path", "")).strip()
        code_file: Path | None = None
        if not code_path_value:
            errors.append(f"{target_id}: provenance must include code_path.")
        else:
            code_file = resolve_project_path(project_dir, code_path_value)
            if not code_file.exists():
                errors.append(f"{target_id}: code_path does not exist at {code_file}.")
            else:
                if not code_file.resolve().is_relative_to(project_dir.resolve()):
                    errors.append(f"{target_id}: code_path must live inside the case-study project.")
                if code_file.resolve().is_relative_to((project_dir / "artifacts").resolve()):
                    errors.append(f"{target_id}: code_path may not live under artifacts/.")
                if code_file.resolve().is_relative_to((project_dir / "paper").resolve()):
                    errors.append(f"{target_id}: code_path may not live under paper/.")
                if source_root is not None and code_file.resolve().is_relative_to(source_root.resolve()):
                    errors.append(f"{target_id}: code_path may not resolve inside the paper source tree.")
                if str(provenance.get("code_sha256", "")).strip() != file_checksum(code_file):
                    errors.append(f"{target_id}: code_sha256 does not match the current code_path file.")

        config_path_value = str(provenance.get("config_path", "")).strip()
        config_file: Path | None = None
        if not config_path_value:
            errors.append(f"{target_id}: provenance must include config_path.")
        else:
            config_file = resolve_project_path(project_dir, config_path_value)
            if not config_file.exists():
                errors.append(f"{target_id}: config_path does not exist at {config_file}.")
            else:
                if not config_file.resolve().is_relative_to(project_dir.resolve()):
                    errors.append(f"{target_id}: config_path must live inside the case-study project.")
                if config_file.resolve().is_relative_to((project_dir / "artifacts").resolve()):
                    errors.append(f"{target_id}: config_path may not live under artifacts/.")
                if config_file.resolve().is_relative_to((project_dir / "paper").resolve()):
                    errors.append(f"{target_id}: config_path may not live under paper/.")
                if source_root is not None and config_file.resolve().is_relative_to(source_root.resolve()):
                    errors.append(f"{target_id}: config_path may not resolve inside the paper source tree.")
                if str(provenance.get("config_sha256", "")).strip() != file_checksum(config_file):
                    errors.append(f"{target_id}: config_sha256 does not match the current config_path file.")
                if row.get("config", "").strip() and row.get("config", "").strip() != config_path_value:
                    errors.append(f"{target_id}: provenance config_path does not match reproduction_matrix.csv.")

        paper_trace_value = str(provenance.get("paper_trace_path", "")).strip()
        if not paper_trace_value:
            errors.append(f"{target_id}: provenance must include paper_trace_path.")
        else:
            paper_trace_file = resolve_project_path(project_dir, paper_trace_value)
            if not paper_trace_file.exists():
                errors.append(f"{target_id}: paper_trace_path does not exist at {paper_trace_file}.")
            else:
                if not paper_trace_file.resolve().is_relative_to((project_dir / "spec").resolve()):
                    errors.append(f"{target_id}: paper_trace_path must live under spec/.")
                if str(provenance.get("paper_trace_sha256", "")).strip() != file_checksum(paper_trace_file):
                    errors.append(f"{target_id}: paper_trace_sha256 does not match the current paper_trace_path file.")

        implementation_kind = str(provenance.get("implementation_kind", "")).strip()
        method_components_raw = provenance.get("method_components", [])
        method_components = []
        if isinstance(method_components_raw, list):
            method_components = [str(component).strip() for component in method_components_raw if str(component).strip()]
        if not method_components:
            errors.append(f"{target_id}: baseline provenance must list method_components.")
        implementation_summary = str(provenance.get("implementation_summary", "")).strip()
        if not implementation_summary:
            errors.append(f"{target_id}: baseline provenance must include implementation_summary.")
        if str(provenance.get("claim_mode", "")).strip() != "baseline":
            errors.append(f"{target_id}: baseline paper targets may not be satisfied by non-baseline provenance.")
        if not bool(provenance.get("baseline_faithful", False)):
            errors.append(f"{target_id}: baseline provenance must declare baseline_faithful=true.")
        if str(provenance.get("deviation_notes", "")).strip().lower() not in {"", "none", "n/a"}:
            errors.append(f"{target_id}: baseline provenance may not carry deviation notes.")
        if implementation_kind != "paper-method":
            errors.append(f"{target_id}: baseline provenance must declare implementation_kind=paper-method.")
        run_id = str(provenance.get("run_id", "")).strip()
        if not run_id:
            errors.append(f"{target_id}: provenance must include run_id.")
        else:
            try:
                record = run_record_by_id(project_dir, run_id)
            except PaperReplicationError:
                errors.append(f"{target_id}: provenance run_id {run_id} is not present in artifacts/runs/run_index.jsonl.")
            else:
                if str(record.get("generated_by", "")).strip() != RUN_WRAPPER_NAME:
                    errors.append(f"{target_id}: run ledger entries must be generated by {RUN_WRAPPER_NAME}.")
                if int(record.get("exit_code", 1)) != 0:
                    errors.append(f"{target_id}: run {run_id} did not finish successfully.")
                expected = {str(item.get('path', '')).strip() for item in record.get("expected_artifacts", [])}
                if expected and output_path not in expected:
                    errors.append(f"{target_id}: run {run_id} does not declare {output_path} as an output.")
                suspicious = suspicious_baseline_hits(
                    implementation_summary,
                    str(provenance.get("method_label", "")),
                    str(record.get("command", "")),
                    read_small_text(code_file) if code_file is not None else "",
                )
                if suspicious:
                    errors.append(
                        f"{target_id}: baseline method evidence contains forbidden pattern-matching markers: {', '.join(suspicious)}."
                    )

    kind = row.get("kind", "").strip().lower() or "figure"
    if kind in {"figure", "table"}:
        comparison_json, comparison_note = comparison_paths(project_dir, target_id, kind)
        if not comparison_json.exists():
            errors.append(f"{target_id}: MATCHED {kind} targets require {comparison_json.relative_to(project_dir)}.")
        else:
            comparison_payload = read_json(comparison_json)
            if str(comparison_payload.get("generated_by", "")).strip() != COMPARISON_WRAPPER_NAME:
                errors.append(f"{target_id}: comparison JSON must be generated by {COMPARISON_WRAPPER_NAME}.")
            acceptance_mode = inferred_acceptance_mode(row, comparison_payload)
            payload_mode = normalized_acceptance_mode(str(comparison_payload.get("acceptance_mode", "") or ""))
            if payload_mode and payload_mode != acceptance_mode:
                errors.append(
                    f"{target_id}: comparison evidence acceptance_mode {payload_mode!r} does not match "
                    f"the target acceptance mode {acceptance_mode!r}."
                )
            metrics = comparison_evidence_metrics(comparison_payload)
            metric_keys = non_metadata_metric_keys(metrics)
            visual_keys = visual_metric_keys(metrics)
            comparison_metric = primary_metric_name(row)
            if acceptance_mode == "exact-visual":
                if comparison_metric and not is_visual_metric_name(comparison_metric):
                    errors.append(f"{target_id}: exact-visual targets must use a visual comparison_metric.")
                if not visual_keys:
                    errors.append(f"{target_id}: exact-visual targets require at least one visual comparison metric.")
                if "reference_path" not in metrics or "candidate_path" not in metrics:
                    errors.append(f"{target_id}: exact-visual targets require reference_path and candidate_path evidence.")
            elif acceptance_mode == "numeric-equivalence":
                if comparison_metric and is_visual_metric_name(comparison_metric):
                    errors.append(
                        f"{target_id}: numeric-equivalence targets may not use a purely visual comparison_metric."
                    )
                if not metric_keys:
                    errors.append(f"{target_id}: numeric-equivalence targets require numeric comparison metrics.")
                elif set(metric_keys).issubset(set(visual_keys)):
                    errors.append(
                        f"{target_id}: numeric-equivalence targets require non-visual metrics beyond SSIM/pixel similarity."
                    )
            elif acceptance_mode == "distributional-equivalence":
                if not metric_keys:
                    errors.append(f"{target_id}: distributional-equivalence targets require comparison metrics.")
                elif set(metric_keys).issubset(set(visual_keys)):
                    errors.append(
                        f"{target_id}: distributional-equivalence targets require non-visual distributional metrics."
                    )
            elif acceptance_mode == "qualitative-structural":
                if comparison_metric and is_visual_metric_name(comparison_metric) and not metric_keys:
                    errors.append(
                        f"{target_id}: qualitative-structural targets may not rely on visual metrics alone without rationale."
                    )
        if not comparison_note.exists():
            errors.append(f"{target_id}: MATCHED {kind} targets require {comparison_note.relative_to(project_dir)}.")
        elif comparison_note_placeholder(comparison_note):
            errors.append(f"{target_id}: comparison note still contains placeholder text.")
    return errors


def validate_required_files(project_dir: Path) -> list[str]:
    errors: list[str] = []
    manifest_path = project_dir / "paper_manifest.json"
    if not manifest_path.exists():
        errors.append("Missing paper_manifest.json.")
    if not (project_dir / "spec" / "reproduction_matrix.csv").exists():
        errors.append("Missing spec/reproduction_matrix.csv.")
    if not (project_dir / "todo.md").exists():
        errors.append("Missing todo.md.")
    for relative in REQUIRED_SPEC_DOCS:
        if not (project_dir / relative).exists():
            errors.append(f"Missing {relative}.")
    return errors


def validate_spec(project_dir: Path) -> dict[str, Any]:
    errors = validate_required_files(project_dir)
    if errors:
        return {"ok": False, "errors": errors}

    manifest = load_manifest(project_dir)
    errors.extend(enforce_policy_contract(manifest))
    missing_manifest_fields = []
    for key in ("paper_title", "paper_slug", "source_mode"):
        if not str(manifest.get(key, "")).strip():
            missing_manifest_fields.append(key)
    source = manifest.get("source", {})
    if not str(source.get("location", "")).strip():
        missing_manifest_fields.append("source.location")
    if missing_manifest_fields:
        errors.append("Missing manifest fields: " + ", ".join(missing_manifest_fields))

    todo_sections = read_todo_sections(project_dir)
    for header in REQUIRED_TODO_HEADERS:
        if header not in todo_sections:
            errors.append(f"todo.md missing section: {header}.")
    inventory_path = project_dir / "spec" / "paper_inventory.json"
    if not inventory_path.exists():
        errors.append("Missing spec/paper_inventory.json. Run inspect-paper first.")
    if not (project_dir / "artifacts" / "paper_figures" / "asset_index.json").exists():
        errors.append("Missing artifacts/paper_figures/asset_index.json. Run inspect-paper first.")
    if not (project_dir / "artifacts" / "paper_figures" / "source_hashes.json").exists():
        errors.append("Missing artifacts/paper_figures/source_hashes.json. Run inspect-paper first.")

    for relative in REQUIRED_SPEC_DOCS:
        if placeholder_file(project_dir / relative):
            errors.append(f"{relative} still contains placeholder content.")

    try:
        rows = matrix_rows(project_dir)
    except PaperReplicationError as exc:
        errors.append(str(exc))
        rows = []

    if not rows:
        errors.append("spec/reproduction_matrix.csv must contain at least one target row.")
    else:
        for row in rows:
            target_id = row.get("target_id", "").strip()
            if not target_id:
                errors.append("All reproduction matrix rows must define target_id.")
            for required in ("kind", "paper_locator", "runner", "output_path", "status"):
                if not row.get(required, "").strip():
                    errors.append(f"{target_id or '<missing-target-id>'}: missing {required} in reproduction_matrix.csv.")
            acceptance_mode = str(row.get("acceptance_mode", "") or "").strip()
            if acceptance_mode and not normalized_acceptance_mode(acceptance_mode):
                errors.append(
                    f"{target_id or '<missing-target-id>'}: invalid acceptance_mode {acceptance_mode!r}. "
                    f"Use one of: {', '.join(sorted(ACCEPTANCE_MODES))}."
                )

    return {"ok": not errors, "errors": errors}


def validate_progress(project_dir: Path) -> dict[str, Any]:
    errors = validate_required_files(project_dir)
    if errors:
        return {"ok": False, "errors": errors}

    manifest = load_manifest(project_dir)
    errors.extend(enforce_policy_contract(manifest))
    rows = matrix_rows(project_dir)
    todo_sections = read_todo_sections(project_dir)

    active_targets = find_active_targets(rows)
    if len(active_targets) > 1:
        errors.append("reproduction_matrix.csv may only contain one ACTIVE target.")
    if not rows:
        errors.append("reproduction_matrix.csv must contain at least one target row.")
    elif not active_targets and any(normalized_status(row.get("status", "")) not in TERMINAL_STATUSES for row in rows):
        errors.append("At least one target must be ACTIVE until every row is terminal.")

    todo_active = todo_sections.get("Active target", "").strip().splitlines()
    todo_active_target = todo_active[0].strip() if todo_active else ""
    if active_targets:
        if todo_active_target != active_targets[0]:
            errors.append(
                f"todo.md Active target ({todo_active_target or 'UNSET'}) does not match matrix ACTIVE target ({active_targets[0]})."
            )
    elif todo_active_target not in {"", "UNSET"}:
        errors.append("todo.md Active target must be UNSET when no matrix row is ACTIVE.")

    reproduce_command = str(manifest.get("reproduce", {}).get("command", "")).strip()
    if not reproduce_command:
        errors.append("paper_manifest.json missing reproduce.command.")
    elif reproduce_command == "bash scripts/reproduce_all.sh" and not (project_dir / "scripts" / "reproduce_all.sh").exists():
        errors.append("scripts/reproduce_all.sh is missing.")

    errors.extend(project_code_matches_source(project_dir))

    for row in matched_rows(rows):
        errors.extend(matched_row_errors(project_dir, row))

    errors.extend(report_errors(project_dir, rows))

    compute_mode = str(manifest.get("compute", {}).get("mode", "auto")).strip().lower()
    delegate = cluster_delegate_info(project_dir, manifest)
    if compute_mode == "cluster" and not delegate["available"] and not running_in_cluster_exec():
        errors.append("compute.mode=cluster but cluster-slurm is not installed or not discoverable.")

    return {"ok": not errors, "errors": errors}


def validate_report(project_dir: Path) -> dict[str, Any]:
    rows = matrix_rows(project_dir)
    errors = report_errors(project_dir, rows)
    return {"ok": not errors, "errors": errors}


def validate_completion(project_dir: Path) -> dict[str, Any]:
    errors: list[str] = []
    spec_result = validate_spec(project_dir)
    progress_result = validate_progress(project_dir)
    report_result = validate_report(project_dir)

    if not spec_result["ok"]:
        errors.extend(spec_result["errors"])
    if not progress_result["ok"]:
        errors.extend(progress_result["errors"])
    if not report_result["ok"]:
        errors.extend(report_result["errors"])

    rows: list[dict[str, str]] = []
    try:
        rows = matrix_rows(project_dir)
    except PaperReplicationError as exc:
        errors.append(str(exc))

    incomplete = incomplete_rows(rows) if rows else []
    if incomplete:
        summary = ", ".join(
            f"{row.get('target_id', '<missing-target-id>')}:{normalized_status(row.get('status', '')) or 'UNSET'}"
            for row in incomplete[:10]
        )
        if len(incomplete) > 10:
            summary += f", ... (+{len(incomplete) - 10} more)"
        errors.append(
            "Paper replication is incomplete. Remaining targets are not MATCHED: "
            + summary
            + "."
        )

    if rows:
        active_targets = find_active_targets(rows)
        if active_targets:
            errors.append("Paper replication is incomplete while an ACTIVE target remains in reproduction_matrix.csv.")

    try:
        todo_sections = read_todo_sections(project_dir)
    except PaperReplicationError as exc:
        errors.append(str(exc))
    else:
        todo_active = todo_sections.get("Active target", "").strip().splitlines()
        todo_active_target = todo_active[0].strip() if todo_active else ""
        if todo_active_target not in {"", "UNSET"}:
            errors.append("Paper replication is incomplete while todo.md Active target is still set.")

    report_pdf = project_dir / "report" / "main.pdf"
    if not report_pdf.exists():
        errors.append("report/main.pdf is missing. Compile the final report PDF before claiming completion.")

    return {
        "ok": not errors,
        "errors": errors,
        "matched_target_count": len(matched_rows(rows)),
        "total_target_count": len(rows),
        "incomplete_targets": [
            {
                "target_id": row.get("target_id", "").strip(),
                "status": normalized_status(row.get("status", "")) or "UNSET",
            }
            for row in incomplete
        ],
    }


def detect_latex_builder() -> list[str] | None:
    if shutil.which("latexmk"):
        return ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error"]
    if shutil.which("tectonic"):
        return ["tectonic"]
    if shutil.which("pdflatex"):
        return ["pdflatex", "-interaction=nonstopmode", "-halt-on-error"]
    return None


def build_paper_pdf(project_dir: Path) -> dict[str, Any]:
    manifest = load_manifest(project_dir)
    source_root = ensure_source_ready(project_dir, manifest)
    source = manifest.setdefault("source", {})
    entrypoint_value = str(source.get("entrypoint", "")).strip()
    entrypoint = (source_root / entrypoint_value).resolve() if entrypoint_value else detect_main_tex(source_root)
    if not entrypoint.exists():
        raise PaperReplicationError(
            f"Configured TeX entrypoint does not exist: {entrypoint}",
            details={"entrypoint": str(entrypoint)},
        )
    builder = detect_latex_builder()
    if builder is None:
        raise PaperReplicationError(
            "No LaTeX builder found. Install latexmk, tectonic, or pdflatex to build the paper PDF."
        )

    build_dir = project_dir / "artifacts" / "paper_build"
    build_dir.mkdir(parents=True, exist_ok=True)

    if builder[0] == "latexmk":
        command = builder + [f"-output-directory={build_dir}", str(entrypoint)]
        cwd = source_root
    elif builder[0] == "tectonic":
        command = builder + ["--outdir", str(build_dir), str(entrypoint)]
        cwd = source_root
    else:
        command = builder + [f"-output-directory={build_dir}", str(entrypoint)]
        cwd = source_root

    proc = subprocess.run(command, cwd=str(cwd), text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        raise PaperReplicationError(
            "Paper PDF build failed.",
            details={"stdout": proc.stdout, "stderr": proc.stderr, "command": command},
        )

    expected_pdf = build_dir / f"{entrypoint.stem}.pdf"
    if not expected_pdf.exists():
        pdf_candidates = sorted(build_dir.glob("*.pdf"))
        if not pdf_candidates:
            raise PaperReplicationError(f"No PDF was produced under {build_dir}.")
        expected_pdf = pdf_candidates[0]

    source["entrypoint"] = str(entrypoint.relative_to(source_root))
    source["compiled_pdf"] = str(expected_pdf.relative_to(project_dir))
    save_manifest(project_dir, manifest)
    return {
        "builder": builder[0],
        "source_root": str(source_root),
        "entrypoint": str(entrypoint),
        "pdf_path": str(expected_pdf),
    }


def render_paper_pages(project_dir: Path, pdf_path: str | None = None) -> dict[str, Any]:
    manifest = load_manifest(project_dir)
    if pdf_path:
        resolved_pdf = Path(pdf_path).expanduser()
        if not resolved_pdf.is_absolute():
            resolved_pdf = (project_dir / resolved_pdf).resolve()
    else:
        compiled_pdf = str(manifest.get("source", {}).get("compiled_pdf", "")).strip()
        if not compiled_pdf:
            build = build_paper_pdf(project_dir)
            resolved_pdf = Path(build["pdf_path"])
        else:
            resolved_pdf = (project_dir / compiled_pdf).resolve()
    if not resolved_pdf.exists():
        raise PaperReplicationError(f"Compiled PDF does not exist: {resolved_pdf}")

    output_dir = project_dir / "artifacts" / "paper_figures" / "pages"
    output_dir.mkdir(parents=True, exist_ok=True)

    if shutil.which("pdftoppm"):
        prefix = output_dir / "page"
        command = ["pdftoppm", "-png", str(resolved_pdf), str(prefix)]
        proc = subprocess.run(command, text=True, capture_output=True, check=False)
        if proc.returncode != 0:
            raise PaperReplicationError(
                "Failed to render PDF pages with pdftoppm.",
                details={"stdout": proc.stdout, "stderr": proc.stderr},
            )
    elif shutil.which("mutool"):
        command = ["mutool", "draw", "-o", str(output_dir / "page-%03d.png"), str(resolved_pdf)]
        proc = subprocess.run(command, text=True, capture_output=True, check=False)
        if proc.returncode != 0:
            raise PaperReplicationError(
                "Failed to render PDF pages with mutool.",
                details={"stdout": proc.stdout, "stderr": proc.stderr},
            )
    else:
        raise PaperReplicationError("Neither pdftoppm nor mutool is available to render PDF pages.")

    rendered = []
    for path in sorted(output_dir.glob("*.png")):
        rendered.append({"path": str(path.relative_to(project_dir)), "sha256": file_checksum(path)})
    page_index_path = project_dir / "artifacts" / "paper_figures" / "page_index.json"
    write_json(page_index_path, {"pdf_path": str(resolved_pdf), "pages": rendered})
    return {"pdf_path": str(resolved_pdf), "rendered_pages": [item["path"] for item in rendered], "page_index_path": str(page_index_path)}


def index_paper_assets(project_dir: Path) -> dict[str, Any]:
    manifest = load_manifest(project_dir)
    source_root = ensure_source_ready(project_dir, manifest)
    entrypoint_value = str(manifest.get("source", {}).get("entrypoint", "")).strip()
    entrypoint = (source_root / entrypoint_value).resolve() if entrypoint_value else detect_main_tex(source_root)
    inventory = inventory_latex_tree(source_root, entrypoint)
    indexed_assets = []
    for figure in inventory["figure_references"]:
        resolved = figure.get("resolved_path", "")
        extension = Path(resolved).suffix.lower() if resolved else ""
        asset_kind = "missing"
        if extension in RASTER_EXTENSIONS:
            asset_kind = "raster"
        elif extension in VECTOR_EXTENSIONS:
            asset_kind = "vector"
        indexed_assets.append(
            {
                "raw_reference": figure["raw_reference"],
                "referenced_from": figure["referenced_from"],
                "resolved_path": resolved,
                "exists": figure["exists"],
                "asset_kind": asset_kind,
                "source_classification": "original_asset",
                "sha256": file_checksum(source_root / resolved) if resolved and (source_root / resolved).exists() else "",
            }
        )
    output = {
        "source_root": str(source_root),
        "entrypoint": inventory["entrypoint"],
        "tex_files": inventory["tex_files"],
        "bibliography_references": inventory["bibliography_references"],
        "appendix_files": inventory["appendix_files"],
        "data_references": inventory["data_references"],
        "figure_assets": indexed_assets,
    }
    output_path = project_dir / "artifacts" / "paper_figures" / "asset_index.json"
    write_json(output_path, output)
    source_hashes = scan_source_hashes(project_dir, source_root)
    return {
        "asset_index_path": str(output_path),
        "figure_count": len(indexed_assets),
        "source_hash_count": len(source_hashes["entries"]),
    }


def inspect_paper(project_dir: Path) -> dict[str, Any]:
    manifest = load_manifest(project_dir)
    source_root = ensure_source_ready(project_dir, manifest)
    source = manifest.setdefault("source", {})
    entrypoint_value = str(source.get("entrypoint", "")).strip()
    entrypoint = (source_root / entrypoint_value).resolve() if entrypoint_value else detect_main_tex(source_root)
    inventory = inventory_latex_tree(source_root, entrypoint)
    source_path = actual_source_path(project_dir, manifest)
    inventory["checksum"] = source_checksum(source_path) if source_path is not None else ""
    inventory_path = project_dir / "spec" / "paper_inventory.json"
    write_json(inventory_path, inventory)
    source["entrypoint"] = str(entrypoint.relative_to(source_root))
    save_manifest(project_dir, manifest)
    asset_index = index_paper_assets(project_dir)
    return {
        "inventory_path": str(inventory_path),
        "tex_file_count": len(inventory["tex_files"]),
        "figure_reference_count": len(inventory["figure_references"]),
        "appendix_count": len(inventory["appendix_files"]),
        "suspect_unreferenced_tex_count": len(inventory.get("suspect_unreferenced_tex_files", [])),
        "asset_index_path": asset_index["asset_index_path"],
    }


def compare_figures(
    project_dir: Path,
    reference_path: str,
    candidate_path: str,
    output_dir: str | None = None,
    target_id: str | None = None,
) -> dict[str, Any]:
    reference = Path(reference_path).expanduser()
    candidate = Path(candidate_path).expanduser()
    if not reference.is_absolute():
        reference = (project_dir / reference).resolve()
    if not candidate.is_absolute():
        candidate = (project_dir / candidate).resolve()
    if not reference.exists():
        raise PaperReplicationError(f"Reference figure does not exist: {reference}")
    if not candidate.exists():
        raise PaperReplicationError(f"Candidate figure does not exist: {candidate}")

    metrics: dict[str, Any] = {
        "reference_path": str(reference),
        "candidate_path": str(candidate),
        "reference_sha256": file_checksum(reference),
        "candidate_sha256": file_checksum(candidate),
        "reference_size_bytes": reference.stat().st_size,
        "candidate_size_bytes": candidate.stat().st_size,
    }
    metrics["bytes_identical"] = metrics["reference_sha256"] == metrics["candidate_sha256"]
    try:
        from PIL import Image, ImageChops, ImageStat  # type: ignore

        ref_image = Image.open(reference).convert("RGB")
        cand_image = Image.open(candidate).convert("RGB")
        metrics["reference_dimensions"] = list(ref_image.size)
        metrics["candidate_dimensions"] = list(cand_image.size)
        if ref_image.size == cand_image.size:
            diff = ImageChops.difference(ref_image, cand_image)
            stat = ImageStat.Stat(diff)
            metrics["mean_abs_diff"] = sum(stat.mean) / len(stat.mean)
        else:
            metrics["mean_abs_diff"] = None
    except Exception:
        metrics["reference_dimensions"] = None
        metrics["candidate_dimensions"] = None
        metrics["mean_abs_diff"] = None

    comparison_dir = project_dir / "artifacts" / "figures" / "comparison"
    if output_dir:
        comparison_dir = Path(output_dir).expanduser()
        if not comparison_dir.is_absolute():
            comparison_dir = (project_dir / comparison_dir).resolve()
    comparison_dir.mkdir(parents=True, exist_ok=True)
    stem = target_id or f"{reference.stem}__vs__{candidate.stem}"
    json_path = comparison_dir / f"{stem}.json"
    md_path = comparison_dir / f"{stem}.md"
    kind = "figure"
    if target_id:
        comparison_result = record_comparison(
            project_dir,
            target_id=target_id,
            kind=kind,
            note="",
            metrics=metrics,
            acceptance_mode="exact-visual",
        )
        return {
            "comparison_json": comparison_result["comparison_json"],
            "comparison_note": comparison_result["comparison_note"],
            "metrics": metrics,
        }
    write_json(json_path, metrics)
    write_text(
        md_path,
        textwrap.dedent(
            f"""\
            # Figure comparison: {stem}

            - Reference: `{reference}`
            - Candidate: `{candidate}`
            - Bytes identical: `{metrics["bytes_identical"]}`
            - Mean absolute diff: `{metrics["mean_abs_diff"]}`
            - Generated by: `{COMPARISON_WRAPPER_NAME}`

            Match rationale:
            - {PLACEHOLDER_COMPARISON_NOTE}
            """
        ),
    )
    return {"comparison_json": str(json_path), "comparison_note": str(md_path), "metrics": metrics}


def tracked_run(
    project_dir: Path,
    shell_command: str,
    *,
    label: str,
    cwd: str | None,
    expected_artifacts: list[str],
) -> dict[str, Any]:
    run_id = next_run_id(project_dir)
    run_dir = project_dir / "artifacts" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    started_at = utc_now_iso()
    execution_cwd = resolve_project_path(project_dir, cwd) if cwd else project_dir
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    if os.name == "nt":
        shell_executable = shutil.which("pwsh") or shutil.which("powershell")
        if shell_executable is None:
            raise PaperReplicationError(
                "track-run requires PowerShell on Windows, but neither pwsh nor "
                "powershell was found."
            )
        command: str | list[str] = [
            shell_executable,
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            shell_command,
        ]
        shell = False
        executable = None
    else:
        command = shell_command
        shell = True
        executable = "/bin/zsh"
    proc = subprocess.run(
        command,
        cwd=str(execution_cwd),
        shell=shell,
        executable=executable,
        text=True,
        capture_output=True,
        check=False,
    )
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")
    expected_outputs = []
    for raw in expected_artifacts:
        artifact_path = resolve_project_path(project_dir, raw)
        expected_outputs.append(
            {
                "path": relative_project_path(project_dir, artifact_path),
                "exists": artifact_path.exists(),
                "sha256": file_checksum(artifact_path) if artifact_path.exists() else "",
            }
        )
    record = {
        "schema_version": SCHEMA_VERSION,
        "generated_by": RUN_WRAPPER_NAME,
        "run_id": run_id,
        "label": label,
        "started_at": started_at,
        "finished_at": utc_now_iso(),
        "cwd": relative_project_path(project_dir, execution_cwd),
        "command": shell_command,
        "exit_code": proc.returncode,
        "status": "success" if proc.returncode == 0 else "failure",
        "stdout_path": relative_project_path(project_dir, stdout_path),
        "stderr_path": relative_project_path(project_dir, stderr_path),
        "expected_artifacts": expected_outputs,
        "record_path": relative_project_path(project_dir, run_record_path(project_dir, run_id)),
    }
    write_json(run_record_path(project_dir, run_id), record)
    append_jsonl(run_index_path(project_dir), record)
    return record


def register_target_artifact(
    project_dir: Path,
    *,
    target_id: str,
    run_id: str,
    artifact_path: str | None,
    claim_mode: str,
    method_label: str,
    code_path: str,
    config_path: str,
    paper_trace_path: str,
    seed: str,
    baseline_faithful: bool,
    deviation_notes: str,
    implementation_kind: str,
    method_components: list[str],
    implementation_summary: str,
) -> dict[str, Any]:
    rows = matrix_rows(project_dir)
    row = row_by_target_id(rows, target_id)
    target_artifact = artifact_path or row.get("output_path", "").strip()
    if not target_artifact:
        raise PaperReplicationError(f"{target_id}: output_path is missing in reproduction_matrix.csv.")
    artifact = resolve_project_path(project_dir, target_artifact)
    if not artifact.exists():
        raise PaperReplicationError(f"{target_id}: artifact does not exist at {artifact}.")

    record = run_record_by_id(project_dir, run_id)
    if int(record.get("exit_code", 1)) != 0:
        raise PaperReplicationError(f"{target_id}: run {run_id} did not finish successfully.")

    expected_paths = {str(item.get("path", "")).strip() for item in record.get("expected_artifacts", [])}
    relative_artifact = relative_project_path(project_dir, artifact)
    if expected_paths and relative_artifact not in expected_paths:
        raise PaperReplicationError(
            f"{target_id}: artifact {relative_artifact} was not declared as an expected output of run {run_id}."
        )

    code_file = resolve_project_path(project_dir, code_path)
    if not code_file.exists():
        raise PaperReplicationError(f"{target_id}: code_path does not exist at {code_file}.")
    config_file = resolve_project_path(project_dir, config_path)
    if not config_file.exists():
        raise PaperReplicationError(f"{target_id}: config_path does not exist at {config_file}.")
    paper_trace_file = resolve_project_path(project_dir, paper_trace_path)
    if not paper_trace_file.exists():
        raise PaperReplicationError(f"{target_id}: paper_trace_path does not exist at {paper_trace_file}.")

    cleaned_components = [component.strip() for component in method_components if component.strip()]
    if not cleaned_components:
        raise PaperReplicationError(f"{target_id}: at least one method component is required.")
    summary = implementation_summary.strip()
    if not summary:
        raise PaperReplicationError(f"{target_id}: implementation_summary is required.")

    if claim_mode == "baseline":
        if implementation_kind.strip() != "paper-method":
            raise PaperReplicationError(f"{target_id}: baseline claims require implementation_kind=paper-method.")
        suspicious = suspicious_baseline_hits(
            method_label,
            summary,
            str(record.get("command", "")),
            read_small_text(code_file),
        )
        if suspicious:
            raise PaperReplicationError(
                f"{target_id}: baseline method evidence contains forbidden pattern-matching markers: {', '.join(suspicious)}."
            )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_by": REGISTER_WRAPPER_NAME,
        "target_id": target_id,
        "kind": row.get("kind", "").strip().lower(),
        "output_path": relative_artifact,
        "artifact_sha256": file_checksum(artifact),
        "run_id": run_id,
        "run_record_path": str(record.get("record_path", "")),
        "claim_mode": claim_mode,
        "method_label": method_label,
        "code_path": relative_project_path(project_dir, code_file),
        "code_sha256": file_checksum(code_file),
        "config_path": relative_project_path(project_dir, config_file),
        "config_sha256": file_checksum(config_file),
        "paper_trace_path": relative_project_path(project_dir, paper_trace_file),
        "paper_trace_sha256": file_checksum(paper_trace_file),
        "seed": seed,
        "baseline_faithful": bool(baseline_faithful),
        "deviation_notes": deviation_notes,
        "implementation_kind": implementation_kind.strip(),
        "method_components": cleaned_components,
        "implementation_summary": summary,
        "created_at": utc_now_iso(),
    }
    write_json(provenance_path(project_dir, target_id), payload)
    return payload


def record_comparison(
    project_dir: Path,
    *,
    target_id: str,
    kind: str,
    note: str,
    metrics: dict[str, Any],
    acceptance_mode: str | None = None,
) -> dict[str, Any]:
    resolved_acceptance_mode = normalized_acceptance_mode(acceptance_mode or "")
    if not resolved_acceptance_mode:
        synthetic_row = {"comparison_metric": "", "acceptance_mode": ""}
        if metrics:
            synthetic_row["comparison_metric"] = next(iter(metrics.keys()))
        resolved_acceptance_mode = inferred_acceptance_mode(synthetic_row)
    json_path, note_path = comparison_paths(project_dir, target_id, kind)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_by": COMPARISON_WRAPPER_NAME,
        "target_id": target_id,
        "kind": kind,
        "acceptance_mode": resolved_acceptance_mode,
        "metrics": metrics,
        "created_at": utc_now_iso(),
    }
    write_json(json_path, payload)
    note_body = note.strip() if note.strip() else PLACEHOLDER_COMPARISON_NOTE
    write_text(
        note_path,
        textwrap.dedent(
            f"""\
            # Comparison note: {target_id}

            {note_body}
            """
        ),
    )
    return {"comparison_json": str(json_path), "comparison_note": str(note_path)}


def render_markdown_template(title: str, body: str) -> str:
    return textwrap.dedent(
        f"""\
        # {title}

        {body}
        """
    )


def project_gitignore() -> str:
    return textwrap.dedent(
        """\
        paper/paper_src.zip
        paper/source_tree/
        data/raw/*
        !data/raw/.gitkeep
        data/processed/*
        !data/processed/.gitkeep
        artifacts/runs/*
        !artifacts/runs/.gitkeep
        !artifacts/runs/run_index.jsonl
        !artifacts/runs/*/
        artifacts/runs/*/*
        !artifacts/runs/*/record.json
        artifacts/paper_build/*
        !artifacts/paper_build/.gitkeep
        report/*.aux
        report/*.bbl
        report/*.blg
        report/*.fdb_latexmk
        report/*.fls
        report/*.log
        report/*.out
        report/*.toc
        """
    )


def todo_template() -> str:
    return textwrap.dedent(
        """\
        # TODO

        ## Current phase
        paper-intake

        ## Active target
        UNSET

        ## Acceptance gates
        - [ ] Run `python3 scripts/paper_harness.py inspect-paper --project-dir .`
        - [ ] Enumerate every target in `spec/reproduction_matrix.csv`
        - [ ] Declare an `acceptance_mode` for targets that should be judged by numeric, distributional, or qualitative equivalence instead of exact visual matching
        - [ ] Replace placeholder sections in `spec/*.md`
        - [ ] Keep exactly one ACTIVE target until all rows are terminal
        - [ ] Register every claimed artifact through the run wrapper and provenance files
        - [ ] Register baseline claims only with paper-method evidence (`code_path`, `config_path`, `paper_trace_path`, `method_components`, `implementation_summary`)
        - [ ] Ensure every MATCHED figure/table has comparison evidence and appears in `report/main.tex`
        - [ ] `validate-completion` passes only when the whole paper is genuinely complete

        ## Open unknowns
        - Capture missing paper details here with explicit hypotheses and evidence.

        ## Completed
        - Bootstrap completed.
        """
    )


def readme_template(paper_title: str, paper_slug: str) -> str:
    return textwrap.dedent(
        f"""\
        # {paper_title}

        This case study was scaffolded for a LaTeX-first paper replication workflow.

        ## Core commands

        ```bash
        python3 scripts/paper_harness.py status --project-dir .
        python3 scripts/paper_harness.py inspect-paper --project-dir .
        python3 scripts/paper_harness.py validate-completion --project-dir .
        bash scripts/reproduce_all.sh
        ```

        ## Expected workflow

        1. Inspect the paper sources and inventory the TeX tree.
        2. Fill out the spec files under `spec/`.
        3. Keep `spec/reproduction_matrix.csv` and `todo.md` in sync, including per-target `acceptance_mode`.
        4. Mark exactly one ACTIVE target at a time.
        5. Use `track-run`, `register-target-artifact`, and comparison evidence before moving any target to MATCHED.
        6. Register baseline targets only after a real paper-method implementation exists under project code, config, and spec trace files.
        7. Use exact visual matching only for targets that truly require it; convergence curves and other stochastic summaries should usually use numeric or structural acceptance.
        8. Move a target to MATCHED only after the artifact exists, provenance is recorded, and `report/main.tex` embeds it.
        9. Treat the paper as done only when `validate-completion` passes.

        ## Case study id

        - Paper slug: `{paper_slug}`
        """
    )


def report_template(paper_title: str) -> str:
    return textwrap.dedent(
        f"""\
        \\documentclass{{article}}
        \\usepackage[margin=1in]{{geometry}}
        \\usepackage{{graphicx}}
        \\usepackage{{booktabs}}
        \\usepackage{{float}}
        \\title{{Replication Report: {paper_title}}}
        \\author{{}}
        \\date{{}}

        \\begin{{document}}
        \\maketitle

        \\section{{Overview}}
        Replace this section with the concise replication summary.

        \\section{{Targets}}
        Embed every matched figure/table using generated artifact paths.
        Keep the `report_anchor` token from `spec/reproduction_matrix.csv` next to each embedded artifact.

        % Example:
        % \\begin{{figure}}[H]
        %   \\centering
        %   \\includegraphics[width=0.9\\linewidth]{{artifacts/figures/example.png}}
        %   \\caption{{Reproduced target.}}
        %   \\label{{fig:example}}
        % \\end{{figure}}
        % report_anchor=fig:example

        \\section{{Notes}}
        Record deviations, residual gaps, and hardware/runtime notes here.

        \\bibliographystyle{{plain}}
        \\bibliography{{refs}}
        \\end{{document}}
        """
    )


def spec_doc_template(title: str, bullets: list[str]) -> str:
    bullet_lines = "\n".join(f"- TODO: {item}" for item in bullets)
    return render_markdown_template(title, bullet_lines)


def matrix_header() -> str:
    return ",".join(REQUIRED_MATRIX_COLUMNS) + "\n"


def manifest_template(
    paper_title: str,
    paper_slug: str,
    paper_source: str,
    source_checksum_value: str,
    main_tex: str,
    source_mode: str,
    author_code_policy: str,
    stack_policy: str,
    compute_mode: str,
    cluster_profile_hint: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "paper_title": paper_title,
        "paper_slug": paper_slug,
        "source_mode": source_mode,
        "source": {
            "location": paper_source,
            "checksum": source_checksum_value,
            "entrypoint": main_tex,
            "compiled_pdf": "",
            "extracted_path": "",
        },
        "policies": {
            "author_code_policy": author_code_policy,
            "author_code_paths": [],
            "stack_policy": stack_policy,
            "target_progression": "single-active-target",
            "paper_assets_reference_only": True,
            "artifact_provenance_required": True,
            "comparison_evidence_required": True,
            "manual_artifact_claims_forbidden": True,
            "baseline_method_policy": "paper-faithful",
            "baseline_method_evidence_required": True,
            "baseline_pattern_matching_forbidden": True,
        },
        "compute": {
            "mode": compute_mode,
            "cluster_profile_hint": cluster_profile_hint,
        },
        "reproduce": {
            "command": "bash scripts/reproduce_all.sh",
            "report_path": "report/main.tex",
        },
        "state": {
            "current_phase": "paper-intake",
        },
    }


def python_wrapper(command_name: str) -> str:
    return textwrap.dedent(
        f"""\
        #!/usr/bin/env python3
        import pathlib
        import subprocess
        import sys

        script_dir = pathlib.Path(__file__).resolve().parent
        project_dir = script_dir.parent
        command = [
            sys.executable,
            str(script_dir / "paper_harness.py"),
            "{command_name}",
            "--project-dir",
            str(project_dir),
            *sys.argv[1:],
        ]
        raise SystemExit(subprocess.run(command, check=False).returncode)
        """
    )


def shell_wrapper(command_name: str) -> str:
    return textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        set -euo pipefail
        SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
        PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
        python3 "$SCRIPT_DIR/paper_harness.py" "{command_name}" --project-dir "$PROJECT_DIR" "$@"
        """
    )


def reproduce_all_script() -> str:
    return textwrap.dedent(
        """\
        #!/usr/bin/env bash
        set -euo pipefail
        SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
        python3 "$SCRIPT_DIR/paper_harness.py" run-reproduce --project-dir "$PROJECT_DIR" "$@"
        """
    )


def sync_generated_scripts(project_dir: Path, *, rewrite_reproduce_all: bool = False) -> list[str]:
    synced_files: list[str] = []
    harness_path = project_dir / "scripts" / "paper_harness.py"
    current_script = Path(__file__).read_text(encoding="utf-8")
    write_text(harness_path, current_script, mode=0o755)
    synced_files.append("scripts/paper_harness.py")

    generated_python_wrappers = {
        "scripts/status.py": "status",
        "scripts/inspect_paper.py": "inspect-paper",
        "scripts/index_paper_assets.py": "index-paper-assets",
        "scripts/extract_paper_figures.py": "render-paper-pages",
        "scripts/render_paper_pages.py": "render-paper-pages",
        "scripts/compare_figures.py": "compare-figures",
        "scripts/track_run.py": "track-run",
        "scripts/register_target_artifact.py": "register-target-artifact",
        "scripts/record_comparison.py": "record-comparison",
        "scripts/check_spec.py": "validate-spec",
        "scripts/check_progress.py": "validate-progress",
        "scripts/check_completion.py": "validate-completion",
        "scripts/check_report_coverage.py": "validate-report",
    }
    for relative_path, command_name in generated_python_wrappers.items():
        write_text(project_dir / relative_path, python_wrapper(command_name), mode=0o755)
        synced_files.append(relative_path)

    generated_shell_wrappers = {
        "scripts/download_paper.sh": "download-paper",
        "scripts/build_paper_pdf.sh": "build-paper-pdf",
    }
    for relative_path, command_name in generated_shell_wrappers.items():
        write_text(project_dir / relative_path, shell_wrapper(command_name), mode=0o755)
        synced_files.append(relative_path)

    reproduce_path = project_dir / "scripts" / "reproduce_all.sh"
    if rewrite_reproduce_all:
        write_text(reproduce_path, reproduce_all_script(), mode=0o755)
    else:
        write_text_if_missing(reproduce_path, reproduce_all_script(), mode=0o755)
    synced_files.append("scripts/reproduce_all.sh")

    cluster_runner_source = cluster_delegate_runner_source()
    if cluster_runner_source is not None:
        write_text(
            project_dir / "scripts" / "cluster_reproduce.py",
            cluster_runner_source.read_text(encoding="utf-8"),
            mode=0o755,
        )
        synced_files.append("scripts/cluster_reproduce.py")

    return synced_files


def bootstrap_project(args: argparse.Namespace) -> dict[str, Any]:
    project_dir = project_root_from(args.project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)
    paper_title = args.paper_title.strip()
    paper_slug = args.paper_slug.strip() if args.paper_slug else slugify(paper_title)
    paper_source = args.paper_source.strip()
    source_checksum_value = args.source_checksum.strip()
    main_tex = args.main_tex.strip()

    scaffold_dirs = [
        "src",
        "configs",
        "spec",
        "paper",
        "report",
        "scripts",
        "data/raw",
        "data/processed",
        "artifacts/paper_figures",
        "artifacts/figures",
        "artifacts/figures/comparison",
        "artifacts/tables",
        "artifacts/tables/comparison",
        "artifacts/provenance",
        "artifacts/runs",
        "artifacts/paper_build",
        "artifacts/status",
    ]
    for relative in scaffold_dirs:
        (project_dir / relative).mkdir(parents=True, exist_ok=True)

    for relative in [
        "data/raw/.gitkeep",
        "data/processed/.gitkeep",
        "artifacts/runs/.gitkeep",
        "artifacts/paper_build/.gitkeep",
        "artifacts/runs/run_index.jsonl",
    ]:
        write_text_if_missing(project_dir / relative, "")

    write_text_if_missing(project_dir / ".gitignore", project_gitignore())
    write_text_if_missing(project_dir / "README.md", readme_template(paper_title, paper_slug))
    write_text_if_missing(project_dir / "todo.md", todo_template())
    default_manifest = manifest_template(
        paper_title,
        paper_slug,
        paper_source,
        source_checksum_value,
        main_tex,
        args.source_mode,
        args.author_code_policy,
        args.stack_policy,
        args.compute_mode,
        args.cluster_profile_hint,
    )
    manifest_path = project_dir / "paper_manifest.json"
    if manifest_path.exists():
        current_manifest = load_manifest(project_dir)
        save_manifest(project_dir, deep_merge(default_manifest, current_manifest))
    else:
        save_manifest(project_dir, default_manifest)

    write_text_if_missing(
        project_dir / "spec" / "targets.md",
        spec_doc_template(
            "Targets",
            [
                "Enumerate every figure/table/example target.",
                "State the paper location and comparison metric for each target.",
                "Note any target-specific ambiguities.",
            ],
        ),
    )
    write_text_if_missing(
        project_dir / "spec" / "math_audit.md",
        spec_doc_template(
            "Math Audit",
            [
                "Rewrite the method in your own words.",
                "Derive or verify the critical update equations.",
                "Document the gradient paths and approximations.",
            ],
        ),
    )
    write_text_if_missing(
        project_dir / "spec" / "implementation_plan.md",
        spec_doc_template(
            "Implementation Plan",
            [
                "List deliverables with file paths.",
                "State dependencies and acceptance criteria.",
                "Define the target-by-target execution order.",
            ],
        ),
    )
    write_text_if_missing(
        project_dir / "spec" / "assumptions_and_unknowns.md",
        spec_doc_template(
            "Assumptions And Unknowns",
            [
                "Log each missing detail as an explicit hypothesis.",
                "Add the test or sweep that will resolve it.",
                "Record the chosen value and evidence after resolution.",
            ],
        ),
    )
    write_text_if_missing(
        project_dir / "spec" / "paper_figure_notes.md",
        spec_doc_template(
            "Paper Figure Notes",
            [
                "Describe the visual content of every paper figure from actual rendered views.",
                "Record axes, scales, smoothing, uncertainty, and panel layout.",
                "Link each note back to reproduction_matrix rows.",
            ],
        ),
    )
    write_text_if_missing(project_dir / "spec" / "reproduction_matrix.csv", matrix_header())
    write_text_if_missing(project_dir / "report" / "main.tex", report_template(paper_title))
    write_text_if_missing(project_dir / "report" / "refs.bib", "")
    write_text_if_missing(
        project_dir / "paper" / "README.md",
        render_markdown_template(
            "Paper Inputs",
            "- Store the downloaded paper zip at `paper/paper_src.zip` when using scripted downloads.\n"
            "- Keep raw paper sources out of version control.\n"
            "- Record the canonical source location and checksum in `paper_manifest.json`.",
        ),
    )
    write_text_if_missing(
        project_dir / "data" / "README.md",
        render_markdown_template(
            "Data Staging",
            "- Keep raw data outside version control.\n"
            "- Stage canonical data sources reproducibly and record checksums.\n"
            "- Use `data/raw/` for staged raw inputs and `data/processed/` for generated derivatives.",
        ),
    )

    harness_path = project_dir / "scripts" / "paper_harness.py"
    synced_files = sync_generated_scripts(project_dir)

    return {
        "project_dir": str(project_dir),
        "paper_slug": paper_slug,
        "manifest_path": str(project_dir / "paper_manifest.json"),
        "harness_path": str(harness_path),
        "synced_files": synced_files,
    }


def sync_harness(project_dir: Path, *, rewrite_reproduce_all: bool = False) -> dict[str, Any]:
    manifest_path = project_dir / "paper_manifest.json"
    if not manifest_path.exists():
        raise PaperReplicationError(f"Missing manifest: {manifest_path}")
    synced_files = sync_generated_scripts(project_dir, rewrite_reproduce_all=rewrite_reproduce_all)
    return {
        "project_dir": str(project_dir),
        "manifest_path": str(manifest_path),
        "harness_path": str(project_dir / "scripts" / "paper_harness.py"),
        "rewrite_reproduce_all": rewrite_reproduce_all,
        "synced_files": synced_files,
    }


def download_paper(project_dir: Path, url: str, checksum_value: str, destination: str) -> dict[str, Any]:
    dest = Path(destination).expanduser()
    if not dest.is_absolute():
        dest = (project_dir / dest).resolve()
    ensure_parent(dest)
    urllib.request.urlretrieve(url, dest)
    if checksum_value:
        actual = file_checksum(dest)
        if actual != checksum_value:
            dest.unlink(missing_ok=True)
            raise PaperReplicationError(
                "Downloaded paper checksum mismatch.",
                details={"expected_checksum": checksum_value, "actual_checksum": actual},
            )
    return {"downloaded_to": str(dest), "checksum": file_checksum(dest)}


def status_payload(project_dir: Path) -> dict[str, Any]:
    manifest = load_manifest(project_dir)
    spec_result = validate_spec(project_dir)
    progress_result = validate_progress(project_dir)
    completion_result = validate_completion(project_dir)
    delegate = cluster_delegate_info(project_dir, manifest)
    transport = cluster_transport_status(project_dir)
    try:
        rows = matrix_rows(project_dir)
    except Exception:
        rows = []
    try:
        todo_sections = read_todo_sections(project_dir)
    except Exception:
        todo_sections = {}
    active_target = find_active_targets(rows)
    counts = count_by_status(rows)
    next_action = ""
    if not (project_dir / "spec" / "paper_inventory.json").exists():
        next_action = "Run inspect-paper to inventory the TeX tree and figure assets."
    elif not spec_result["ok"]:
        next_action = "Fill the spec docs and reproduction matrix until validate-spec passes."
    elif not progress_result["ok"]:
        next_action = "Resolve the progress validator failures before continuing."
    elif not completion_result["ok"]:
        if (
            active_target
            and delegate["available"]
            and bool(transport)
            and not bool(transport.get("ok"))
            and bool(transport.get("retryable"))
        ):
            next_action = (
                f"Retry the cluster delegate for the ACTIVE target `{active_target[0]}` first. "
                "The last cluster transport failure was retryable and should not be treated as a durable blocker."
            )
        elif active_target:
            next_action = (
                f"Continue the ACTIVE target `{active_target[0]}` until it is genuinely MATCHED, "
                "then advance the remaining incomplete targets."
            )
        elif completion_result["incomplete_targets"]:
            next_id = completion_result["incomplete_targets"][0]["target_id"] or "<missing-target-id>"
            next_action = (
                f"Select the next incomplete target `{next_id}`, mark it ACTIVE, and continue until every target is MATCHED."
            )
        else:
            next_action = "Resolve the remaining completion blockers and rerun validate-completion."
    else:
        next_action = "Completion gate passed. The case study is ready for final review."

    payload = {
        "project_dir": str(project_dir),
        "paper_slug": manifest.get("paper_slug", ""),
        "paper_title": manifest.get("paper_title", ""),
        "phase": todo_sections.get("Current phase", "").strip() or manifest.get("state", {}).get("current_phase", ""),
        "active_target": active_target[0] if active_target else "UNSET",
        "status_counts": counts,
        "spec_ok": spec_result["ok"],
        "progress_ok": progress_result["ok"],
        "completion_ok": completion_result["ok"],
        "spec_errors": spec_result["errors"],
        "progress_errors": progress_result["errors"],
        "completion_errors": completion_result["errors"],
        "incomplete_targets": completion_result["incomplete_targets"],
        "cluster_delegate": delegate,
        "cluster_transport": transport,
        "next_action": next_action,
    }
    status_path = project_dir / "artifacts" / "status" / "status.json"
    write_json(status_path, payload)
    payload["status_path"] = str(status_path)
    return payload


def run_reproduce(project_dir: Path, local_exec: bool = False) -> dict[str, Any]:
    manifest = load_manifest(project_dir)
    compute_mode = str(manifest.get("compute", {}).get("mode", "auto")).strip().lower()
    if compute_mode == "cluster" and not local_exec:
        delegate = cluster_delegate_info(project_dir, manifest)
        if not delegate["available"]:
            raise PaperReplicationError(
                "compute.mode=cluster but cluster-slurm is not discoverable. "
                "Install the cluster-slurm skill or rerun with --local-exec."
            )
        proc = subprocess.run(delegate["command"], text=True, capture_output=True, check=False)
        if proc.returncode != 0:
            raise PaperReplicationError(
                "Cluster delegation failed.",
                details={"stdout": proc.stdout, "stderr": proc.stderr, "command": delegate["command"]},
            )
        return {"delegated": True, "command": delegate["command"], "stdout": proc.stdout}

    local_steps = []
    if str(manifest.get("source_mode", "latex-first")).strip().lower() == "latex-first":
        try:
            build_result = build_paper_pdf(project_dir)
            local_steps.append({"step": "build-paper-pdf", "result": build_result})
        except PaperReplicationError as exc:
            local_steps.append({"step": "build-paper-pdf", "warning": str(exc), "details": exc.details})

    spec_result = validate_spec(project_dir)
    if not spec_result["ok"]:
        raise PaperReplicationError("validate-spec failed during reproduce.", details={"errors": spec_result["errors"]})
    progress_result = validate_progress(project_dir)
    if not progress_result["ok"]:
        raise PaperReplicationError(
            "validate-progress failed during reproduce.", details={"errors": progress_result["errors"]}
        )
    completion_result = validate_completion(project_dir)
    if not completion_result["ok"]:
        raise PaperReplicationError(
            "validate-completion failed during reproduce.",
            details={"errors": completion_result["errors"], "incomplete_targets": completion_result["incomplete_targets"]},
        )
    return {"delegated": False, "steps": local_steps, "spec_ok": True, "progress_ok": True, "completion_ok": True}


def json_result(result: dict[str, Any]) -> None:
    print(json.dumps({"ok": True, "result": result}, indent=2, sort_keys=True))


def json_error(message: str, details: dict[str, Any] | None = None) -> None:
    payload = {"ok": False, "error": message}
    if details:
        payload["details"] = details
    print(json.dumps(payload, indent=2, sort_keys=True), file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Paper replication harness CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    bootstrap = sub.add_parser("bootstrap", help="Create or refresh a per-paper replication scaffold")
    bootstrap.add_argument("--project-dir", required=True)
    bootstrap.add_argument("--paper-title", required=True)
    bootstrap.add_argument("--paper-slug", default="")
    bootstrap.add_argument("--paper-source", required=True)
    bootstrap.add_argument("--source-checksum", default="")
    bootstrap.add_argument("--main-tex", default="")
    bootstrap.add_argument("--source-mode", default="latex-first")
    bootstrap.add_argument("--author-code-policy", default="forbid_by_default")
    bootstrap.add_argument("--stack-policy", default="paper-driven")
    bootstrap.add_argument("--compute-mode", default="auto")
    bootstrap.add_argument("--cluster-profile-hint", default="")

    sync_cmd = sub.add_parser("sync-harness", help="Refresh generated harness files in an existing case study")
    sync_cmd.add_argument("--project-dir", required=True)
    sync_cmd.add_argument("--rewrite-reproduce-all", action="store_true")

    inspect = sub.add_parser("inspect-paper", help="Inventory TeX sources, figures, appendices, and data references")
    inspect.add_argument("--project-dir", required=True)

    status = sub.add_parser("status", help="Summarize current project state from repo files")
    status.add_argument("--project-dir", required=True)

    validate_spec_cmd = sub.add_parser("validate-spec", help="Fail if the spec files are missing or incomplete")
    validate_spec_cmd.add_argument("--project-dir", required=True)

    validate_progress_cmd = sub.add_parser("validate-progress", help="Fail if target/artifact/report state is inconsistent")
    validate_progress_cmd.add_argument("--project-dir", required=True)

    validate_completion_cmd = sub.add_parser(
        "validate-completion",
        help="Fail unless the entire paper target set is genuinely complete",
    )
    validate_completion_cmd.add_argument("--project-dir", required=True)

    validate_report_cmd = sub.add_parser("validate-report", help="Fail if matched artifacts are not embedded in the report")
    validate_report_cmd.add_argument("--project-dir", required=True)

    build_pdf = sub.add_parser("build-paper-pdf", help="Compile the paper PDF from LaTeX sources")
    build_pdf.add_argument("--project-dir", required=True)

    render_pages = sub.add_parser("render-paper-pages", help="Render paper PDF pages to PNG files")
    render_pages.add_argument("--project-dir", required=True)
    render_pages.add_argument("--pdf-path", default="")

    index_assets = sub.add_parser("index-paper-assets", help="Write a machine-readable inventory of referenced paper assets")
    index_assets.add_argument("--project-dir", required=True)

    compare = sub.add_parser("compare-figures", help="Compare a reference figure to a reproduced figure")
    compare.add_argument("--project-dir", required=True)
    compare.add_argument("--reference-path", required=True)
    compare.add_argument("--candidate-path", required=True)
    compare.add_argument("--output-dir", default="")
    compare.add_argument("--target-id", default="")

    track_run_cmd = sub.add_parser("track-run", help="Execute a reproduction command and record a run ledger entry")
    track_run_cmd.add_argument("--project-dir", required=True)
    track_run_cmd.add_argument("--label", required=True)
    track_run_cmd.add_argument("--shell-command", required=True)
    track_run_cmd.add_argument("--cwd", default="")
    track_run_cmd.add_argument("--expected-artifact", action="append", default=[])

    register_artifact_cmd = sub.add_parser(
        "register-target-artifact",
        help="Register a MATCHED candidate artifact against a successful tracked run",
    )
    register_artifact_cmd.add_argument("--project-dir", required=True)
    register_artifact_cmd.add_argument("--target-id", required=True)
    register_artifact_cmd.add_argument("--run-id", required=True)
    register_artifact_cmd.add_argument("--artifact-path", default="")
    register_artifact_cmd.add_argument("--claim-mode", default="baseline")
    register_artifact_cmd.add_argument("--method-label", required=True)
    register_artifact_cmd.add_argument("--code-path", required=True)
    register_artifact_cmd.add_argument("--config-path", required=True)
    register_artifact_cmd.add_argument("--paper-trace-path", required=True)
    register_artifact_cmd.add_argument("--seed", required=True)
    register_artifact_cmd.add_argument("--implementation-kind", default="paper-method")
    register_artifact_cmd.add_argument("--method-component", action="append", default=[])
    register_artifact_cmd.add_argument("--implementation-summary", required=True)
    register_artifact_cmd.add_argument("--baseline-faithful", action="store_true")
    register_artifact_cmd.add_argument("--deviation-notes", default="")

    record_comparison_cmd = sub.add_parser(
        "record-comparison",
        help="Write standardized comparison evidence for a target",
    )
    record_comparison_cmd.add_argument("--project-dir", required=True)
    record_comparison_cmd.add_argument("--target-id", required=True)
    record_comparison_cmd.add_argument("--kind", required=True, choices=["figure", "table"])
    record_comparison_cmd.add_argument("--note", default="")
    record_comparison_cmd.add_argument("--metric", action="append", default=[])
    record_comparison_cmd.add_argument("--acceptance-mode", default="")
    record_comparison_cmd.add_argument("--reference-path", default="")
    record_comparison_cmd.add_argument("--candidate-path", default="")

    download = sub.add_parser("download-paper", help="Download a paper archive into the scaffold")
    download.add_argument("--project-dir", required=True)
    download.add_argument("--url", required=True)
    download.add_argument("--checksum", default="")
    download.add_argument("--destination", default="paper/paper_src.zip")

    reproduce = sub.add_parser("run-reproduce", help="Run the local or cluster reproduction entrypoint")
    reproduce.add_argument("--project-dir", required=True)
    reproduce.add_argument("--local-exec", action="store_true")

    return parser


def dispatch(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "bootstrap":
        return bootstrap_project(args)
    project_dir = project_root_from(args.project_dir)
    if args.command == "sync-harness":
        return sync_harness(project_dir, rewrite_reproduce_all=bool(args.rewrite_reproduce_all))
    if args.command == "inspect-paper":
        return inspect_paper(project_dir)
    if args.command == "status":
        return status_payload(project_dir)
    if args.command == "validate-spec":
        result = validate_spec(project_dir)
        if not result["ok"]:
            raise PaperReplicationError("validate-spec failed.", details={"errors": result["errors"]})
        return result
    if args.command == "validate-progress":
        result = validate_progress(project_dir)
        if not result["ok"]:
            raise PaperReplicationError("validate-progress failed.", details={"errors": result["errors"]})
        return result
    if args.command == "validate-completion":
        result = validate_completion(project_dir)
        if not result["ok"]:
            raise PaperReplicationError(
                "validate-completion failed.",
                details={"errors": result["errors"], "incomplete_targets": result["incomplete_targets"]},
            )
        return result
    if args.command == "validate-report":
        result = validate_report(project_dir)
        if not result["ok"]:
            raise PaperReplicationError("validate-report failed.", details={"errors": result["errors"]})
        return result
    if args.command == "build-paper-pdf":
        return build_paper_pdf(project_dir)
    if args.command == "render-paper-pages":
        return render_paper_pages(project_dir, args.pdf_path or None)
    if args.command == "index-paper-assets":
        return index_paper_assets(project_dir)
    if args.command == "compare-figures":
        return compare_figures(
            project_dir,
            args.reference_path,
            args.candidate_path,
            args.output_dir or None,
            args.target_id or None,
        )
    if args.command == "track-run":
        return tracked_run(
            project_dir,
            args.shell_command,
            label=args.label,
            cwd=args.cwd or None,
            expected_artifacts=args.expected_artifact,
        )
    if args.command == "register-target-artifact":
        return register_target_artifact(
            project_dir,
            target_id=args.target_id,
            run_id=args.run_id,
            artifact_path=args.artifact_path or None,
            claim_mode=args.claim_mode,
            method_label=args.method_label,
            code_path=args.code_path,
            config_path=args.config_path,
            paper_trace_path=args.paper_trace_path,
            seed=args.seed,
            baseline_faithful=bool(args.baseline_faithful),
            deviation_notes=args.deviation_notes,
            implementation_kind=args.implementation_kind,
            method_components=args.method_component,
            implementation_summary=args.implementation_summary,
        )
    if args.command == "record-comparison":
        metrics: dict[str, Any] = {}
        for item in args.metric:
            key, sep, value = item.partition("=")
            if not sep:
                raise PaperReplicationError(f"Invalid metric format: {item}. Use key=value.")
            metrics[key.strip()] = value.strip()
        if args.reference_path:
            metrics["reference_path"] = args.reference_path
        if args.candidate_path:
            metrics["candidate_path"] = args.candidate_path
        return record_comparison(
            project_dir,
            target_id=args.target_id,
            kind=args.kind,
            note=args.note,
            metrics=metrics,
            acceptance_mode=args.acceptance_mode or None,
        )
    if args.command == "download-paper":
        return download_paper(project_dir, args.url, args.checksum, args.destination)
    if args.command == "run-reproduce":
        return run_reproduce(project_dir, local_exec=bool(args.local_exec))
    raise PaperReplicationError(f"Unknown command: {args.command}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = dispatch(args)
    except PaperReplicationError as exc:
        json_error(str(exc), exc.details)
        return 1
    json_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
