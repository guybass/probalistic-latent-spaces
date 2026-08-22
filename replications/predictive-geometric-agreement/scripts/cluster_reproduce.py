#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
STAGE_DIRECTORIES = ("configs", "data", "report", "scripts", "spec", "src")
STAGE_FILES = ("README.md", "paper_manifest.json", "todo.md")
ARTIFACT_DIRECTORIES = (
    "artifacts/figures",
    "artifacts/figures/comparison",
    "artifacts/tables",
    "artifacts/tables/comparison",
    "artifacts/provenance",
    "artifacts/status",
    "artifacts/runs",
    "artifacts/paper_build",
    "artifacts/paper_figures",
)
PAPER_FIGURE_INDEX_FILES = ("asset_index.json", "source_hashes.json", "page_index.json")
IGNORE_PATTERNS = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".DS_Store")
CLUSTER_EXEC_ENV_VAR = "PAPER_REPLICATION_RUNNING_IN_CLUSTER_EXEC"
CLUSTER_TRANSPORT_STATUS_RELATIVE = Path("artifacts") / "status" / "cluster_transport.json"
RETRYABLE_TRANSPORT_PATTERNS = {
    "hostname_resolution": (
        "could not resolve hostname",
        "name or service not known",
        "temporary failure in name resolution",
        "nodename nor servname provided",
    ),
    "connect_timeout": ("connection timed out", "operation timed out", "timed out"),
    "connection_refused": ("connection refused",),
    "connection_reset": ("connection reset", "connection closed by remote host"),
    "network_unreachable": ("network is unreachable", "no route to host"),
}
NONRETRYABLE_TRANSPORT_PATTERNS = (
    "permission denied",
    "host key verification failed",
    "too many authentication failures",
    "invalid account",
    "unknown option",
)
MAX_TRANSPORT_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 2.0


class ClusterDelegateError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        command: list[str] | None = None,
        stdout: str = "",
        stderr: str = "",
        payload: dict[str, Any] | None = None,
        returncode: int | None = None,
        transport_state: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.command = command or []
        self.stdout = stdout
        self.stderr = stderr
        self.payload = payload or {}
        self.returncode = returncode
        self.transport_state = transport_state or {}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def truncate_text(value: str, limit: int = 2000) -> str:
    text = value.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def cluster_transport_status_path(project_dir: Path) -> Path:
    return project_dir / CLUSTER_TRANSPORT_STATUS_RELATIVE


def write_cluster_transport_status(project_dir: Path, payload: dict[str, Any]) -> Path:
    path = cluster_transport_status_path(project_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def payload_messages(value: Any) -> list[str]:
    messages: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, str) and key in {"error", "stderr", "stdout", "message", "name"}:
                messages.append(item)
            else:
                messages.extend(payload_messages(item))
    elif isinstance(value, list):
        for item in value:
            messages.extend(payload_messages(item))
    return messages


def classify_retryable_transport_failure(
    returncode: int | None,
    stdout: str,
    stderr: str,
    payload: dict[str, Any] | None,
) -> dict[str, Any] | None:
    combined = "\n".join(
        part for part in [stdout, stderr, *payload_messages(payload or {})] if part and part.strip()
    ).lower()
    if not combined:
        return None
    if any(marker in combined for marker in NONRETRYABLE_TRANSPORT_PATTERNS):
        return None
    for category, markers in RETRYABLE_TRANSPORT_PATTERNS.items():
        for marker in markers:
            if marker in combined:
                return {"retryable": True, "category": category, "matched_text": marker}
    if returncode == 255 and "ssh" in combined:
        return {"retryable": True, "category": "ssh_transport", "matched_text": "ssh transport failure"}
    return None


def build_cluster_transport_status(
    *,
    ok: bool,
    profile: str,
    step: str,
    attempt: int,
    max_attempts: int,
    command: list[str],
    retryable: bool,
    category: str,
    summary: str,
    returncode: int | None,
    stdout: str,
    stderr: str,
    payload: dict[str, Any] | None = None,
    recovered_from_retryable_error: bool = False,
    previous_retryable_failures: int = 0,
    last_failure: dict[str, Any] | None = None,
) -> dict[str, Any]:
    status = {
        "ok": ok,
        "profile": profile,
        "step": step,
        "attempt": attempt,
        "max_attempts": max_attempts,
        "retryable": retryable,
        "category": category,
        "summary": summary,
        "returncode": returncode,
        "command": command,
        "stdout": truncate_text(stdout),
        "stderr": truncate_text(stderr),
        "updated_at": utc_now(),
        "recovered_from_retryable_error": recovered_from_retryable_error,
        "previous_retryable_failures": previous_retryable_failures,
    }
    if payload:
        status["payload"] = payload
    if last_failure:
        status["last_failure"] = last_failure
    return status


def project_root_from(raw_path: str | None) -> Path:
    if not raw_path:
        return PROJECT_DIR
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path.resolve()


def project_path_from(project_dir: Path, raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (project_dir / path).resolve()
    return path.resolve()


def load_manifest(project_dir: Path) -> dict[str, Any]:
    manifest_path = project_dir / "paper_manifest.json"
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def cluster_slurm_script() -> Path:
    env_override = os.environ.get("PAPER_REPLICATION_CLUSTER_SLURM")
    if env_override:
        path = Path(env_override).expanduser().resolve()
        if path.exists():
            return path
    default = Path("~/.codex/skills/cluster-slurm/scripts/cluster_slurm.py").expanduser().resolve()
    if default.exists():
        return default
    raise ClusterDelegateError("cluster-slurm is not discoverable.")


def active_target_row(project_dir: Path) -> dict[str, str]:
    matrix_path = project_dir / "spec" / "reproduction_matrix.csv"
    with matrix_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    active = [row for row in rows if row.get("status", "").strip().upper() == "ACTIVE"]
    if len(active) != 1:
        raise ClusterDelegateError(f"Expected exactly one ACTIVE target, found {len(active)}.")
    return active[0]


def copy_path(source: Path, destination: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, destination, ignore=IGNORE_PATTERNS)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def stage_project(project_dir: Path, stage_root: Path) -> tuple[Path, dict[str, Any]]:
    staged_project = stage_root / project_dir.name
    copied_paths: list[str] = []
    staged_project.mkdir(parents=True, exist_ok=True)

    for relative in STAGE_FILES:
        source = project_dir / relative
        if not source.exists():
            continue
        copy_path(source, staged_project / relative)
        copied_paths.append(relative)

    for relative in STAGE_DIRECTORIES:
        source = project_dir / relative
        if not source.exists():
            continue
        copy_path(source, staged_project / relative)
        copied_paths.append(relative)

    for relative in ARTIFACT_DIRECTORIES:
        path = staged_project / relative
        path.mkdir(parents=True, exist_ok=True)

    (staged_project / "artifacts" / "runs" / "run_index.jsonl").write_text("", encoding="utf-8")
    (staged_project / "artifacts" / "runs" / ".gitkeep").write_text("", encoding="utf-8")
    (staged_project / "artifacts" / "paper_build" / ".gitkeep").write_text("", encoding="utf-8")

    paper_figures_root = project_dir / "artifacts" / "paper_figures"
    staged_paper_figures_root = staged_project / "artifacts" / "paper_figures"
    copied_indexes: list[str] = []
    for filename in PAPER_FIGURE_INDEX_FILES:
        source = paper_figures_root / filename
        if not source.exists():
            continue
        copy_path(source, staged_paper_figures_root / filename)
        copied_indexes.append(f"artifacts/paper_figures/{filename}")

    manifest = {
        "project_dir": str(project_dir),
        "staged_project_dir": str(staged_project),
        "copied_paths": copied_paths,
        "copied_paper_figure_indexes": copied_indexes,
    }
    return staged_project, manifest


def parse_payload(stdout: str, stderr: str, command: list[str]) -> dict[str, Any]:
    candidates = [("stdout", stdout.strip()), ("stderr", stderr.strip())]
    for stream_name, text in candidates:
        if not text:
            continue
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            if stream_name == "stderr":
                raise ClusterDelegateError(
                    f"Failed to parse cluster-slurm output: {exc}",
                    command=command,
                    stdout=stdout,
                    stderr=stderr,
                ) from exc
    raise ClusterDelegateError("cluster-slurm produced no JSON output.", command=command, stdout=stdout, stderr=stderr)


def run_cluster_command(cluster_script: Path, args: list[str], *, cwd: Path) -> dict[str, Any]:
    command = [sys.executable, str(cluster_script), *args]
    proc = subprocess.run(command, cwd=str(cwd), text=True, capture_output=True, check=False)
    try:
        payload = parse_payload(proc.stdout, proc.stderr, command)
    except ClusterDelegateError as exc:
        exc.returncode = proc.returncode
        raise
    if proc.returncode != 0 or not bool(payload.get("ok")):
        raise ClusterDelegateError(
            str(payload.get("error") or f"cluster-slurm command failed with exit code {proc.returncode}"),
            command=command,
            stdout=proc.stdout,
            stderr=proc.stderr,
            payload=payload,
            returncode=proc.returncode,
        )
    return payload["result"]


def run_cluster_command_with_retries(
    cluster_script: Path,
    args: list[str],
    *,
    cwd: Path,
    project_dir: Path,
    profile: str,
    step: str,
    max_attempts: int = MAX_TRANSPORT_ATTEMPTS,
) -> dict[str, Any]:
    retryable_failures: list[dict[str, Any]] = []
    for attempt in range(1, max_attempts + 1):
        try:
            result = run_cluster_command(cluster_script, args, cwd=cwd)
        except ClusterDelegateError as exc:
            classification = classify_retryable_transport_failure(
                exc.returncode,
                exc.stdout,
                exc.stderr,
                exc.payload,
            )
            if classification:
                state = build_cluster_transport_status(
                    ok=False,
                    profile=profile,
                    step=step,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    command=exc.command,
                    retryable=True,
                    category=classification["category"],
                    summary=f"Retryable cluster transport failure during {step}.",
                    returncode=exc.returncode,
                    stdout=exc.stdout,
                    stderr=exc.stderr,
                    payload=exc.payload,
                )
                write_cluster_transport_status(project_dir, state)
                retryable_failures.append(
                    {
                        "category": state["category"],
                        "summary": state["summary"],
                        "returncode": state["returncode"],
                        "stderr": state["stderr"],
                        "updated_at": state["updated_at"],
                    }
                )
                if attempt < max_attempts:
                    time.sleep(RETRY_BACKOFF_SECONDS * attempt)
                    continue
                exc.transport_state = state
                raise
            state = build_cluster_transport_status(
                ok=False,
                profile=profile,
                step=step,
                attempt=attempt,
                max_attempts=max_attempts,
                command=exc.command,
                retryable=False,
                category="cluster_error",
                summary=f"Cluster step {step} failed with a non-retryable error.",
                returncode=exc.returncode,
                stdout=exc.stdout,
                stderr=exc.stderr,
                payload=exc.payload,
            )
            write_cluster_transport_status(project_dir, state)
            exc.transport_state = state
            raise
        success_state = build_cluster_transport_status(
            ok=True,
            profile=profile,
            step=step,
            attempt=attempt,
            max_attempts=max_attempts,
            command=[],
            retryable=False,
            category="",
            summary=f"Cluster step {step} succeeded.",
            returncode=0,
            stdout="",
            stderr="",
            recovered_from_retryable_error=bool(retryable_failures),
            previous_retryable_failures=len(retryable_failures),
            last_failure=retryable_failures[-1] if retryable_failures else None,
        )
        write_cluster_transport_status(project_dir, success_state)
        return result
    raise AssertionError("run_cluster_command_with_retries exhausted attempts without returning or raising.")


def extract_run_id(init_result: dict[str, Any]) -> str:
    direct = str(init_result.get("run_id", "")).strip()
    if direct:
        return direct
    nested_run = init_result.get("run")
    if isinstance(nested_run, dict):
        nested = str(nested_run.get("run_id", "")).strip()
        if nested:
            return nested
    raise ClusterDelegateError("cluster-slurm init-run returned no run_id.", payload={"init_run": init_result})


def workload_command_lines(stage_name: str, reproduce_args: list[str]) -> list[str]:
    lines = [
        f"cd {json.dumps(stage_name)}",
        'export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"',
        'export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/pycache}"',
        'export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"',
        f"export {CLUSTER_EXEC_ENV_VAR}=1",
        'mkdir -p "$MPLCONFIGDIR" "$(dirname "$PYTHONPYCACHEPREFIX")"',
    ]
    reproduce_command = ["bash", "scripts/reproduce_all.sh", "--local-exec", *reproduce_args]
    lines.append(" ".join(json.dumps(arg) for arg in reproduce_command))
    return lines


def download_artifacts(
    cluster_script: Path,
    project_dir: Path,
    run_id: str,
    stage_name: str,
    profile: str,
    *,
    cwd: Path,
) -> list[dict[str, Any]]:
    downloads: list[dict[str, Any]] = []
    local_artifact_root = project_dir / "artifacts"
    for remote_subpath in ("artifacts/figures", "artifacts/tables"):
        result = run_cluster_command_with_retries(
            cluster_script,
            [
                "download",
                "--run-id",
                run_id,
                "--remote-path",
                f"{stage_name}/{remote_subpath}",
                "--local-path",
                str(local_artifact_root),
            ],
            cwd=cwd,
            project_dir=project_dir,
            profile=profile,
            step="download",
        )
        downloads.append(result)
    return downloads


def write_success_marker(
    marker_path: Path,
    *,
    active_target: dict[str, str],
    profile: str,
    run_id: str,
    reproduce_args: list[str],
    workload_result: dict[str, Any],
) -> None:
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_payload = {
        "ok": True,
        "active_target": active_target.get("target_id", "").strip(),
        "kind": active_target.get("kind", "").strip(),
        "profile": profile,
        "run_id": run_id,
        "reproduce_args": reproduce_args,
        "workload_result": workload_result,
    }
    marker_path.write_text(json.dumps(marker_payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage and run the current paper-replication project on cluster-slurm.")
    parser.add_argument("--project-dir", default=str(PROJECT_DIR))
    parser.add_argument("--profile", default="")
    parser.add_argument("--reproduce-arg", action="append", default=[])
    parser.add_argument("--tail", type=int, default=200)
    parser.add_argument("--timeout-seconds", type=int, default=0)
    parser.add_argument("--poll-interval-seconds", type=int, default=10)
    parser.add_argument("--success-marker-path", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    project_dir = project_root_from(args.project_dir)
    manifest = load_manifest(project_dir)
    active_target = active_target_row(project_dir)
    cluster_script = cluster_slurm_script()
    success_marker_path = project_path_from(project_dir, args.success_marker_path)
    slug = str(manifest.get("paper_slug", "paper-replication")).strip() or "paper-replication"
    workload = f"paper replication: {slug}"
    profile = str(args.profile).strip() or str(manifest.get("compute", {}).get("cluster_profile_hint", "")).strip()
    if not profile:
        raise ClusterDelegateError("No cluster profile was provided and paper_manifest.json has no compute.cluster_profile_hint.")
    if success_marker_path is not None and not args.dry_run:
        success_marker_path.unlink(missing_ok=True)

    with tempfile.TemporaryDirectory(prefix=f"{project_dir.name}-cluster-stage-") as temp_dir:
        stage_root = Path(temp_dir)
        staged_project, stage_manifest = stage_project(project_dir, stage_root)
        stage_name = staged_project.name

        init_args = ["init-run", "--prefix", slug, "--profile", profile]
        if args.dry_run:
            init_args.insert(0, "--dry-run")
        init_result = run_cluster_command_with_retries(
            cluster_script,
            init_args,
            cwd=project_dir,
            project_dir=project_dir,
            profile=profile,
            step="init-run",
        )
        run_id = extract_run_id(init_result)

        upload_args = ["upload", "--run-id", run_id, "--local-path", str(staged_project)]
        if args.dry_run:
            upload_args.insert(0, "--dry-run")
        upload_result = run_cluster_command_with_retries(
            cluster_script,
            upload_args,
            cwd=project_dir,
            project_dir=project_dir,
            profile=profile,
            step="upload",
        )

        workload_args = [
            "run-workload",
            "--run-id",
            run_id,
            "--prefix",
            slug,
            "--workload",
            workload,
            "--profile",
            profile,
            "--wait",
            "--fetch-logs",
            "--tail",
            str(args.tail),
            "--poll-interval-seconds",
            str(args.poll_interval_seconds),
        ]
        if args.timeout_seconds > 0:
            workload_args.extend(["--timeout-seconds", str(args.timeout_seconds)])
        for line in workload_command_lines(stage_name, args.reproduce_arg):
            workload_args.extend(["--command", line])
        if args.dry_run:
            workload_args.insert(0, "--dry-run")
        workload_result = run_cluster_command_with_retries(
            cluster_script,
            workload_args,
            cwd=project_dir,
            project_dir=project_dir,
            profile=profile,
            step="run-workload",
        )

        payload: dict[str, Any] = {
            "project_dir": str(project_dir),
            "profile": profile,
            "cluster_script": str(cluster_script),
            "active_target": active_target.get("target_id", "").strip(),
            "active_target_kind": active_target.get("kind", "").strip(),
            "reproduce_args": args.reproduce_arg,
            "run_id": run_id,
            "success_marker_path": str(success_marker_path) if success_marker_path is not None else "",
            "staged_project_name": stage_name,
            "stage_manifest": stage_manifest,
            "init_run": init_result,
            "upload": upload_result,
            "workload": workload_result,
        }

        if not args.dry_run:
            run_payload = workload_result.get("run", {}) if isinstance(workload_result, dict) else {}
            state = str((run_payload.get("status") if isinstance(run_payload, dict) else {}).get("state", "")).strip()
            if state != "COMPLETED":
                raise ClusterDelegateError(
                    f"Cluster workload did not complete successfully (state={state or 'unknown'}).",
                    payload=payload,
                )
            payload["downloads"] = download_artifacts(
                cluster_script,
                project_dir,
                run_id,
                stage_name,
                profile,
                cwd=project_dir,
            )
            if success_marker_path is not None:
                write_success_marker(
                    success_marker_path,
                    active_target=active_target,
                    profile=profile,
                    run_id=run_id,
                    reproduce_args=args.reproduce_arg,
                    workload_result=workload_result,
                )

    print(json.dumps({"ok": True, "result": payload}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ClusterDelegateError as exc:
        details = {
            "command": exc.command,
            "stdout": exc.stdout,
            "stderr": exc.stderr,
        }
        if exc.returncode is not None:
            details["returncode"] = exc.returncode
        if exc.payload:
            details["payload"] = exc.payload
        if exc.transport_state:
            details["transport_state"] = exc.transport_state
        print(json.dumps({"ok": False, "error": str(exc), "details": details}, indent=2), file=sys.stderr)
        raise SystemExit(1)
    except Exception as exc:  # pragma: no cover - top-level guard for cluster orchestration failures
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        raise SystemExit(1)
