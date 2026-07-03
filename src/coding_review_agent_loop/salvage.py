"""Local salvage artifacts for failed mutating agent runs."""

from __future__ import annotations

import datetime
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .agents.base import AgentName, AgentResult
from .errors import AgentLoopError
from .runner import ensure_log_dir_ignored

if TYPE_CHECKING:
    from .runner import Runner


@dataclass(frozen=True)
class SalvageContext:
    repo: str
    issue_number: int | None
    scope: str
    agent: AgentName
    run_id: str | None = None
    approved_plan_hash: str | None = None


@dataclass(frozen=True)
class SalvageArtifacts:
    directory: Path
    patch_path: Path
    changed_files_path: Path
    diff_stat_path: Path
    diff_check_path: Path
    summary_path: Path
    metadata_path: Path
    summary: str


class SalvageCaptureError(AgentLoopError):
    """Raised when mandatory salvage inspection fails."""


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-") or "unknown"


def _artifact_dir(log_dir: Path, context: SalvageContext, created_at_ns: int) -> Path:
    run_id = _slug(context.run_id or str(created_at_ns))
    base = (
        log_dir
        / "salvage"
        / f"{run_id}-{_slug(context.agent)}-{_slug(context.scope)}"
    )
    if not base.exists():
        return base
    for index in range(2, 1000):
        candidate = base.with_name(f"{base.name}-{index}")
        if not candidate.exists():
            return candidate
    raise SalvageCaptureError(f"Unable to allocate a unique salvage directory under {base.parent}")


def _required_git_output(
    runner: Runner,
    checkout: Path,
    args: tuple[str, ...],
    *,
    label: str,
) -> str:
    result = runner.run(("git", *args), cwd=checkout, check=False)
    if result.returncode != 0:
        details = (result.stderr or result.stdout or "").strip()
        suffix = f": {details}" if details else ""
        raise SalvageCaptureError(
            f"git {' '.join(args)} failed while collecting {label} "
            f"(exit {result.returncode}){suffix}"
        )
    return result.stdout


def _best_effort_diff_check(runner: Runner, checkout: Path) -> tuple[str, int | None]:
    try:
        result = runner.run(("git", "diff", "--check"), cwd=checkout, check=False)
    except AgentLoopError as exc:
        return f"git diff --check could not run: {exc}\n", None

    output = ""
    if result.stdout:
        output += result.stdout
    if result.stderr:
        if output and not output.endswith("\n"):
            output += "\n"
        output += result.stderr
    if not output:
        output = "git diff --check produced no output.\n"
    elif not output.endswith("\n"):
        output += "\n"
    return output, result.returncode


def _write_text(path: Path, text: str, *, trailing_newline: bool = True) -> None:
    if trailing_newline and text and not text.endswith("\n"):
        text += "\n"
    path.write_text(text, encoding="utf-8")


def _summary_text(
    *,
    context: SalvageContext,
    artifacts_dir: Path,
    patch_path: Path,
    changed_files_path: Path,
    diff_stat_path: Path,
    diff_check_path: Path,
    metadata_path: Path,
    failure_category: str,
    failure_reason: str,
    required_marker: str,
    result: AgentResult | None,
    status_text: str,
    diff_check_returncode: int | None,
) -> str:
    response_file_missing = result is None or result.response_file_text is None
    response_status = (
        "missing; stdout or agent message fallback was not a valid successful response"
        if response_file_missing
        else "present but failed validation"
    )
    returncode = "none" if result is None or result.returncode is None else str(result.returncode)
    log_path = str(result.log_path) if result is not None and result.log_path is not None else "(none)"
    diff_check_status = (
        "not run"
        if diff_check_returncode is None
        else "passed"
        if diff_check_returncode == 0
        else f"reported diagnostics (exit {diff_check_returncode})"
    )
    untracked_note = ""
    if any(line.startswith("??") for line in status_text.splitlines()):
        untracked_note = (
            "\n- Untracked files appear in `changed-files.txt`; `partial.patch` "
            "contains only `git diff HEAD --binary` output."
        )
    approved_hash_line = (
        f"\n- Approved plan hash: `{context.approved_plan_hash}`"
        if context.approved_plan_hash
        else ""
    )
    return f"""# Salvage Summary

The {context.agent} agent failed before producing a valid public response. No
successful response, review result, or pull request should be inferred from this
artifact.

This patch is incomplete context only. A later attempt may cherry-pick or ignore
it selectively, but must not treat it as validated, complete, or automatically
applicable.

- Repository: `{context.repo}`
- Issue: `{context.issue_number if context.issue_number is not None else "unknown"}`
- Scope: `{context.scope}`
- Agent: `{context.agent}`{approved_hash_line}
- Failure category: `{failure_category}`
- Failure reason: {failure_reason}
- Agent exit code: `{returncode}`
- Attempt log: `{log_path}`
- Public response file: {response_status}
- Required marker: `{required_marker}`
- Required marker status: missing or invalid; the failed response did not satisfy this requirement.
- `git diff --check`: {diff_check_status}{untracked_note}

## Local Artifacts

- Salvage directory: `{artifacts_dir}`
- Partial patch: `{patch_path}`
- Changed files/status: `{changed_files_path}`
- Diff stat: `{diff_stat_path}`
- Diff check: `{diff_check_path}`
- Metadata: `{metadata_path}`
"""


def capture_salvage_artifacts(
    runner: Runner,
    *,
    checkout: Path,
    log_dir: Path,
    context: SalvageContext,
    failure_category: str,
    failure_reason: str,
    required_marker: str,
    result: AgentResult | None,
) -> SalvageArtifacts | None:
    """Capture cheap local diagnostics for a failed mutating implementation run.

    Returns None when the checkout has no tracked/staged diff against HEAD. This
    intentionally avoids turning untracked-only work into a misleading patch.
    """

    diff_text = _required_git_output(
        runner,
        checkout,
        ("diff", "HEAD", "--binary"),
        label="partial patch",
    )
    if not diff_text.strip():
        return None

    status_text = _required_git_output(
        runner,
        checkout,
        ("status", "--short"),
        label="changed file status",
    )
    diff_stat_text = _required_git_output(
        runner,
        checkout,
        ("diff", "--stat", "HEAD"),
        label="diff stat",
    )
    diff_check_text, diff_check_returncode = _best_effort_diff_check(runner, checkout)

    created_at_ns = time.time_ns()
    created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    log_dir.mkdir(parents=True, exist_ok=True)
    ensure_log_dir_ignored(log_dir)
    artifacts_dir = _artifact_dir(log_dir, context, created_at_ns)
    artifacts_dir.mkdir(parents=True, exist_ok=False)

    patch_path = artifacts_dir / "partial.patch"
    changed_files_path = artifacts_dir / "changed-files.txt"
    diff_stat_path = artifacts_dir / "diff-stat.txt"
    diff_check_path = artifacts_dir / "diff-check.txt"
    summary_path = artifacts_dir / "salvage-summary.md"
    metadata_path = artifacts_dir / "metadata.json"

    _write_text(patch_path, diff_text, trailing_newline=False)
    _write_text(changed_files_path, status_text or "(no status output)\n")
    _write_text(diff_stat_path, diff_stat_text or "(no diff stat output)\n")
    _write_text(diff_check_path, diff_check_text)

    summary = _summary_text(
        context=context,
        artifacts_dir=artifacts_dir,
        patch_path=patch_path,
        changed_files_path=changed_files_path,
        diff_stat_path=diff_stat_path,
        diff_check_path=diff_check_path,
        metadata_path=metadata_path,
        failure_category=failure_category,
        failure_reason=failure_reason,
        required_marker=required_marker,
        result=result,
        status_text=status_text,
        diff_check_returncode=diff_check_returncode,
    )
    _write_text(summary_path, summary)

    metadata: dict[str, Any] = {
        "schema_version": 1,
        "created_at": created_at,
        "created_at_ns": created_at_ns,
        "repo": context.repo,
        "issue_number": context.issue_number,
        "scope": context.scope,
        "agent": context.agent,
        "run_id": context.run_id,
        "approved_plan_hash": context.approved_plan_hash,
        "failure_category": failure_category,
        "failure_reason": failure_reason,
        "required_marker": required_marker,
        "response_file_missing": result is None or result.response_file_text is None,
        "returncode": None if result is None else result.returncode,
        "attempt_log": None if result is None or result.log_path is None else str(result.log_path),
        "directory": str(artifacts_dir),
        "partial_patch": str(patch_path),
        "changed_files": str(changed_files_path),
        "diff_stat": str(diff_stat_path),
        "diff_check": str(diff_check_path),
        "diff_check_returncode": diff_check_returncode,
        "summary": str(summary_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return SalvageArtifacts(
        directory=artifacts_dir,
        patch_path=patch_path,
        changed_files_path=changed_files_path,
        diff_stat_path=diff_stat_path,
        diff_check_path=diff_check_path,
        summary_path=summary_path,
        metadata_path=metadata_path,
        summary=summary,
    )


def format_salvage_artifacts_for_error(artifacts: SalvageArtifacts) -> str:
    return (
        "\nSalvage artifacts:\n"
        f"- directory: {artifacts.directory}\n"
        f"- summary: {artifacts.summary_path}\n"
        f"- patch: {artifacts.patch_path}"
    )


def _metadata_matches(
    metadata: dict[str, Any],
    *,
    repo: str,
    issue_number: int,
    scope: str,
    approved_plan_hash: str | None,
) -> bool:
    if metadata.get("repo") != repo:
        return False
    try:
        metadata_issue = int(metadata.get("issue_number"))
    except (TypeError, ValueError):
        return False
    if metadata_issue != issue_number:
        return False
    if metadata.get("scope") != scope:
        return False
    metadata_hash = metadata.get("approved_plan_hash")
    if approved_plan_hash is not None:
        return metadata_hash == approved_plan_hash
    return metadata_hash in (None, "")


def latest_salvage_summary(
    log_dir: Path,
    *,
    repo: str,
    issue_number: int,
    scope: str,
    approved_plan_hash: str | None = None,
) -> str | None:
    root = log_dir / "salvage"
    if not root.is_dir():
        return None

    newest: tuple[int, Path, str] | None = None
    for metadata_path in root.glob("**/metadata.json"):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(metadata, dict):
            continue
        if not _metadata_matches(
            metadata,
            repo=repo,
            issue_number=issue_number,
            scope=scope,
            approved_plan_hash=approved_plan_hash,
        ):
            continue
        raw_summary_path = metadata.get("summary")
        summary_path = (
            Path(raw_summary_path)
            if isinstance(raw_summary_path, str) and raw_summary_path
            else metadata_path.with_name("salvage-summary.md")
        )
        if not summary_path.is_absolute():
            summary_path = metadata_path.parent / summary_path
        try:
            summary = summary_path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if not summary:
            continue
        try:
            created_at_ns = int(metadata.get("created_at_ns"))
        except (TypeError, ValueError):
            try:
                created_at_ns = metadata_path.stat().st_mtime_ns
            except OSError:
                continue
        candidate = (created_at_ns, metadata_path, summary)
        if newest is None or candidate[:2] > newest[:2]:
            newest = candidate

    return None if newest is None else newest[2]
