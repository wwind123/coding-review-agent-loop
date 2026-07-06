"""Local salvage artifacts for failed mutating agent runs.

Also implements durable GitHub-backed salvage breadcrumbs (#507): when a
mutating implementation attempt fails, a hidden ``AGENT_SALVAGE`` marker
comment is posted alongside the local artifacts so a rerun with a different
coder, workdir, or machine can still discover the latest matching salvage
context.
"""

from __future__ import annotations

import base64
import datetime
import json
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .agents.base import AgentName, AgentResult
from .errors import AgentLoopError
from .github import IssueComment, post_issue_comment
from .logging import log
from .runner import ensure_log_dir_ignored

if TYPE_CHECKING:
    from .config import AgentLoopConfig
    from .runner import Runner

# Scopes whose rerun prompts actually consume a salvage summary today (#507).
# Posting is narrowed to these so pr-followup/task-implementation salvage
# stays local-only until their rerun prompts are wired to inject one.
SALVAGE_COMMENT_SCOPES = frozenset({"issue-implementation", "approved-plan-implementation"})

AGENT_SALVAGE_MARKER_RE = re.compile(
    r"<!--\s*AGENT_SALVAGE:\s*(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->",
    re.I,
)

_SALVAGE_SCHEMA_VERSION = 1
_MAX_COMMENT_CHARS = 60_000
_GIT_BINARY_PATCH_MARKER = "GIT binary patch"
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"gh[opsu]_[A-Za-z0-9]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"(?i)authorization:\s*bearer\s+\S+"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._-]{10,}"),
    re.compile(r"(?i)\b(password|secret|api[_-]?key)\s*[:=]\s*\S+"),
)


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


def _latest_local_salvage_record(
    log_dir: Path,
    *,
    repo: str,
    issue_number: int,
    scope: str,
    approved_plan_hash: str | None = None,
) -> tuple[int, str] | None:
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

    return None if newest is None else (newest[0], newest[2])


def latest_salvage_summary(
    log_dir: Path,
    *,
    repo: str,
    issue_number: int,
    scope: str,
    approved_plan_hash: str | None = None,
) -> str | None:
    record = _latest_local_salvage_record(
        log_dir,
        repo=repo,
        issue_number=issue_number,
        scope=scope,
        approved_plan_hash=approved_plan_hash,
    )
    return None if record is None else record[1]


def _truncate_field(text: str, *, max_lines: int = 20, max_chars: int = 4000) -> str:
    """Bound a text field before encoding, mirroring `_compact_failure_reason`'s tail-keeping rule."""
    if not text:
        return text
    lines = text.splitlines()
    if len(lines) > max_lines:
        text = "\n".join(lines[-max_lines:])
    if len(text) > max_chars:
        text = text[-max_chars:]
    return text


def _patch_secret_scan_flagged(patch_text: str) -> bool:
    return any(pattern.search(patch_text) for pattern in _SECRET_PATTERNS)


def _evaluate_patch_policy(patch_text: str, *, max_bytes: int) -> str | None:
    """Return the patch text to embed, or None if policy excludes it."""
    if not patch_text.strip():
        return None
    if len(patch_text.encode("utf-8")) > max_bytes:
        return None
    if _GIT_BINARY_PATCH_MARKER in patch_text:
        return None
    if _patch_secret_scan_flagged(patch_text):
        return None
    return patch_text


def _encode_json_payload(payload: dict[str, object]) -> str:
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii")


def _decode_json_payload(encoded: str) -> dict[str, object] | None:
    """Decode a marker payload, returning None instead of raising.

    Issue comments can come from third parties or a future incompatible
    schema, so discovery must skip malformed/unknown payloads silently
    rather than aborting a rerun.
    """
    try:
        payload = json.loads(base64.urlsafe_b64decode(encoded.encode("ascii")).decode("utf-8"))
    except (ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _build_salvage_payload(
    *,
    artifacts: SalvageArtifacts,
    context: SalvageContext,
    failure_category: str,
    failure_reason: str,
    patch_max_bytes: int,
) -> dict[str, object]:
    try:
        changed_files_text = artifacts.changed_files_path.read_text(encoding="utf-8")
    except OSError:
        changed_files_text = ""
    try:
        diff_stat_text = artifacts.diff_stat_path.read_text(encoding="utf-8")
    except OSError:
        diff_stat_text = ""
    try:
        patch_text = artifacts.patch_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        patch_text = ""

    included_patch_text = _evaluate_patch_policy(patch_text, max_bytes=patch_max_bytes)

    payload: dict[str, object] = {
        "schema_version": _SALVAGE_SCHEMA_VERSION,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "created_at_ns": time.time_ns(),
        "repo": context.repo,
        "issue_number": context.issue_number,
        "scope": context.scope,
        "agent": context.agent,
        "run_id": context.run_id,
        "approved_plan_hash": context.approved_plan_hash,
        "failure_category": failure_category,
        "failure_reason": _truncate_field(failure_reason),
        "changed_files": _truncate_field(changed_files_text),
        "diff_stat": _truncate_field(diff_stat_text),
        "local_directory": str(artifacts.directory),
        "patch_exists": bool(patch_text.strip()),
        "patch_included": included_patch_text is not None,
    }
    if included_patch_text is not None:
        payload["patch_text"] = included_patch_text
    return payload


def _render_salvage_comment_body(payload: dict[str, object]) -> str:
    approved_hash_line = (
        f"\n- Approved plan hash: `{payload['approved_plan_hash']}`"
        if payload.get("approved_plan_hash")
        else ""
    )
    patch_note = (
        "Patch content is embedded in the machine-readable marker below, within the "
        "configured size and safety limits."
        if payload.get("patch_included")
        else (
            "Patch content is local-only at "
            f"`{payload['local_directory']}/partial.patch` on the machine that produced "
            "this attempt; it was omitted from this comment (over the size limit, "
            "binary, or flagged by a secret scan) and was not included below."
        )
    )
    issue_number = payload.get("issue_number")
    return f"""### Implementation salvage breadcrumb

The {payload['agent']} agent failed before producing a valid public response. No
successful response, review result, or pull request should be inferred from this
comment. It exists only so a later rerun (possibly with a different coder, workdir,
or machine) can recover partial context; do not auto-apply anything from it.

- Repository: `{payload['repo']}`
- Issue: `{issue_number if issue_number is not None else "unknown"}`
- Scope: `{payload['scope']}`
- Agent: `{payload['agent']}`{approved_hash_line}
- Failure category: `{payload['failure_category']}`
- Failure reason: {payload['failure_reason']}
- Local artifact directory (on the original run's machine): `{payload['local_directory']}`
- {patch_note}

<!-- AGENT_SALVAGE: {_encode_json_payload(payload)} -->
"""


def _render_clamped_salvage_comment(payload: dict[str, object]) -> str:
    body = _render_salvage_comment_body(payload)
    if len(body) > _MAX_COMMENT_CHARS and payload.get("patch_included"):
        payload = dict(payload)
        payload.pop("patch_text", None)
        payload["patch_included"] = False
        body = _render_salvage_comment_body(payload)
    return body


def post_salvage_comment(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    artifacts: SalvageArtifacts,
    context: SalvageContext,
    failure_category: str,
    failure_reason: str,
) -> bool:
    """Best-effort post of a GitHub salvage breadcrumb comment.

    Returns whether a comment was posted. Never raises: posting is a
    durability nice-to-have and must never mask the original agent failure
    that triggered local salvage capture.
    """
    if context.issue_number is None or context.scope not in SALVAGE_COMMENT_SCOPES:
        return False
    if config.dry_run or not config.salvage_comments:
        return False
    try:
        payload = _build_salvage_payload(
            artifacts=artifacts,
            context=context,
            failure_category=failure_category,
            failure_reason=failure_reason,
            patch_max_bytes=config.salvage_comment_patch_max_bytes,
        )
        body = _render_clamped_salvage_comment(payload)
        post_issue_comment(runner, config=config, issue_number=context.issue_number, body=body)
    except (AgentLoopError, OSError, UnicodeDecodeError) as exc:
        log(config, f"salvage comment posting failed ({exc}); preserving original agent failure")
        return False
    return True


@dataclass(frozen=True)
class RemoteSalvageRecord:
    created_at_ns: int
    repo: str
    issue_number: int | None
    scope: str
    agent: str
    run_id: str | None
    approved_plan_hash: str | None
    failure_category: str
    failure_reason: str
    changed_files: str
    diff_stat: str
    local_directory: str
    patch_exists: bool
    patch_included: bool
    patch_text: str | None


def _remote_payload_matches(
    payload: dict[str, Any],
    *,
    repo: str,
    issue_number: int,
    scope: str,
    approved_plan_hash: str | None,
) -> bool:
    if payload.get("repo") != repo:
        return False
    try:
        payload_issue = int(payload.get("issue_number"))
    except (TypeError, ValueError):
        return False
    if payload_issue != issue_number:
        return False
    if payload.get("scope") != scope:
        return False
    payload_hash = payload.get("approved_plan_hash")
    if approved_plan_hash is not None:
        return payload_hash == approved_plan_hash
    return payload_hash in (None, "")


def _record_from_payload(payload: dict[str, Any]) -> RemoteSalvageRecord | None:
    if payload.get("schema_version") != _SALVAGE_SCHEMA_VERSION:
        return None
    try:
        created_at_ns = int(payload["created_at_ns"])
        repo = str(payload["repo"])
        scope = str(payload["scope"])
        agent = str(payload["agent"])
    except (KeyError, TypeError, ValueError):
        return None
    issue_number_raw = payload.get("issue_number")
    issue_number = issue_number_raw if isinstance(issue_number_raw, int) else None
    run_id = payload.get("run_id")
    run_id = run_id if isinstance(run_id, str) else None
    approved_plan_hash = payload.get("approved_plan_hash")
    approved_plan_hash = approved_plan_hash if isinstance(approved_plan_hash, str) else None
    patch_included = bool(payload.get("patch_included", False))
    patch_text = payload.get("patch_text")
    patch_text = patch_text if patch_included and isinstance(patch_text, str) else None
    return RemoteSalvageRecord(
        created_at_ns=created_at_ns,
        repo=repo,
        issue_number=issue_number,
        scope=scope,
        agent=agent,
        run_id=run_id,
        approved_plan_hash=approved_plan_hash,
        failure_category=str(payload.get("failure_category", "")),
        failure_reason=str(payload.get("failure_reason", "")),
        changed_files=str(payload.get("changed_files", "")),
        diff_stat=str(payload.get("diff_stat", "")),
        local_directory=str(payload.get("local_directory", "")),
        patch_exists=bool(payload.get("patch_exists", False)),
        patch_included=patch_included,
        patch_text=patch_text,
    )


def find_latest_remote_salvage(
    comments: Sequence[IssueComment],
    *,
    repo: str,
    issue_number: int,
    scope: str,
    approved_plan_hash: str | None = None,
) -> RemoteSalvageRecord | None:
    best: RemoteSalvageRecord | None = None
    for comment in comments:
        body = getattr(comment, "body", None)
        if not isinstance(body, str):
            continue
        for match in AGENT_SALVAGE_MARKER_RE.finditer(body):
            payload = _decode_json_payload(match.group("payload"))
            if payload is None:
                continue
            if not _remote_payload_matches(
                payload,
                repo=repo,
                issue_number=issue_number,
                scope=scope,
                approved_plan_hash=approved_plan_hash,
            ):
                continue
            record = _record_from_payload(payload)
            if record is None:
                continue
            if best is None or record.created_at_ns >= best.created_at_ns:
                best = record
    return best


def render_remote_salvage_summary(record: RemoteSalvageRecord) -> str:
    approved_hash_line = (
        f"\n- Approved plan hash: `{record.approved_plan_hash}`" if record.approved_plan_hash else ""
    )
    if record.patch_text is not None:
        patch_section = (
            "\n## Patch (embedded in the GitHub salvage comment)\n\n"
            "```diff\n"
            f"{record.patch_text}\n"
            "```\n"
        )
    elif record.patch_exists:
        patch_section = (
            "\n## Patch\n\n"
            f"The partial patch is local-only at `{record.local_directory}/partial.patch` "
            "on the machine that produced this attempt and was not included in the "
            "GitHub comment.\n"
        )
    else:
        patch_section = ""
    return f"""# Salvage Summary (recovered from a GitHub issue comment)

The {record.agent} agent failed before producing a valid public response on a
prior run. No successful response, review result, or pull request should be
inferred from this artifact. Local artifact paths below were written on that
run's machine/workdir and may not exist here.

This is incomplete context only. A later attempt may cherry-pick or ignore it
selectively, but must not treat it as validated, complete, or automatically
applicable.

- Repository: `{record.repo}`
- Issue: `{record.issue_number if record.issue_number is not None else "unknown"}`
- Scope: `{record.scope}`
- Agent: `{record.agent}`{approved_hash_line}
- Failure category: `{record.failure_category}`
- Failure reason: {record.failure_reason}

## Local Artifacts (from the original run; may not exist here)

- Salvage directory: `{record.local_directory}`
- Changed files/status:

{record.changed_files}

- Diff stat:

{record.diff_stat}
{patch_section}"""


def latest_salvage_context(
    log_dir: Path,
    comments: Sequence[IssueComment],
    *,
    repo: str,
    issue_number: int,
    scope: str,
    approved_plan_hash: str | None = None,
) -> str | None:
    """Merge local and GitHub-backed salvage discovery, newest wins (ties prefer local)."""
    local_record = _latest_local_salvage_record(
        log_dir,
        repo=repo,
        issue_number=issue_number,
        scope=scope,
        approved_plan_hash=approved_plan_hash,
    )
    remote_record = find_latest_remote_salvage(
        comments,
        repo=repo,
        issue_number=issue_number,
        scope=scope,
        approved_plan_hash=approved_plan_hash,
    )
    if remote_record is None:
        return None if local_record is None else local_record[1]
    if local_record is None:
        return render_remote_salvage_summary(remote_record)
    local_created_at_ns, local_summary = local_record
    if remote_record.created_at_ns > local_created_at_ns:
        return render_remote_salvage_summary(remote_record)
    return local_summary
