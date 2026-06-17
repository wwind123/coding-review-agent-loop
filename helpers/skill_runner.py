"""
Run one complete review round for a plan or PR from within a Claude Code skill session.

Subcommands:

  run-plan-round
    --issue N --repo OWNER/REPO
    --plan-file PATH
    --reviewers REVIEWER [REVIEWER ...]
    [--workdir-codex PATH] [--workdir-gemini PATH] [--workdir PATH]
    [--dry-run]

  run-pr-round
    --pr N --repo OWNER/REPO
    --reviewers REVIEWER [REVIEWER ...]
    [--head-sha SHA]
    [--workdir PATH] [--workdir-codex PATH] [--workdir-gemini PATH]
    [--dry-run]

  retry-validate
    --repair-dir PATH
    [--dry-run]

    Re-validate an already-written reviewer response without re-running the agent.
    On validation failure, skill_runner saves the raw response plus context to a
    stable repair dir and prints the retry command. Edit {repair_dir}/raw.md, then
    run this subcommand to complete the round. A repair dir tagged `role: coder`
    (a malformed external coder plan) is routed to the coder-completion path.

  complete-host-review
    --dir PATH
    [--dry-run]

    Complete a host (Claude) reviewer turn in reversed-roles mode. When `claude`
    is among --reviewers, run-plan-round writes a review-request dir and marks the
    round pending; write your plan_review JSON to {dir}/host-review.md, then run
    this subcommand to validate, render, attach (--agent Claude), and post it.

  run-implement
    --issue N --repo OWNER/REPO
    --coder {codex,gemini}
    --plan-file PATH
    [--base BRANCH] [--workdir PATH] [--dry-run]

    Reversed roles: an external coder implements an approved plan and opens a PR
    (validated for the PR marker, human-requirements ack, in-workdir tests, an
    advanced HEAD, and that the PR is open and references the issue), then posts a
    role: coder PR comment so run-pr-round can review it. Idempotent per plan via
    the one-shot handoff marker. Requires a push-capable workdir.

  run-implement-by-phase
    --issue N --repo OWNER/REPO
    --coder {codex,gemini}
    --plan-file PATH
    [--base BRANCH] [--workdir PATH] [--workdir-codex PATH] [--workdir-gemini PATH]
    [--dry-run]

    Reversed roles: decompose an approved plan into child phase issues, then
    implement phase 1 when it is agent-pr. Reruns reuse decomposition and phase
    handoff markers. Requires a push-capable workdir for live implementation.

  run-decompose
    --issue N --repo OWNER/REPO
    --coder {codex,gemini}
    --plan-file PATH
    [--workdir PATH] [--workdir-codex PATH] [--workdir-gemini PATH]
    [--dry-run]

    Reversed roles: an external coder decomposes an approved plan into child
    phase issues. Live runs create real GitHub issues and post an idempotency
    marker on the parent. Dry-run invokes a parseable stub and creates nothing.

Prints a JSON result to stdout and exits 0:
  {"state": "approved"|"blocking", "round_number": N,
   "blocking_items": [...], "approved_reviewers": [...]}
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from coding_review_agent_loop.agents.base import AgentName
from coding_review_agent_loop.agents.registry import agent_display_name
from coding_review_agent_loop.config import ensure_temp_checkout
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.followups import (
    _approved_followup_from_unresolved_item,
    _publish_approved_followups,
)
from coding_review_agent_loop.memory import AgentMemoryContext, prepare_agent_memory
from coding_review_agent_loop.runner import Runner, tail_text
from coding_review_agent_loop.usage import RunUsageContext, UsageMetadata
from coding_review_agent_loop.protocol import ReviewItemDisposition, UnresolvedReviewItem
from coding_review_agent_loop.round_state import (
    PostedRoundMetadata,
    _attach_round_metadata,
    _deserialize_disposition,
    _deserialize_unresolved_item,
    _plan_subject,
    _serialize_unresolved_item,
    _serialize_disposition,
)
from coding_review_agent_loop.unresolved_items import apply_item_dispositions

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HELPERS = Path(__file__).parent
_REPAIR_BASE = Path(tempfile.gettempdir()) / "coding-review-agent-loop" / "repair"


class _ValidationError(Exception):
    """Raised by _complete_reviewer_turn when the reviewer response fails validation."""


def _run_helper(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [sys.executable, "-m", *args],
        capture_output=False,
        text=True,
        cwd=_HELPERS.parent,
        check=False,
    )
    if check and result.returncode != 0:
        print(f"skill_runner: helper {args[0]} failed (exit {result.returncode})", file=sys.stderr)
        sys.exit(result.returncode)
    return result


def _run_helper_capture(*args: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [sys.executable, "-m", *args],
        capture_output=True,
        text=True,
        cwd=_HELPERS.parent,
        check=False,
    )
    return result


def _plan_subject_of_file(path: Path) -> str:
    return hashlib.sha256(path.read_text(encoding="utf-8").strip().encode("utf-8")).hexdigest()


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


_ITEM_ID_RE = re.compile(r"-(\d+)$")
_DISPOSITION_ALIASES: dict[str, str] = {
    "still blocking": "blocking",
    "remains blocking": "blocking",
    "keep blocking": "blocking",
    "stay blocking": "blocking",
    "still same-pr": "same-pr",
    "still same-plan": "same-plan",
    "still future": "future",
}
_STATE_MARKER_RE = re.compile(r"<!-- AGENT(?:_PLAN)?_STATE: \w+ -->")
_SIGNATURE_RE = re.compile(r"^--\s+\S[^\n]*", re.MULTILINE)


def _normalize_disposition_values(text: str) -> str:
    """Fix known bad disposition values in the review JSON blob.

    Agents occasionally write natural-language variants like 'still blocking'
    instead of the canonical 'blocking'. This normalizes them before validation.
    """
    try:
        stripped = text.lstrip()
        decoder = json.JSONDecoder()
        data, end_idx = decoder.raw_decode(stripped)
        footer = stripped[end_idx:]
    except json.JSONDecodeError:
        return text
    changed = False
    for key in ("prior_plan_item_dispositions", "prior_item_dispositions"):
        for disp in data.get(key, []):
            raw = str(disp.get("disposition", ""))
            canonical = _DISPOSITION_ALIASES.get(raw.lower())
            if canonical:
                disp["disposition"] = canonical
                changed = True
    # Flatten list fields whose items are dicts instead of plain strings.
    # Agents occasionally write {"title": "...", "detail": "..."} objects.
    for key in (
        "blocking_plan_issues", "same_plan_followups",
        "blocking_items", "same_pr_followups", "future_followups",
    ):
        if key not in data:
            continue
        flattened = []
        any_dict = False
        for item in data[key]:
            if isinstance(item, dict):
                any_dict = True
                title = str(
                    item.get("title") or item.get("text")
                    or item.get("issue") or item.get("summary") or ""
                )
                detail = str(item.get("detail") or item.get("description") or "")
                flattened.append(f"{title}: {detail}".strip(": ") if detail else title)
            else:
                flattened.append(item)
        if any_dict:
            data[key] = flattened
            changed = True
    # Blocking plan/PR reviews may not include future_followups per protocol.
    # Strip them rather than failing validation on an otherwise-valid review.
    if data.get("state") == "blocking" and data.get("future_followups"):
        data["future_followups"] = []
        changed = True
    if not changed:
        return text
    return json.dumps(data, indent=2) + "\n" + footer.lstrip()


_HR_RESOLVED_LINE_RE = re.compile(r"^\s*<!--\s*HUMAN_REQUIREMENTS_RESOLVED\s*-->\s*$")


def _normalize_raw_response(text: str) -> str:
    """Strip content the protocol validator would reject.

    1. Leading prose before the JSON object (Gemini sometimes writes a summary line first).
    2. Between the closing } and <!-- AGENT_STATE -->: keep only the one recognized optional
       marker (<!-- HUMAN_REQUIREMENTS_RESOLVED -->); strip everything else (e.g. Codex's
       stray <!-- HUMAN_REQUIREMENTS_RESOLVED --> mixed with other unrecognized lines, or
       prose that predates the structured format).
    3. Any content after the first agent signature line (e.g. a duplicate state marker).

    This function is only called for reviewer responses (pr_review / plan_review).
    Plan-revision responses (plan_state) are never passed here; their
    <!-- HUMAN_REQUIREMENTS_ADDRESSED --> + ### Human requirements sections are not at risk.
    """
    # Strip leading prose before the JSON block
    brace_idx = text.find("{")
    if brace_idx > 0:
        text = text[brace_idx:]

    # Use raw_decode to find exact end of the JSON object
    try:
        _, json_end_idx = json.JSONDecoder().raw_decode(text)
        json_text = text[:json_end_idx]
        remainder = text[json_end_idx:]
    except json.JSONDecodeError:
        json_text = text
        remainder = ""

    # Find state marker and signature. Either order may appear (Gemini sometimes
    # places the signature before the state marker).
    m = _STATE_MARKER_RE.search(remainder)
    if m is None:
        return text

    sig_m = _SIGNATURE_RE.search(remainder, m.end())
    if sig_m is not None:
        # Standard order: STATE_MARKER … SIGNATURE
        # Keep HR_RESOLVED from between-section; drop everything else.
        between = remainder[: m.start()]
        state_and_sig = remainder[m.start() : sig_m.end()]
    else:
        # Non-standard order: SIGNATURE … STATE_MARKER (Gemini legacy)
        sig_m = _SIGNATURE_RE.search(remainder)
        if sig_m is None or sig_m.start() >= m.start():
            return text  # Cannot normalize — return as-is
        # Reorder to standard form.
        between = remainder[: sig_m.start()]
        state_text = remainder[m.start() : m.end()].strip()
        sig_text = remainder[sig_m.start() : sig_m.end()].strip()
        state_and_sig = state_text + "\n" + sig_text

    # Reconstruct: JSON + (HR_RESOLVED if present) + state-marker + signature.
    # All other between-section content (prose, unrecognized HTML comments) is dropped.
    hr_resolved = next(
        (line.strip() for line in between.splitlines() if _HR_RESOLVED_LINE_RE.match(line)),
        None,
    )
    prefix = (hr_resolved + "\n") if hr_resolved else ""
    return json_text.rstrip() + "\n" + prefix + state_and_sig.rstrip() + "\n"


def _save_raw_to_repair_dir(
    *,
    agent: str,
    agent_cap: str,
    flow: str,
    issue: int,
    repo: str,
    new_round_number: int,
    round_subject: str,
    item_id_offset: int,
    validate_kind: str,
    dry_run: bool,
    raw_output: Path,
    context_file: Path,
    prior_items_file: Path,
) -> Path:
    """Copy raw response + context to a stable repair dir before normalization/validation."""
    repair_dir = _REPAIR_BASE / f"{issue}-r{new_round_number}-{agent}"
    repair_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(raw_output, repair_dir / "raw.md")
    shutil.copy2(context_file, repair_dir / "context.json")
    shutil.copy2(prior_items_file, repair_dir / "prior_items.json")
    manifest = {
        "agent": agent, "agent_cap": agent_cap, "flow": flow,
        "issue": issue, "repo": repo,
        "new_round_number": new_round_number, "round_subject": round_subject,
        "item_id_offset": item_id_offset, "validate_kind": validate_kind,
        "dry_run": dry_run,
    }
    (repair_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return repair_dir


def _max_item_number(item_lists: list[list[dict]]) -> int:
    """Return the highest numeric suffix found across all item dicts, or 0 if none."""
    max_n = 0
    for items in item_lists:
        for item in items:
            m = _ITEM_ID_RE.search(str(item.get("item_id", "")))
            if m:
                max_n = max(max_n, int(m.group(1)))
    return max_n


def _workdir_for_agent(agent: str, args: argparse.Namespace) -> str:
    specific = getattr(args, f"workdir_{agent}", None)
    if specific:
        return specific
    generic = getattr(args, "workdir", None)
    if generic:
        return generic
    # Auto-clone to a temp path; run_external validates/re-clones via --repo
    tmp = Path(tempfile.gettempdir()) / "coding-review-agent-loop" / f"skill-runner-{agent}"
    tmp.mkdir(parents=True, exist_ok=True)
    return str(tmp)


def _model_used_from_usage(usage_file: Path) -> str:
    """Model the agent ran, read from run_external's usage sidecar (#332).

    Returns "" when the sidecar is missing/unreadable or carries no model, so the
    signature falls back to the generic provider name.
    """
    if usage_file.exists():
        try:
            return json.loads(usage_file.read_text(encoding="utf-8")).get("model_used") or ""
        except (OSError, json.JSONDecodeError, AttributeError):
            return ""
    return ""


def _agent_memory_dir(repo: str) -> Path:
    slug = repo.replace("/", "-").replace(":", "-")
    state_home = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    return state_home / "coding-review-agent-loop" / "skill-sessions" / slug / "agent-memory"


def _prepare_skill_memory(
    repo: str,
    reviewers: list[str],
    memory_workdir: str,
    *,
    refresh: bool,
    dry_run: bool,
) -> AgentMemoryContext | None:
    """Build repo-scoped agent memory once for a round (advisory; never crashes).

    Memory generation is deterministic (git + static analysis) but needs a
    checked-out target repo, so this ensures `memory_workdir` exists first. The
    config's coder is set to ``reviewers[0]`` with its dir = `memory_workdir` so
    `active_workdir(config)` resolves there. Failures degrade to None.
    """
    if dry_run or not reviewers:
        return None
    from helpers.prompt_builders import make_minimal_config
    first = reviewers[0]
    try:
        config = dataclasses.replace(
            make_minimal_config(
                repo, first, tuple(reviewers),
                reviewer=first, workdir=memory_workdir,
            ),
            agent_memory=True,
            agent_memory_dir=_agent_memory_dir(repo),
            refresh_agent_memory=refresh,
        )
        runner = Runner(dry_run=False)
        ensure_temp_checkout(Path(memory_workdir), agent=first, config=config, runner=runner)
        return prepare_agent_memory(runner, config)
    except Exception as exc:  # noqa: BLE001 — advisory only
        print(f"skill_runner: agent memory unavailable ({exc}); continuing without it",
              file=sys.stderr)
        return None


def _run_test_gate(command: str, workdir: str, *, dry_run: bool) -> dict:
    """Run an optional test command and report the outcome. Never raises.

    Distinguishes "ran and failed" (``exit_code`` int, ``output_tail``) from
    "could not run" (``exit_code`` null, ``error``). Setup failures (bad quoting,
    empty command, missing executable, bad workdir) are reported, not propagated,
    so the review round is never crashed by the gate.
    """
    if dry_run:
        return {"command": command, "skipped": True, "reason": "dry-run"}
    base = {"command": command, "passed": False, "exit_code": None}
    try:
        argv = shlex.split(command)
    except ValueError as exc:
        return {**base, "error": f"could not parse --test-command: {exc}"}
    if not argv:
        return {**base, "error": "--test-command is empty"}
    try:
        proc = subprocess.run(
            argv,
            cwd=workdir,
            text=True,
            errors="replace",  # test output may contain bytes invalid for the locale;
                               # decode lossily so the gate never raises UnicodeDecodeError
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    except OSError as exc:  # FileNotFoundError, NotADirectoryError, ...
        return {**base, "error": f"could not run --test-command: {exc}"}
    return {
        "command": command,
        "passed": proc.returncode == 0,
        "exit_code": proc.returncode,
        "output_tail": tail_text(proc.stdout or "", max_lines=50),
    }


def _maybe_test_gate(
    test_command: str | None, test_workdir: str, *, dry_run: bool
) -> dict | None:
    """Return the test-gate result, or None when no --test-command was provided.

    Distinguishes "not requested" (``None`` -> no gate, no ``tests`` field) from a
    provided-but-empty command (``--test-command ""``), which flows through to
    _run_test_gate and is reported as a setup error rather than silently
    disabling the gate.
    """
    if test_command is None:
        return None
    return _run_test_gate(test_command, test_workdir, dry_run=dry_run)


def _mint_new_items(
    *,
    blocking_texts: list[str],
    same_texts: list[str],
    future_texts: list[str],
    flow: str,
    agent_cap: str,
    new_round_number: int,
    item_id_offset: int,
) -> list[dict]:
    """Serialize a reviewer's items with globally-unique IDs.

    Future follow-ups are minted with status="future" so they flow through
    AGENT_LOOP_META -> build-resume -> the resume path (resume-safe) and can be
    published on an approved round (#300). Normalize clears future_followups for
    blocking reviews, so blocking rounds pass an empty future_texts.
    """
    same_s = "same-plan" if flow == "plan" else "same-pr"
    groups = [
        ("blocking", "blocking", blocking_texts),
        (same_s, same_s, same_texts),
        ("future", "future", future_texts),
    ]
    items: list[dict] = []
    counter = item_id_offset + 1
    for status, source_status, texts in groups:
        for text in texts:
            items.append({
                "item_id": f"item-{counter}",
                "reviewer": agent_cap,
                "source_round": new_round_number,
                "text": str(text),
                "status": status,
                "source_status": source_status,
                "notes": [],
            })
            counter += 1
    return items


def _aggregate_reviewer_usage(records: list[dict]) -> dict | None:
    """Aggregate external-agent usage across reviewer records (#308).

    Each record's ``usage`` is the sidecar dict written by run_external
    (``{agent, session_id, returncode, usage: <UsageMetadata dict>}``). Returns a
    labeled, external-agents-only summary, or None if no record carried usage.
    ``success_count`` is intentionally dropped: it is a validation metric not
    tracked per usage record here, and would otherwise read 0.
    """
    ctx = RunUsageContext(run_id="skill", summary_path=Path(tempfile.gettempdir()) / "skill-usage.json")
    found = False
    for rec in records:
        sidecar = rec.get("usage") if isinstance(rec, dict) else None
        if not isinstance(sidecar, dict) or not isinstance(sidecar.get("usage"), dict):
            continue
        found = True
        ctx.add_record(
            agent=sidecar.get("agent", "codex"),
            session_id=sidecar.get("session_id"),
            returncode=int(sidecar.get("returncode", 0) or 0),
            usage=UsageMetadata(**sidecar["usage"]),
        )
    if not found:
        return None

    def _strip(d: dict) -> dict:
        d.pop("success_count", None)
        return d

    return {
        "scope": "external-agents-only",
        "note": "Host (Claude) coder/plan turns run in the interactive session and are not counted.",
        "totals": _strip(ctx.totals().to_dict()),
        "per_agent": {agent: _strip(t.to_dict()) for agent, t in ctx.per_agent_totals().items()},
    }


def _publish_pr_followups(
    repo: str,
    pr: int,
    head_sha: str | None,
    mode: str,
    future_items: list[dict],
    pr_comments: list,
    *,
    dry_run: bool,
) -> dict:
    """Publish approved future follow-ups for a PR via the library logic (#300).

    Reuses ``followups._publish_approved_followups`` (reconcile, dedupe,
    idempotency marker, post-comment / file-issues). Returns a small dict for the
    round result. Never publishes for mode 'ignore' or when there are no items.
    """
    if mode == "ignore" or not future_items:
        return {"mode": mode, "published": False, "count": 0}
    approved = [
        _approved_followup_from_unresolved_item(_deserialize_unresolved_item(item))
        for item in future_items
    ]
    if dry_run:
        return {"mode": mode, "dry_run": True, "count": len(approved)}
    from helpers.prompt_builders import make_minimal_config
    from coding_review_agent_loop.workdirs import active_workdir
    config = dataclasses.replace(
        make_minimal_config(repo, "claude", ("codex", "gemini")),
        approved_followups=mode,
    )
    # The library posts via `gh --repo`, but runs gh with cwd=active_workdir(config)
    # (the coder dir). The Codex+Gemini skill flow never creates a Claude checkout,
    # so ensure that directory exists or gh raises FileNotFoundError (#300).
    Path(active_workdir(config)).mkdir(parents=True, exist_ok=True)
    published = _publish_approved_followups(
        Runner(dry_run=False),
        config=config,
        pr_number=pr,
        head_sha=head_sha,
        pr_comments=pr_comments,
        followups=approved,
    )
    return {"mode": mode, "published": bool(published), "count": len(approved)}


def _fetch_issue_json(repo: str, issue: int, gh_cmd: str = "gh") -> dict:
    result = subprocess.run(
        [gh_cmd, "issue", "view", str(issue), "--repo", repo,
         "--json", "number,title,body,url"],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        print(f"skill_runner: gh issue view failed: {result.stderr.strip()}", file=sys.stderr)
        sys.exit(1)
    return json.loads(result.stdout)


def _fetch_pr_json(repo: str, pr: int, gh_cmd: str = "gh") -> dict:
    result = subprocess.run(
        [gh_cmd, "pr", "view", str(pr), "--repo", repo,
         "--json", "number,title,body,url,headRefOid"],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        print(f"skill_runner: gh pr view failed: {result.stderr.strip()}", file=sys.stderr)
        sys.exit(1)
    return json.loads(result.stdout)


def _fetch_pr_diff(repo: str, pr: int, gh_cmd: str = "gh") -> str:
    result = subprocess.run(
        [gh_cmd, "pr", "diff", str(pr), "--repo", repo],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        print(f"skill_runner: gh pr diff failed: {result.stderr.strip()}", file=sys.stderr)
        sys.exit(1)
    return result.stdout


def _fetch_issue_comments_raw(repo: str, issue: int, gh_cmd: str = "gh") -> list[str]:
    # Use `gh issue view --json comments` to get correctly structured comment bodies
    # without the multiline-splitting bug that `--jq .[].body` produces.
    result = subprocess.run(
        [gh_cmd, "issue", "view", str(issue), "--repo", repo, "--json", "comments"],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        return []
    try:
        data = json.loads(result.stdout)
        return [c["body"] for c in data.get("comments", []) if isinstance(c, dict) and "body" in c]
    except (json.JSONDecodeError, KeyError):
        return []


def _fetch_pr_comments_raw(repo: str, pr: int, gh_cmd: str = "gh") -> list[str]:
    # PR conversation comments live behind `gh pr view` (not `gh issue view`,
    # which rejects PR numbers). Used for the approved-followups idempotency marker.
    result = subprocess.run(
        [gh_cmd, "pr", "view", str(pr), "--repo", repo, "--json", "comments"],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        return []
    try:
        data = json.loads(result.stdout)
        return [c["body"] for c in data.get("comments", []) if isinstance(c, dict) and "body" in c]
    except (json.JSONDecodeError, KeyError):
        return []


def _build_resume(
    issue_or_pr: int,
    repo: str,
    reviewers: list[str],
    flow: str,
    *,
    head_sha: str | None = None,
    pr: int | None = None,
) -> dict:
    extra: list[str] = []
    if flow == "pr":
        if head_sha:
            extra += ["--head-sha", head_sha]
        if pr:
            extra += ["--pr", str(pr)]
    result = _run_helper_capture(
        "helpers.state_manager", "build-resume",
        "--issue", str(issue_or_pr),
        "--repo", repo,
        "--reviewers", *reviewers,
        "--flow", flow,
        *extra,
    )
    if result.returncode != 0:
        print(f"skill_runner: build-resume failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    return json.loads(result.stdout)


# ---------------------------------------------------------------------------
# Pending-comment reconciliation
# ---------------------------------------------------------------------------

def _reconcile_pending_comment(
    resume: dict,
    issue: int,
    repo: str,
    dry_run: bool,
) -> None:
    body = resume.get("pending_comment_body")
    if not body:
        return
    body_path = Path(str(body))
    if not body_path.exists():
        # Stale reference; clear it
        _run_helper(
            "helpers.state_manager", "clear-pending-comment",
            "--issue", str(issue), "--repo", repo,
        )
        return
    body_text = body_path.read_text(encoding="utf-8")
    # Check if already posted by comparing against raw GitHub comment bodies.
    # Both body_text and the posted comment contain the full AGENT_LOOP_META block,
    # so a direct (stripped) comparison is sufficient and avoids any parsing dependency.
    existing_bodies = _fetch_issue_comments_raw(repo, issue)
    already_posted = body_text.strip() in {b.strip() for b in existing_bodies}
    if already_posted:
        _run_helper(
            "helpers.state_manager", "clear-pending-comment",
            "--issue", str(issue), "--repo", repo,
        )
        return
    # Not posted yet; post it now
    if not dry_run:
        _run_helper(
            "helpers.gh_ops", "post-issue-comment",
            "--issue", str(issue), "--file", str(body_path), "--repo", repo,
        )
    else:
        print(f"[dry-run] would post pending comment from {body_path}")
    _run_helper(
        "helpers.state_manager", "clear-pending-comment",
        "--issue", str(issue), "--repo", repo,
    )


# ---------------------------------------------------------------------------
# Ledger transition (new round)
# ---------------------------------------------------------------------------

def _compute_next_prior_items(
    prior_items_raw: list[dict],
    completed_reviewer_data: list[dict],
    same_status: str,
    retain_future: bool,
) -> list[dict]:
    """Apply completed-round reviewer data to produce next round's prior_items list."""
    prior_items = [_deserialize_unresolved_item(item) for item in prior_items_raw]

    # Build dispositions_by_item from all completed reviewers
    dispositions_by_item: dict[str, list[ReviewItemDisposition]] = {}
    for record in completed_reviewer_data:
        for d in record.get("dispositions", []):
            item_id = str(d["item_id"])
            dispositions_by_item.setdefault(item_id, []).append(_deserialize_disposition(d))

    remaining, _future = apply_item_dispositions(
        prior_items,
        dispositions_by_item,
        same_status=same_status,
        retain_future=retain_future,
    )

    # Add new must-fix items from completed reviewers (blocking and same-status only)
    must_fix_statuses = {"blocking", same_status}
    round_new: list[UnresolvedReviewItem] = [
        _deserialize_unresolved_item(item)
        for record in completed_reviewer_data
        for item in record.get("new_items", [])
        if item.get("status") in must_fix_statuses
    ]

    next_items = [
        item for item in [*remaining, *round_new]
        if item.status in must_fix_statuses
    ]
    return [_serialize_unresolved_item(item) for item in next_items]


# ---------------------------------------------------------------------------
# Per-reviewer turn completion (shared by _run_reviewer and cmd_retry_validate)
# ---------------------------------------------------------------------------

def _complete_reviewer_turn(
    *,
    agent: str,
    agent_cap: str,
    flow: str,
    validate_kind: str,
    issue: int,
    repo: str,
    new_round_number: int,
    round_subject: str,
    next_prior_items_raw: list[dict],
    item_id_offset: int,
    dry_run: bool,
    raw_output: Path,
    context_file: Path,
    work_dir: Path,
) -> dict:
    """Normalize, validate, render, parse, mint IDs, attach metadata, and post.

    Normalizes raw_output in-place. Raises _ValidationError on validation failure.
    Returns {reviewer_name, state, blocking_items, new_items}.
    """
    rendered_output   = work_dir / f"{agent}-review-rendered.md"
    tagged_output     = work_dir / f"{agent}-review-tagged.md"
    prior_items_file  = work_dir / f"{agent}-prior-items.json"
    dispositions_file = work_dir / f"{agent}-dispositions.json"
    new_items_file    = work_dir / f"{agent}-new-items.json"

    _write_json(prior_items_file, next_prior_items_raw)

    # --- Normalize ---
    raw_text = raw_output.read_text(encoding="utf-8")
    raw_text = _normalize_raw_response(raw_text)
    raw_text = _normalize_disposition_values(raw_text)
    raw_output.write_text(raw_text, encoding="utf-8")

    # --- Validate ---
    result = _run_helper_capture(
        "helpers.validate_response",
        "--file", str(raw_output),
        "--kind", validate_kind,
        "--context-file", str(context_file),
    )
    if result.returncode != 0:
        raise _ValidationError(
            f"skill_runner: {agent} review validation failed: {result.stderr.strip()}\n"
            f"skill_runner: raw response at: {raw_output}"
        )

    # --- Render ---
    # The model the reviewer actually ran is in run_external's usage sidecar (#332);
    # pass it so the rendered signature reads e.g. "Google Antigravity: Gemini 3.1 Pro (High)".
    reviewer_model = _model_used_from_usage(work_dir / f"{agent}-usage.json")
    _run_helper(
        "helpers.render_response",
        "--file", str(raw_output),
        "--kind", validate_kind,
        "--reviewer", agent_cap,
        "--context-file", str(context_file),
        "--output", str(rendered_output),
        *(["--model", reviewer_model] if reviewer_model else []),
    )

    # --- Parse review JSON for metadata ---
    raw_text = raw_output.read_text(encoding="utf-8")
    try:
        decoder = json.JSONDecoder()
        review_json, _ = decoder.raw_decode(raw_text.lstrip())
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"skill_runner: cannot parse {agent} review JSON: {exc}", file=sys.stderr)
        sys.exit(1)

    parsed_state = str(review_json.get("state", "approved"))
    disp_key = "prior_plan_item_dispositions" if flow == "plan" else "prior_item_dispositions"
    raw_dispositions = review_json.get(disp_key, [])
    for d in raw_dispositions:
        if "reviewer" not in d:
            d["reviewer"] = agent_cap

    blocking_key = "blocking_plan_issues" if flow == "plan" else "blocking_items"
    blocking_texts: list[str] = list(review_json.get(blocking_key, []))
    same_key = "same_plan_followups" if flow == "plan" else "same_pr_followups"
    new_unresolved_texts: list[str] = list(review_json.get(same_key, []))

    # When a reviewer blocks via same-round followups only (no explicit blocking issues),
    # surface those followups as blocking_items for the caller so the overall state is correct.
    reported_blocking: list[str] = blocking_texts
    if parsed_state == "blocking" and not blocking_texts:
        reported_blocking = new_unresolved_texts

    _write_json(dispositions_file, raw_dispositions)

    # --- Mint IDs globally unique across rounds ---
    new_items_serialized = _mint_new_items(
        blocking_texts=blocking_texts,
        same_texts=new_unresolved_texts,
        future_texts=[str(t) for t in review_json.get("future_followups", [])],
        flow=flow,
        agent_cap=agent_cap,
        new_round_number=new_round_number,
        item_id_offset=item_id_offset,
    )
    _write_json(new_items_file, new_items_serialized)

    # External-agent usage sidecar written by run_external (#308); persist it in
    # AGENT_LOOP_META so resumed rounds can aggregate the complete cost.
    usage_file = work_dir / f"{agent}-usage.json"
    reviewer_usage: dict | None = None
    if usage_file.exists():
        try:
            reviewer_usage = json.loads(usage_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            reviewer_usage = None
    usage_args = ["--usage-file", str(usage_file)] if usage_file.exists() else []

    # --- Attach metadata ---
    _run_helper(
        "helpers.state_manager", "attach-metadata",
        "--body-file", str(rendered_output),
        "--output", str(tagged_output),
        "--flow", flow,
        "--role", "reviewer",
        "--agent", agent_cap,
        "--round-number", str(new_round_number),
        "--state", parsed_state,
        "--prior-items-file", str(prior_items_file),
        "--dispositions-file", str(dispositions_file),
        "--new-items-file", str(new_items_file),
        "--subject", round_subject,
        *usage_args,
    )

    # --- Post ---
    if not dry_run:
        _run_helper(
            "helpers.state_manager", "write-pending-comment",
            "--issue", str(issue), "--repo", repo,
            "--body", str(tagged_output),
        )
        _run_helper(
            "helpers.gh_ops", "post-issue-comment",
            "--issue", str(issue), "--file", str(tagged_output), "--repo", repo,
        )
        _run_helper(
            "helpers.state_manager", "clear-pending-comment",
            "--issue", str(issue), "--repo", repo,
        )
    else:
        print(f"[dry-run] would post {agent} review for {repo}#{issue}")

    return {
        "reviewer_name": agent_cap,
        "state": parsed_state,
        "blocking_items": [{"text": t} for t in reported_blocking],
        "new_items": new_items_serialized,
        "usage": reviewer_usage,
    }


# ---------------------------------------------------------------------------
# Per-reviewer run
# ---------------------------------------------------------------------------

# Signatures of an external agent/CLI *failure* — a turn that produced no usable
# review (e.g. Gemini reaching for a missing tool and returning an empty/malformed
# stream, #322) — as opposed to a malformed-but-content-bearing review the user can
# repair. Kept narrow and CLI-specific so a genuine review (even a plain-text one
# with findings) is never misclassified; matched only after validation has failed.
_AGENT_UNAVAILABLE_SIGNATURES = (
    "invalid stream",
    "empty response or malformed tool call",
    "error executing tool",
    "not found. did you mean one of",
)


def _is_agent_unavailable_output(text: str) -> str | None:
    """Return a short reason if ``text`` is an agent/CLI failure (no usable review),
    else ``None``. Conservative: only truly-empty output or a known tooling-failure
    signature qualifies — a content-bearing response stays on the repair path."""
    stripped = text.strip()
    if not stripped:
        return "empty response"
    lowered = stripped.lower()
    for signature in _AGENT_UNAVAILABLE_SIGNATURES:
        if signature in lowered:
            return signature
    return None


def _round_overall_state(
    *,
    pending_reviewers: list[str],
    any_reviewer_blocked: bool,
    unavailable_reviewers: list[str],
) -> str:
    """Top-level round state by precedence (#314, #322): pending (a host review is
    outstanding) > blocking > incomplete (a configured reviewer was unavailable) >
    approved. Never reports approved/blocking while a host review is pending, and
    never approved while a reviewer was unavailable."""
    if pending_reviewers:
        return "pending"
    if any_reviewer_blocked:
        return "blocking"
    if unavailable_reviewers:
        return "incomplete"
    return "approved"


def _run_reviewer(
    *,
    agent: str,
    prompt_text: str,
    context: dict,
    round_subject: str,
    next_prior_items_raw: list[dict],
    new_round_number: int,
    issue: int,
    repo: str,
    flow: str,
    role: str,
    state_key: str,
    workdir: str,
    dry_run: bool,
    tmpdir: Path,
    item_id_offset: int = 0,
) -> dict:
    """Run one reviewer turn; return {reviewer_name, state, blocking_items, new_items}."""
    agent_cap = agent_display_name(agent)  # type: ignore[arg-type]
    validate_kind = "pr_review" if flow == "pr" else "plan_review"

    prompt_file      = tmpdir / f"{agent}-prompt.md"
    raw_output       = tmpdir / f"{agent}-review-raw.md"
    context_file     = tmpdir / f"{agent}-context.json"
    prior_items_file = tmpdir / f"{agent}-prior-items.json"

    _write_text(prompt_file, prompt_text)
    _write_json(context_file, context)
    _write_json(prior_items_file, next_prior_items_raw)

    # --- Run agent ---
    # check=False: a non-zero run_external exit (agent/CLI failure, which now writes
    # its failure text to --output, #322) must not abort the whole round here — let
    # the validation + agent-unavailable classification below decide whether to skip
    # this reviewer and continue.
    usage_file = tmpdir / f"{agent}-usage.json"
    _run_helper(
        "helpers.run_external",
        "--agent", agent,
        "--prompt-file", str(prompt_file),
        "--output", str(raw_output),
        "--workdir", workdir,
        "--repo", repo,
        "--flow", flow,
        "--usage-output", str(usage_file),
        *(["--dry-run"] if dry_run else []),
        check=False,
    )
    if not raw_output.exists():
        # run_external failed before writing anything; synthesize an empty body so
        # the classifier marks the reviewer unavailable instead of crashing.
        _write_text(raw_output, "")

    # Save raw response to stable repair dir BEFORE normalization
    repair_dir = _save_raw_to_repair_dir(
        agent=agent, agent_cap=agent_cap, flow=flow,
        issue=issue, repo=repo, new_round_number=new_round_number,
        round_subject=round_subject, item_id_offset=item_id_offset,
        validate_kind=validate_kind, dry_run=dry_run,
        raw_output=raw_output, context_file=context_file,
        prior_items_file=prior_items_file,
    )

    try:
        return _complete_reviewer_turn(
            agent=agent, agent_cap=agent_cap, flow=flow,
            validate_kind=validate_kind, issue=issue, repo=repo,
            new_round_number=new_round_number, round_subject=round_subject,
            next_prior_items_raw=next_prior_items_raw,
            item_id_offset=item_id_offset, dry_run=dry_run,
            raw_output=raw_output, context_file=context_file,
            work_dir=tmpdir,
        )
    except _ValidationError as exc:
        # Distinguish an agent/CLI failure (no usable review -> skip this reviewer
        # for the round, re-attempt on rerun) from a fixable malformed review (keep
        # the retry-validate handoff) (#322).
        raw_text = raw_output.read_text(encoding="utf-8") if raw_output.exists() else ""
        reason = _is_agent_unavailable_output(raw_text)
        if reason is not None:
            print(
                f"skill_runner: {agent_cap} unavailable: agent/CLI failure ({reason}) — "
                f"not a fixable review; it will be re-attempted on rerun. "
                f"Raw response saved to: {repair_dir}/raw.md",
                file=sys.stderr,
            )
            reviewer_usage = None
            if usage_file.exists():
                try:
                    reviewer_usage = json.loads(usage_file.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    reviewer_usage = None
            return {
                "reviewer_name": agent_cap,
                "state": "unavailable",
                "reason": reason,
                "blocking_items": [],
                "new_items": [],
                "usage": reviewer_usage,
            }
        print(str(exc), file=sys.stderr)
        print(f"skill_runner: raw response saved to: {repair_dir}/raw.md", file=sys.stderr)
        dry_run_flag = " --dry-run" if dry_run else ""
        print(
            f"skill_runner: fix raw.md, then run: "
            f"python -m helpers.skill_runner retry-validate --repair-dir {repair_dir}{dry_run_flag}",
            file=sys.stderr,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# run-task-round  (free-form task -> scratch issue -> plan round)
# ---------------------------------------------------------------------------

def _resolve_task_text(args: argparse.Namespace) -> str:
    task = getattr(args, "task", None)
    task_file = getattr(args, "task_file", None)
    if task is not None and task_file is not None:
        print("skill_runner: pass either --task or --task-file, not both", file=sys.stderr)
        sys.exit(1)
    if task_file is not None:
        text = sys.stdin.read() if task_file == "-" else Path(task_file).read_text(encoding="utf-8")
    elif task is not None:
        text = task
    else:
        print("skill_runner: provide --task or --task-file", file=sys.stderr)
        sys.exit(1)
    if not text.strip():
        print("skill_runner: task description is empty", file=sys.stderr)
        sys.exit(1)
    return text


def _task_key(task_text: str) -> str:
    return hashlib.sha256(task_text.strip().encode("utf-8")).hexdigest()


def _task_index_path(repo: str) -> Path:
    slug = repo.replace("/", "-").replace(":", "-")
    state_home = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    return state_home / "coding-review-agent-loop" / "skill-sessions" / slug / "task-index.json"


def _load_task_index(repo: str) -> dict:
    path = _task_index_path(repo)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _issue_is_open(repo: str, issue: int, gh_cmd: str = "gh") -> bool:
    result = subprocess.run(
        [gh_cmd, "issue", "view", str(issue), "--repo", repo, "--json", "state"],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        return False
    try:
        return json.loads(result.stdout).get("state", "").upper() == "OPEN"
    except json.JSONDecodeError:
        return False


def _lookup_task_issue(repo: str, task_key: str) -> int | None:
    number = _load_task_index(repo).get(task_key)
    if isinstance(number, int) and _issue_is_open(repo, number):
        return number
    return None


def _record_task_issue(repo: str, task_key: str, number: int) -> None:
    path = _task_index_path(repo)
    path.parent.mkdir(parents=True, exist_ok=True)
    index = _load_task_index(repo)
    index[task_key] = number
    path.write_text(json.dumps(index, indent=2), encoding="utf-8")


def _title_from_task(task_text: str) -> str:
    for line in task_text.splitlines():
        stripped = line.lstrip("#").strip()
        if stripped:
            return stripped[:80]
    return "Agent task"


def _create_task_issue(repo: str, task_text: str, task_key: str, gh_cmd: str = "gh") -> int:
    body = f"{task_text.rstrip()}\n\n<!-- AGENT_TASK_KEY: {task_key} -->\n"
    result = subprocess.run(
        [gh_cmd, "issue", "create", "--repo", repo,
         "--title", _title_from_task(task_text), "--body", body],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        print(f"skill_runner: gh issue create failed: {result.stderr.strip()}", file=sys.stderr)
        sys.exit(1)
    match = re.search(r"/issues/(\d+)", result.stdout)
    if not match:
        print(f"skill_runner: could not parse issue number from: {result.stdout.strip()}", file=sys.stderr)
        sys.exit(1)
    number = int(match.group(1))
    _record_task_issue(repo, task_key, number)
    return number


def cmd_run_task_round(args: argparse.Namespace) -> None:
    repo: str = args.repo
    dry_run: bool = args.dry_run
    if getattr(args, "coder", "claude") != "claude":
        print(
            "skill_runner: run-task-round supports only --coder claude (host coder) for now; "
            "external coders are available via run-plan-round (#307).",
            file=sys.stderr,
        )
        sys.exit(1)
    task_text = _resolve_task_text(args)
    key = _task_key(task_text)
    issue = _lookup_task_issue(repo, key)

    if dry_run:
        # Dry-run is a pure preview of the task issue: never create, never delegate
        # to the plan round (which would fetch issue state and touch session state).
        if issue is None:
            print(json.dumps({
                "would_create_issue": True,
                "task_key": key,
                "title": _title_from_task(task_text),
            }, indent=2))
        else:
            print(json.dumps({"would_reuse_issue": issue, "task_key": key}, indent=2))
        return

    if issue is None:
        issue = _create_task_issue(repo, task_text, key)
        print(f"created scratch issue #{issue} for task {key[:12]}")
    else:
        print(f"reusing issue #{issue} for task {key[:12]}")

    args.issue = issue
    cmd_run_plan_round(args)


# ---------------------------------------------------------------------------
# External-coder plan turn (reversed roles: Codex/Gemini writes the plan) (#307)
# ---------------------------------------------------------------------------

def _external_coder_phase(resume: dict, reviewers: list[str]) -> dict:
    """Decide what an external-coder plan round should do from the resume state.

    With no host ``--plan-file`` to discriminate rounds, the phase is read entirely
    from the posted ledger. Returns ``{"phase": ..., ...}``:

      ``coder-round-1``    — no coder record yet; run the coder for round 1.
      ``resume-reviewers`` — plan posted, some reviewers incomplete; run them.
      ``coder-round-next`` — round complete with blocking/same-plan findings; run
                             the coder for round N+1 (carries ``next_prior_items_raw``).
      ``approved``         — round complete and all reviewers approved.
    """
    if not resume.get("current_plan_subject"):
        return {"phase": "coder-round-1"}

    completed = {str(n) for n in resume.get("completed_reviewer_names", [])}
    configured = {agent_display_name(r) for r in reviewers}  # type: ignore[arg-type]
    if not configured.issubset(completed):
        return {"phase": "resume-reviewers"}

    # All configured reviewers completed this round; decide approve vs. revise.
    completed_data = list(resume.get("completed_reviewer_data", []))
    any_blocking = any(rec.get("state") == "blocking" for rec in completed_data)
    next_prior = _compute_next_prior_items(
        list(resume.get("prior_items", [])),
        completed_data,
        same_status="same-plan",
        retain_future=False,
    )
    if any_blocking or next_prior:
        return {"phase": "coder-round-next", "next_prior_items_raw": next_prior}
    return {"phase": "approved"}


def _aggregate_blocking_feedback(resume: dict) -> str:
    """Assemble prose reviewer feedback for the coder's revision prompt.

    The skill ledger keeps each completed reviewer's must-fix items (not the full
    review prose), so synthesize the feedback from the blocking / same-plan items.
    """
    parts: list[str] = []
    for rec in resume.get("completed_reviewer_data", []):
        name = str(rec.get("reviewer_name", "Reviewer"))
        texts = [
            str(item.get("text", ""))
            for item in rec.get("new_items", [])
            if item.get("status") in ("blocking", "same-plan")
        ]
        if texts:
            body = "\n".join(f"- {t}" for t in texts)
            parts.append(f"{name} plan review:\n{body}")
    return "\n\n".join(parts)


def _save_coder_raw_to_repair_dir(
    *,
    coder: str,
    coder_cap: str,
    issue: int,
    repo: str,
    new_round_number: int,
    kind: str,
    next_prior_items_raw: list[dict],
    dry_run: bool,
    raw_output: Path,
    usage_file: Path,
) -> Path:
    """Copy the coder's raw response + context to a stable repair dir.

    The manifest is tagged ``role: coder`` so ``retry-validate`` routes to the
    coder-completion path rather than the reviewer one.
    """
    repair_dir = _REPAIR_BASE / f"{issue}-r{new_round_number}-{coder}-coder"
    repair_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(raw_output, repair_dir / "raw.md")
    _write_json(repair_dir / "prior_items.json", next_prior_items_raw)
    if usage_file.exists():
        shutil.copy2(usage_file, repair_dir / "coder-usage.json")
    manifest = {
        "role": "coder",
        "agent": coder, "agent_cap": coder_cap, "flow": "plan",
        "issue": issue, "repo": repo,
        "new_round_number": new_round_number, "kind": kind,
        "dry_run": dry_run,
    }
    (repair_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return repair_dir


def _complete_coder_turn(
    *,
    coder: str,
    coder_cap: str,
    issue: int,
    repo: str,
    new_round_number: int,
    next_prior_items_raw: list[dict],
    kind: str,
    dry_run: bool,
    raw_output: Path,
    work_dir: Path,
    usage_file: Path,
) -> dict:
    """Validate, render, canonicalize, attach (role coder), and post a coder plan.

    ``kind`` is ``plan_state`` (round 1) or ``plan_revision`` (round N+1). For a
    revision, the structured response is rendered to a public comment, the
    canonical plan is extracted for reviewers, and the raw JSON is persisted in
    AGENT_LOOP_META. Raises ``_ValidationError`` on validation failure. Returns
    ``{plan_text, subject, usage}`` (``plan_text`` is the canonical plan reviewers
    will review).
    """
    result = _run_helper_capture(
        "helpers.validate_response", "--file", str(raw_output), "--kind", kind,
    )
    if result.returncode != 0:
        raise _ValidationError(
            f"skill_runner: {coder_cap} {kind} validation failed: {result.stderr.strip()}\n"
            f"skill_runner: raw response at: {raw_output}"
        )

    raw_text = raw_output.read_text(encoding="utf-8")
    canonical_file = work_dir / "coder-canonical.md"
    raw_structured_args: list[str] = []

    if kind == "plan_state":
        canonical_text = raw_text
        public_file = raw_output
    else:
        from coding_review_agent_loop.agents.registry import agent_signature
        from coding_review_agent_loop.comment_rendering import (
            _render_public_plan_revision_comment,
            render_canonical_plan_revision,
        )
        from coding_review_agent_loop.protocol import validate_structured_plan_revision

        parsed = validate_structured_plan_revision(raw_text)
        if parsed is None:
            raise _ValidationError(
                f"skill_runner: {coder_cap} plan_revision did not parse\n"
                f"skill_runner: raw response at: {raw_output}"
            )
        prior_items = [_deserialize_unresolved_item(i) for i in next_prior_items_raw]
        # The coder's actual model is in run_external's usage sidecar (#332); stamp it
        # so a skill-mode plan revision is signed e.g. "-- Google Antigravity: Gemini
        # 3.1 Pro (High)" rather than the bare provider name.
        coder_model = _model_used_from_usage(usage_file)
        canonical_text = render_canonical_plan_revision(parsed, prior_items)
        public_body = _render_public_plan_revision_comment(
            parsed,
            prior_items=prior_items,
            raw_text=raw_text,
            signature=agent_signature(coder, None, coder_model or None),  # type: ignore[arg-type]
        )
        public_file = work_dir / "coder-public.md"
        _write_text(public_file, public_body)
        raw_structured_file = work_dir / "coder-raw-structured.json"
        _write_text(raw_structured_file, raw_text)
        raw_structured_args = ["--raw-structured-coder-response-file", str(raw_structured_file)]

    _write_text(canonical_file, canonical_text)
    subject = _plan_subject(canonical_text)

    prior_file = work_dir / "coder-prior-items.json"
    _write_json(prior_file, next_prior_items_raw)

    coder_usage: dict | None = None
    usage_args: list[str] = []
    if usage_file.exists():
        usage_args = ["--usage-file", str(usage_file)]
        try:
            coder_usage = json.loads(usage_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            coder_usage = None

    tagged = work_dir / "coder-tagged.md"
    _run_helper(
        "helpers.state_manager", "attach-metadata",
        "--body-file", str(public_file),
        "--output", str(tagged),
        "--flow", "plan",
        "--role", "coder",
        "--agent", coder_cap,
        "--round-number", str(new_round_number),
        "--state", "approved",
        "--subject", subject,
        "--canonical-plan-file", str(canonical_file),
        "--prior-items-file", str(prior_file),
        *raw_structured_args,
        *usage_args,
    )

    if not dry_run:
        _run_helper(
            "helpers.state_manager", "write-pending-comment",
            "--issue", str(issue), "--repo", repo, "--body", str(tagged),
        )
        _run_helper(
            "helpers.gh_ops", "post-issue-comment",
            "--issue", str(issue), "--file", str(tagged), "--repo", repo,
        )
        _run_helper(
            "helpers.state_manager", "clear-pending-comment",
            "--issue", str(issue), "--repo", repo,
        )
    else:
        print(f"[dry-run] would post {coder_cap} plan for {repo}#{issue} (round {new_round_number})")

    return {"plan_text": canonical_text, "subject": subject, "usage": coder_usage}


def _run_external_coder_phase(
    args: argparse.Namespace,
    coder: str,
    resume: dict,
    reviewers: list[str],
    memory: AgentMemoryContext | None,
    dry_run: bool,
) -> dict:
    """Drive the external-coder side of a plan round and return shared round vars.

    Returns ``{"done": True, "result": {...}}`` when the round is already approved
    (nothing left to do), otherwise ``{"done": False, ...}`` with the variables the
    shared reviewer loop needs (plan text/subject, round number, prior items,
    new-round flag, locally-completed reviewers, and the coder usage sidecar).
    """
    issue: int = args.issue
    repo: str = args.repo
    phase_info = _external_coder_phase(resume, reviewers)
    phase = phase_info["phase"]

    if phase == "approved":
        completed_data = list(resume.get("completed_reviewer_data", []))
        result = {
            "state": "approved",
            "round_number": int(resume.get("round_number", 1)),
            "blocking_items": [],
            "approved_reviewers": [str(r.get("reviewer_name", "")) for r in completed_data],
        }
        usage = _aggregate_reviewer_usage(completed_data)
        if usage is not None:
            result["usage"] = usage
        return {"done": True, "result": result}

    if phase == "resume-reviewers":
        return {
            "done": False,
            "plan_text": str(resume.get("current_plan", "")),
            "plan_subject": str(resume.get("current_plan_subject")),
            "new_round_number": int(resume.get("round_number", 1)),
            "next_prior_items_raw": list(resume.get("prior_items", [])),
            "is_new_round": False,
            "local_completed": {str(n) for n in resume.get("completed_reviewer_names", [])},
            "coder_usage": None,
        }

    # coder-round-1 or coder-round-next: run the external coder turn.
    from helpers.prompt_builders import (
        build_plan_prompt_for_skill,
        build_plan_revision_prompt_for_skill,
    )
    from coding_review_agent_loop.protocol import is_clarification_request

    new_round_number = int(resume.get("completed_round_number", 0)) + 1
    workdir = _workdir_for_agent(coder, args)
    issue_dict = _fetch_issue_json(repo, issue)
    coder_cap = agent_display_name(coder)  # type: ignore[arg-type]

    if phase == "coder-round-1":
        next_prior_items_raw: list[dict] = []
        prompt_text = build_plan_prompt_for_skill(
            issue_dict, repo=repo, coder=coder,  # type: ignore[arg-type]
            reviewers=reviewers, workdir=workdir, memory=memory,  # type: ignore[arg-type]
        )
        kind = "plan_state"
    else:  # coder-round-next
        next_prior_items_raw = list(phase_info["next_prior_items_raw"])
        prompt_text = build_plan_revision_prompt_for_skill(
            issue_dict, repo=repo, coder=coder,  # type: ignore[arg-type]
            reviewers=reviewers, workdir=workdir,  # type: ignore[arg-type]
            round_number=new_round_number,
            previous_plan=str(resume.get("current_plan", "")),
            reviewer_feedback=_aggregate_blocking_feedback(resume),
            prior_items_raw=next_prior_items_raw,
            memory=memory,
        )
        kind = "plan_revision"

    with tempfile.TemporaryDirectory() as tmpstr:
        tmpdir = Path(tmpstr)
        prompt_file = tmpdir / "coder-prompt.md"
        raw_output = tmpdir / "coder-plan-raw.md"
        usage_file = tmpdir / "coder-usage.json"
        _write_text(prompt_file, prompt_text)

        _run_helper(
            "helpers.run_external",
            "--agent", coder,
            "--role", "coder",
            "--prompt-file", str(prompt_file),
            "--output", str(raw_output),
            "--workdir", workdir,
            "--repo", repo,
            "--flow", "plan",
            "--usage-output", str(usage_file),
            *(["--dry-run"] if dry_run else []),
        )

        raw_text = raw_output.read_text(encoding="utf-8")
        if is_clarification_request(raw_text):
            print(
                f"skill_runner: {coder_cap} requested clarification during planning; "
                f"human intervention required.\n\n{raw_text}",
                file=sys.stderr,
            )
            sys.exit(2)

        # Save raw to a stable repair dir before validation so a malformed plan is
        # recoverable via `retry-validate` (manifest role: coder).
        repair_dir = _save_coder_raw_to_repair_dir(
            coder=coder, coder_cap=coder_cap, issue=issue, repo=repo,
            new_round_number=new_round_number, kind=kind,
            next_prior_items_raw=next_prior_items_raw, dry_run=dry_run,
            raw_output=raw_output, usage_file=usage_file,
        )

        try:
            completed = _complete_coder_turn(
                coder=coder, coder_cap=coder_cap, issue=issue, repo=repo,
                new_round_number=new_round_number,
                next_prior_items_raw=next_prior_items_raw, kind=kind,
                dry_run=dry_run, raw_output=raw_output, work_dir=tmpdir,
                usage_file=usage_file,
            )
        except _ValidationError as exc:
            print(str(exc), file=sys.stderr)
            dry_run_flag = " --dry-run" if dry_run else ""
            print(
                f"skill_runner: fix raw.md, then run: "
                f"python -m helpers.skill_runner retry-validate --repair-dir {repair_dir}{dry_run_flag}",
                file=sys.stderr,
            )
            sys.exit(1)

    return {
        "done": False,
        "plan_text": completed["plan_text"],
        "plan_subject": completed["subject"],
        "new_round_number": new_round_number,
        "next_prior_items_raw": next_prior_items_raw,
        "is_new_round": True,
        "local_completed": set(),
        "coder_usage": completed.get("usage"),
    }


# ---------------------------------------------------------------------------
# Host-as-reviewer handoff (reversed roles: Claude reviews the posted plan/PR)
# (plan flow #307; PR flow #314)
# ---------------------------------------------------------------------------

def _write_host_review_request(
    *,
    flow: str,
    validate_kind: str,
    issue: int,
    repo: str,
    new_round_number: int,
    round_subject: str,
    review_material: str,
    material_filename: str,
    next_prior_items_raw: list[dict],
    current_round_items: list[dict],
    item_id_offset: int,
    dry_run: bool,
) -> Path:
    """Write a review-request dir for the host (Claude) reviewer turn.

    Flow-agnostic: ``review_material`` (the plan text or the PR diff) is written to
    ``{dir}/{material_filename}`` for the host to read, alongside the review
    context, prior items, and a manifest mirroring the reviewer repair manifest so
    ``complete-host-review`` can drive ``_complete_reviewer_turn`` once the host
    writes ``host-review.md``.
    """
    request_dir = _REPAIR_BASE / f"{issue}-r{new_round_number}-claude"
    request_dir.mkdir(parents=True, exist_ok=True)
    _write_text(request_dir / material_filename, review_material)
    _write_json(request_dir / "prior_items.json", next_prior_items_raw)
    _write_json(request_dir / "context.json", {
        "reviewer": "Claude",
        "prior_items": next_prior_items_raw,
        "current_round_items": current_round_items,
    })
    manifest = {
        "role": "reviewer",
        "agent": "claude", "agent_cap": "Claude", "flow": flow,
        "issue": issue, "repo": repo,
        "new_round_number": new_round_number, "round_subject": round_subject,
        "item_id_offset": item_id_offset, "validate_kind": validate_kind,
        "material_filename": material_filename,
        "dry_run": dry_run,
    }
    (request_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return request_dir


# ---------------------------------------------------------------------------
# run-plan-round
# ---------------------------------------------------------------------------

def _run_host_coder_phase(
    args: argparse.Namespace,
    plan_file: Path,
    resume: dict,
    dry_run: bool,
) -> dict:
    """Host-coder (Claude) plan turn: detect new-round vs resume and post the plan.

    This is the default behavior — the host supplies ``--plan-file`` — extracted
    verbatim so ``cmd_run_plan_round`` can dispatch symmetrically with the external
    coder. Returns the shared round vars the reviewer loop consumes.
    """
    issue: int = args.issue
    repo: str = args.repo

    # New-round vs resume detection
    plan_subject = _plan_subject_of_file(plan_file)
    current_plan_subject = resume.get("current_plan_subject")
    is_new_round = current_plan_subject != plan_subject

    if is_new_round:
        completed_round_number = int(resume.get("completed_round_number", 0))
        new_round_number = completed_round_number + 1
        next_prior_items_raw = _compute_next_prior_items(
            list(resume.get("prior_items", [])),
            list(resume.get("completed_reviewer_data", [])),
            same_status="same-plan",
            retain_future=False,
        )
        local_completed: set[str] = set()
    else:
        # Resume: plan already posted; only run remaining reviewers
        new_round_number = int(resume.get("round_number", 1))
        next_prior_items_raw = list(resume.get("prior_items", []))
        local_completed = {str(n) for n in resume.get("completed_reviewer_names", [])}

    # Validate and post plan (new rounds only; resume skips this)
    if is_new_round:
        result = _run_helper_capture(
            "helpers.validate_response",
            "--file", str(plan_file),
            "--kind", "plan_state",
        )
        if result.returncode != 0:
            print(f"skill_runner: plan validation failed: {result.stderr.strip()}", file=sys.stderr)
            sys.exit(1)
        print(f"validation passed: plan_state")

        if not dry_run:
            with tempfile.TemporaryDirectory() as tmpstr:
                tmpdir = Path(tmpstr)
                prior_items_file = tmpdir / "prior_items.json"
                tagged_plan = tmpdir / "plan_tagged.md"
                _write_json(prior_items_file, next_prior_items_raw)
                _run_helper(
                    "helpers.state_manager", "attach-metadata",
                    "--body-file", str(plan_file),
                    "--output", str(tagged_plan),
                    "--flow", "plan",
                    "--role", "coder",
                    "--agent", "Claude",
                    "--round-number", str(new_round_number),
                    "--state", "approved",
                    "--subject-plan-file", str(plan_file),
                    "--canonical-plan-file", str(plan_file),
                    "--prior-items-file", str(prior_items_file),
                )
                _run_helper(
                    "helpers.state_manager", "write-pending-comment",
                    "--issue", str(issue), "--repo", repo,
                    "--body", str(tagged_plan),
                )
                _run_helper(
                    "helpers.gh_ops", "post-issue-comment",
                    "--issue", str(issue), "--file", str(tagged_plan), "--repo", repo,
                )
                _run_helper(
                    "helpers.state_manager", "clear-pending-comment",
                    "--issue", str(issue), "--repo", repo,
                )
        else:
            print(f"[dry-run] would post plan for {repo}#{issue} (round {new_round_number})")

    return {
        "plan_text": plan_file.read_text(encoding="utf-8"),
        "plan_subject": plan_subject,
        "new_round_number": new_round_number,
        "next_prior_items_raw": next_prior_items_raw,
        "is_new_round": is_new_round,
        "local_completed": local_completed,
        "coder_usage": None,
    }


def cmd_run_plan_round(args: argparse.Namespace) -> None:
    issue: int = args.issue
    repo: str = args.repo
    reviewers: list[str] = args.reviewers
    dry_run: bool = args.dry_run
    coder: str = getattr(args, "coder", "claude")

    plan_file: Path | None = None
    if coder == "claude":
        if not args.plan_file:
            print("skill_runner: run-plan-round --coder claude requires --plan-file", file=sys.stderr)
            sys.exit(1)
        plan_file = Path(args.plan_file)
        if not plan_file.exists():
            print(f"skill_runner: plan file not found: {plan_file}", file=sys.stderr)
            sys.exit(1)

    # Step 1 — build_resume
    resume = _build_resume(issue, repo, reviewers, flow="plan")

    # Step 2 — pending-comment reconciliation
    _reconcile_pending_comment(resume, issue, repo, dry_run)

    # Repo-scoped agent memory for coder/reviewer orientation (#306), prepared once
    # and shared by the coder turn (external mode) and the reviewer turns.
    memory = None
    if getattr(args, "agent_memory", True):
        memory = _prepare_skill_memory(
            repo, reviewers, _workdir_for_agent(reviewers[0], args),
            refresh=getattr(args, "refresh_agent_memory", False), dry_run=dry_run,
        )

    # Step 3+4 — produce and post the round's plan (host coder or external coder)
    if coder == "claude":
        assert plan_file is not None
        phase = _run_host_coder_phase(args, plan_file, resume, dry_run)
    else:
        phase = _run_external_coder_phase(args, coder, resume, reviewers, memory, dry_run)
        if phase["done"]:
            print(json.dumps(phase["result"], indent=2))
            return

    plan_text: str = phase["plan_text"]
    plan_subject: str = phase["plan_subject"]
    new_round_number: int = phase["new_round_number"]
    next_prior_items_raw: list[dict] = phase["next_prior_items_raw"]
    is_new_round: bool = phase["is_new_round"]
    local_completed: set[str] = phase["local_completed"]
    coder_usage: dict | None = phase["coder_usage"]

    # Step 5 — reconstruct round state from already-completed reviewers (RESUME path only).
    # In a new round, completed_reviewer_data belongs to the previous round and must not
    # contaminate the current round's result.
    current_round_items: list[dict] = []
    round_reviewer_records: list[dict] = []
    round_blocking_items: list[dict] = []
    round_approved_reviewers: list[str] = []
    unavailable_reviewers: list[str] = []
    any_reviewer_blocked = False
    if not is_new_round:
        for record in resume.get("completed_reviewer_data", []):
            current_round_items.extend(record.get("new_items", []))
            round_reviewer_records.append(record)
            if record.get("state") == "blocking":
                any_reviewer_blocked = True
                round_blocking_items.extend(
                    {"text": item.get("text", "")}
                    for item in record.get("new_items", [])
                    if item.get("status") in ("blocking", "same-plan")
                )
            else:
                round_approved_reviewers.append(str(record.get("reviewer_name", "")))

    # Step 6 — run pending reviewers.
    # External reviewers (Codex/Gemini) always run first, regardless of --reviewers
    # order; a configured host reviewer ("claude") is authored by the host after it
    # reads the posted plan, so it becomes a pending handoff after the loop (#307).
    pending_reviewers: list[str] = []
    external_reviewers = [r for r in reviewers if r != "claude"]
    with tempfile.TemporaryDirectory() as tmpstr:
        tmpdir = Path(tmpstr)
        for reviewer in external_reviewers:
            reviewer_cap = agent_display_name(reviewer)  # type: ignore[arg-type]
            if reviewer_cap in local_completed or reviewer in local_completed:
                print(f"[skip] {reviewer_cap} already completed this round")
                continue

            # Build prompt
            issue_dict = _fetch_issue_json(repo, issue)
            workdir = _workdir_for_agent(reviewer, args)
            try:
                from helpers.prompt_builders import build_plan_review_prompt_for_skill
                prompt_text = build_plan_review_prompt_for_skill(
                    issue_dict,
                    plan_text,
                    next_prior_items_raw,
                    new_round_number,
                    reviewer,  # type: ignore[arg-type]
                    repo=repo,
                    all_reviewers=[r for r in reviewers],  # type: ignore[misc]
                    coder=coder,  # type: ignore[arg-type]
                    workdir=workdir,
                    memory=memory,
                )
            except Exception as exc:  # noqa: BLE001
                prompt_text = (
                    f"Review the following implementation plan for issue #{issue} in {repo}.\n\n"
                    f"Round: {new_round_number}\n\n"
                    f"{plan_text}\n"
                )

            context: dict = {
                "reviewer": reviewer_cap,
                "prior_items": next_prior_items_raw,
                "current_round_items": current_round_items,
            }

            # item_id_offset: highest numeric suffix across prior + current-round items
            item_id_offset = _max_item_number([next_prior_items_raw, current_round_items])
            record = _run_reviewer(
                agent=reviewer,
                prompt_text=prompt_text,
                context=context,
                round_subject=plan_subject,
                next_prior_items_raw=next_prior_items_raw,
                new_round_number=new_round_number,
                issue=issue,
                repo=repo,
                flow="plan",
                role="reviewer",
                state_key="plan_review",
                workdir=workdir,
                dry_run=dry_run,
                tmpdir=tmpdir,
                item_id_offset=item_id_offset,
            )
            if record["state"] == "unavailable":
                # Agent/CLI failure: skip this reviewer for the round (re-attempted
                # on rerun); still fold its usage into the aggregate (#322).
                unavailable_reviewers.append(record["reviewer_name"])
                round_reviewer_records.append(record)
                continue
            current_round_items.extend(record["new_items"])
            round_reviewer_records.append(record)
            if record["state"] == "blocking":
                any_reviewer_blocked = True
                round_blocking_items.extend(record["blocking_items"])
            else:
                round_approved_reviewers.append(record["reviewer_name"])

    # Step 6b — host-as-reviewer handoff. The host (Claude) reviews the posted plan
    # itself, so write a review-request dir and mark it pending; the host writes its
    # plan_review JSON there and completes it via `complete-host-review` (#307).
    if "claude" in reviewers and "Claude" not in local_completed:
        item_id_offset = _max_item_number([next_prior_items_raw, current_round_items])
        request_dir = _write_host_review_request(
            flow="plan", validate_kind="plan_review",
            issue=issue, repo=repo, new_round_number=new_round_number,
            round_subject=plan_subject, review_material=plan_text,
            material_filename="plan.md",
            next_prior_items_raw=next_prior_items_raw,
            current_round_items=current_round_items,
            item_id_offset=item_id_offset, dry_run=dry_run,
        )
        pending_reviewers.append("Claude")
        dry_run_flag = " --dry-run" if dry_run else ""
        print(
            f"skill_runner: host review pending — read the plan in {request_dir}/plan.md, "
            f"write your plan_review JSON to {request_dir}/host-review.md, then run: "
            f"python -m helpers.skill_runner complete-host-review --dir {request_dir}{dry_run_flag}",
            file=sys.stderr,
        )

    # Step 7 — write session
    if not dry_run:
        _run_helper(
            "helpers.state_manager", "write-session",
            "--issue", str(issue), "--repo", repo,
            "--fields", json.dumps({
                "last_completed_step": "post_review",
                "round_number": new_round_number,
            }),
        )

    # Step 8 — print result.
    # A configured-but-incomplete host reviewer makes the round "pending": never
    # report approved/blocking while a reviewer is still outstanding (#307).
    # Otherwise use any_reviewer_blocked to catch reviewers that blocked via
    # same-plan followups only.
    # Precedence: pending (host review outstanding) > blocking > incomplete
    # (a configured reviewer was unavailable) > approved. Both reviewer lists are
    # always surfaced when non-empty so neither the host handoff nor a failed
    # reviewer is ever hidden, and a round is never falsely reported approved (#322).
    overall_state = _round_overall_state(
        pending_reviewers=pending_reviewers,
        any_reviewer_blocked=any_reviewer_blocked,
        unavailable_reviewers=unavailable_reviewers,
    )
    result_json = {
        "state": overall_state,
        "round_number": new_round_number,
        "blocking_items": round_blocking_items,
        "approved_reviewers": round_approved_reviewers,
    }
    if pending_reviewers:
        result_json["pending_reviewers"] = pending_reviewers
    if unavailable_reviewers:
        result_json["unavailable_reviewers"] = unavailable_reviewers
    # Include the external coder's usage (if any) alongside the reviewers'.
    usage_records = list(round_reviewer_records)
    if coder_usage is not None:
        usage_records.append({"usage": coder_usage})
    usage = _aggregate_reviewer_usage(usage_records)
    if usage is not None:
        result_json["usage"] = usage
    print(json.dumps(result_json, indent=2))


# ---------------------------------------------------------------------------
# run-pr-round
# ---------------------------------------------------------------------------

def cmd_run_pr_round(args: argparse.Namespace) -> None:
    pr: int = args.pr
    repo: str = args.repo
    reviewers: list[str] = args.reviewers
    dry_run: bool = args.dry_run
    head_sha: str | None = getattr(args, "head_sha", None)

    # Step 1 — build_resume; fetch head SHA if not provided
    if not head_sha:
        pr_info = _fetch_pr_json(repo, pr)
        head_sha = pr_info.get("headRefOid") or ""

    resume = _build_resume(pr, repo, reviewers, flow="pr", head_sha=head_sha, pr=pr)

    # Step 2 — pending-comment reconciliation
    _reconcile_pending_comment(resume, pr, repo, dry_run)

    # Step 3 — detect new vs resume (for PR, head SHA mismatch = new round)
    current_subject = resume.get("current_plan_subject")
    is_new_round = current_subject != head_sha

    if is_new_round:
        completed_round_number = int(resume.get("completed_round_number", 0))
        new_round_number = completed_round_number + 1
        next_prior_items_raw = _compute_next_prior_items(
            list(resume.get("prior_items", [])),
            list(resume.get("completed_reviewer_data", [])),
            same_status="same-pr",
            retain_future=True,
        )
        local_completed: set[str] = set()
    else:
        new_round_number = int(resume.get("round_number", 1))
        next_prior_items_raw = list(resume.get("prior_items", []))
        local_completed = {str(n) for n in resume.get("completed_reviewer_names", [])}

    # Step 5 — reconstruct round state from already-completed reviewers (RESUME path only).
    # In a new round, completed_reviewer_data belongs to the previous round and must not
    # contaminate the current round's result.
    current_round_items: list[dict] = []
    round_reviewer_records: list[dict] = []
    round_blocking_items: list[dict] = []
    round_approved_reviewers: list[str] = []
    unavailable_reviewers: list[str] = []
    any_reviewer_blocked = False
    if not is_new_round:
        for record in resume.get("completed_reviewer_data", []):
            current_round_items.extend(record.get("new_items", []))
            round_reviewer_records.append(record)
            if record.get("state") == "blocking":
                any_reviewer_blocked = True
                round_blocking_items.extend(
                    {"text": item.get("text", "")}
                    for item in record.get("new_items", [])
                    if item.get("status") in ("blocking", "same-pr")
                )
            else:
                round_approved_reviewers.append(str(record.get("reviewer_name", "")))

    pr_diff = _fetch_pr_diff(repo, pr)
    issue_dict = _fetch_pr_json(repo, pr)

    # Repo-scoped agent memory for reviewer orientation (#306), prepared once.
    memory = None
    if getattr(args, "agent_memory", True):
        memory = _prepare_skill_memory(
            repo, reviewers, _workdir_for_agent(reviewers[0], args),
            refresh=getattr(args, "refresh_agent_memory", False), dry_run=dry_run,
        )

    # Step 6 — run pending reviewers.
    # External reviewers (Codex/Gemini) always run first, regardless of --reviewers
    # order; a configured host reviewer ("claude") reviews the PR itself and becomes
    # a pending handoff after the loop (#314).
    pending_reviewers: list[str] = []
    external_reviewers = [r for r in reviewers if r != "claude"]
    with tempfile.TemporaryDirectory() as tmpstr:
        tmpdir = Path(tmpstr)
        for reviewer in external_reviewers:
            reviewer_cap = agent_display_name(reviewer)  # type: ignore[arg-type]
            if reviewer_cap in local_completed or reviewer in local_completed:
                print(f"[skip] {reviewer_cap} already completed this round")
                continue

            workdir = _workdir_for_agent(reviewer, args)
            try:
                from helpers.prompt_builders import build_review_prompt_for_skill
                prompt_text = build_review_prompt_for_skill(
                    issue_dict,
                    pr_diff,
                    next_prior_items_raw,
                    new_round_number,
                    reviewer,  # type: ignore[arg-type]
                    repo=repo,
                    pr_number=pr,
                    all_reviewers=[r for r in reviewers],  # type: ignore[misc]
                    workdir=workdir,
                    approved_followups=getattr(args, "approved_followups", "ignore"),
                    memory=memory,
                )
            except Exception as exc:  # noqa: BLE001
                prompt_text = (
                    f"Review the following PR #{pr} in {repo}.\n\n"
                    f"Round: {new_round_number}\n\n"
                    f"```diff\n{pr_diff[:8000]}\n```\n"
                )

            context: dict = {
                "reviewer": reviewer_cap,
                "prior_items": next_prior_items_raw,
                "current_round_items": current_round_items,
            }

            item_id_offset = _max_item_number([next_prior_items_raw, current_round_items])
            record = _run_reviewer(
                agent=reviewer,
                prompt_text=prompt_text,
                context=context,
                round_subject=head_sha,
                next_prior_items_raw=next_prior_items_raw,
                new_round_number=new_round_number,
                issue=pr,
                repo=repo,
                flow="pr",
                role="reviewer",
                state_key="pr_review",
                workdir=workdir,
                dry_run=dry_run,
                tmpdir=tmpdir,
                item_id_offset=item_id_offset,
            )
            if record["state"] == "unavailable":
                # Agent/CLI failure: skip this reviewer for the round (re-attempted
                # on rerun); still fold its usage into the aggregate (#322).
                unavailable_reviewers.append(record["reviewer_name"])
                round_reviewer_records.append(record)
                continue
            current_round_items.extend(record["new_items"])
            round_reviewer_records.append(record)
            if record["state"] == "blocking":
                any_reviewer_blocked = True
                round_blocking_items.extend(record["blocking_items"])
            else:
                round_approved_reviewers.append(record["reviewer_name"])

    # Step 6b — host-as-reviewer handoff. The host (Claude) reviews the PR itself,
    # so write a review-request dir (the PR diff for the host to read) and mark it
    # pending; the host writes its pr_review JSON there and completes it via
    # `complete-host-review` (#314).
    if "claude" in reviewers and "Claude" not in local_completed:
        item_id_offset = _max_item_number([next_prior_items_raw, current_round_items])
        request_dir = _write_host_review_request(
            flow="pr", validate_kind="pr_review",
            issue=pr, repo=repo, new_round_number=new_round_number,
            round_subject=head_sha or "", review_material=pr_diff,
            material_filename="pr-diff.diff",
            next_prior_items_raw=next_prior_items_raw,
            current_round_items=current_round_items,
            item_id_offset=item_id_offset, dry_run=dry_run,
        )
        pending_reviewers.append("Claude")
        dry_run_flag = " --dry-run" if dry_run else ""
        print(
            f"skill_runner: host review pending — read the PR diff in {request_dir}/pr-diff.diff, "
            f"write your pr_review JSON to {request_dir}/host-review.md, then run: "
            f"python -m helpers.skill_runner complete-host-review --dir {request_dir}{dry_run_flag}",
            file=sys.stderr,
        )

    # Step 7 — write session
    if not dry_run:
        _run_helper(
            "helpers.state_manager", "write-session",
            "--issue", str(pr), "--repo", repo,
            "--fields", json.dumps({
                "last_completed_step": "post_review",
                "round_number": new_round_number,
            }),
        )

    # Step 8 — print result.
    # A configured-but-incomplete host reviewer makes the round "pending": never
    # report approved/blocking (or publish followups / run the test gate, which
    # belong to a settled round) while a reviewer is still outstanding (#314).
    # Precedence: pending (host review outstanding) > blocking > incomplete
    # (a configured reviewer was unavailable) > approved. Both reviewer lists are
    # always surfaced when non-empty so neither the host handoff nor a failed
    # reviewer is ever hidden, and a round is never falsely reported approved (#322).
    overall_state = _round_overall_state(
        pending_reviewers=pending_reviewers,
        any_reviewer_blocked=any_reviewer_blocked,
        unavailable_reviewers=unavailable_reviewers,
    )
    result_json = {
        "state": overall_state,
        "round_number": new_round_number,
        "blocking_items": round_blocking_items,
        "approved_reviewers": round_approved_reviewers,
    }
    if pending_reviewers:
        result_json["pending_reviewers"] = pending_reviewers
    if unavailable_reviewers:
        result_json["unavailable_reviewers"] = unavailable_reviewers
    # Optional test gate: reports pass/fail; never blocks or merges. An explicit
    # empty --test-command "" is reported as a setup error, not silently skipped.
    # Skipped while the round is unsettled (pending a host review, or a reviewer
    # was unavailable) so it isn't read as a terminal result (#314, #322).
    gate = None if (pending_reviewers or unavailable_reviewers) else _maybe_test_gate(
        getattr(args, "test_command", None),
        getattr(args, "test_workdir", "."),
        dry_run=dry_run,
    )
    if gate is not None:
        result_json["tests"] = gate

    # Optional approved-followups publishing (#300): only on an approved round and
    # when a non-ignore mode is set. PR comments are fetched lazily here (for the
    # idempotency marker) so blocking/ignore rounds make no extra network call.
    followups_mode = getattr(args, "approved_followups", "ignore")
    if overall_state == "approved" and followups_mode != "ignore":
        round_future_items = [
            item for item in current_round_items if item.get("status") == "future"
        ]
        pr_comments = [
            types.SimpleNamespace(body=body)
            for body in _fetch_pr_comments_raw(repo, pr)
        ]
        result_json["approved_followups"] = _publish_pr_followups(
            repo, pr, head_sha, followups_mode, round_future_items, pr_comments,
            dry_run=dry_run,
        )
    usage = _aggregate_reviewer_usage(round_reviewer_records)
    if usage is not None:
        result_json["usage"] = usage
    print(json.dumps(result_json, indent=2))


# ---------------------------------------------------------------------------
# retry-validate
# ---------------------------------------------------------------------------

def cmd_retry_validate(args: argparse.Namespace) -> None:
    """Re-validate an already-written reviewer response from a repair dir."""
    repair_dir = Path(args.repair_dir)
    if not repair_dir.exists():
        print(f"skill_runner: repair dir not found: {repair_dir}", file=sys.stderr)
        sys.exit(1)

    manifest = json.loads((repair_dir / "manifest.json").read_text(encoding="utf-8"))

    # Reversed-roles coder repair: a malformed external coder plan_state/plan_revision
    # is recoverable here, distinct from the reviewer completion path below (#307).
    if manifest.get("role") == "coder":
        coder            = manifest["agent"]
        coder_cap        = manifest["agent_cap"]
        issue            = manifest["issue"]
        repo             = manifest["repo"]
        new_round_number = manifest["new_round_number"]
        kind             = manifest["kind"]
        next_prior_items_raw = json.loads(
            (repair_dir / "prior_items.json").read_text(encoding="utf-8")
        )
        try:
            result = _complete_coder_turn(
                coder=coder, coder_cap=coder_cap, issue=issue, repo=repo,
                new_round_number=new_round_number,
                next_prior_items_raw=next_prior_items_raw, kind=kind,
                dry_run=args.dry_run, raw_output=repair_dir / "raw.md",
                work_dir=repair_dir, usage_file=repair_dir / "coder-usage.json",
            )
        except _ValidationError as exc:
            print(str(exc), file=sys.stderr)
            sys.exit(1)
        print(json.dumps({
            "role": "coder",
            "agent": coder_cap,
            "round_number": new_round_number,
            "subject": result["subject"],
        }, indent=2))
        print(
            "hint: coder plan posted — re-run run-plan-round to drive the reviewers.",
            file=sys.stderr,
        )
        return

    agent            = manifest["agent"]
    agent_cap        = manifest["agent_cap"]
    flow             = manifest["flow"]
    validate_kind    = manifest["validate_kind"]
    issue            = manifest["issue"]
    repo             = manifest["repo"]
    new_round_number = manifest["new_round_number"]
    round_subject    = manifest["round_subject"]
    item_id_offset   = manifest["item_id_offset"]
    dry_run          = args.dry_run

    raw_output      = repair_dir / "raw.md"
    context_file    = repair_dir / "context.json"
    prior_items_raw = json.loads((repair_dir / "prior_items.json").read_text(encoding="utf-8"))

    try:
        result = _complete_reviewer_turn(
            agent=agent, agent_cap=agent_cap, flow=flow,
            validate_kind=validate_kind, issue=issue, repo=repo,
            new_round_number=new_round_number, round_subject=round_subject,
            next_prior_items_raw=prior_items_raw,
            item_id_offset=item_id_offset, dry_run=dry_run,
            raw_output=raw_output, context_file=context_file,
            work_dir=repair_dir,
        )
    except _ValidationError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    print(json.dumps(result, indent=2))
    if result["state"] in ("approved", "approved-with-notes"):
        print("hint: reviewer approved — re-run the parent round command to continue.", file=sys.stderr)
    else:
        print(f"hint: reviewer state={result['state']} — fix issues in raw.md and retry.", file=sys.stderr)


# ---------------------------------------------------------------------------
# complete-host-review  (reversed roles: host (Claude) reviewer turn)
# (plan flow #307; PR flow #314)
# ---------------------------------------------------------------------------

def cmd_complete_host_review(args: argparse.Namespace) -> None:
    """Complete a host (Claude) reviewer turn from a review-request dir.

    The host reads the posted plan/PR (``{dir}/<material>``), writes its
    ``plan_review``/``pr_review`` JSON to ``{dir}/host-review.md``, then runs this
    command. Reuses ``_complete_reviewer_turn`` (agent=claude → display ``Claude``)
    to validate, render, attach metadata, and post — so ``build-resume`` recognizes
    the review and re-running the round command recomputes the final state. Works
    for both flows; the manifest's ``flow``/``validate_kind`` drive the behavior and
    the user-facing wording.
    """
    request_dir = Path(args.dir)
    if not request_dir.exists():
        print(f"skill_runner: host review dir not found: {request_dir}", file=sys.stderr)
        sys.exit(1)
    manifest = json.loads((request_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("agent") != "claude" or manifest.get("role") != "reviewer":
        print(
            f"skill_runner: {request_dir}/manifest.json is not a host (Claude) review request",
            file=sys.stderr,
        )
        sys.exit(1)

    flow             = manifest["flow"]
    validate_kind    = manifest["validate_kind"]
    round_command    = f"run-{flow}-round"

    review_file = request_dir / "host-review.md"
    if not review_file.exists():
        print(
            f"skill_runner: write your {validate_kind} JSON to {review_file} first, then re-run.",
            file=sys.stderr,
        )
        sys.exit(1)

    issue            = manifest["issue"]
    repo             = manifest["repo"]
    new_round_number = manifest["new_round_number"]
    dry_run          = args.dry_run
    prior_items_raw  = json.loads((request_dir / "prior_items.json").read_text(encoding="utf-8"))

    try:
        result = _complete_reviewer_turn(
            agent="claude", agent_cap=agent_display_name("claude"),
            flow=flow, validate_kind=validate_kind,
            issue=issue, repo=repo,
            new_round_number=new_round_number, round_subject=manifest["round_subject"],
            next_prior_items_raw=prior_items_raw,
            item_id_offset=manifest["item_id_offset"], dry_run=dry_run,
            raw_output=review_file, context_file=request_dir / "context.json",
            work_dir=request_dir,
        )
    except _ValidationError as exc:
        print(str(exc), file=sys.stderr)
        print(
            f"skill_runner: fix {review_file}, then re-run complete-host-review.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not dry_run:
        _run_helper(
            "helpers.state_manager", "write-session",
            "--issue", str(issue), "--repo", repo,
            "--fields", json.dumps({
                "last_completed_step": "post_review",
                "round_number": new_round_number,
            }),
        )

    print(json.dumps(result, indent=2))
    print(
        f"hint: host review posted — re-run {round_command} to recompute the round.",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# run-implement  (reversed roles: external coder implements an approved plan) (#316)
# ---------------------------------------------------------------------------

def _git_head(workdir: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", workdir, "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _validate_coder_implementation_response(
    coder_output: str,
    *,
    workdir: str,
    human_requirements,
) -> int:
    """Validate an external coder's implementation response; return the PR number.

    Mirrors the guards in ``orchestrator._implement_approved_issue``: a valid
    ``<!-- AGENT_PR: N -->`` marker, the signed-human-requirements acknowledgement
    (when any were surfaced), and that the coder's reported test commands ran inside
    the assigned workdir. Raises ``AgentLoopError`` on any failure (the caller treats
    that as non-retryable). Factored out so the guards are directly unit-testable.
    """
    from coding_review_agent_loop.orchestrator import (
        _require_pr_number,
        _validate_response_with_human_requirements,
    )
    from coding_review_agent_loop.workdir_guard import validate_response_tests_within_workdir

    pr_number = _validate_response_with_human_requirements(
        coder_output,
        marker_validator=_require_pr_number,
        human_requirements=human_requirements,
        requirement_scope="implementation requirements",
        full_omission_fallback="Fetch the issue discussion directly before implementing.",
    )
    validate_response_tests_within_workdir(coder_output, assigned_workdir=Path(workdir))
    return int(pr_number)


def _load_required_plan_file(path_text: str, *, action: str) -> str:
    plan_file = Path(path_text)
    if not plan_file.exists():
        print(f"skill_runner: plan file not found: {plan_file}", file=sys.stderr)
        sys.exit(1)
    approved_plan = plan_file.read_text(encoding="utf-8")
    if not approved_plan.strip():
        print(f"skill_runner: plan file is empty; cannot {action} an empty plan", file=sys.stderr)
        sys.exit(1)
    return approved_plan


def _implementation_config(
    *,
    repo: str,
    coder: str,
    workdir: str,
    base: str,
    dry_run: bool,
):
    from helpers.prompt_builders import make_minimal_config

    # auto_agent_dirs=() so the explicit, user-provided push-capable workdir is
    # treated as user-owned: sync_coder_base_before_implementation then *rejects* a
    # dirty checkout instead of `git reset --hard`/`clean -fd`-ing the user's clone (#316).
    return dataclasses.replace(
        make_minimal_config(repo, coder, (coder,), reviewer=coder, workdir=str(workdir), base=base),  # type: ignore[arg-type]
        dry_run=dry_run,
        auto_agent_dirs=(),
    )


def _run_child_or_one_shot_implementation(
    *,
    issue: int,
    repo: str,
    coder: str,
    base: str,
    dry_run: bool,
    workdir: str,
    approved_plan: str,
    issue_context,
    config,
    runner: Runner,
    post_one_shot_handoff: bool,
    one_shot_mode: str = "implement-one-shot",
) -> dict:
    from coding_review_agent_loop.config import sync_coder_base_before_implementation
    from coding_review_agent_loop.decomposition import (
        approved_plan_hash,
        format_one_shot_impl_handoff_comment,
    )
    from coding_review_agent_loop.github import (
        get_pr_review_context,
        validate_open_pr,
        validate_pr_references_issue,
    )
    from coding_review_agent_loop.workdir_guard import validate_assigned_head_advanced
    from helpers.prompt_builders import build_implementation_prompt_for_skill

    coder_cap = agent_display_name(coder)  # type: ignore[arg-type]
    prompt_text = build_implementation_prompt_for_skill(
        issue_context,
        approved_plan,
        repo=repo,
        coder=coder,  # type: ignore[arg-type]
        workdir=str(workdir),
        base=base,
    )

    if dry_run:
        print(f"[dry-run] would sync {workdir} to base '{base}' and run {coder_cap} implementation")
        head_before = None
    else:
        sync_coder_base_before_implementation(config, runner)
        head_before = _git_head(str(workdir))

    with tempfile.TemporaryDirectory() as tmpstr:
        tmpdir = Path(tmpstr)
        prompt_file = tmpdir / "impl-prompt.md"
        raw_output = tmpdir / "impl-raw.md"
        usage_file = tmpdir / "impl-usage.json"
        _write_text(prompt_file, prompt_text)

        # No --repo: run_external would otherwise ensure_temp_checkout the workdir
        # against its own base="main" config, resetting it away from --base right
        # before the coder runs (#316).
        _run_helper(
            "helpers.run_external",
            "--agent", coder,
            "--role", "coder",
            "--prompt-file", str(prompt_file),
            "--output", str(raw_output),
            "--workdir", str(workdir),
            "--flow", "pr",
            "--usage-output", str(usage_file),
            *(["--dry-run"] if dry_run else []),
        )
        coder_output = raw_output.read_text(encoding="utf-8")
        coder_usage: dict | None = None
        if usage_file.exists():
            try:
                coder_usage = json.loads(usage_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                coder_usage = None

        try:
            pr_number = _validate_coder_implementation_response(
                coder_output,
                workdir=str(workdir),
                human_requirements=issue_context.human_requirements,
            )
        except AgentLoopError as exc:
            debug_dir = _REPAIR_BASE / f"{issue}-implement-debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(raw_output, debug_dir / "raw.md")
            print(f"skill_runner: implementation response rejected: {exc}", file=sys.stderr)
            print(
                f"skill_runner: raw response saved to {debug_dir}/raw.md — fix the "
                f"underlying issue and re-run run-implement (non-retryable).",
                file=sys.stderr,
            )
            sys.exit(1)

        if dry_run:
            head_sha = "dry-run-sha"
        else:
            head_after = _git_head(str(workdir))
            validate_assigned_head_advanced(
                before_head=head_before, after_head=head_after,
                assigned_workdir=Path(str(workdir)),
            )
            validate_open_pr(runner, config=config, pr_number=pr_number)
            validate_pr_references_issue(
                runner, config=config, pr_number=pr_number, issue_number=issue,
            )
            head_sha = get_pr_review_context(
                runner, config=config, pr_number=pr_number,
            ).metadata.head_sha or "unknown"

        tagged = tmpdir / "impl-tagged.md"
        _run_helper(
            "helpers.state_manager", "attach-metadata",
            "--body-file", str(raw_output),
            "--output", str(tagged),
            "--flow", "pr",
            "--role", "coder",
            "--agent", coder_cap,
            "--round-number", "1",
            "--subject", str(head_sha),
            "--state", "approved",
        )
        if not dry_run:
            _run_helper(
                "helpers.gh_ops", "post-issue-comment",
                "--issue", str(pr_number), "--file", str(tagged), "--repo", repo,
            )
            if post_one_shot_handoff:
                handoff = tmpdir / "impl-handoff.md"
                _write_text(handoff, format_one_shot_impl_handoff_comment(
                    parent_issue=issue,
                    mode=one_shot_mode,
                    plan_hash=approved_plan_hash(approved_plan),
                    plan_subject=_plan_subject(approved_plan),
                    pr_number=pr_number,
                    pr_head_sha=head_sha,
                ))
                _run_helper(
                    "helpers.gh_ops", "post-issue-comment",
                    "--issue", str(issue), "--file", str(handoff), "--repo", repo,
                )
        else:
            print(f"[dry-run] would post coder PR comment for PR #{pr_number}")
            if post_one_shot_handoff:
                print(f"[dry-run] would post one-shot handoff for PR #{pr_number}")

    result_json: dict = {"pr": pr_number, "head_sha": head_sha, "issue": issue}
    usage = _aggregate_reviewer_usage([{"usage": coder_usage}] if coder_usage else [])
    if usage is not None:
        result_json["usage"] = usage
    return result_json


def cmd_run_implement(args: argparse.Namespace) -> None:
    issue: int = args.issue
    repo: str = args.repo
    coder: str = args.coder
    base: str = args.base
    dry_run: bool = args.dry_run

    approved_plan = _load_required_plan_file(args.plan_file, action="implement")

    explicit_workdir = getattr(args, f"workdir_{coder}", None) or getattr(args, "workdir", None)
    if not dry_run and not explicit_workdir:
        print(
            "skill_runner: run-implement requires an explicit push-capable "
            "--workdir (or --workdir-codex/--workdir-gemini); the external coder "
            "commits, pushes a branch, and opens a PR there.",
            file=sys.stderr,
        )
        sys.exit(1)
    workdir = _workdir_for_agent(coder, args)

    from coding_review_agent_loop.decomposition import (
        approved_plan_hash,
        find_existing_one_shot_impl_handoff,
    )
    from coding_review_agent_loop.github import (
        get_issue_context,
        validate_open_pr,
        validate_pr_references_issue,
    )

    plan_hash = approved_plan_hash(approved_plan)
    mode = "implement-one-shot"

    config = _implementation_config(
        repo=repo, coder=coder, workdir=str(workdir), base=base, dry_run=dry_run,
    )
    runner = Runner(dry_run=dry_run)

    # Idempotency — reuse the durable one-shot handoff marker (no duplicate PRs).
    comments = [types.SimpleNamespace(body=b) for b in _fetch_issue_comments_raw(repo, issue)]
    existing = find_existing_one_shot_impl_handoff(
        comments, parent_issue=issue, plan_hash=plan_hash, mode=mode,
    )
    if existing is not None:
        # Only reuse a recorded PR that is still open AND still references this issue
        # — a stale/injected same-plan-hash marker must not return an unrelated PR.
        reusable = True
        try:
            validate_open_pr(runner, config=config, pr_number=existing.pr_number)
            validate_pr_references_issue(
                runner, config=config, pr_number=existing.pr_number, issue_number=issue,
            )
        except AgentLoopError:
            reusable = False
        if reusable:
            print(json.dumps({
                "pr": existing.pr_number,
                "head_sha": existing.pr_head_sha,
                "issue": issue,
                "reused": True,
            }, indent=2))
            print(
                f"hint: PR #{existing.pr_number} already implements this plan — "
                f"review it with run-pr-round.",
                file=sys.stderr,
            )
            return

    issue_context = get_issue_context(runner, config=config, issue_number=issue)
    result_json = _run_child_or_one_shot_implementation(
        issue=issue,
        repo=repo,
        coder=coder,
        base=base,
        dry_run=dry_run,
        workdir=str(workdir),
        approved_plan=approved_plan,
        issue_context=issue_context,
        config=config,
        runner=runner,
        post_one_shot_handoff=True,
        one_shot_mode=mode,
    )
    print(json.dumps(result_json, indent=2))
    print(
        f"hint: PR #{result_json['pr']} opened — review it with: python -m helpers.skill_runner "
        f"run-pr-round --pr {result_json['pr']} --repo {repo} --reviewers claude gemini",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# run-decompose  (external coder decomposes an approved plan into child issues) (#318)
# ---------------------------------------------------------------------------

def _issue_context_for_decompose_preview(repo: str, issue: int):
    from coding_review_agent_loop.github import IssueContext, IssueComment

    try:
        issue_json = _fetch_issue_json(repo, issue)
    except SystemExit:
        return IssueContext(
            number=issue,
            repo=repo,
            title=None,
            body=None,
            url=None,
            comments=(),
            human_requirements=(),
        )
    comments = tuple(
        IssueComment(author=None, created_at=None, body=body)
        for body in _fetch_issue_comments_raw(repo, issue)
    )
    return IssueContext(
        number=int(issue_json.get("number") or issue),
        repo=repo,
        title=issue_json.get("title"),
        body=issue_json.get("body"),
        url=issue_json.get("url"),
        comments=comments,
        human_requirements=(),
    )


def _decomposition_phase_json(index: int, phase, *, issue_url: str | None = None,
                              issue_number: int | None = None) -> dict:
    return {
        "index": index,
        "title": phase.title,
        "phase_title": f"Phase {index}: {phase.title}",
        "automation": phase.automation,
        "depends_on": list(getattr(phase, "depends_on", ()) or ()),
        "scope": getattr(phase, "scope", None),
        "non_goals": getattr(phase, "non_goals", None),
        "dependency_notes": getattr(phase, "dependency_notes", None),
        "rollout_risk": phase.rollout_risk,
        "validation": getattr(phase, "validation", None),
        "parent_context": phase.parent_context,
        "issue_url": issue_url,
        "issue_number": issue_number,
    }


def _existing_decomposition_json(existing) -> list[dict]:
    phases = []
    for index, (title, url, number) in enumerate(existing.children, start=1):
        phases.append({
            "index": index,
            "title": title,
            "phase_title": f"Phase {index}: {title}",
            "automation": existing.automation[index - 1] if index <= len(existing.automation) else None,
            "issue_url": url,
            "issue_number": number,
        })
    return phases


def _created_phase_issues_from_existing(existing):
    from coding_review_agent_loop.decomposition import CreatedPhaseIssue, RecordedPhase

    return tuple(
        CreatedPhaseIssue(
            phase=RecordedPhase(title=title, automation=automation),
            issue_url=url,
            issue_number=number,
        )
        for (title, url, number), automation in zip(existing.children, existing.automation, strict=False)
    )


def _run_decomposition_for_skill(
    *,
    issue: int,
    repo: str,
    coder: str,
    plan_file: str,
    args: argparse.Namespace,
    dry_run: bool,
    mode: str,
    approved_plan: str | None = None,
) -> tuple[dict, tuple, object, object, Runner, str, str, str]:
    if approved_plan is None:
        approved_plan = _load_required_plan_file(plan_file, action="decompose")
    workdir = _workdir_for_agent(coder, args)
    Path(workdir).mkdir(parents=True, exist_ok=True)

    from coding_review_agent_loop.decomposition import (
        approved_plan_hash,
        CreatedPhaseIssue,
        create_decomposition_child_issues,
        find_existing_decomposition,
        parse_plan_decomposition,
        post_decomposition_parent_summary,
    )
    from coding_review_agent_loop.github import get_issue_context
    from helpers.prompt_builders import (
        build_plan_decomposition_prompt_for_skill,
        make_minimal_config,
    )

    config = dataclasses.replace(
        make_minimal_config(repo, coder, (coder,), reviewer=coder, workdir=str(workdir)),  # type: ignore[arg-type]
        dry_run=dry_run,
    )
    runner = Runner(dry_run=dry_run)
    plan_hash = approved_plan_hash(approved_plan)

    if dry_run:
        issue_context = _issue_context_for_decompose_preview(repo, issue)
    else:
        issue_context = get_issue_context(runner, config=config, issue_number=issue)

    existing = find_existing_decomposition(
        issue_context.comments,
        parent_issue=issue,
        plan_hash=plan_hash,
        mode=mode,
    )
    if existing is not None:
        created = _created_phase_issues_from_existing(existing)
        result = {
            "issue": issue,
            "plan_hash": plan_hash,
            "mode": mode,
            "dry_run": dry_run,
            "reused": True,
            "phase_count": existing.phase_count,
            "phases": _existing_decomposition_json(existing),
        }
        print(
            f"hint: issue #{issue} already has a {mode} decomposition for this plan.",
            file=sys.stderr,
        )
        return result, created, issue_context, config, runner, plan_hash, approved_plan, str(workdir)

    prompt_text = build_plan_decomposition_prompt_for_skill(
        issue_context, approved_plan, repo=repo, coder=coder, workdir=str(workdir),  # type: ignore[arg-type]
    )

    with tempfile.TemporaryDirectory() as tmpstr:
        tmpdir = Path(tmpstr)
        prompt_file = tmpdir / "decompose-prompt.md"
        raw_output = tmpdir / "decompose-raw.md"
        usage_file = tmpdir / "decompose-usage.json"
        _write_text(prompt_file, prompt_text)

        _run_helper(
            "helpers.run_external",
            "--agent", coder,
            "--role", "coder",
            "--prompt-file", str(prompt_file),
            "--output", str(raw_output),
            "--workdir", str(workdir),
            "--repo", repo,
            "--flow", "decompose",
            "--usage-output", str(usage_file),
            *(["--dry-run"] if dry_run else []),
        )
        coder_output = raw_output.read_text(encoding="utf-8")
        coder_usage: dict | None = None
        if usage_file.exists():
            try:
                coder_usage = json.loads(usage_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                coder_usage = None

        try:
            decomposition = parse_plan_decomposition(coder_output)
        except AgentLoopError as exc:
            debug_dir = _REPAIR_BASE / f"{issue}-decompose-debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(raw_output, debug_dir / "raw.md")
            shutil.copy2(prompt_file, debug_dir / "prompt.md")
            print(f"skill_runner: decomposition response rejected: {exc}", file=sys.stderr)
            print(
                f"skill_runner: raw response saved to {debug_dir}/raw.md — fix the "
                f"underlying issue and re-run run-decompose (non-retryable).",
                file=sys.stderr,
            )
            sys.exit(1)

        result_json: dict = {
            "issue": issue,
            "plan_hash": plan_hash,
            "mode": mode,
            "dry_run": dry_run,
            "reused": False,
        }
        if dry_run:
            result_json["would_post_parent_summary"] = True
            created = tuple(
                CreatedPhaseIssue(phase=phase, issue_url=None, issue_number=None)
                for phase in decomposition.phases
            )
            result_json["phases"] = [
                _decomposition_phase_json(index, item.phase)
                for index, item in enumerate(created, start=1)
            ]
            print("[dry-run] would create child phase issues and post parent decomposition marker")
        else:
            created = create_decomposition_child_issues(
                runner,
                config=config,
                parent_issue=issue,
                approved_plan=approved_plan,
                decomposition=decomposition,
            )
            post_decomposition_parent_summary(
                runner,
                config=config,
                parent_issue=issue,
                mode=mode,
                plan_hash=plan_hash,
                created=created,
            )
            result_json["phases"] = [
                _decomposition_phase_json(
                    index, created_issue.phase,
                    issue_url=created_issue.issue_url,
                    issue_number=created_issue.issue_number,
                )
                for index, created_issue in enumerate(created, start=1)
            ]
        result_json["phase_count"] = len(result_json["phases"])
        usage = _aggregate_reviewer_usage([{"usage": coder_usage}] if coder_usage else [])
        if usage is not None:
            result_json["usage"] = usage
        return result_json, created, issue_context, config, runner, plan_hash, approved_plan, str(workdir)


def cmd_run_decompose(args: argparse.Namespace) -> None:
    result_json, *_ = _run_decomposition_for_skill(
        issue=args.issue,
        repo=args.repo,
        coder=args.coder,
        plan_file=args.plan_file,
        args=args,
        dry_run=args.dry_run,
        mode="decompose-only",
    )
    print(json.dumps(result_json, indent=2))


# ---------------------------------------------------------------------------
# run-implement-by-phase  (decompose, then implement phase 1 when agent-pr) (#319)
# ---------------------------------------------------------------------------

def cmd_run_implement_by_phase(args: argparse.Namespace) -> None:
    issue: int = args.issue
    repo: str = args.repo
    coder: str = args.coder
    base: str = args.base
    dry_run: bool = args.dry_run
    mode = "implement-by-phase"

    approved_plan = _load_required_plan_file(args.plan_file, action="decompose")
    explicit_workdir = getattr(args, f"workdir_{coder}", None) or getattr(args, "workdir", None)
    if not dry_run and not explicit_workdir:
        print(
            "skill_runner: run-implement-by-phase requires an explicit push-capable "
            "--workdir (or --workdir-codex/--workdir-gemini); the external coder "
            "commits, pushes a branch, and opens a PR for the first agent phase there.",
            file=sys.stderr,
        )
        sys.exit(1)

    (
        decomposition_result,
        created,
        parent_issue_context,
        decompose_config,
        decompose_runner,
        plan_hash,
        approved_plan,
        workdir,
    ) = _run_decomposition_for_skill(
        issue=issue,
        repo=repo,
        coder=coder,
        plan_file=args.plan_file,
        args=args,
        dry_run=dry_run,
        mode=mode,
        approved_plan=approved_plan,
    )

    if not created:
        raise AgentLoopError("Plan decomposition produced no phases.")

    first_phase = created[0]
    result_json: dict = {
        "issue": issue,
        "plan_hash": plan_hash,
        "mode": mode,
        "dry_run": dry_run,
        "decomposition_reused": bool(decomposition_result.get("reused")),
        "phase": _decomposition_phase_json(
            1,
            first_phase.phase,
            issue_url=first_phase.issue_url,
            issue_number=first_phase.issue_number,
        ),
    }

    if first_phase.phase.automation != "agent-pr":
        result_json.update({
            "state": "stopped",
            "reason": "first phase requires human work",
            "automation": first_phase.phase.automation,
        })
        print(json.dumps(result_json, indent=2))
        print(
            f"hint: issue #{issue} decomposed; phase 1 requires human work "
            f"({first_phase.phase.automation}), so implementation is stopping.",
            file=sys.stderr,
        )
        return

    phase_parent_context = getattr(first_phase.phase, "parent_context", None) or approved_plan

    if dry_run:
        from coding_review_agent_loop.github import IssueContext

        preview_issue_number = first_phase.issue_number or 0
        preview_context = IssueContext(
            number=preview_issue_number,
            repo=repo,
            title=first_phase.phase.title,
            body=phase_parent_context,
            url=first_phase.issue_url,
            comments=(),
            human_requirements=getattr(parent_issue_context, "human_requirements", ()),
        )
        impl_config = _implementation_config(
            repo=repo, coder=coder, workdir=workdir, base=base, dry_run=True,
        )
        impl_runner = Runner(dry_run=True)
        impl_result = _run_child_or_one_shot_implementation(
            issue=preview_issue_number,
            repo=repo,
            coder=coder,
            base=base,
            dry_run=True,
            workdir=workdir,
            approved_plan=phase_parent_context,
            issue_context=preview_context,
            config=impl_config,
            runner=impl_runner,
            post_one_shot_handoff=False,
        )
        result_json.update({
            "state": "would-implement",
            "would_post_phase_handoff": True,
            "child_implementation_preview": impl_result,
        })
        print(json.dumps(result_json, indent=2))
        return

    if first_phase.issue_number is None:
        raise AgentLoopError(
            "Cannot implement first decomposed phase because its child issue number "
            "was not available from GitHub CLI output."
        )

    from coding_review_agent_loop.decomposition import (
        find_existing_phase_implementation_handoff,
        post_phase_implementation_handoff_comment,
    )
    from coding_review_agent_loop.github import get_issue_context

    handoff = find_existing_phase_implementation_handoff(
        parent_issue_context.comments,
        parent_issue=issue,
        plan_hash=plan_hash,
        mode=mode,
        phase_index=1,
        child_issue_number=first_phase.issue_number,
    )
    if handoff is not None:
        result_json.update({
            "state": "handoff-exists",
            "handoff_reused": True,
            "child_issue": handoff.child_issue_number,
            "child_issue_url": handoff.child_issue_url,
        })
        print(json.dumps(result_json, indent=2))
        print(
            f"hint: issue #{issue} already handed off phase 1 to child issue "
            f"#{handoff.child_issue_number}; resume directly with "
            f"`agent-loop issue {handoff.child_issue_number}`.",
            file=sys.stderr,
        )
        return

    post_phase_implementation_handoff_comment(
        decompose_runner,
        config=decompose_config,
        parent_issue=issue,
        mode=mode,
        plan_hash=plan_hash,
        phase_index=1,
        created=first_phase,
    )

    impl_config = _implementation_config(
        repo=repo, coder=coder, workdir=workdir, base=base, dry_run=False,
    )
    impl_runner = Runner(dry_run=False)
    child_issue_context = get_issue_context(
        impl_runner,
        config=impl_config,
        issue_number=first_phase.issue_number,
    )
    impl_result = _run_child_or_one_shot_implementation(
        issue=first_phase.issue_number,
        repo=repo,
        coder=coder,
        base=base,
        dry_run=False,
        workdir=workdir,
        approved_plan=phase_parent_context,
        issue_context=child_issue_context,
        config=impl_config,
        runner=impl_runner,
        post_one_shot_handoff=False,
    )
    result_json.update({
        "state": "implemented",
        "handoff_posted": True,
        "child_issue": first_phase.issue_number,
        "child_implementation": impl_result,
    })
    print(json.dumps(result_json, indent=2))
    print(
        f"hint: phase 1 PR #{impl_result['pr']} opened from child issue "
        f"#{first_phase.issue_number}; review it with run-pr-round.",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one complete review round for a plan or PR."
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # run-plan-round
    p_plan = subparsers.add_parser("run-plan-round", help="Run one plan review round.")
    p_plan.add_argument("--issue", type=int, required=True)
    p_plan.add_argument("--repo", required=True)
    p_plan.add_argument(
        "--coder", choices=["claude", "codex", "gemini", "antigravity"], default="claude",
        help="Who writes the plan: 'claude' (default, host supplies --plan-file) or "
             "an external coder ('codex'/'gemini') run by the skill (reversed roles, #307).",
    )
    p_plan.add_argument(
        "--plan-file", default=None,
        help="Plan markdown file. Required for --coder claude; ignored for an external coder.",
    )
    p_plan.add_argument("--reviewers", nargs="+", required=True)
    p_plan.add_argument("--workdir-codex", default=None)
    p_plan.add_argument("--workdir-gemini", default=None)
    p_plan.add_argument("--workdir", default=None)
    p_plan_mem = p_plan.add_mutually_exclusive_group()
    p_plan_mem.add_argument("--agent-memory", dest="agent_memory", action="store_true", default=True,
                            help="Include repo-scoped agent memory in reviewer prompts (default).")
    p_plan_mem.add_argument("--no-agent-memory", dest="agent_memory", action="store_false",
                            help="Disable repo-scoped agent memory.")
    p_plan.add_argument("--refresh-agent-memory", action="store_true",
                        help="Regenerate the agent-memory cache before this round.")
    p_plan.add_argument("--dry-run", action="store_true")

    # run-pr-round
    p_pr = subparsers.add_parser("run-pr-round", help="Run one PR review round.")
    p_pr.add_argument("--pr", type=int, required=True)
    p_pr.add_argument("--repo", required=True)
    p_pr.add_argument("--reviewers", nargs="+", required=True)
    p_pr.add_argument("--head-sha", default=None)
    p_pr.add_argument("--workdir", default=None)
    p_pr.add_argument("--workdir-codex", default=None)
    p_pr.add_argument("--workdir-gemini", default=None)
    p_pr.add_argument(
        "--test-command",
        default=None,
        help="Optional command to run as a test gate after the reviewer turns. "
             "Result (pass/fail) is reported in the JSON 'tests' field; it never "
             "blocks or merges. 'Ready to merge' = state approved AND tests.passed.",
    )
    p_pr.add_argument(
        "--test-workdir",
        default=".",
        help="Directory to run --test-command in (default: current directory, "
             "where the host coder has the PR branch checked out).",
    )
    p_pr.add_argument(
        "--approved-followups",
        choices=("ignore", "summarize", "issue"),
        default="ignore",
        help="On an approved round, publish reviewers' future follow-ups: "
             "'summarize' posts one PR comment; 'issue' files follow-up issues; "
             "'ignore' (default) discards them. Never blocks or merges.",
    )
    p_pr_mem = p_pr.add_mutually_exclusive_group()
    p_pr_mem.add_argument("--agent-memory", dest="agent_memory", action="store_true", default=True,
                          help="Include repo-scoped agent memory in reviewer prompts (default).")
    p_pr_mem.add_argument("--no-agent-memory", dest="agent_memory", action="store_false",
                          help="Disable repo-scoped agent memory.")
    p_pr.add_argument("--refresh-agent-memory", action="store_true",
                      help="Regenerate the agent-memory cache before this round.")
    p_pr.add_argument("--dry-run", action="store_true")

    # retry-validate
    p_retry = subparsers.add_parser(
        "retry-validate",
        help="Re-validate an already-written reviewer response without re-running the agent.",
    )
    p_retry.add_argument(
        "--repair-dir", required=True,
        help="Path to repair dir written by skill_runner on validation failure.",
    )
    p_retry.add_argument("--dry-run", action="store_true")

    # complete-host-review
    p_host = subparsers.add_parser(
        "complete-host-review",
        help="Complete a host (Claude) reviewer turn from a review-request dir "
             "(plan flow #307; PR flow #314).",
    )
    p_host.add_argument(
        "--dir", required=True,
        help="Path to the host-review request dir printed by the round command "
             "(run-plan-round or run-pr-round). Write your review JSON "
             "(plan_review or pr_review) to {dir}/host-review.md first.",
    )
    p_host.add_argument("--dry-run", action="store_true")

    # run-implement
    p_impl = subparsers.add_parser(
        "run-implement",
        help="Reversed roles: an external coder implements an approved plan and "
             "opens a PR, which you then review with run-pr-round (#316).",
    )
    p_impl.add_argument("--issue", type=int, required=True)
    p_impl.add_argument("--repo", required=True)
    p_impl.add_argument(
        "--coder", choices=["codex", "gemini", "antigravity"], required=True,
        help="External coder that implements the plan (the host implements in-session).",
    )
    p_impl.add_argument("--plan-file", required=True, help="Approved plan to implement.")
    p_impl.add_argument(
        "--base", default="main", help="Branch to base the PR on (default: main).",
    )
    p_impl.add_argument(
        "--workdir", default=None,
        help="Push-capable clone where the coder commits/pushes/opens the PR "
             "(required for a real run).",
    )
    p_impl.add_argument("--workdir-codex", default=None)
    p_impl.add_argument("--workdir-gemini", default=None)
    p_impl.add_argument("--dry-run", action="store_true")

    # run-implement-by-phase
    p_impl_phase = subparsers.add_parser(
        "run-implement-by-phase",
        help="Reversed roles: decompose an approved plan, then implement phase 1 "
             "when it is agent-pr (#319).",
    )
    p_impl_phase.add_argument("--issue", type=int, required=True)
    p_impl_phase.add_argument("--repo", required=True)
    p_impl_phase.add_argument(
        "--coder", choices=["codex", "gemini", "antigravity"], required=True,
        help="External coder that decomposes and implements the first agent phase.",
    )
    p_impl_phase.add_argument("--plan-file", required=True, help="Approved plan to decompose and implement.")
    p_impl_phase.add_argument(
        "--base", default="main", help="Branch to base the child PR on (default: main).",
    )
    p_impl_phase.add_argument(
        "--workdir", default=None,
        help="Push-capable clone where the coder commits/pushes/opens the child PR "
             "(required for a real run).",
    )
    p_impl_phase.add_argument("--workdir-codex", default=None)
    p_impl_phase.add_argument("--workdir-gemini", default=None)
    p_impl_phase.add_argument("--dry-run", action="store_true")

    # run-decompose
    p_decompose = subparsers.add_parser(
        "run-decompose",
        help="Reversed roles: an external coder decomposes an approved plan into "
             "child phase issues (#318).",
    )
    p_decompose.add_argument("--issue", type=int, required=True)
    p_decompose.add_argument("--repo", required=True)
    p_decompose.add_argument(
        "--coder", choices=["codex", "gemini", "antigravity"], required=True,
        help="External coder that decomposes the approved plan.",
    )
    p_decompose.add_argument("--plan-file", required=True, help="Approved plan to decompose.")
    p_decompose.add_argument("--workdir", default=None)
    p_decompose.add_argument("--workdir-codex", default=None)
    p_decompose.add_argument("--workdir-gemini", default=None)
    p_decompose.add_argument("--dry-run", action="store_true")

    # run-task-round
    p_task = subparsers.add_parser(
        "run-task-round",
        help="Create (or reuse) a scratch issue from free-form task text, then run a plan round.",
    )
    task_src = p_task.add_mutually_exclusive_group(required=True)
    task_src.add_argument("--task", default=None, help="Free-form task description.")
    task_src.add_argument(
        "--task-file", default=None,
        help="Read task description from this file ('-' for stdin).",
    )
    p_task.add_argument("--repo", required=True)
    p_task.add_argument(
        "--coder", choices=["claude", "codex", "gemini", "antigravity"], default="claude",
        help="Host coder only for now; external coders are rejected in task mode (#307).",
    )
    p_task.add_argument("--plan-file", required=True)
    p_task.add_argument("--reviewers", nargs="+", required=True)
    p_task.add_argument("--workdir-codex", default=None)
    p_task.add_argument("--workdir-gemini", default=None)
    p_task.add_argument("--workdir", default=None)
    p_task_mem = p_task.add_mutually_exclusive_group()
    p_task_mem.add_argument("--agent-memory", dest="agent_memory", action="store_true", default=True,
                            help="Include repo-scoped agent memory in reviewer prompts (default).")
    p_task_mem.add_argument("--no-agent-memory", dest="agent_memory", action="store_false",
                            help="Disable repo-scoped agent memory.")
    p_task.add_argument("--refresh-agent-memory", action="store_true",
                        help="Regenerate the agent-memory cache before this round.")
    p_task.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()
    dispatch = {
        "run-plan-round": cmd_run_plan_round,
        "run-pr-round": cmd_run_pr_round,
        "retry-validate": cmd_retry_validate,
        "complete-host-review": cmd_complete_host_review,
        "run-implement": cmd_run_implement,
        "run-implement-by-phase": cmd_run_implement_by_phase,
        "run-decompose": cmd_run_decompose,
        "run-task-round": cmd_run_task_round,
    }
    dispatch[args.subcommand](args)


if __name__ == "__main__":
    main()
