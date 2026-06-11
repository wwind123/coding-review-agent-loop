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

Prints a JSON result to stdout and exits 0:
  {"state": "approved"|"blocking", "round_number": N,
   "blocking_items": [...], "approved_reviewers": [...]}
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from coding_review_agent_loop.agents.base import AgentName
from coding_review_agent_loop.errors import AgentLoopError
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
    json_end = text.find("<!-- AGENT")
    if json_end < 0:
        json_end = len(text)
    json_part = text[:json_end].strip()
    footer = text[json_end:]
    try:
        data = json.loads(json_part)
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
    if not changed:
        return text
    return json.dumps(data, indent=2) + "\n" + footer.lstrip()


def _normalize_raw_response(text: str) -> str:
    """Strip prose the protocol validator would reject.

    1. Leading prose before the JSON object (Gemini sometimes writes a summary line first).
    2. Any content after the first agent signature line (e.g. a duplicate state marker).
    """
    # Strip leading prose before the JSON block
    brace_idx = text.find("{")
    if brace_idx > 0:
        text = text[brace_idx:]

    # Strip trailing content after the first signature line
    m = _STATE_MARKER_RE.search(text)
    if m is None:
        return text
    sig_m = _SIGNATURE_RE.search(text, m.end())
    if sig_m is None:
        return text
    return text[: sig_m.end()].rstrip() + "\n"


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
    # Auto-clone to a temp path; run_external handles this via --workdir
    tmp = Path(tempfile.gettempdir()) / "coding-review-agent-loop" / f"skill-runner-{agent}"
    tmp.mkdir(parents=True, exist_ok=True)
    return str(tmp)


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
# Per-reviewer run
# ---------------------------------------------------------------------------

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
    agent_cap = agent.capitalize() if agent in ("codex", "gemini") else agent
    prompt_file = tmpdir / f"{agent}-prompt.md"
    raw_output = tmpdir / f"{agent}-review-raw.md"
    rendered_output = tmpdir / f"{agent}-review-rendered.md"
    tagged_output = tmpdir / f"{agent}-review-tagged.md"
    context_file = tmpdir / f"{agent}-context.json"
    prior_items_file = tmpdir / f"{agent}-prior-items.json"
    dispositions_file = tmpdir / f"{agent}-dispositions.json"
    new_items_file = tmpdir / f"{agent}-new-items.json"

    _write_text(prompt_file, prompt_text)
    _write_json(context_file, context)
    _write_json(prior_items_file, next_prior_items_raw)

    # --- Run agent ---
    _run_helper(
        "helpers.run_external",
        "--agent", agent,
        "--prompt-file", str(prompt_file),
        "--output", str(raw_output),
        "--workdir", workdir,
        "--flow", flow,
        *(["--dry-run"] if dry_run else []),
    )

    # Normalize: fix prose prefix / duplicate state markers / bad disposition values
    raw_text = raw_output.read_text(encoding="utf-8")
    raw_text = _normalize_raw_response(raw_text)
    raw_text = _normalize_disposition_values(raw_text)
    raw_output.write_text(raw_text, encoding="utf-8")

    # --- Validate ---
    validate_kind = "pr_review" if flow == "pr" else "plan_review"
    result = _run_helper_capture(
        "helpers.validate_response",
        "--file", str(raw_output),
        "--kind", validate_kind,
        "--context-file", str(context_file),
    )
    if result.returncode != 0:
        print(f"skill_runner: {agent} review validation failed: {result.stderr.strip()}", file=sys.stderr)
        sys.exit(1)

    # --- Render ---
    _run_helper(
        "helpers.render_response",
        "--file", str(raw_output),
        "--kind", validate_kind,
        "--reviewer", agent_cap,
        "--context-file", str(context_file),
        "--output", str(rendered_output),
    )

    # --- Parse review JSON for metadata ---
    raw_text = raw_output.read_text(encoding="utf-8")
    try:
        json_part = raw_text.split("<!-- AGENT")[0].strip()
        review_json = json.loads(json_part)
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"skill_runner: cannot parse {agent} review JSON: {exc}", file=sys.stderr)
        sys.exit(1)

    parsed_state = str(review_json.get("state", "approved"))
    disp_key = "prior_plan_item_dispositions" if flow == "plan" else "prior_item_dispositions"
    raw_dispositions = review_json.get(disp_key, [])
    # Add reviewer field required by _deserialize_disposition
    for d in raw_dispositions:
        if "reviewer" not in d:
            d["reviewer"] = agent_cap

    blocking_key = "blocking_plan_issues" if flow == "plan" else "blocking_items"
    blocking_texts = list(review_json.get(blocking_key, []))
    same_key = "same_plan_followups" if flow == "plan" else "same_pr_followups"
    new_unresolved_texts = list(review_json.get(same_key, []))

    # When a reviewer blocks via same-round followups only (no explicit blocking issues),
    # surface those followups as blocking_items for the caller so the overall state is correct.
    reported_blocking = blocking_texts
    if parsed_state == "blocking" and not blocking_texts:
        reported_blocking = new_unresolved_texts

    # Serialize dispositions (with reviewer) for attach-metadata
    _write_json(dispositions_file, raw_dispositions)

    # Mint IDs that are globally unique across rounds by starting from item_id_offset.
    # item_id_offset = max numeric suffix seen across prior_items + already-minted round items.
    new_items_serialized: list[dict] = []
    item_counter = item_id_offset + 1
    for text in blocking_texts:
        new_items_serialized.append({
            "item_id": f"item-{item_counter}",
            "reviewer": agent_cap,
            "source_round": new_round_number,
            "text": str(text),
            "status": "blocking",
            "source_status": "blocking",
            "notes": [],
        })
        item_counter += 1
    for text in new_unresolved_texts:
        same_s = "same-plan" if flow == "plan" else "same-pr"
        new_items_serialized.append({
            "item_id": f"item-{item_counter}",
            "reviewer": agent_cap,
            "source_round": new_round_number,
            "text": str(text),
            "status": same_s,
            "source_status": same_s,
            "notes": [],
        })
        item_counter += 1
    _write_json(new_items_file, new_items_serialized)

    # --- Attach metadata ---
    attach_args = [
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
    ]
    attach_args += ["--subject", round_subject]
    _run_helper(*attach_args)

    # --- Post (skip when dry_run) ---
    if not dry_run:
        _run_helper(
            "helpers.state_manager", "write-pending-comment",
            "--issue", str(issue), "--repo", repo,
            "--body", str(tagged_output),
        )
        post_cmd = (
            "helpers.gh_ops", "post-issue-comment",
            "--issue", str(issue), "--file", str(tagged_output), "--repo", repo,
        )
        _run_helper(*post_cmd)
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
    }


# ---------------------------------------------------------------------------
# run-plan-round
# ---------------------------------------------------------------------------

def cmd_run_plan_round(args: argparse.Namespace) -> None:
    issue: int = args.issue
    repo: str = args.repo
    reviewers: list[str] = args.reviewers
    plan_file = Path(args.plan_file)
    dry_run: bool = args.dry_run

    if not plan_file.exists():
        print(f"skill_runner: plan file not found: {plan_file}", file=sys.stderr)
        sys.exit(1)

    # Step 1 — build_resume
    resume = _build_resume(issue, repo, reviewers, flow="plan")

    # Step 2 — pending-comment reconciliation
    _reconcile_pending_comment(resume, issue, repo, dry_run)

    # Step 3 — new-round vs resume detection
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

    # Step 4 — validate and post plan (new rounds only; resume skips this)
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

    # Step 5 — reconstruct round state from already-completed reviewers (RESUME path only).
    # In a new round, completed_reviewer_data belongs to the previous round and must not
    # contaminate the current round's result.
    current_round_items: list[dict] = []
    round_blocking_items: list[dict] = []
    round_approved_reviewers: list[str] = []
    any_reviewer_blocked = False
    if not is_new_round:
        for record in resume.get("completed_reviewer_data", []):
            current_round_items.extend(record.get("new_items", []))
            if record.get("state") == "blocking":
                any_reviewer_blocked = True
                round_blocking_items.extend(
                    {"text": item.get("text", "")}
                    for item in record.get("new_items", [])
                    if item.get("status") in ("blocking", "same-plan")
                )
            else:
                round_approved_reviewers.append(str(record.get("reviewer_name", "")))

    plan_text = plan_file.read_text(encoding="utf-8")

    # Step 6 — run pending reviewers
    with tempfile.TemporaryDirectory() as tmpstr:
        tmpdir = Path(tmpstr)
        for reviewer in reviewers:
            reviewer_cap = reviewer.capitalize() if reviewer in ("codex", "gemini") else reviewer
            if reviewer_cap in local_completed or reviewer in local_completed:
                print(f"[skip] {reviewer_cap} already completed this round")
                continue

            # Build prompt
            issue_dict = _fetch_issue_json(repo, issue)
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
            workdir = _workdir_for_agent(reviewer, args)
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
            current_round_items.extend(record["new_items"])
            if record["state"] == "blocking":
                any_reviewer_blocked = True
                round_blocking_items.extend(record["blocking_items"])
            else:
                round_approved_reviewers.append(record["reviewer_name"])

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

    # Step 8 — print result
    # Use any_reviewer_blocked to catch reviewers that blocked via same-plan followups only.
    overall_state = "blocking" if any_reviewer_blocked else "approved"
    result_json = {
        "state": overall_state,
        "round_number": new_round_number,
        "blocking_items": round_blocking_items,
        "approved_reviewers": round_approved_reviewers,
    }
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
    round_blocking_items: list[dict] = []
    round_approved_reviewers: list[str] = []
    any_reviewer_blocked = False
    if not is_new_round:
        for record in resume.get("completed_reviewer_data", []):
            current_round_items.extend(record.get("new_items", []))
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

    # Step 6 — run pending reviewers
    with tempfile.TemporaryDirectory() as tmpstr:
        tmpdir = Path(tmpstr)
        for reviewer in reviewers:
            reviewer_cap = reviewer.capitalize() if reviewer in ("codex", "gemini") else reviewer
            if reviewer_cap in local_completed or reviewer in local_completed:
                print(f"[skip] {reviewer_cap} already completed this round")
                continue

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
            workdir = _workdir_for_agent(reviewer, args)
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
            current_round_items.extend(record["new_items"])
            if record["state"] == "blocking":
                any_reviewer_blocked = True
                round_blocking_items.extend(record["blocking_items"])
            else:
                round_approved_reviewers.append(record["reviewer_name"])

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

    # Step 8 — print result
    overall_state = "blocking" if any_reviewer_blocked else "approved"
    result_json = {
        "state": overall_state,
        "round_number": new_round_number,
        "blocking_items": round_blocking_items,
        "approved_reviewers": round_approved_reviewers,
    }
    print(json.dumps(result_json, indent=2))


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
    p_plan.add_argument("--plan-file", required=True)
    p_plan.add_argument("--reviewers", nargs="+", required=True)
    p_plan.add_argument("--workdir-codex", default=None)
    p_plan.add_argument("--workdir-gemini", default=None)
    p_plan.add_argument("--workdir", default=None)
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
    p_pr.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()
    dispatch = {
        "run-plan-round": cmd_run_plan_round,
        "run-pr-round": cmd_run_pr_round,
    }
    dispatch[args.subcommand](args)


if __name__ == "__main__":
    main()
