"""
Manage Claude Code skill session state and build resume descriptors.

Subcommands:

  build-resume
    --issue N --repo REPO --reviewers REVIEWER [REVIEWER ...]
    [--flow plan|pr] [--head-sha SHA | --pr PR_NUMBER]

    Reads GitHub issue comments, extracts AGENT_LOOP_META records, and calls
    _resume_plan_round or _resume_pr_round from the existing library.  Outputs
    a JSON resume descriptor to stdout.

  attach-metadata
    --body-file PATH --output PATH
    --flow plan|pr --role coder|reviewer --agent NAME
    --round-number N --state approved|blocking
    (--subject SHA | --subject-plan-file PATH)
    [--prior-items-file PATH]
    [--dispositions-file PATH]
    [--new-items-file PATH]
    [--canonical-plan-file PATH]

    Reads the comment body from --body-file, builds a PostedRoundMetadata
    object, and writes the body with AGENT_LOOP_META appended to --output.
    The resulting file can be posted via gh_ops post-issue-comment and will
    be recognized by build-resume's _resume_plan_round / _resume_pr_round.

  write-session
    --issue N --repo REPO --fields JSON

    Writes (or merges) session state to
    ~/.local/state/coding-review-agent-loop/skill-sessions/{repo-slug}/{issue}.json

  read-session
    --issue N --repo REPO

    Reads session state JSON to stdout.

  write-pending-comment
    --issue N --repo REPO --body PATH

    Writes a pending comment body path to session state.

  clear-pending-comment
    --issue N --repo REPO

    Clears the pending_comment_body field from session state.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from coding_review_agent_loop.agents.base import AgentName
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.round_state import (
    PostedRoundMetadata,
    _attach_round_metadata,
    _deserialize_disposition,
    _deserialize_unresolved_item,
    _plan_subject,
    _resume_plan_round,
    _resume_pr_round,
    _serialize_disposition,
    _serialize_unresolved_item,
)
from coding_review_agent_loop.protocol import ReviewItemDisposition, UnresolvedReviewItem


def _session_path(repo: str, issue: int) -> Path:
    slug = repo.replace("/", "-").replace(":", "-")
    state_home = Path(
        os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state")
    )
    return state_home / "coding-review-agent-loop" / "skill-sessions" / slug / f"{issue}.json"


def _load_session(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_session(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


@dataclass
class _FakeComment:
    """Minimal object satisfying the comment.body duck-type expected by round_state helpers."""
    body: str


def _fetch_issue_comments(repo: str, issue: int, gh_cmd: str = "gh") -> list[_FakeComment]:
    result = subprocess.run(
        [
            gh_cmd,
            "api",
            f"repos/{repo}/issues/{issue}/comments",
            "--paginate",
            "--jq",
            ".[].body",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print(
            f"state_manager: gh api failed: {result.stderr.strip()}",
            file=sys.stderr,
        )
        sys.exit(1)
    bodies = result.stdout.strip().split("\n")
    return [_FakeComment(body=b) for b in bodies if b]


def _fetch_pr_head_sha(repo: str, pr_number: int, gh_cmd: str = "gh") -> str:
    result = subprocess.run(
        [
            gh_cmd,
            "pr",
            "view",
            str(pr_number),
            "--repo",
            repo,
            "--json",
            "headRefOid",
            "--jq",
            ".headRefOid",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print(
            f"state_manager: could not fetch PR head SHA: {result.stderr.strip()}",
            file=sys.stderr,
        )
        sys.exit(1)
    return result.stdout.strip()


def cmd_build_resume(args: argparse.Namespace) -> None:
    repo: str = args.repo
    issue: int = args.issue
    reviewers: list[AgentName] = args.reviewers
    flow: str = args.flow
    gh_cmd: str = getattr(args, "gh_cmd", "gh")

    comments = _fetch_issue_comments(repo, issue, gh_cmd=gh_cmd)
    session = _load_session(_session_path(repo, issue))

    descriptor: dict[str, object] = {
        "round_number": 1,
        "prior_items": [],
        "compact_prior_summaries": [],
        "completed_reviewer_names": [],
        "completed_reviewer_data": [],
        "completed_round_number": 0,
        "current_plan_subject": None,
        "pending_comment_body": session.get("pending_comment_body"),
    }

    try:
        if flow == "plan":
            result = _resume_plan_round(comments, configured_reviewers=reviewers)
            if result is not None:
                plan_text, resumed = result
                descriptor["round_number"] = resumed.round_number
                descriptor["prior_items"] = [
                    _serialize_unresolved_item(item) for item in resumed.prior_items
                ]
                descriptor["compact_prior_summaries"] = list(resumed.compact_prior_summaries)
                completed_names = [record.metadata.agent for record in resumed.completed_reviews]
                descriptor["completed_reviewer_names"] = completed_names
                descriptor["current_plan"] = plan_text
                descriptor["current_plan_subject"] = _plan_subject(plan_text)
                descriptor["completed_reviewer_data"] = [
                    {
                        "reviewer_name": record.metadata.agent,
                        "state": record.metadata.state or "approved",
                        "dispositions": [
                            _serialize_disposition(d) for d in record.metadata.dispositions
                        ],
                        "new_items": [
                            _serialize_unresolved_item(item)
                            for item in record.metadata.new_items
                        ],
                    }
                    for record in resumed.completed_reviews
                ]
                if len(completed_names) == len(reviewers):
                    descriptor["completed_round_number"] = resumed.round_number
                elif resumed.round_number > 1:
                    descriptor["completed_round_number"] = resumed.round_number - 1
        else:
            # PR flow
            head_sha: str | None = getattr(args, "head_sha", None)
            pr_number: int | None = getattr(args, "pr", None)
            if not head_sha and pr_number:
                head_sha = _fetch_pr_head_sha(repo, pr_number, gh_cmd=gh_cmd)
            result = _resume_pr_round(
                comments,
                head_sha=head_sha,
                configured_reviewers=reviewers,
            )
            if result is not None:
                descriptor["round_number"] = result.round_number
                descriptor["prior_items"] = [
                    _serialize_unresolved_item(item) for item in result.prior_items
                ]
                descriptor["compact_prior_summaries"] = list(result.compact_prior_summaries)
                completed_names = [record.metadata.agent for record in result.completed_reviews]
                descriptor["completed_reviewer_names"] = completed_names
                descriptor["current_plan_subject"] = head_sha
                descriptor["completed_reviewer_data"] = [
                    {
                        "reviewer_name": record.metadata.agent,
                        "state": record.metadata.state or "approved",
                        "dispositions": [
                            _serialize_disposition(d) for d in record.metadata.dispositions
                        ],
                        "new_items": [
                            _serialize_unresolved_item(item)
                            for item in record.metadata.new_items
                        ],
                    }
                    for record in result.completed_reviews
                ]
                if len(completed_names) == len(reviewers):
                    descriptor["completed_round_number"] = result.round_number
                elif result.round_number > 1:
                    descriptor["completed_round_number"] = result.round_number - 1
    except AgentLoopError as exc:
        print(f"state_manager: resume error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(json.dumps(descriptor, indent=2))


def _load_item_list(path: str | None) -> list[object]:
    if not path:
        return []
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"state_manager: cannot read {path}: {exc}", file=sys.stderr)
        sys.exit(1)


def cmd_attach_metadata(args: argparse.Namespace) -> None:
    try:
        body = Path(args.body_file).read_text(encoding="utf-8")
    except OSError as exc:
        print(f"state_manager: cannot read body file: {exc}", file=sys.stderr)
        sys.exit(1)

    # Compute subject
    if args.subject:
        subject = args.subject
    elif args.subject_plan_file:
        try:
            plan_text = Path(args.subject_plan_file).read_text(encoding="utf-8")
        except OSError as exc:
            print(f"state_manager: cannot read subject plan file: {exc}", file=sys.stderr)
            sys.exit(1)
        subject = _plan_subject(plan_text)
    else:
        print("state_manager attach-metadata: provide --subject or --subject-plan-file", file=sys.stderr)
        sys.exit(1)

    # Load optional item lists
    raw_prior = _load_item_list(args.prior_items_file)
    raw_dispositions = _load_item_list(args.dispositions_file)
    raw_new_items = _load_item_list(args.new_items_file)

    prior_items = tuple(_deserialize_unresolved_item(item) for item in raw_prior)
    dispositions = tuple(_deserialize_disposition(d) for d in raw_dispositions)
    new_items = tuple(_deserialize_unresolved_item(item) for item in raw_new_items)

    canonical_plan: str | None = None
    if args.canonical_plan_file:
        try:
            canonical_plan = Path(args.canonical_plan_file).read_text(encoding="utf-8")
        except OSError as exc:
            print(f"state_manager: cannot read canonical plan file: {exc}", file=sys.stderr)
            sys.exit(1)

    metadata = PostedRoundMetadata(
        flow=args.flow,
        role=args.role,
        agent=args.agent,
        round_number=args.round_number,
        subject=subject,
        prior_items=prior_items,
        dispositions=dispositions,
        new_items=new_items,
        state=args.state,
        canonical_plan=canonical_plan,
    )
    augmented = _attach_round_metadata(body, metadata)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(augmented, encoding="utf-8")
    print(f"metadata attached: {output_path}")


def cmd_write_session(args: argparse.Namespace) -> None:
    path = _session_path(args.repo, args.issue)
    existing = _load_session(path)
    try:
        updates: dict[str, object] = json.loads(args.fields)
    except json.JSONDecodeError as exc:
        print(f"state_manager: invalid --fields JSON: {exc}", file=sys.stderr)
        sys.exit(1)
    existing.update(updates)
    _save_session(path, existing)
    print(f"session written: {path}")


def cmd_read_session(args: argparse.Namespace) -> None:
    path = _session_path(args.repo, args.issue)
    data = _load_session(path)
    print(json.dumps(data, indent=2))


def cmd_write_pending_comment(args: argparse.Namespace) -> None:
    path = _session_path(args.repo, args.issue)
    existing = _load_session(path)
    existing["pending_comment_body"] = str(args.body)
    _save_session(path, existing)
    print(f"pending comment path written: {path}")


def cmd_clear_pending_comment(args: argparse.Namespace) -> None:
    path = _session_path(args.repo, args.issue)
    existing = _load_session(path)
    existing.pop("pending_comment_body", None)
    _save_session(path, existing)
    print(f"pending comment cleared: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage skill session state.")
    parser.add_argument("--gh-cmd", default="gh")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # attach-metadata
    p_meta = subparsers.add_parser(
        "attach-metadata",
        help="Attach AGENT_LOOP_META to a comment body so build-resume can reconstruct the round.",
    )
    p_meta.add_argument("--body-file", required=True, help="Input comment body file.")
    p_meta.add_argument("--output", required=True, help="Output file with AGENT_LOOP_META attached.")
    p_meta.add_argument("--flow", required=True, choices=["plan", "pr"])
    p_meta.add_argument("--role", required=True, choices=["coder", "reviewer"])
    p_meta.add_argument("--agent", required=True, help="Agent display name (e.g. 'Claude', 'Codex').")
    p_meta.add_argument("--round-number", type=int, required=True)
    p_meta.add_argument("--state", required=True, choices=["approved", "blocking"])
    p_meta.add_argument("--subject", default=None, help="Pre-computed subject SHA256 hex string.")
    p_meta.add_argument("--subject-plan-file", default=None,
                        help="Compute subject as sha256(plan text) from this file.")
    p_meta.add_argument("--prior-items-file", default=None,
                        help="JSON array of serialized UnresolvedReviewItem.")
    p_meta.add_argument("--dispositions-file", default=None,
                        help="JSON array of serialized ReviewItemDisposition.")
    p_meta.add_argument("--new-items-file", default=None,
                        help="JSON array of serialized UnresolvedReviewItem (new items from this turn).")
    p_meta.add_argument("--canonical-plan-file", default=None,
                        help="Plan text file (written as canonical_plan for coder turns).")

    # build-resume
    p_resume = subparsers.add_parser("build-resume", help="Build a resume descriptor from GitHub comments.")
    p_resume.add_argument("--issue", type=int, required=True)
    p_resume.add_argument("--repo", required=True)
    p_resume.add_argument("--reviewers", nargs="+", required=True)
    p_resume.add_argument("--flow", choices=["plan", "pr"], default="plan")
    p_resume.add_argument("--head-sha", default=None)
    p_resume.add_argument("--pr", type=int, default=None)

    # write-session
    p_write = subparsers.add_parser("write-session", help="Write session state fields.")
    p_write.add_argument("--issue", type=int, required=True)
    p_write.add_argument("--repo", required=True)
    p_write.add_argument("--fields", required=True, help="JSON object of fields to merge.")

    # read-session
    p_read = subparsers.add_parser("read-session", help="Read session state.")
    p_read.add_argument("--issue", type=int, required=True)
    p_read.add_argument("--repo", required=True)

    # write-pending-comment
    p_pending = subparsers.add_parser("write-pending-comment", help="Record a pending comment body path.")
    p_pending.add_argument("--issue", type=int, required=True)
    p_pending.add_argument("--repo", required=True)
    p_pending.add_argument("--body", required=True)

    # clear-pending-comment
    p_clear = subparsers.add_parser("clear-pending-comment", help="Clear the pending comment body path.")
    p_clear.add_argument("--issue", type=int, required=True)
    p_clear.add_argument("--repo", required=True)

    args = parser.parse_args()
    dispatch = {
        "attach-metadata": cmd_attach_metadata,
        "build-resume": cmd_build_resume,
        "write-session": cmd_write_session,
        "read-session": cmd_read_session,
        "write-pending-comment": cmd_write_pending_comment,
        "clear-pending-comment": cmd_clear_pending_comment,
    }
    dispatch[args.subcommand](args)


if __name__ == "__main__":
    main()
