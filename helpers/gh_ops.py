"""
GitHub CLI wrapper for skill orchestration.

Subcommands:

  fetch-issue   --issue N --repo REPO
  post-issue-comment --issue N --file PATH --repo REPO [--dry-run]
  fetch-pr      --pr N --repo REPO
  post-pr-comment --pr N --file PATH --repo REPO [--dry-run]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _gh(*args_list: str, gh_cmd: str = "gh") -> str:
    result = subprocess.run(
        [gh_cmd, *args_list],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print(f"gh_ops: gh error: {result.stderr.strip()}", file=sys.stderr)
        sys.exit(1)
    return result.stdout


def cmd_fetch_issue(args: argparse.Namespace) -> None:
    output = _gh(
        "issue",
        "view",
        str(args.issue),
        "--repo",
        args.repo,
        "--json",
        "number,title,body,comments,state",
        gh_cmd=args.gh_cmd,
    )
    print(output, end="")


def cmd_post_issue_comment(args: argparse.Namespace) -> None:
    try:
        body = Path(args.file).read_text(encoding="utf-8")
    except OSError as exc:
        print(f"gh_ops: cannot read comment file: {exc}", file=sys.stderr)
        sys.exit(1)
    if args.dry_run:
        print(f"[dry-run] would post issue comment to {args.repo}#{args.issue}:")
        print(body[:400])
        return
    _gh(
        "issue",
        "comment",
        str(args.issue),
        "--repo",
        args.repo,
        "--body-file",
        args.file,
        gh_cmd=args.gh_cmd,
    )
    print(f"comment posted to {args.repo}#{args.issue}")


def cmd_fetch_pr(args: argparse.Namespace) -> None:
    output = _gh(
        "pr",
        "view",
        str(args.pr),
        "--repo",
        args.repo,
        "--json",
        "number,title,body,headRefOid,state,comments",
        gh_cmd=args.gh_cmd,
    )
    print(output, end="")


def cmd_post_pr_comment(args: argparse.Namespace) -> None:
    try:
        body = Path(args.file).read_text(encoding="utf-8")
    except OSError as exc:
        print(f"gh_ops: cannot read comment file: {exc}", file=sys.stderr)
        sys.exit(1)
    if args.dry_run:
        print(f"[dry-run] would post PR comment to {args.repo}#{args.pr}:")
        print(body[:400])
        return
    _gh(
        "pr",
        "comment",
        str(args.pr),
        "--repo",
        args.repo,
        "--body-file",
        args.file,
        gh_cmd=args.gh_cmd,
    )
    print(f"comment posted to {args.repo}#PR{args.pr}")


def main() -> None:
    parser = argparse.ArgumentParser(description="GitHub CLI wrapper for skill orchestration.")
    parser.add_argument("--gh-cmd", default="gh")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    p_fi = subparsers.add_parser("fetch-issue")
    p_fi.add_argument("--issue", type=int, required=True)
    p_fi.add_argument("--repo", required=True)

    p_pic = subparsers.add_parser("post-issue-comment")
    p_pic.add_argument("--issue", type=int, required=True)
    p_pic.add_argument("--file", required=True)
    p_pic.add_argument("--repo", required=True)
    p_pic.add_argument("--dry-run", action="store_true")

    p_fp = subparsers.add_parser("fetch-pr")
    p_fp.add_argument("--pr", type=int, required=True)
    p_fp.add_argument("--repo", required=True)

    p_ppc = subparsers.add_parser("post-pr-comment")
    p_ppc.add_argument("--pr", type=int, required=True)
    p_ppc.add_argument("--file", required=True)
    p_ppc.add_argument("--repo", required=True)
    p_ppc.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()
    dispatch = {
        "fetch-issue": cmd_fetch_issue,
        "post-issue-comment": cmd_post_issue_comment,
        "fetch-pr": cmd_fetch_pr,
        "post-pr-comment": cmd_post_pr_comment,
    }
    dispatch[args.subcommand](args)


if __name__ == "__main__":
    main()
