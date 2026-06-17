"""
Render a validated structured agent response to human-readable markdown.

Usage:
  python -m helpers.render_response \\
    --file PATH --kind KIND --output PATH \\
    [--reviewer REVIEWER] \\
    [--context-file PATH]

Kinds: pr_review, plan_review, coder_followup, plan_revision, plan_state

Exit 0 on success; exit 1 with diagnostic on failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from coding_review_agent_loop.comment_rendering import (
    render_public_agent_comment,
)
from coding_review_agent_loop.protocol import (
    parse_pr_review,
    parse_plan_review,
    validate_structured_coder_followup,
    validate_structured_plan_state,
    validate_structured_plan_revision,
)
from coding_review_agent_loop.round_state import _deserialize_unresolved_item


def _load_context(path: str | None) -> dict:
    if path is None:
        return {}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"render_response: could not read context file: {exc}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a structured agent response to markdown.")
    parser.add_argument("--file", required=True, help="Path to the validated response file.")
    parser.add_argument(
        "--kind",
        required=True,
        choices=["pr_review", "plan_review", "coder_followup", "plan_revision", "plan_state"],
    )
    parser.add_argument("--output", required=True, help="Path to write rendered markdown.")
    parser.add_argument("--reviewer", default="Codex", help="Reviewer display name.")
    parser.add_argument("--context-file", default=None, help="Optional JSON context file.")
    parser.add_argument(
        "--model",
        default="",
        help="Model the agent actually ran, stamped into the signature (#332).",
    )
    args = parser.parse_args()
    model_used = args.model or None

    try:
        text = Path(args.file).read_text(encoding="utf-8")
    except OSError as exc:
        print(f"render_response: cannot read {args.file}: {exc}", file=sys.stderr)
        sys.exit(1)

    ctx = _load_context(args.context_file)
    prior_items = [_deserialize_unresolved_item(i) for i in ctx.get("prior_items", [])]

    try:
        if args.kind == "pr_review":
            parsed = parse_pr_review(text, reviewer=args.reviewer)
            rendered = render_public_agent_comment(
                kind="pr_review",
                parsed=parsed,
                agent=args.reviewer,
                prior_items=prior_items,
                dispositions=parsed.dispositions,
                model_used=model_used,
            )
        elif args.kind == "plan_review":
            parsed = parse_plan_review(text, reviewer=args.reviewer)
            rendered = render_public_agent_comment(
                kind="plan_review",
                parsed=parsed,
                agent=args.reviewer,
                prior_items=prior_items,
                dispositions=parsed.dispositions,
                model_used=model_used,
            )
        elif args.kind == "coder_followup":
            parsed = validate_structured_coder_followup(text)
            if parsed is None:
                print("render_response: coder_followup did not parse", file=sys.stderr)
                sys.exit(1)
            rendered = render_public_agent_comment(
                kind="coder_followup",
                parsed=parsed,
                agent=args.reviewer,
                prior_items=prior_items,
                model_used=model_used,
            )
        elif args.kind == "plan_revision":
            parsed = validate_structured_plan_revision(text)
            if parsed is None:
                print("render_response: plan_revision did not parse", file=sys.stderr)
                sys.exit(1)
            rendered = render_public_agent_comment(
                kind="plan_revision",
                parsed=parsed,
                agent=args.reviewer,
                prior_items=prior_items,
                raw_text=text,
                model_used=model_used,
            )
        elif args.kind == "plan_state":
            parsed = validate_structured_plan_state(text)
            if parsed is None:
                rendered = text
            else:
                rendered = render_public_agent_comment(
                    kind="plan_state",
                    parsed=parsed,
                    agent=args.reviewer,
                    model_used=model_used,
                )
        else:
            print(f"render_response: unknown kind {args.kind}", file=sys.stderr)
            sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        print(f"render_response: render failed: {exc}", file=sys.stderr)
        sys.exit(1)

    Path(args.output).write_text(rendered, encoding="utf-8")
    print(f"rendered: {args.output}")


if __name__ == "__main__":
    main()
