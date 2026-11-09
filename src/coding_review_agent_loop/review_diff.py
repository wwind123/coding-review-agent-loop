"""The base ref every reviewer-facing diff instruction names (#1977).

Reviewers inspect ``git diff <ref>...HEAD``.  The bare local base branch may
be missing from the shared store, checked out in another worktree (so its
fast-forward is skipped) or lagging, which makes that diff stale or
unresolvable.  ``run_worktrees._fetch_and_pin`` writes the remote-tracking
``refs/remotes/origin/<base>`` into the shared store on every fetch, and
per-run worktrees share it, so prompts name ``origin/<base>`` instead.
"""

from __future__ import annotations

# Placeholder shown when the base branch is not resolved for this prompt.
UNRESOLVED_REVIEW_DIFF_BASE_REF = "origin/<base>"


def review_diff_base_ref(base: str) -> str:
    """Remote-tracking ref for ``base`` that resolves in every per-run worktree."""
    return f"origin/{base}"


def review_diff_target(base: str | None) -> str:
    """``origin/<base>...HEAD`` for a resolved base, else the placeholder form."""
    resolved = (base or "").strip()
    ref = review_diff_base_ref(resolved) if resolved else UNRESOLVED_REVIEW_DIFF_BASE_REF
    return f"{ref}...HEAD"
