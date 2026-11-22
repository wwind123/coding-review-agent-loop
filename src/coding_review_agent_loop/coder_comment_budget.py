"""Size-bound the PR-loop coder follow-up round comment (#2059).

A long-running PR can carry a ledger whose ordinary coder comment no longer
fits GitHub's body limit.  The coder's commit is already pushed when the
comment is posted, so failing the post loses the round record.  This helper
posts the ordinary rendering whenever it fits (byte-identical to the previous
behaviour) and otherwise steps through bounded projections of the visible body.
The complete coder response stays in the round metadata, spilled as usual; only
the visible projection shrinks.  The transport overflow error remains the last
resort, raised only when the bounded fixed core plus metadata cannot fit, and it
then names the recovery command.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from .comment_rendering import MAX_CODER_BOUND_LEVEL, render_public_agent_comment
from .protocol_common import HTML_COMMENT_RE, SIGNATURE_RE
from .round_metadata import PostedRoundMetadata, _attach_round_metadata
from .round_transport import RoundCommentOverflowError, prepare_round_comment, round_comment_fits

LOGGER = logging.getLogger(__name__)

FREEFORM_OMISSION_NOTE = (
    "> Orchestrator note: the rest of this coder follow-up was omitted to fit GitHub's "
    "comment size limit; the full text is in the round metadata."
)
_AUDIT_NOTE_CHARS = {3: 500, 4: 200}


@dataclass(frozen=True)
class BoundedCoderComment:
    """The exact body to post, and the bound level that produced it (0 = unbounded)."""

    posted_body: str
    level: int
    metadata: PostedRoundMetadata


def recovery_suffix(*, pr_number: int, head_sha: str | None) -> str:
    commit = f"the coder's commit {head_sha} is pushed; " if head_sha else ""
    return f" {commit}recover with: agent-loop pr {pr_number} --review-unrecorded-head"


def _with_audit_note(audit_note: str, body: str, level: int) -> str:
    if not audit_note:
        return body
    limit = _AUDIT_NOTE_CHARS.get(level)
    note = audit_note
    if limit is not None and len(note) > limit:
        cut = note[: limit - 1]
        opener = cut.rfind("<!--")
        if opener != -1 and cut.rfind("-->") < opener:
            cut = cut[:opener]
        note = cut.rstrip() + "…"
    return f"{note}\n\n{body}"


def _raise_with_recovery(body: str, *, pr_number: int, head_sha: str | None) -> None:
    try:
        prepare_round_comment(body)
    except RoundCommentOverflowError as error:
        raise RoundCommentOverflowError(
            str(error) + recovery_suffix(pr_number=pr_number, head_sha=head_sha)
        ) from error
    raise RoundCommentOverflowError(  # pragma: no cover - fits() and prepare() agree
        "Coder round comment does not fit the GitHub body budget."
        + recovery_suffix(pr_number=pr_number, head_sha=head_sha)
    )


def bounded_coder_round_comment(
    full_body: str,
    metadata: PostedRoundMetadata,
    *,
    pr_number: int,
    head_sha: str | None,
    audit_note: str = "",
    render_kwargs: Mapping[str, object] | None = None,
    freeform_text: str | None = None,
    evidence_heads: Sequence[str] = (),
) -> BoundedCoderComment:
    """Return the coder round comment to post, bounded only when it cannot fit.

    ``full_body`` is the ordinary visible rendering (without ``audit_note``).
    ``render_kwargs`` re-renders a structured follow-up at a bound level; for
    free-form output it is ``None`` and ``freeform_text`` is the original text.
    """
    full = _attach_round_metadata(_with_audit_note(audit_note, full_body, 0), metadata)
    if round_comment_fits(full):
        return BoundedCoderComment(full, 0, metadata)
    if render_kwargs is None:
        return _bounded_freeform(
            full_body, metadata, pr_number=pr_number, head_sha=head_sha,
            audit_note=audit_note, freeform_text=freeform_text,
        )
    candidate = full
    for level in range(1, MAX_CODER_BOUND_LEVEL + 1):
        visible = render_public_agent_comment(
            **render_kwargs,  # type: ignore[arg-type]
            bound_level=level,
            head_sha=head_sha,
            evidence_heads=evidence_heads,
        )
        candidate = _attach_round_metadata(
            _with_audit_note(audit_note, visible, level),
            metadata,
            degradation_projection="full" if level < 3 else "compact",
        )
        if round_comment_fits(candidate):
            LOGGER.info("PR #%s: coder round comment posted at bound level %s", pr_number, level)
            return BoundedCoderComment(candidate, level, metadata)
    _raise_with_recovery(candidate, pr_number=pr_number, head_sha=head_sha)
    raise AssertionError("unreachable")  # pragma: no cover


def _split_trailer(body: str) -> tuple[list[str], str]:
    """Content lines and the trailing signature/marker block, as posted."""
    lines = body.rstrip("\n").splitlines()
    index = len(lines)
    while index > 0:
        candidate = lines[index - 1]
        if not candidate.strip() or HTML_COMMENT_RE.match(candidate) or SIGNATURE_RE.match(candidate):
            index -= 1
            continue
        break
    return lines[:index], "\n".join(lines[index:]).strip("\n")


def _close_comments(text: str) -> str:
    opener = text.rfind("<!--")
    if opener != -1 and text.rfind("-->") < opener:
        return text[:opener].rstrip()
    return text


def _bounded_freeform(
    full_body: str,
    metadata: PostedRoundMetadata,
    *,
    pr_number: int,
    head_sha: str | None,
    audit_note: str,
    freeform_text: str | None,
) -> BoundedCoderComment:
    """Keep whole leading lines, then an omission note and the signature."""
    if freeform_text is not None:
        # Store the complete original text in the spillable response field, so
        # resume yields it as the coder output rather than the shortened body.
        metadata = replace(metadata, raw_structured_coder_response=freeform_text)
    content, trailer = _split_trailer(full_body)

    def build(kept: int, cut: int | None = None) -> str:
        lines = list(content[:kept])
        if cut is not None:
            lines.append(content[kept][:cut].rstrip() + "…")
        text = _close_comments("\n".join(lines).rstrip())
        visible = "\n\n".join(part for part in (text, FREEFORM_OMISSION_NOTE, trailer) if part)
        return _attach_round_metadata(_with_audit_note(audit_note, visible, 4), metadata)

    low, high = 0, len(content)
    if not round_comment_fits(build(0)):
        _raise_with_recovery(build(0), pr_number=pr_number, head_sha=head_sha)
    while low < high:
        middle = (low + high + 1) // 2
        if round_comment_fits(build(middle)):
            low = middle
        else:
            high = middle - 1
    best = build(low)
    if low == 0 and content:
        # The first line alone does not fit: cut it at a character boundary.
        cut_low, cut_high = 0, len(content[0]) - 1
        while cut_low < cut_high:
            middle = (cut_low + cut_high + 1) // 2
            if round_comment_fits(build(0, middle)):
                cut_low = middle
            else:
                cut_high = middle - 1
        if cut_low > 0:
            best = build(0, cut_low)
    LOGGER.info("PR #%s: free-form coder round comment posted bounded", pr_number)
    return BoundedCoderComment(best, MAX_CODER_BOUND_LEVEL, metadata)
