"""Size-bound the PR-loop coder follow-up round comment (#2059).

A long-running PR can carry a ledger whose ordinary coder comment no longer
fits GitHub's body limit.  The coder's commit is already pushed when the
comment is posted, so failing the post loses the round record.  This helper
posts the ordinary rendering whenever it fits (byte-identical to the previous
behaviour) and otherwise steps through bounded projections of the visible body.
The complete coder response stays in the round metadata, spilled as usual; only
the visible projection shrinks.  When even the last bound level cannot fit, the
record becomes a claim-free orchestrator-synthesized fallback (#2086): the
ledger stays authoritative in ``prior_items`` (spilled as usual), the visible
item listing and diffstat are capped, and nonessential metadata is omitted.
The transport overflow error is only the internal fit probe and never escapes.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import MISSING, dataclass, fields, replace

from .errors import AgentLoopError
from .comment_rendering import MAX_CODER_BOUND_LEVEL, render_public_agent_comment
from .protocol_common import HTML_COMMENT_RE, SIGNATURE_RE
from .round_metadata import PostedRoundMetadata, _attach_round_metadata
from .round_transport import round_comment_fits

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
    # Set only for the synthesized fallback: the claim-free coder output the
    # caller adopts with ``metadata`` (#2086).
    coder_output: str | None = None


def admitted_evidence_heads(runner, config, pr_number: int, comments, *current_heads: str | None) -> tuple[str, ...]:
    """Every admitted PR head (and the given current heads) for evidence grouping.

    Uses the same authenticated subject source as evidence ingestion, never SHA
    mentions in untrusted prose.  Grouping is display-only, so an unavailable
    history degrades to the current heads instead of failing the post.
    """
    from .native_evidence import known_pr_heads, native_history

    current = tuple(head for head in current_heads if head)
    try:
        history = native_history(runner, config, pr_number, comments)
    except AgentLoopError as error:
        LOGGER.info("PR #%s: evidence grouping uses current heads only (%s)", pr_number, error)
        history = ()
    return tuple(dict.fromkeys((*current, *known_pr_heads(history))))


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


FALLBACK_LEVEL = MAX_CODER_BOUND_LEVEL + 1
# Nonessential, non-spillable metadata the synthesized fallback omits; the
# ledger, round, subject, checkpoint and identity fields are kept.
_FALLBACK_CLEARED_FIELDS = {
    "raw_structured_coder_response": None,
    "compact_prior_summaries": (),
    "step_back_entries": (),
    "local_test_evidence": None,
    "risk_test_matrix_evidence": None,
    "risk_test_matrix_evidence_full_round": None,
    "risk_test_matrix_diagnostics": (),
    "risk_test_matrix_capture_classification": None,
    "scheduler_reasons": (),
    "scheduler_paused_reviewers": (),
    "scheduler_selected_reviewers": (),
    "scheduler_approved_reviewers": (),
    "scheduler_active_owners": (),
    "architecture_impact_degradations": (),
    "test_observation_degradations": (),
    "addressed_sub_items": (),
    "sub_item_claim_degradations": (),
    "sub_item_degradations": (),
    # The discarded response's assessment: unbounded prose, never spilled.
    "architecture_impact": None,
}
# Fields the minimal fallback keeps when the first tier still cannot fit; every
# other field is reset to its default (always a valid, empty value).
_MINIMAL_FALLBACK_KEPT_FIELDS = frozenset({
    "flow", "role", "agent", "round_number", "subject", "prior_items", "state", "phase",
    "qualification_checkpoint", "followup_dispatch_head", "seat_binding",
    "pr_board_root", "pr_board_identity", "pr_board_binding_init",
    "reviewer_board_amendment_digest", "reviewer_board_change_digest",
    "architecture_identity", "architecture_contract_version", "evidence_stall",
    "coder_record_source", "coder_record_degradations", "human_requirements_degraded",
    "post_merge_registry", "post_merge_diagnostics", "model_used",
})
FALLBACK_NOTE = (
    "The coder record did not fit GitHub's comment size limit at any bound level; "
    "this claim-free record was synthesized by the orchestrator."
)


def _synthesized_fallback(
    metadata: PostedRoundMetadata,
    *,
    pr_number: int,
    head_sha: str | None,
    old_head: str | None,
    diffstat: Callable[[str], str] | None,
    human_requirements_in_force: bool,
    ledger: Callable[[Sequence[object]], Sequence[object]] | None = None,
) -> BoundedCoderComment:
    """The last bound level: a whole postable, claim-free synthesized record.

    ``ledger`` rolls back ledger annotations the discarded response introduced
    (#2086), so the fallback carries no claim of it.  Only a candidate that
    ``round_comment_fits`` is returned.
    """
    from .coder_followup_degradation import bound_diffstat_text, synthesize_coder_record_text

    notes = tuple(dict.fromkeys((*metadata.coder_record_degradations, FALLBACK_NOTE)))[-16:]
    prior_items = tuple(ledger(metadata.prior_items)) if ledger is not None else metadata.prior_items
    fallback = replace(
        metadata,
        **_FALLBACK_CLEARED_FIELDS,
        prior_items=prior_items,
        coder_record_source="synthesized",
        coder_record_degradations=notes,
        human_requirements_degraded=bool(
            metadata.human_requirements_degraded or human_requirements_in_force
        ),
    )
    item_ids = tuple(
        item.item_id for item in metadata.prior_items
        if item.status in {"blocking", "same-pr", "future"}
    )
    full_diffstat = "(diffstat unavailable)"
    if diffstat is not None and head_sha:
        try:
            full_diffstat = diffstat(head_sha)
        except Exception:  # noqa: BLE001 - display-only
            full_diffstat = "(diffstat unavailable)"
    attempts = ((2000, 40), (500, 10), (120, 3), (0, 0))
    for tier in (fallback, _minimal_fallback(fallback)):
        for id_chars, diff_lines in attempts:
            text = synthesize_coder_record_text(
                old_head=old_head, new_head=head_sha, item_ids=item_ids,
                diffstat=bound_diffstat_text(full_diffstat, diff_lines) if diff_lines else "(omitted)",
                reason="its record exceeded the GitHub comment size limit",
                id_listing_chars=id_chars,
            )
            candidate = _attach_round_metadata(text, tier, degradation_projection="compact")
            if round_comment_fits(candidate):
                LOGGER.info("PR #%s: coder round comment posted as the synthesized fallback", pr_number)
                return BoundedCoderComment(candidate, FALLBACK_LEVEL, tier, coder_output=text)
    # Only the authoritative ledger and record identity remain, and even they
    # cannot be transported: there is no postable record to return.
    raise AgentLoopError(
        f"PR #{pr_number}: even the minimal synthesized coder record cannot be transported "
        "(the ledger itself exceeds the spill bound)."
    )


def _minimal_fallback(metadata: PostedRoundMetadata) -> PostedRoundMetadata:
    """Reset every field outside the minimal kept set to its (valid) default."""
    changes: dict[str, object] = {}
    for field in fields(PostedRoundMetadata):
        if field.name in _MINIMAL_FALLBACK_KEPT_FIELDS or not field.init:
            continue
        if field.default is not MISSING:
            changes[field.name] = field.default
        elif field.default_factory is not MISSING:  # type: ignore[misc]
            changes[field.name] = field.default_factory()  # type: ignore[misc]
    return replace(metadata, **changes)


def bounded_coder_round_comment(
    full_body: str,
    metadata: PostedRoundMetadata,
    *,
    pr_number: int,
    head_sha: str | None,
    audit_note: str = "",
    render_kwargs: Mapping[str, object] | None = None,
    freeform_text: str | None = None,
    evidence_heads: Sequence[str] | Callable[[], Sequence[str]] = (),
    fallback_old_head: str | None = None,
    fallback_diffstat: Callable[[str], str] | None = None,
    human_requirements_in_force: bool = False,
    fallback_ledger: Callable[[Sequence[object]], Sequence[object]] | None = None,
) -> BoundedCoderComment:
    """Return the coder round comment to post, bounded only when it cannot fit.

    ``full_body`` is the ordinary visible rendering (without ``audit_note``).
    ``render_kwargs`` re-renders a structured follow-up at a bound level; for
    free-form output it is ``None`` and ``freeform_text`` is the original text.
    """
    full = _attach_round_metadata(_with_audit_note(audit_note, full_body, 0), metadata)
    if round_comment_fits(full):
        return BoundedCoderComment(full, 0, metadata)
    fallback = lambda: _synthesized_fallback(  # noqa: E731
        metadata, pr_number=pr_number, head_sha=head_sha, old_head=fallback_old_head,
        diffstat=fallback_diffstat, human_requirements_in_force=human_requirements_in_force,
        ledger=fallback_ledger,
    )
    if render_kwargs is None:
        return _bounded_freeform(
            full_body, metadata, pr_number=pr_number, head_sha=head_sha,
            audit_note=audit_note, freeform_text=freeform_text, fallback=fallback,
        )
    candidate = full
    heads: Sequence[str] = ()
    for level in range(1, MAX_CODER_BOUND_LEVEL + 1):
        if level == 2 and callable(evidence_heads):
            heads = tuple(evidence_heads())  # resolved only when grouping is needed
        elif level == 2:
            heads = tuple(evidence_heads)
        visible = render_public_agent_comment(
            **render_kwargs,  # type: ignore[arg-type]
            bound_level=level,
            head_sha=head_sha,
            evidence_heads=heads,
        )
        candidate = _attach_round_metadata(
            _with_audit_note(audit_note, visible, level),
            metadata,
            degradation_projection="full" if level < 3 else "compact",
        )
        if round_comment_fits(candidate):
            LOGGER.info("PR #%s: coder round comment posted at bound level %s", pr_number, level)
            return BoundedCoderComment(candidate, level, metadata)
    return fallback()


_FOOTER_LINE_CHARS = 200


def _split_trailer(body: str) -> tuple[list[str], str]:
    """Content lines and the required footer: the final signature line and at
    most one short state-marker comment directly above it.

    The footer is bounded by construction (at most two short lines); every
    other trailing line, whatever its shape or count, is ordinary content
    under the same budget.
    """
    lines = body.rstrip("\n").splitlines()
    index = len(lines)
    while index > 0 and not lines[index - 1].strip():
        index -= 1
    footer_start = index
    if index > 0 and len(lines[index - 1]) <= _FOOTER_LINE_CHARS and SIGNATURE_RE.match(lines[index - 1]):
        footer_start = index - 1
        above = footer_start
        while above > 0 and not lines[above - 1].strip():
            above -= 1
        if (
            above > 0
            and len(lines[above - 1]) <= _FOOTER_LINE_CHARS
            and HTML_COMMENT_RE.match(lines[above - 1])
        ):
            footer_start = above - 1
    trailer = "\n".join(line for line in lines[footer_start:index] if line.strip())
    return lines[:footer_start], trailer


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
    fallback: Callable[[], BoundedCoderComment],
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
        return fallback()
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
