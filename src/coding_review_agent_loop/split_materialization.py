"""Materialize discuss `split` proposals and plan `deferred_stages` into child issues (#476).

When a discuss consensus lands on `outcome: split`, or an approved plan-first
plan narrows scope via `deferred_stages`, the proposed follow-up work should
not remain text buried in issue comments: it should become concrete GitHub
issues linked back to the parent, so an implementation run that scopes down to
one stage cannot silently leave the rest unfiled.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass

from .config import AgentLoopConfig
from .decomposition import _decode_json_payload, _encode_json_payload, _issue_number_from_url
from .errors import AgentLoopError
from .github import FoundIssue, create_issue, post_issue_comment, search_issues
from .logging import log
from .protocol import DeferredStage
from .runner import Runner

MAX_SPLIT_CHILDREN = 8

DISCUSS_SPLIT_MARKER_RE = re.compile(
    r"<!--\s*AGENT_DISCUSS_SPLIT:\s*(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->",
    re.I,
)
SPLIT_CHILD_MARKER_RE = re.compile(
    r"<!--\s*AGENT_SPLIT_CHILD:\s*parent=(?P<parent>\d+)\s+key=(?P<key>[0-9a-f]{64})\s*-->",
    re.I,
)
SPLIT_STAGE_HANDOFF_MARKER_RE = re.compile(
    r"<!--\s*AGENT_SPLIT_STAGE_HANDOFF:\s*(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->",
    re.I,
)
UNFILED_SPLIT_WARNING_MARKER_RE = re.compile(
    r"<!--\s*AGENT_SPLIT_UNFILED_WARNING:\s*issue=(?P<issue>\d+)\s+subject=(?P<subject>\S+)\s*-->",
    re.I,
)


@dataclass(frozen=True)
class SplitStageProposal:
    """A candidate follow-up stage, normalized from either a discuss split
    proposal (free text) or a plan's declared `deferred_stages` entry."""

    title: str
    body: str
    key: str


@dataclass(frozen=True)
class MaterializedSplitChild:
    title: str
    key: str
    url: str | None
    number: int | None
    origin: str  # "created" or "adopted"


@dataclass(frozen=True)
class SplitMaterializationMetadata:
    parent_issue: int
    subject: str
    children: tuple[MaterializedSplitChild, ...]
    selected_stage: int | None = None


@dataclass(frozen=True)
class SplitStageHandoffMetadata:
    parent_issue: int
    plan_hash: str
    child_issue_number: int
    child_issue_url: str | None


def _normalize_stage_key(text: str) -> str:
    key = re.sub(r"`([^`]+)`", r"\1", text)
    key = re.sub(r"\*\*([^*]+)\*\*", r"\1", key)
    key = re.sub(r"[_*#>]+", " ", key)
    key = re.sub(r"[^\w\s]+", " ", key.lower())
    return " ".join(key.split())


def _stage_key(title: str) -> str:
    return hashlib.sha256(_normalize_stage_key(title).encode("utf-8")).hexdigest()


def split_stage_proposal_from_text(text: str) -> SplitStageProposal:
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), text.strip())
    title = " ".join(first_line.split())
    return SplitStageProposal(title=title, body=text.strip(), key=_stage_key(title))


def split_stage_proposal_from_deferred_stage(stage: DeferredStage) -> SplitStageProposal:
    return SplitStageProposal(title=stage.title, body=stage.summary, key=_stage_key(stage.title))


def dedupe_split_stage_proposals(
    proposals: Sequence[SplitStageProposal],
) -> list[SplitStageProposal]:
    seen: set[str] = set()
    deduped: list[SplitStageProposal] = []
    for proposal in proposals:
        if proposal.key in seen:
            continue
        seen.add(proposal.key)
        deduped.append(proposal)
    return deduped


def _child_issue_title(parent_issue: int, proposal: SplitStageProposal) -> str:
    return f"[#{parent_issue} stage] {proposal.title}"[:120]


def _strip_child_title_prefix(title: str, parent_issue: int) -> str:
    prefix = f"[#{parent_issue} stage] "
    return title[len(prefix):] if title.startswith(prefix) else title


def _sibling_lines(children: Sequence[MaterializedSplitChild]) -> list[str]:
    if not children:
        return ["- None yet."]
    lines = []
    for child in children:
        ref = child.url or (f"#{child.number}" if child.number is not None else "unavailable")
        lines.append(f"- {child.title}: {ref}")
    return lines


def _format_child_issue_body(
    *,
    parent_issue: int,
    proposal: SplitStageProposal,
    rationale: Sequence[tuple[str, str]],
    siblings_so_far: Sequence[MaterializedSplitChild],
) -> str:
    lines = [
        f"Part of #{parent_issue}",
        "",
        f"Child stage issue split out of parent #{parent_issue}.",
        "",
        "## Proposed scope",
        proposal.body,
        "",
        "## Split rationale from parent discussion",
    ]
    if rationale:
        for reviewer, text in rationale:
            lines.append(f"- {reviewer}: {text}")
    else:
        lines.append("- No structured rationale was recorded; see the parent issue discussion.")
    lines.extend(
        [
            "",
            "## Other stages split from the same parent",
            *_sibling_lines(siblings_so_far),
            "",
            "## Execution instructions",
            f"Run `agent-loop issue <this issue number>` to implement this stage in its own PR. "
            "Keep the PR scoped to this stage. Do not use closing keywords "
            f"(Closes/Fixes/Resolves) against parent #{parent_issue}; reference it with `Refs #{parent_issue}` "
            "instead so the parent stays open until every stage is implemented.",
            "",
            f"<!-- AGENT_SPLIT_CHILD: parent={parent_issue} key={proposal.key} -->",
            "-- coding-review-agent-loop",
        ]
    )
    return "\n".join(lines)


def _encode_split_metadata(metadata: SplitMaterializationMetadata) -> str:
    return _encode_json_payload(
        {
            "parent_issue": metadata.parent_issue,
            "subject": metadata.subject,
            "children": [
                {
                    "title": child.title,
                    "key": child.key,
                    "url": child.url,
                    "number": child.number,
                    "origin": child.origin,
                }
                for child in metadata.children
            ],
            "selected_stage": metadata.selected_stage,
        }
    )


def _decode_split_metadata(encoded: str) -> SplitMaterializationMetadata:
    payload = _decode_json_payload(encoded, marker_name="AGENT_DISCUSS_SPLIT")
    children_payload = payload.get("children")
    if not isinstance(children_payload, list):
        raise AgentLoopError("Invalid AGENT_DISCUSS_SPLIT payload.")
    children: list[MaterializedSplitChild] = []
    for child in children_payload:
        if not isinstance(child, dict) or not isinstance(child.get("title"), str):
            raise AgentLoopError("Invalid AGENT_DISCUSS_SPLIT payload.")
        number = child.get("number")
        children.append(
            MaterializedSplitChild(
                title=str(child["title"]),
                key=str(child.get("key") or _stage_key(str(child["title"]))),
                url=child.get("url") if isinstance(child.get("url"), str) else None,
                number=int(number) if isinstance(number, int) else None,
                origin=str(child.get("origin") or "created"),
            )
        )
    try:
        selected_stage = payload.get("selected_stage")
        return SplitMaterializationMetadata(
            parent_issue=int(payload["parent_issue"]),
            subject=str(payload["subject"]),
            children=tuple(children),
            selected_stage=int(selected_stage) if isinstance(selected_stage, int) else None,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise AgentLoopError("Invalid AGENT_DISCUSS_SPLIT payload.") from exc


def find_existing_split_materialization(
    comments: Sequence[object],
    *,
    parent_issue: int,
) -> SplitMaterializationMetadata | None:
    """Return the latest recorded split-materialization state for `parent_issue`.

    Each materialization comment records the FULL cumulative set of known
    children (not just newly created ones), so the latest marker for a parent
    is always a complete snapshot regardless of which `subject` produced it.
    This is what makes per-parent normalized-title dedupe hold even when the
    subject hash drifts between a discuss consensus and a later plan-first
    run (#476).
    """
    found: SplitMaterializationMetadata | None = None
    for comment in comments:
        body = getattr(comment, "body", None)
        if not isinstance(body, str):
            continue
        for match in DISCUSS_SPLIT_MARKER_RE.finditer(body):
            metadata = _decode_split_metadata(match.group("payload"))
            if metadata.parent_issue == parent_issue:
                found = metadata
    return found


def format_split_materialization_summary(
    *,
    parent_issue: int,
    metadata: SplitMaterializationMetadata,
) -> str:
    lines = [
        f"Split follow-up stages materialized for issue #{parent_issue}.",
        "",
        "| Stage | Child issue | Origin |",
        "| --- | --- | --- |",
    ]
    for child in metadata.children:
        ref = child.url or (f"#{child.number}" if child.number is not None else "Created issue URL unavailable from GitHub CLI output.")
        lines.append(f"| {child.title} | {ref} | {child.origin} |")
    lines.extend(
        [
            "",
            "Every stage above has a GitHub child issue linked back to this parent. "
            "Continue a stage with `agent-loop issue <child issue number>`.",
            "",
            f"<!-- AGENT_DISCUSS_SPLIT: {_encode_split_metadata(metadata)} -->",
            "-- coding-review-agent-loop",
        ]
    )
    return "\n".join(lines)


def _adopt_from_search(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    parent_issue: int,
    remaining: Sequence[SplitStageProposal],
) -> dict[str, FoundIssue]:
    """Crash-window recovery: look for children created before a prior run
    crashed between `create_issue` and posting the parent marker (#476)."""
    if not remaining:
        return {}
    results = search_issues(
        runner,
        config=config,
        search=f'"[#{parent_issue} stage]" in:title',
        state="all",
    )
    by_key: dict[str, FoundIssue] = {}
    remaining_keys = {proposal.key for proposal in remaining}
    for found in results:
        if not found.title:
            continue
        marker_match = SPLIT_CHILD_MARKER_RE.search(found.body or "")
        if marker_match and int(marker_match.group("parent")) == parent_issue:
            key = marker_match.group("key")
        else:
            # Fallback for results missing the AGENT_SPLIT_CHILD body marker
            # (e.g. `gh issue list --search` didn't return a body): titles are
            # created as `[#<parent> stage] <proposal title>`, so strip that
            # prefix before hashing to match a bare proposal key.
            key = _stage_key(_strip_child_title_prefix(found.title, parent_issue))
        if key in remaining_keys and key not in by_key:
            by_key[key] = found
    return by_key


def materialize_split_proposals(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    parent_issue: int,
    subject: str,
    proposals: Sequence[SplitStageProposal],
    rationale: Sequence[tuple[str, str]] = (),
    issue_comments: Sequence[object] = (),
) -> SplitMaterializationMetadata | None:
    """File one child issue per remaining stage in `proposals`, idempotently.

    `proposals` must already exclude the stage the current run is
    implementing (if any) — materialization only ever files REMAINING stages,
    never the selected one. Returns None when there is nothing to file.
    """
    deduped = dedupe_split_stage_proposals(proposals)
    if not deduped:
        return None

    existing = find_existing_split_materialization(issue_comments, parent_issue=parent_issue)
    known_by_key: dict[str, MaterializedSplitChild] = (
        {child.key: child for child in existing.children} if existing is not None else {}
    )
    to_resolve = [proposal for proposal in deduped if proposal.key not in known_by_key]

    if not to_resolve:
        return existing

    # Remaining capacity is derived from children already recorded in the
    # parent's cumulative metadata, not just this call's unresolved proposals,
    # so repeated reruns with the same over-cap proposal set cannot keep
    # filing new children past MAX_SPLIT_CHILDREN (#492 review).
    remaining_capacity = max(0, MAX_SPLIT_CHILDREN - len(known_by_key))
    capped = to_resolve[:remaining_capacity]
    skipped = to_resolve[remaining_capacity:]

    if not capped:
        skipped_titles = ", ".join(proposal.title for proposal in skipped)
        log(
            config,
            f"Split materialization for issue #{parent_issue}: skipped {len(skipped)} "
            f"stage(s) by cap (MAX_SPLIT_CHILDREN={MAX_SPLIT_CHILDREN}, already at cap): "
            f"{skipped_titles}",
        )
        return existing

    adopted_by_key = _adopt_from_search(
        runner, config=config, parent_issue=parent_issue, remaining=capped
    )

    new_children: list[MaterializedSplitChild] = []
    resolved_so_far: list[MaterializedSplitChild] = list(known_by_key.values())

    for proposal in capped:
        found = adopted_by_key.get(proposal.key)
        if found is not None:
            child = MaterializedSplitChild(
                title=proposal.title,
                key=proposal.key,
                url=found.url,
                number=found.number,
                origin="adopted",
            )
            new_children.append(child)
            resolved_so_far.append(child)
            continue
        issue_url = create_issue(
            runner,
            config=config,
            title=_child_issue_title(parent_issue, proposal),
            body=_format_child_issue_body(
                parent_issue=parent_issue,
                proposal=proposal,
                rationale=rationale,
                siblings_so_far=resolved_so_far,
            ),
        )
        child = MaterializedSplitChild(
            title=proposal.title,
            key=proposal.key,
            url=issue_url,
            number=_issue_number_from_url(issue_url),
            origin="created",
        )
        new_children.append(child)
        resolved_so_far.append(child)

    if skipped:
        skipped_titles = ", ".join(proposal.title for proposal in skipped)
        # Not filed; surfaced via log only, matching decomposition's cap note style.
        log(
            config,
            f"Split materialization for issue #{parent_issue}: skipped {len(skipped)} "
            f"stage(s) by cap (MAX_SPLIT_CHILDREN={MAX_SPLIT_CHILDREN}): {skipped_titles}",
        )

    all_children = tuple(known_by_key.values()) + tuple(new_children)
    metadata = SplitMaterializationMetadata(
        parent_issue=parent_issue,
        subject=subject,
        children=all_children,
        selected_stage=existing.selected_stage if existing is not None else None,
    )
    post_issue_comment(
        runner,
        config=config,
        issue_number=parent_issue,
        body=format_split_materialization_summary(parent_issue=parent_issue, metadata=metadata),
    )
    return metadata


def _encode_stage_handoff_metadata(metadata: SplitStageHandoffMetadata) -> str:
    return _encode_json_payload(
        {
            "parent_issue": metadata.parent_issue,
            "plan_hash": metadata.plan_hash,
            "child_issue_number": metadata.child_issue_number,
            "child_issue_url": metadata.child_issue_url,
        }
    )


def _decode_stage_handoff_metadata(encoded: str) -> SplitStageHandoffMetadata:
    payload = _decode_json_payload(encoded, marker_name="AGENT_SPLIT_STAGE_HANDOFF")
    try:
        child_issue_url = payload.get("child_issue_url")
        return SplitStageHandoffMetadata(
            parent_issue=int(payload["parent_issue"]),
            plan_hash=str(payload["plan_hash"]),
            child_issue_number=int(payload["child_issue_number"]),
            child_issue_url=child_issue_url if isinstance(child_issue_url, str) else None,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise AgentLoopError("Invalid AGENT_SPLIT_STAGE_HANDOFF payload.") from exc


def find_existing_split_stage_handoff(
    comments: Sequence[object],
    *,
    parent_issue: int,
    plan_hash: str,
) -> SplitStageHandoffMetadata | None:
    found: SplitStageHandoffMetadata | None = None
    for comment in comments:
        body = getattr(comment, "body", None)
        if not isinstance(body, str):
            continue
        for match in SPLIT_STAGE_HANDOFF_MARKER_RE.finditer(body):
            metadata = _decode_stage_handoff_metadata(match.group("payload"))
            if metadata.parent_issue == parent_issue and metadata.plan_hash == plan_hash:
                found = metadata
    return found


def resolve_selected_stage_child(
    children: Sequence[MaterializedSplitChild],
    *,
    parent_issue: int,
    plan_title_or_subject: str,
    split_stage_flag: int | None,
) -> MaterializedSplitChild:
    """Resolve which already-materialized child the approved plan implements.

    Every proposal was already filed as a child (discuss materialized
    everything up front), so an implement run on the parent must pick one:
    either the explicit `--split-stage <child>` flag, or a unique
    normalized-key match between the plan's subject/title and a child title.
    """
    if split_stage_flag is not None:
        for child in children:
            if child.number == split_stage_flag:
                return child
        raise AgentLoopError(
            f"--split-stage {split_stage_flag} does not match any child issue materialized "
            f"from parent #{parent_issue}. Known children: "
            + ", ".join(f"#{child.number}" for child in children if child.number is not None)
        )
    plan_key = _stage_key(plan_title_or_subject)
    matches = [child for child in children if child.key == plan_key]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise AgentLoopError(
            f"Could not resolve which materialized child stage of #{parent_issue} this approved "
            "plan implements (no title match). Run `agent-loop issue <child>` for the stage you "
            "want, or rerun with --split-stage <child>."
        )
    raise AgentLoopError(
        f"Ambiguous match between the approved plan and materialized child stages of #{parent_issue}. "
        "Run `agent-loop issue <child>` for the stage you want, or rerun with --split-stage <child>."
    )


def format_split_stage_handoff_comment(
    *,
    parent_issue: int,
    plan_hash: str,
    child: MaterializedSplitChild,
) -> str:
    if child.number is None:
        raise AgentLoopError(
            "Cannot record split-stage implementation handoff because the child issue number "
            "is unavailable."
        )
    metadata = SplitStageHandoffMetadata(
        parent_issue=parent_issue,
        plan_hash=plan_hash,
        child_issue_number=child.number,
        child_issue_url=child.url,
    )
    ref = child.url or f"#{child.number}"
    lines = [
        f"Approved plan implementation for issue #{parent_issue} handed off to split stage: {ref}.",
        "",
        f"Stage: {child.title}",
        "",
        "Parent reruns will not automatically re-run this child implementation. "
        f"Resume directly with `agent-loop issue {child.number}`.",
        "",
        f"<!-- AGENT_SPLIT_STAGE_HANDOFF: {_encode_stage_handoff_metadata(metadata)} -->",
        "-- coding-review-agent-loop",
    ]
    return "\n".join(lines)


def post_split_stage_handoff_comment(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    parent_issue: int,
    plan_hash: str,
    child: MaterializedSplitChild,
) -> None:
    post_issue_comment(
        runner,
        config=config,
        issue_number=parent_issue,
        body=format_split_stage_handoff_comment(
            parent_issue=parent_issue, plan_hash=plan_hash, child=child
        ),
    )


def format_unfiled_split_warning(proposals: Sequence[SplitStageProposal]) -> str:
    """Explicit UNFILED warning so the #467/#474 gap can't hide in a neutral listing (#476)."""
    lines = [
        "### Split follow-ups are NOT filed as issues",
        "",
        "The following stage(s) remain unfiled as GitHub issues:",
        "",
    ]
    lines.extend(f"- {proposal.title}" for proposal in proposals)
    lines.extend(
        [
            "",
            "Rerun with `--materialize-split-issues` to file them as child issues linked back to "
            "this parent, or file them manually. Without filing, this scope will not be tracked "
            "once the current stage's PR is implemented.",
        ]
    )
    return "\n".join(lines)


def _unfiled_split_warning_marker(issue_number: int, subject: str) -> str:
    return f"<!-- AGENT_SPLIT_UNFILED_WARNING: issue={issue_number} subject={subject} -->"


def has_unfiled_split_warning(
    comments: Sequence[object],
    *,
    issue_number: int,
    subject: str,
) -> bool:
    for comment in comments:
        body = getattr(comment, "body", None)
        if not isinstance(body, str):
            continue
        for match in UNFILED_SPLIT_WARNING_MARKER_RE.finditer(body):
            if int(match.group("issue")) == issue_number and match.group("subject") == subject:
                return True
    return False


def post_unfiled_split_warning(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    issue_number: int,
    subject: str,
    proposals: Sequence[SplitStageProposal],
) -> None:
    body = "\n".join(
        [
            format_unfiled_split_warning(proposals),
            "",
            _unfiled_split_warning_marker(issue_number, subject),
            "-- coding-review-agent-loop",
        ]
    )
    post_issue_comment(runner, config=config, issue_number=issue_number, body=body)
