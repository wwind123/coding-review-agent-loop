"""Approved-plan decomposition parsing and publishing helpers."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import zlib
from collections.abc import Sequence
from dataclasses import dataclass

from .config import AgentLoopConfig
from .child_topology import (
    NeedsHumanDecision,
    merge_found_issues,
    parent_child_search_queries,
    preflight_flat_child_count,
)
from .errors import AgentLoopError
from .github import FoundIssue, create_issue, post_issue_comment, search_issues
from .runner import Runner
from .protocol_markers import TrustedBody, sanitize_historical_text
from .round_transport import MAX_GITHUB_BODY_CHARS

AUTOMATION_CLASSES = {"agent-pr", "human-action", "manual-close"}
DECOMPOSITION_MARKER_RE = re.compile(
    r"<!--\s*AGENT_PLAN_DECOMPOSITION:\s*(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->",
    re.I,
)
PHASE_IMPLEMENTATION_MARKER_RE = re.compile(
    r"<!--\s*AGENT_PLAN_PHASE_IMPLEMENTATION:\s*(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->",
    re.I,
)
TOPOLOGY_CHECKPOINT_MARKER_RE = re.compile(
    r"<!--\s*AGENT_PLAN_TOPOLOGY_CHECKPOINT:\s*(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->",
    re.I,
)
PHASE_IDENTITY_MARKER_RE = re.compile(
    r"<!--\s*AGENT_PLAN_PHASE_IDENTITY:\s*(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->",
    re.I,
)
LEGACY_SPLIT_IDENTITY_RE = re.compile(
    r"<!--\s*AGENT_SPLIT_CHILD:\s*parent=(?P<parent>\d+)\s+key=(?P<key>[0-9a-f]{64})\s*-->",
    re.I,
)
ONE_SHOT_IMPL_HANDOFF_MARKER_RE = re.compile(
    r"<!--\s*AGENT_PLAN_ONE_SHOT_IMPL:\s*(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->",
    re.I,
)
ISSUE_NUMBER_RE = re.compile(r"/issues/(\d+)(?:\b|$)|#(\d+)\b")


@dataclass(frozen=True)
class PlanPhase:
    title: str
    scope: str
    non_goals: str
    dependency_notes: str
    rollout_risk: str
    validation: str
    parent_context: str
    automation: str
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlanDecomposition:
    phases: tuple[PlanPhase, ...]


@dataclass(frozen=True)
class RecordedPhase:
    title: str
    automation: str
    rollout_risk: str = "recorded"
    parent_context: str | None = None


@dataclass(frozen=True)
class CreatedPhaseIssue:
    phase: PlanPhase | RecordedPhase
    issue_url: str | None
    issue_number: int | None
    origin: str = "created"


@dataclass(frozen=True)
class RetainedParentScope:
    plan_subject: str
    plan_hash: str
    excerpt: str


@dataclass(frozen=True)
class DecompositionMetadata:
    parent_issue: int
    plan_hash: str
    mode: str
    phase_count: int
    phase_titles: tuple[str, ...]
    automation: tuple[str, ...]
    children: tuple[tuple[str, str | None, int | None], ...]
    topology_source: str = "model"
    retained_parent_scope: RetainedParentScope | None = None


@dataclass(frozen=True)
class TopologyCheckpoint:
    parent_issue: int
    plan_hash: str
    mode: str
    topology_source: str
    phases: tuple[PlanPhase, ...]
    retained_parent_scope: RetainedParentScope | None = None


@dataclass(frozen=True)
class PhaseImplementationHandoffMetadata:
    parent_issue: int
    plan_hash: str
    mode: str
    phase_index: int
    phase_title: str
    automation: str
    child_issue_number: int
    child_issue_url: str | None


@dataclass(frozen=True)
class OneShotImplementationHandoffMetadata:
    parent_issue: int
    plan_hash: str
    plan_subject: str
    mode: str
    pr_number: int
    pr_head_sha: str | None


def approved_plan_hash(approved_plan: str) -> str:
    return hashlib.sha256(approved_plan.strip().encode("utf-8")).hexdigest()[:16]


def _extract_json_object(text: str) -> dict[str, object]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^\s*```(?:json)?\s*", "", stripped, count=1, flags=re.I)
        stripped = re.sub(r"\s*```\s*$", "", stripped, count=1)
    decoder = json.JSONDecoder()
    try:
        payload, end = decoder.raw_decode(stripped)
    except json.JSONDecodeError as exc:
        raise AgentLoopError(f"Invalid plan decomposition JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise AgentLoopError("Invalid plan decomposition JSON: top-level payload must be an object.")
    trailing = stripped[end:].strip()
    if trailing and not trailing.startswith("<!--"):
        raise AgentLoopError("Invalid plan decomposition JSON: unexpected text after JSON payload.")
    return payload


def _required_text(payload: dict[str, object], key: str, *, phase_title: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise AgentLoopError(f"Invalid plan decomposition: phase {phase_title!r} is missing `{key}`.")
    return value.strip()


def parse_plan_decomposition(text: str) -> PlanDecomposition:
    payload = _extract_json_object(text)
    if payload.get("kind") not in (None, "plan_decomposition"):
        raise AgentLoopError("Invalid plan decomposition: `kind` must be `plan_decomposition`.")
    phases_payload = payload.get("phases")
    if not isinstance(phases_payload, list) or not phases_payload:
        raise AgentLoopError("Invalid plan decomposition: `phases` must be a non-empty list.")
    title_keys = [
        " ".join(phase_payload["title"].lower().split())
        for phase_payload in phases_payload
        if isinstance(phase_payload, dict)
        and isinstance(phase_payload.get("title"), str)
        and phase_payload["title"].strip()
    ]
    all_title_keys = set(title_keys)
    phases: list[PlanPhase] = []
    seen_titles: set[str] = set()
    for index, phase_payload in enumerate(phases_payload, start=1):
        if not isinstance(phase_payload, dict):
            raise AgentLoopError(f"Invalid plan decomposition: phase {index} must be an object.")
        title = _required_text(phase_payload, "title", phase_title=f"#{index}")
        title_key = " ".join(title.lower().split())
        if title_key in seen_titles:
            raise AgentLoopError(f"Invalid plan decomposition: duplicate phase title {title!r}.")
        seen_titles.add(title_key)
        automation = _required_text(phase_payload, "automation", phase_title=title)
        if automation not in AUTOMATION_CLASSES:
            raise AgentLoopError(
                f"Invalid plan decomposition: phase {title!r} has invalid automation {automation!r}."
            )
        depends_on_payload = phase_payload.get("depends_on", [])
        if depends_on_payload is None:
            depends_on_payload = []
        if not isinstance(depends_on_payload, list) or not all(
            isinstance(value, str) and value.strip() for value in depends_on_payload
        ):
            raise AgentLoopError(
                f"Invalid plan decomposition: phase {title!r} `depends_on` must be a list of titles."
            )
        for dependency in depends_on_payload:
            dependency_key = " ".join(dependency.lower().split())
            if dependency_key == title_key:
                raise AgentLoopError(
                    f"Invalid plan decomposition: phase {title!r} cannot depend on itself."
                )
            if dependency_key not in all_title_keys:
                raise AgentLoopError(
                    f"Invalid plan decomposition: phase {title!r} depends on unknown phase {dependency!r}."
                )
            if dependency_key not in seen_titles:
                raise AgentLoopError(
                    f"Invalid plan decomposition: phase {title!r} depends on {dependency!r}, "
                    "but dependencies must reference an earlier phase."
                )
        phases.append(
            PlanPhase(
                title=title,
                scope=_required_text(phase_payload, "scope", phase_title=title),
                non_goals=_required_text(phase_payload, "non_goals", phase_title=title),
                dependency_notes=_required_text(phase_payload, "dependency_notes", phase_title=title),
                rollout_risk=_required_text(phase_payload, "rollout_risk", phase_title=title),
                validation=_required_text(phase_payload, "validation", phase_title=title),
                parent_context=_required_text(phase_payload, "parent_context", phase_title=title),
                automation=automation,
                depends_on=tuple(value.strip() for value in depends_on_payload),
            )
        )
    return PlanDecomposition(phases=tuple(phases))


def _issue_number_from_url(issue_url: str | None) -> int | None:
    if not issue_url:
        return None
    match = ISSUE_NUMBER_RE.search(issue_url)
    if not match:
        return None
    return int(match.group(1) or match.group(2))


def _phase_issue_title(parent_issue: int, index: int, phase: PlanPhase) -> str:
    prefix = "[Human] " if phase.automation in {"human-action", "manual-close"} else ""
    return f"{prefix}Phase {index}: {phase.title} (from #{parent_issue})"[:120]


def _phase_payload(phase: PlanPhase) -> dict[str, object]:
    return {
        "title": phase.title,
        "scope": phase.scope,
        "non_goals": phase.non_goals,
        "dependency_notes": phase.dependency_notes,
        "rollout_risk": phase.rollout_risk,
        "validation": phase.validation,
        "parent_context": phase.parent_context,
        "automation": phase.automation,
        "depends_on": list(phase.depends_on),
    }


def phase_identity(
    *,
    parent_issue: int,
    plan_hash: str,
    topology_source: str,
    phase_index: int,
    phase: PlanPhase,
) -> str:
    """Return a stable identity independent of the display title truncation."""
    material = {
        "parent_issue": parent_issue,
        "plan_hash": plan_hash,
        "source": topology_source,
        "stage_id": phase_index,
        "phase": _phase_payload(phase),
    }
    encoded = json.dumps(material, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _phase_identity_marker(identity: str, *, parent_issue: int, plan_hash: str, source: str, index: int) -> str:
    return f"<!-- AGENT_PLAN_PHASE_IDENTITY: {_encode_json_payload({'identity': identity, 'parent_issue': parent_issue, 'plan_hash': plan_hash, 'source': source, 'stage_id': index})} -->"


def adapt_typed_child_stages(
    stages: Sequence[object],
    *,
    approved_plan: str,
    plan_subject: str,
) -> tuple[PlanDecomposition, RetainedParentScope]:
    """Adapt the two-field typed remainder into the decomposition contract."""
    excerpt = sanitize_historical_text(approved_plan.strip())
    retained = RetainedParentScope(
        plan_subject=sanitize_historical_text(plan_subject),
        plan_hash=approved_plan_hash(approved_plan),
        excerpt=excerpt,
    )
    phases = tuple(
        PlanPhase(
            title=str(stage.title),
            scope=str(stage.summary),
            non_goals=(
                "No stage-specific non-goals were declared; refer to the neutralized "
                "parent constraints above."
            ),
            dependency_notes=(
                "No stage-specific dependency notes were declared; this typed stage "
                "has no explicit inter-stage dependency."
            ),
            rollout_risk=(
                "No stage-specific rollout risk was declared; follow the neutralized "
                "parent constraints above."
            ),
            validation=(
                "No stage-specific validation state was declared; follow the parent "
                "plan's validation requirements."
            ),
            parent_context=excerpt,
            automation="agent-pr",
            depends_on=(),
        )
        for stage in stages
    )
    return PlanDecomposition(phases=phases), retained


def _checkpoint_payload(checkpoint: TopologyCheckpoint) -> dict[str, object]:
    contexts = tuple(phase.parent_context for phase in checkpoint.phases)
    shared_context = (
        contexts[0]
        if contexts and all(context == contexts[0] for context in contexts)
        else None
    )
    phase_payloads: list[dict[str, object]] = []
    for phase in checkpoint.phases:
        payload = _phase_payload(phase)
        if shared_context is not None:
            payload.pop("parent_context", None)
        phase_payloads.append(payload)
    retained_payload = None
    if checkpoint.retained_parent_scope is not None:
        retained_payload = {
            "plan_subject": checkpoint.retained_parent_scope.plan_subject,
            "plan_hash": checkpoint.retained_parent_scope.plan_hash,
            "excerpt": checkpoint.retained_parent_scope.excerpt,
        }
        # Typed phases and the retained parent scope deliberately share this
        # excerpt. Keep it in one place in the checkpoint marker.
        if (
            shared_context is not None
            and checkpoint.retained_parent_scope.excerpt == shared_context
        ):
            retained_payload.pop("excerpt")
    return {
        "parent_issue": checkpoint.parent_issue,
        "plan_hash": checkpoint.plan_hash,
        "mode": checkpoint.mode,
        "topology_source": checkpoint.topology_source,
        "shared_parent_context": shared_context,
        "phases": phase_payloads,
        "retained_parent_scope": retained_payload,
    }


def _phase_from_payload(payload: object, *, shared_parent_context: str | None = None) -> PlanPhase:
    if not isinstance(payload, dict):
        raise AgentLoopError("Invalid AGENT_PLAN_TOPOLOGY_CHECKPOINT payload.")
    depends = payload.get("depends_on", [])
    if not isinstance(depends, list):
        raise AgentLoopError("Invalid AGENT_PLAN_TOPOLOGY_CHECKPOINT payload.")
    values = {}
    for key in (
        "title",
        "scope",
        "non_goals",
        "dependency_notes",
        "rollout_risk",
        "validation",
        "automation",
    ):
        value = payload.get(key)
        if not isinstance(value, str):
            raise AgentLoopError("Invalid AGENT_PLAN_TOPOLOGY_CHECKPOINT payload.")
        values[key] = value
    parent_context = payload.get("parent_context", shared_parent_context)
    if not isinstance(parent_context, str):
        raise AgentLoopError("Invalid AGENT_PLAN_TOPOLOGY_CHECKPOINT payload.")
    values["parent_context"] = parent_context
    return PlanPhase(**values, depends_on=tuple(str(item) for item in depends))


def _decode_checkpoint(encoded: str) -> TopologyCheckpoint:
    try:
        if encoded.startswith("v1_"):
            raw = zlib.decompress(
                base64.urlsafe_b64decode(encoded[3:].encode("ascii"))
            )
            payload = json.loads(raw.decode("utf-8"))
        else:
            # Read the original uncompressed representation for checkpoints
            # already posted before the compact payload format was introduced.
            payload = _decode_json_payload(
                encoded, marker_name="AGENT_PLAN_TOPOLOGY_CHECKPOINT"
            )
    except (ValueError, json.JSONDecodeError, zlib.error) as exc:
        raise AgentLoopError("Invalid AGENT_PLAN_TOPOLOGY_CHECKPOINT payload.") from exc
    if not isinstance(payload, dict):
        raise AgentLoopError("Invalid AGENT_PLAN_TOPOLOGY_CHECKPOINT payload.")
    phases_payload = payload.get("phases")
    if not isinstance(phases_payload, list) or not phases_payload:
        raise AgentLoopError("Invalid AGENT_PLAN_TOPOLOGY_CHECKPOINT payload.")
    retained_payload = payload.get("retained_parent_scope")
    shared_parent_context = payload.get("shared_parent_context")
    if shared_parent_context is not None and not isinstance(shared_parent_context, str):
        raise AgentLoopError("Invalid AGENT_PLAN_TOPOLOGY_CHECKPOINT payload.")
    retained = None
    if retained_payload is not None:
        if not isinstance(retained_payload, dict):
            raise AgentLoopError("Invalid AGENT_PLAN_TOPOLOGY_CHECKPOINT payload.")
        retained = RetainedParentScope(
            plan_subject=str(retained_payload.get("plan_subject") or ""),
            plan_hash=str(retained_payload.get("plan_hash") or ""),
            excerpt=str(retained_payload.get("excerpt") or shared_parent_context or ""),
        )
    try:
        return TopologyCheckpoint(
            parent_issue=int(payload["parent_issue"]),
            plan_hash=str(payload["plan_hash"]),
            mode=str(payload["mode"]),
            topology_source=str(payload["topology_source"]),
            phases=tuple(
                _phase_from_payload(item, shared_parent_context=shared_parent_context)
                for item in phases_payload
            ),
            retained_parent_scope=retained,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise AgentLoopError("Invalid AGENT_PLAN_TOPOLOGY_CHECKPOINT payload.") from exc


def find_existing_topology_checkpoint(
    comments: Sequence[object], *, parent_issue: int, plan_hash: str, mode: str
) -> TopologyCheckpoint | None:
    found: TopologyCheckpoint | None = None
    for comment in comments:
        body = getattr(comment, "body", None)
        if not isinstance(body, str):
            continue
        for match in TOPOLOGY_CHECKPOINT_MARKER_RE.finditer(body):
            checkpoint = _decode_checkpoint(match.group("payload"))
            if (
                checkpoint.parent_issue == parent_issue
                and checkpoint.plan_hash == plan_hash
                and checkpoint.mode == mode
            ):
                found = checkpoint
    return found


def format_topology_checkpoint(checkpoint: TopologyCheckpoint) -> str:
    raw = json.dumps(
        _checkpoint_payload(checkpoint),
        separators=(",", ":"),
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    encoded = "v1_" + base64.urlsafe_b64encode(zlib.compress(raw, 9)).decode("ascii")
    body = "\n".join(
        [
            f"Topology checkpoint recorded for issue #{checkpoint.parent_issue}.",
            "",
            f"Source: {checkpoint.topology_source}",
            f"Mode: {checkpoint.mode}",
            f"Stages: {len(checkpoint.phases)}",
            "",
            f"<!-- AGENT_PLAN_TOPOLOGY_CHECKPOINT: {encoded} -->",
            "-- coding-review-agent-loop",
        ]
    )
    if len(body) > MAX_GITHUB_BODY_CHARS:
        raise AgentLoopError(
            "Topology checkpoint exceeds the GitHub comment size limit after compact encoding; "
            "shorten the approved plan or use a smaller flat topology."
        )
    return body


def post_topology_checkpoint(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    checkpoint: TopologyCheckpoint,
) -> None:
    post_issue_comment(
        runner,
        config=config,
        issue_number=checkpoint.parent_issue,
        body=TrustedBody.canonical(
            format_topology_checkpoint(checkpoint),
            expected_tokens=("AGENT_PLAN_TOPOLOGY_CHECKPOINT",),
        ),
    )


def _dependency_lines(
    phase: PlanPhase,
    created: Sequence[CreatedPhaseIssue],
    *,
    placeholders: bool = False,
) -> list[str]:
    if not phase.depends_on:
        return ["- None."]
    by_title = {
        " ".join(item.phase.title.lower().split()): item
        for item in created
    }
    lines: list[str] = []
    for dependency in phase.depends_on:
        created_issue = by_title.get(" ".join(dependency.lower().split()))
        if created_issue and created_issue.issue_number is not None:
            lines.append(f"- depends on #{created_issue.issue_number}: {dependency}")
        elif created_issue and created_issue.issue_url:
            lines.append(f"- depends on {created_issue.issue_url}: {dependency}")
        elif created_issue:
            if placeholders:
                lines.append(f"- depends on __ORCHESTRATOR_ISSUE_NUMBER__: {dependency}")
            else:
                lines.append(f"- depends on previously created phase with unavailable issue URL: {dependency}")
        else:
            lines.append(f"- depends on prior phase: {dependency}")
    return lines


def _unresolved_dependencies(
    phase: PlanPhase,
    created: Sequence[CreatedPhaseIssue],
) -> tuple[str, ...]:
    by_title = {
        " ".join(item.phase.title.lower().split()): item
        for item in created
    }
    return tuple(
        dependency
        for dependency in phase.depends_on
        if (
            (item := by_title.get(" ".join(dependency.lower().split()))) is None
            or (item.issue_number is None and not item.issue_url)
        )
    )


def format_phase_issue_body(
    *,
    repo: str,
    parent_issue: int,
    approved_plan: str,
    phase: PlanPhase,
    created_so_far: Sequence[CreatedPhaseIssue],
    phase_identity_value: str | None = None,
    dependency_placeholders: bool = False,
    topology_source: str = "model",
    phase_index: int = 0,
    phase_plan_hash: str | None = None,
) -> str:
    parent_url = f"https://github.com/{repo}/issues/{parent_issue}"
    if phase.automation == "agent-pr":
        execution = (
            "Run `agent-loop issue <this issue number>` to implement this phase in its own PR. "
            "Keep the PR scoped to this phase."
        )
    elif phase.automation == "human-action":
        execution = (
            "This phase requires human action before agent implementation continues. A human should perform "
            "the work, add a remark/update describing the result, and close this issue."
        )
    else:
        execution = (
            "This phase is a manual closure/checkpoint. A human should add the required remark/update and "
            "close this issue when the checkpoint is satisfied."
        )
    body = "\n".join(
        [
            f"Child phase issue for parent #{parent_issue}: {parent_url}",
            "",
            "## Approved parent-plan excerpt for this phase",
            sanitize_historical_text(phase.parent_context),
            "",
            "## Scope",
            phase.scope,
            "",
            "## Non-goals",
            phase.non_goals,
            "",
            "## Constraints and invariants from the parent plan",
            "The linked parent issue is the source of truth for the complete approved-plan constraints and invariants; the excerpt above is the phase-specific context supplied to this issue.",
            "",
            "## Dependency notes",
            phase.dependency_notes,
            "",
            "## Dependency links",
            *_dependency_lines(phase, created_so_far, placeholders=dependency_placeholders),
            "",
            "## Rollout risk",
            phase.rollout_risk,
            "",
            "## Validation / soak requirement",
            phase.validation,
            "",
            "## Automation classification",
            phase.automation,
            "",
            "## Execution instructions",
            execution,
        ]
    )
    if phase_identity_value is not None:
        body += "\n\n" + _phase_identity_marker(
            phase_identity_value,
            parent_issue=parent_issue,
            plan_hash=phase_plan_hash or approved_plan_hash(approved_plan),
            source=topology_source,
            index=phase_index,
        )
    return body


def create_decomposition_child_issues(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    parent_issue: int,
    approved_plan: str,
    decomposition: PlanDecomposition,
    topology_source: str = "model",
    issue_comments: Sequence[object] = (),
    mode: str = "decompose-only",
    retained_parent_scope: RetainedParentScope | None = None,
) -> tuple[CreatedPhaseIssue, ...] | NeedsHumanDecision:
    """Preflight, recover, and create one immutable decomposition topology."""
    plan_hash = approved_plan_hash(approved_plan)
    phases = tuple(decomposition.phases)
    if not phases:
        raise AgentLoopError("Plan decomposition produced no phases.")

    # Search is read-only and intentionally includes every issue state.  It
    # closes the create-before-summary crash window without trusting authorship.
    found = merge_found_issues(
        search_issues(
            runner,
            config=config,
            search=query,
            state="all",
        )
        for query in parent_child_search_queries(parent_issue)
    )
    expected_ids = {
        index: phase_identity(
            parent_issue=parent_issue,
            plan_hash=plan_hash,
            topology_source=topology_source,
            phase_index=index,
            phase=phase,
        )
        for index, phase in enumerate(phases, start=1)
    }
    exact: dict[str, FoundIssue] = {}
    recognized: set[str] = set()
    for candidate in found:
        marker = PHASE_IDENTITY_MARKER_RE.search(candidate.body or "")
        candidate_identity: str | None = None
        candidate_parent: int | None = None
        candidate_plan_hash: str | None = None
        candidate_source: str | None = None
        candidate_stage_id: int | None = None
        if marker:
            payload = _decode_json_payload(marker.group("payload"), marker_name="AGENT_PLAN_PHASE_IDENTITY")
            if isinstance(payload.get("identity"), str):
                candidate_identity = payload["identity"]
            if isinstance(payload.get("parent_issue"), int) and not isinstance(payload.get("parent_issue"), bool):
                candidate_parent = payload["parent_issue"]
            if isinstance(payload.get("plan_hash"), str):
                candidate_plan_hash = payload["plan_hash"]
            if isinstance(payload.get("source"), str):
                candidate_source = payload["source"]
            if isinstance(payload.get("stage_id"), int) and not isinstance(payload.get("stage_id"), bool):
                candidate_stage_id = payload["stage_id"]
        if candidate_identity is not None and candidate_parent == parent_issue:
            recognized.add(candidate_identity)
            for index, expected in expected_ids.items():
                if candidate_identity == expected:
                    if (
                        candidate_plan_hash != plan_hash
                        or candidate_source != topology_source
                        or candidate_stage_id != index
                    ):
                        raise AgentLoopError(
                            f"Invalid decomposition recovery identity metadata for phase {index}."
                        )
                    if expected in exact:
                        raise AgentLoopError(
                            f"Ambiguous decomposition recovery: multiple child issues carry identity {expected}."
                        )
                    exact[expected] = candidate
        else:
            legacy = LEGACY_SPLIT_IDENTITY_RE.search(candidate.body or "")
            if legacy and int(legacy.group("parent")) == parent_issue:
                recognized.add("legacy:" + legacy.group("key").lower())
            elif candidate.body and f"#{parent_issue}" in candidate.body and candidate.title:
                # Count a parent-linked canonical child from another workflow
                # toward the parent budget, without adopting it as a desired
                # decomposition phase.
                recognized.add("linked:" + " ".join(candidate.title.casefold().split()))
        if not candidate.body and candidate.title:
            # Some GitHub search responses omit bodies.  The generated parent
            # prefixed title is a canonical recovery key in that narrow case.
            for index, phase in enumerate(phases, start=1):
                if candidate.title == _phase_issue_title(parent_issue, index, phase):
                    identity = expected_ids[index]
                    if identity in exact:
                        raise AgentLoopError(
                            f"Ambiguous decomposition recovery: multiple title matches for phase {index}."
                        )
                    exact[identity] = candidate
                    recognized.add(identity)

    count = preflight_flat_child_count(
        parent_issue=parent_issue,
        source=topology_source,
        desired_keys=expected_ids.values(),
        recognized_keys=recognized,
        configured_limit=config.flat_child_limit,
    )
    if isinstance(count, NeedsHumanDecision):
        return count

    # Validate every title and body before any checkpoint or create.  Dependency
    # slots are orchestrator-owned placeholders and can only be replaced later
    # by adopted/created issue references.
    empty_prior = [
        CreatedPhaseIssue(phase=phase, issue_url=None, issue_number=None)
        for phase in phases
    ]
    for index, phase in enumerate(phases, start=1):
        title = _phase_issue_title(parent_issue, index, phase)
        TrustedBody.current_untrusted_visible(title)
        draft = format_phase_issue_body(
            repo=config.repo,
            parent_issue=parent_issue,
            approved_plan=approved_plan,
            phase=phase,
            created_so_far=empty_prior[: index - 1],
            phase_identity_value=expected_ids[index],
            dependency_placeholders=True,
            topology_source=topology_source,
            phase_index=index,
            phase_plan_hash=plan_hash,
        )
        TrustedBody.canonical(draft, expected_tokens=("AGENT_PLAN_PHASE_IDENTITY",))

    # The checkpoint is the durable handoff between complete preflight and
    # child creation.  It is deliberately not posted for dry-run previews.
    if not config.dry_run and find_existing_topology_checkpoint(
        issue_comments,
        parent_issue=parent_issue,
        plan_hash=plan_hash,
        mode=mode,
    ) is None:
        post_topology_checkpoint(
            runner,
            config=config,
            checkpoint=TopologyCheckpoint(
                parent_issue=parent_issue,
                plan_hash=plan_hash,
                mode=mode,
                topology_source=topology_source,
                phases=phases,
                retained_parent_scope=retained_parent_scope,
            ),
        )

    created: list[CreatedPhaseIssue] = []
    for index, phase in enumerate(phases, start=1):
        identity = expected_ids[index]
        found = exact.get(identity)
        if found is not None:
            created.append(
                CreatedPhaseIssue(
                    phase=phase,
                    issue_url=found.url,
                    issue_number=found.number or _issue_number_from_url(found.url),
                    origin="adopted",
                )
            )
            continue
        unresolved = () if config.dry_run else _unresolved_dependencies(phase, created)
        if unresolved:
            raise AgentLoopError(
                "Cannot create decomposition phase because dependency issue references are unavailable: "
                + ", ".join(unresolved)
            )
        title = _phase_issue_title(parent_issue, index, phase)
        body = format_phase_issue_body(
            repo=config.repo,
            parent_issue=parent_issue,
            approved_plan=approved_plan,
            phase=phase,
            created_so_far=created,
            phase_identity_value=identity,
            topology_source=topology_source,
            phase_index=index,
            phase_plan_hash=plan_hash,
        )
        if "__ORCHESTRATOR_ISSUE_NUMBER__" in body:
            raise AgentLoopError(
                "Cannot create decomposition phase because a dependency reference was not resolved."
            )
        issue_url = create_issue(
            runner,
            config=config,
            title=title,
            body=TrustedBody.canonical(
                body,
                expected_tokens=("AGENT_PLAN_PHASE_IDENTITY",),
            ),
        )
        if config.dry_run:
            # A test double may return a URL even though the real dry-run
            # Runner intentionally returns no remote issue reference.
            issue_url = None
        created.append(
            CreatedPhaseIssue(
                phase=phase,
                issue_url=issue_url,
                issue_number=_issue_number_from_url(issue_url),
                origin="created",
            )
        )
    return tuple(created)


def _encode_json_payload(payload: dict[str, object]) -> str:
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii")


def _decode_json_payload(encoded: str, *, marker_name: str) -> dict[str, object]:
    try:
        payload = json.loads(base64.urlsafe_b64decode(encoded.encode("ascii")).decode("utf-8"))
    except (ValueError, json.JSONDecodeError) as exc:
        raise AgentLoopError(f"Invalid {marker_name} payload.") from exc
    if not isinstance(payload, dict):
        raise AgentLoopError(f"Invalid {marker_name} payload.")
    return payload


def _encode_metadata(metadata: DecompositionMetadata) -> str:
    return _encode_json_payload(
        {
            "parent_issue": metadata.parent_issue,
            "plan_hash": metadata.plan_hash,
            "mode": metadata.mode,
            "phase_count": metadata.phase_count,
            "phase_titles": list(metadata.phase_titles),
            "automation": list(metadata.automation),
            "children": [
                {"title": title, "url": url, "number": number}
                for title, url, number in metadata.children
            ],
            "topology_source": metadata.topology_source,
            "retained_parent_scope": (
                {
                    "plan_subject": metadata.retained_parent_scope.plan_subject,
                    "plan_hash": metadata.retained_parent_scope.plan_hash,
                    "excerpt": metadata.retained_parent_scope.excerpt,
                }
                if metadata.retained_parent_scope is not None
                else None
            ),
        }
    )


def _decode_metadata(encoded: str) -> DecompositionMetadata:
    payload = _decode_json_payload(encoded, marker_name="AGENT_PLAN_DECOMPOSITION")
    children_payload = payload.get("children")
    if not isinstance(children_payload, list):
        raise AgentLoopError("Invalid AGENT_PLAN_DECOMPOSITION payload.")
    children: list[tuple[str, str | None, int | None]] = []
    for child in children_payload:
        if not isinstance(child, dict) or not isinstance(child.get("title"), str):
            raise AgentLoopError("Invalid AGENT_PLAN_DECOMPOSITION payload.")
        url = child.get("url")
        number = child.get("number")
        children.append(
            (
                child["title"],
                url if isinstance(url, str) else None,
                number if isinstance(number, int) else None,
            )
        )
    try:
        retained_payload = payload.get("retained_parent_scope")
        retained = None
        if isinstance(retained_payload, dict):
            retained = RetainedParentScope(
                plan_subject=str(retained_payload.get("plan_subject") or ""),
                plan_hash=str(retained_payload.get("plan_hash") or ""),
                excerpt=str(retained_payload.get("excerpt") or ""),
            )
        return DecompositionMetadata(
            parent_issue=int(payload["parent_issue"]),
            plan_hash=str(payload["plan_hash"]),
            mode=str(payload["mode"]),
            phase_count=int(payload["phase_count"]),
            phase_titles=tuple(str(value) for value in payload["phase_titles"]),
            automation=tuple(str(value) for value in payload["automation"]),
            children=tuple(children),
            topology_source=str(payload.get("topology_source") or "model"),
            retained_parent_scope=retained,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise AgentLoopError("Invalid AGENT_PLAN_DECOMPOSITION payload.") from exc


def _encode_phase_implementation_handoff_metadata(
    metadata: PhaseImplementationHandoffMetadata,
) -> str:
    return _encode_json_payload(
        {
            "parent_issue": metadata.parent_issue,
            "plan_hash": metadata.plan_hash,
            "mode": metadata.mode,
            "phase_index": metadata.phase_index,
            "phase_title": metadata.phase_title,
            "automation": metadata.automation,
            "child_issue_number": metadata.child_issue_number,
            "child_issue_url": metadata.child_issue_url,
        }
    )


def _decode_phase_implementation_handoff_metadata(encoded: str) -> PhaseImplementationHandoffMetadata:
    payload = _decode_json_payload(encoded, marker_name="AGENT_PLAN_PHASE_IMPLEMENTATION")
    try:
        child_issue_url = payload.get("child_issue_url")
        return PhaseImplementationHandoffMetadata(
            parent_issue=int(payload["parent_issue"]),
            plan_hash=str(payload["plan_hash"]),
            mode=str(payload["mode"]),
            phase_index=int(payload["phase_index"]),
            phase_title=str(payload["phase_title"]),
            automation=str(payload["automation"]),
            child_issue_number=int(payload["child_issue_number"]),
            child_issue_url=child_issue_url if isinstance(child_issue_url, str) else None,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise AgentLoopError("Invalid AGENT_PLAN_PHASE_IMPLEMENTATION payload.") from exc


def find_existing_decomposition(
    comments: Sequence[object],
    *,
    parent_issue: int,
    plan_hash: str,
    mode: str,
) -> DecompositionMetadata | None:
    found: DecompositionMetadata | None = None
    for comment in comments:
        body = getattr(comment, "body", None)
        if not isinstance(body, str):
            continue
        for match in DECOMPOSITION_MARKER_RE.finditer(body):
            metadata = _decode_metadata(match.group("payload"))
            if (
                metadata.parent_issue == parent_issue
                and metadata.plan_hash == plan_hash
                and metadata.mode == mode
            ):
                found = metadata
    if found is None:
        return None
    if found.phase_count != len(found.children):
        known = ", ".join(url or f"#{number}" for _title, url, number in found.children if url or number)
        raise AgentLoopError(
            "Existing plan decomposition metadata is incomplete; manual recovery required before rerun. "
            f"Known child issues: {known or 'none'}."
        )
    return found


def find_existing_phase_implementation_handoff(
    comments: Sequence[object],
    *,
    parent_issue: int,
    plan_hash: str,
    mode: str,
    phase_index: int,
    child_issue_number: int,
) -> PhaseImplementationHandoffMetadata | None:
    found: PhaseImplementationHandoffMetadata | None = None
    for comment in comments:
        body = getattr(comment, "body", None)
        if not isinstance(body, str):
            continue
        for match in PHASE_IMPLEMENTATION_MARKER_RE.finditer(body):
            metadata = _decode_phase_implementation_handoff_metadata(match.group("payload"))
            if (
                metadata.parent_issue == parent_issue
                and metadata.plan_hash == plan_hash
                and metadata.mode == mode
                and metadata.phase_index == phase_index
                and metadata.child_issue_number == child_issue_number
            ):
                found = metadata
    return found


def format_decomposition_parent_summary(
    *,
    parent_issue: int,
    mode: str,
    plan_hash: str,
    created: Sequence[CreatedPhaseIssue],
    topology_source: str = "model",
    retained_parent_scope: RetainedParentScope | None = None,
) -> str:
    metadata = DecompositionMetadata(
        parent_issue=parent_issue,
        plan_hash=plan_hash,
        mode=mode,
        phase_count=len(created),
        phase_titles=tuple(item.phase.title for item in created),
        automation=tuple(item.phase.automation for item in created),
        children=tuple(
            (item.phase.title, item.issue_url, item.issue_number)
            for item in created
        ),
        topology_source=topology_source,
        retained_parent_scope=retained_parent_scope,
    )
    lines = [
        f"Approved plan decomposed for issue #{parent_issue}.",
        "",
        f"Mode: {mode}",
        f"Topology source: {topology_source}",
        "",
    ]
    if retained_parent_scope is not None:
        lines.extend(
            [
                "## Retained parent scope",
                f"Plan subject: {sanitize_historical_text(retained_parent_scope.plan_subject)}",
                f"Plan hash: {retained_parent_scope.plan_hash}",
                "The approved plan's primary scope remains owned by the parent; "
                "the typed stages below are its declared remainder.",
                "",
                retained_parent_scope.excerpt,
                "",
            ]
        )
    lines.extend(
        [
            "| Phase | Automation | Child issue | Risk |",
            "| --- | --- | --- | --- |",
        ]
    )
    for index, item in enumerate(created, start=1):
        if item.issue_url:
            child = item.issue_url
        elif item.issue_number is not None:
            child = f"#{item.issue_number}"
        else:
            child = "Created issue URL unavailable from GitHub CLI output."
        human_note = " Human remark and closure required." if item.phase.automation != "agent-pr" else ""
        lines.append(
            f"| {index}. {item.phase.title} | {item.phase.automation} | {child} | {item.phase.rollout_risk}{human_note} |"
        )
    lines.extend(
        [
            "",
            "Every phase above has a GitHub child issue; this table is only a summary.",
            "",
            f"<!-- AGENT_PLAN_DECOMPOSITION: {_encode_metadata(metadata)} -->",
            "-- coding-review-agent-loop",
        ]
    )
    return "\n".join(lines)


def format_phase_implementation_handoff_comment(
    *,
    parent_issue: int,
    mode: str,
    plan_hash: str,
    phase_index: int,
    created: CreatedPhaseIssue,
) -> str:
    if created.issue_number is None:
        raise AgentLoopError(
            "Cannot record decomposed phase implementation handoff because the child issue number is unavailable."
        )
    metadata = PhaseImplementationHandoffMetadata(
        parent_issue=parent_issue,
        plan_hash=plan_hash,
        mode=mode,
        phase_index=phase_index,
        phase_title=created.phase.title,
        automation=created.phase.automation,
        child_issue_number=created.issue_number,
        child_issue_url=created.issue_url,
    )
    child = created.issue_url or f"#{created.issue_number}"
    lines = [
        f"Approved plan implementation for issue #{parent_issue} handed off to phase {phase_index}: {child}.",
        "",
        f"Mode: {mode}",
        f"Phase: {created.phase.title}",
        f"Automation: {created.phase.automation}",
        "",
        "Parent reruns will not automatically re-run this child implementation. "
        f"Resume directly with `agent-loop issue {created.issue_number}`.",
        "",
        f"<!-- AGENT_PLAN_PHASE_IMPLEMENTATION: {_encode_phase_implementation_handoff_metadata(metadata)} -->",
        "-- coding-review-agent-loop",
    ]
    return "\n".join(lines)


def post_phase_implementation_handoff_comment(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    parent_issue: int,
    mode: str,
    plan_hash: str,
    phase_index: int,
    created: CreatedPhaseIssue,
) -> None:
    post_issue_comment(
        runner,
        config=config,
        issue_number=parent_issue,
        body=TrustedBody.canonical(
            format_phase_implementation_handoff_comment(
                parent_issue=parent_issue,
                mode=mode,
                plan_hash=plan_hash,
                phase_index=phase_index,
                created=created,
            ),
            expected_tokens=("AGENT_PLAN_PHASE_IMPLEMENTATION",),
        ),
    )


def post_decomposition_parent_summary(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    parent_issue: int,
    mode: str,
    plan_hash: str,
    created: Sequence[CreatedPhaseIssue],
    topology_source: str = "model",
    retained_parent_scope: RetainedParentScope | None = None,
) -> None:
    post_issue_comment(
        runner,
        config=config,
        issue_number=parent_issue,
        body=TrustedBody.canonical(
            format_decomposition_parent_summary(
                parent_issue=parent_issue,
                mode=mode,
                plan_hash=plan_hash,
                created=created,
                topology_source=topology_source,
                retained_parent_scope=retained_parent_scope,
            ),
            expected_tokens=("AGENT_PLAN_DECOMPOSITION",),
        ),
    )


def _encode_one_shot_impl_handoff_metadata(
    metadata: OneShotImplementationHandoffMetadata,
) -> str:
    return _encode_json_payload(
        {
            "parent_issue": metadata.parent_issue,
            "plan_hash": metadata.plan_hash,
            "plan_subject": metadata.plan_subject,
            "mode": metadata.mode,
            "pr_number": metadata.pr_number,
            "pr_head_sha": metadata.pr_head_sha,
        }
    )


def _decode_one_shot_impl_handoff_metadata(encoded: str) -> OneShotImplementationHandoffMetadata:
    payload = _decode_json_payload(encoded, marker_name="AGENT_PLAN_ONE_SHOT_IMPL")
    try:
        pr_head_sha = payload.get("pr_head_sha")
        return OneShotImplementationHandoffMetadata(
            parent_issue=int(payload["parent_issue"]),
            plan_hash=str(payload["plan_hash"]),
            plan_subject=str(payload.get("plan_subject") or ""),
            mode=str(payload["mode"]),
            pr_number=int(payload["pr_number"]),
            pr_head_sha=pr_head_sha if isinstance(pr_head_sha, str) else None,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise AgentLoopError("Invalid AGENT_PLAN_ONE_SHOT_IMPL payload.") from exc


def find_existing_one_shot_impl_handoff(
    comments: Sequence[object],
    *,
    parent_issue: int,
    plan_hash: str,
    mode: str,
) -> OneShotImplementationHandoffMetadata | None:
    found: OneShotImplementationHandoffMetadata | None = None
    for comment in comments:
        body = getattr(comment, "body", None)
        if not isinstance(body, str):
            continue
        for match in ONE_SHOT_IMPL_HANDOFF_MARKER_RE.finditer(body):
            metadata = _decode_one_shot_impl_handoff_metadata(match.group("payload"))
            if (
                metadata.parent_issue == parent_issue
                and metadata.plan_hash == plan_hash
                and metadata.mode == mode
            ):
                found = metadata
    return found


def find_latest_one_shot_impl_handoff(
    comments: Sequence[object], *, parent_issue: int, mode: str
) -> OneShotImplementationHandoffMetadata | None:
    """Return the latest valid one-shot handoff regardless of plan hash.

    Plan-first recovery uses this to distinguish an open handoff for an older
    plan from an absent handoff; silently ignoring the former would create a
    duplicate implementation PR.
    """
    found: OneShotImplementationHandoffMetadata | None = None
    for comment in comments:
        body = getattr(comment, "body", None)
        if not isinstance(body, str):
            continue
        for match in ONE_SHOT_IMPL_HANDOFF_MARKER_RE.finditer(body):
            metadata = _decode_one_shot_impl_handoff_metadata(match.group("payload"))
            if metadata.parent_issue == parent_issue and metadata.mode == mode:
                found = metadata
    return found


def format_one_shot_impl_handoff_comment(
    *,
    parent_issue: int,
    mode: str,
    plan_hash: str,
    plan_subject: str,
    pr_number: int,
    pr_head_sha: str | None,
) -> str:
    metadata = OneShotImplementationHandoffMetadata(
        parent_issue=parent_issue,
        plan_hash=plan_hash,
        plan_subject=plan_subject,
        mode=mode,
        pr_number=pr_number,
        pr_head_sha=pr_head_sha,
    )
    lines = [
        f"Approved plan for issue #{parent_issue} handed off to PR #{pr_number} for one-shot implementation.",
        "",
        f"Mode: {mode}",
        f"Plan hash: {plan_hash}",
        f"Plan subject: {plan_subject}",
        "",
        "Parent reruns will resume the PR review loop for this PR instead of re-implementing.",
        "",
        f"<!-- AGENT_PLAN_ONE_SHOT_IMPL: {_encode_one_shot_impl_handoff_metadata(metadata)} -->",
        "-- coding-review-agent-loop",
    ]
    return "\n".join(lines)


def post_one_shot_impl_handoff_comment(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    parent_issue: int,
    mode: str,
    plan_hash: str,
    plan_subject: str,
    pr_number: int,
    pr_head_sha: str | None,
) -> None:
    post_issue_comment(
        runner,
        config=config,
        issue_number=parent_issue,
        body=TrustedBody.canonical(
            format_one_shot_impl_handoff_comment(
                parent_issue=parent_issue,
                mode=mode,
                plan_hash=plan_hash,
                plan_subject=plan_subject,
                pr_number=pr_number,
                pr_head_sha=pr_head_sha,
            ),
            expected_tokens=("AGENT_PLAN_ONE_SHOT_IMPL",),
        ),
    )
