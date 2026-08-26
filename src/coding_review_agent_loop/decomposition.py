"""Approved-plan decomposition parsing and publishing helpers."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass

from .config import AgentLoopConfig
from .errors import AgentLoopError
from .github import create_issue, post_issue_comment
from .runner import Runner
from .protocol_markers import TrustedBody

MAX_DECOMPOSITION_PHASES = 8
AUTOMATION_CLASSES = {"agent-pr", "human-action", "manual-close"}
DECOMPOSITION_MARKER_RE = re.compile(
    r"<!--\s*AGENT_PLAN_DECOMPOSITION:\s*(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->",
    re.I,
)
PHASE_IMPLEMENTATION_MARKER_RE = re.compile(
    r"<!--\s*AGENT_PLAN_PHASE_IMPLEMENTATION:\s*(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->",
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


@dataclass(frozen=True)
class DecompositionMetadata:
    parent_issue: int
    plan_hash: str
    mode: str
    phase_count: int
    phase_titles: tuple[str, ...]
    automation: tuple[str, ...]
    children: tuple[tuple[str, str | None, int | None], ...]


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
    if len(phases_payload) > MAX_DECOMPOSITION_PHASES:
        raise AgentLoopError(
            f"Invalid plan decomposition: {len(phases_payload)} phases exceeds "
            f"MAX_DECOMPOSITION_PHASES={MAX_DECOMPOSITION_PHASES}; consolidate phases."
        )

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


def _dependency_lines(phase: PlanPhase, created: Sequence[CreatedPhaseIssue]) -> list[str]:
    if not phase.depends_on:
        return ["- None."]
    by_title = {item.phase.title: item for item in created}
    lines: list[str] = []
    for dependency in phase.depends_on:
        created_issue = by_title.get(dependency)
        if created_issue and created_issue.issue_number is not None:
            lines.append(f"- depends on #{created_issue.issue_number}: {dependency}")
        elif created_issue and created_issue.issue_url:
            lines.append(f"- depends on {created_issue.issue_url}: {dependency}")
        elif created_issue:
            lines.append(f"- depends on previously created phase with unavailable issue URL: {dependency}")
        else:
            lines.append(f"- depends on prior phase: {dependency}")
    return lines


def format_phase_issue_body(
    *,
    repo: str,
    parent_issue: int,
    approved_plan: str,
    phase: PlanPhase,
    created_so_far: Sequence[CreatedPhaseIssue],
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
    return "\n".join(
        [
            f"Child phase issue for parent #{parent_issue}: {parent_url}",
            "",
            "## Approved parent-plan excerpt for this phase",
            phase.parent_context,
            "",
            "## Scope",
            phase.scope,
            "",
            "## Non-goals",
            phase.non_goals,
            "",
            "## Constraints and invariants from the parent plan",
            approved_plan.strip(),
            "",
            "## Dependency notes",
            phase.dependency_notes,
            "",
            "## Dependency links",
            *_dependency_lines(phase, created_so_far),
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


def create_decomposition_child_issues(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    parent_issue: int,
    approved_plan: str,
    decomposition: PlanDecomposition,
) -> tuple[CreatedPhaseIssue, ...]:
    created: list[CreatedPhaseIssue] = []
    for index, phase in enumerate(decomposition.phases, start=1):
        issue_url = create_issue(
            runner,
            config=config,
            title=_phase_issue_title(parent_issue, index, phase),
            body=format_phase_issue_body(
                repo=config.repo,
                parent_issue=parent_issue,
                approved_plan=approved_plan,
                phase=phase,
                created_so_far=created,
            ),
        )
        created.append(
            CreatedPhaseIssue(
                phase=phase,
                issue_url=issue_url,
                issue_number=_issue_number_from_url(issue_url),
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
        return DecompositionMetadata(
            parent_issue=int(payload["parent_issue"]),
            plan_hash=str(payload["plan_hash"]),
            mode=str(payload["mode"]),
            phase_count=int(payload["phase_count"]),
            phase_titles=tuple(str(value) for value in payload["phase_titles"]),
            automation=tuple(str(value) for value in payload["automation"]),
            children=tuple(children),
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
    )
    lines = [
        f"Approved plan decomposed for issue #{parent_issue}.",
        "",
        f"Mode: {mode}",
        "",
        "| Phase | Automation | Child issue | Risk |",
        "| --- | --- | --- | --- |",
    ]
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
