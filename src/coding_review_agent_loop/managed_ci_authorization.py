"""Issue-created managed-CI authorization, continuity and recovery."""
from __future__ import annotations

from .managed_ci_tuple import MANAGED_LABEL, read_opening_pr, compare_opening_pr
from .review_scope import is_complete_code_review
import base64
import json
import re
import secrets
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from typing import Literal
from urllib.parse import quote, urlparse
from .config import AgentLoopConfig
from .errors import AgentLoopError
from .github import (
    PullRequestMetadata,
    parse_strong_issue_reference_evidence,
    post_verified_trusted_pr_protocol_comment,
)
from .logging import log
from .managed_ci_recovery_commands import render_managed_ci_resume_command
from .runner import Runner
from .workdirs import github_api_cwd
from .protocol_markers import (
    MARKER_BY_TOKEN,
    PR_BODY_SURFACE,
    PR_COMMENT_SURFACE,
    TrustedBody,
    protocol_record_label,
    scan_reserved_markers,
)
from .round_transport import ROUND_RESUME_MARKER_RE, decode_mapping, hydrate_mapping
from . import managed_ci_api
from .managed_ci_types import (
    UNPROTECTED_OVERRIDE_TRAILER,
    ISSUE_AUTHORIZATION_MARKER,
    ManagedCiProbeContext,
    _OVERRIDE_PROTECTION_STATES,
    ManagedCiCreationIntent,
    ManagedCiOverrideRecord,
    AuthenticatedIssueCreatedHandoff,
    ManagedCiIssueAuthorization,
    _SUPERSEDING_AUTHORIZATION_KINDS,
)
from .managed_ci_api import (
    _last_list_failure,
    _surfaces_list_failures,
    _advertised_managed_actor,
    _historical_managed_label_event,
    _managed_label_event_history,
    _api_list_detailed,
    _with_reason,
    _api_list,
    _active_managed_label_event,
)
from .managed_ci_protection import (
    waivable_protection_states,
    waiver_flags_for_protection,
    _waiver_for_protection,
    reconcile_live_protection,
    assess_exact_head_protection,
)
from .managed_ci_api import _normalized_comment_body


def _encode_issue_authorization_payload(payload: dict[str, object]) -> str:
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii")


def format_issue_created_authorization_comment(
    authorization: ManagedCiIssueAuthorization,
) -> TrustedBody:
    """Render the complete v1 authorization record for a PR comment only."""
    encoded = _encode_issue_authorization_payload(authorization.to_payload())
    # The label is a pure function of the already-authenticated record, so every
    # call site and every retry renders a byte-identical body.  It sits outside
    # the marker span, which is unchanged.
    label = protocol_record_label(
        "managed_ci_authorization",
        kind=authorization.kind,
        issue_number=authorization.issue_number,
        pr_number=authorization.pr_number,
        head_sha=authorization.head_sha,
    )
    return TrustedBody.canonical(
        f"{label}\n\n<!-- {ISSUE_AUTHORIZATION_MARKER}: {encoded} -->",
        surface=PR_COMMENT_SURFACE,
        expected_tokens=(ISSUE_AUTHORIZATION_MARKER,),
    )


def parse_issue_created_authorization_comment(
    body: str,
) -> ManagedCiIssueAuthorization | None:
    """Parse one strict authorization record without accepting body copies."""
    occurrences = scan_reserved_markers(body)
    matches = [
        occurrence for occurrence in occurrences
        if occurrence.definition.token == ISSUE_AUTHORIZATION_MARKER
    ]
    if not matches:
        return None
    if len(matches) != 1:
        raise AgentLoopError("Managed-CI issue authorization contains duplicate records.")
    TrustedBody.canonical(
        body,
        surface=PR_COMMENT_SURFACE,
        expected_tokens=(ISSUE_AUTHORIZATION_MARKER,),
    )
    match = MARKER_BY_TOKEN[ISSUE_AUTHORIZATION_MARKER].pattern.fullmatch(matches[0].text)
    if match is None:
        raise AgentLoopError("Managed-CI issue authorization record is malformed.")
    try:
        payload = json.loads(
            base64.urlsafe_b64decode(match.group("payload").encode("ascii")).decode("utf-8")
        )
    except (ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise AgentLoopError("Managed-CI issue authorization record is not valid JSON.") from exc
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise AgentLoopError("Managed-CI issue authorization record has an invalid version.")
    kind = payload.get("kind")
    if kind not in {"creation", "fresh", "continuity"}:
        raise AgentLoopError("Managed-CI issue authorization record has an invalid kind.")
    required_string_fields = (
        "repository", "base", "head", "actor", "protection", "waiver", "nonce",
    )
    if any(not isinstance(payload.get(field), str) or not payload[field] for field in required_string_fields):
        raise AgentLoopError("Managed-CI issue authorization record is missing required fields.")
    integer_fields = ("issue", "pr", "actor_id", "label_event_id")
    if any(not isinstance(payload.get(field), int) or payload[field] <= 0 for field in integer_fields):
        raise AgentLoopError("Managed-CI issue authorization record has invalid identity fields.")
    round_comment_ids = payload.get("round_comment_ids", [])
    if not isinstance(round_comment_ids, list) or any(
        not isinstance(value, int) or value <= 0 for value in round_comment_ids
    ):
        raise AgentLoopError("Managed-CI issue authorization record has invalid round metadata IDs.")
    predecessor_head = payload.get("predecessor_head")
    if predecessor_head is not None and (not isinstance(predecessor_head, str) or not predecessor_head):
        raise AgentLoopError("Managed-CI issue authorization record has an invalid predecessor head.")
    predecessor_comment_id = payload.get("predecessor_comment_id")
    if predecessor_comment_id is not None and (
        not isinstance(predecessor_comment_id, int) or predecessor_comment_id <= 0
    ):
        raise AgentLoopError("Managed-CI issue authorization record has an invalid predecessor comment.")
    plan_hash = payload.get("approved_plan_hash")
    if plan_hash is not None and (not isinstance(plan_hash, str) or not plan_hash):
        raise AgentLoopError("Managed-CI issue authorization record has an invalid plan hash.")
    superseded_comment_ids = payload.get("superseded_comment_ids", [])
    if (
        not isinstance(superseded_comment_ids, list)
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in superseded_comment_ids
        )
        or superseded_comment_ids != sorted(set(superseded_comment_ids))
        or ("superseded_comment_ids" in payload and not superseded_comment_ids)
    ):
        raise AgentLoopError(
            "Managed-CI issue authorization record has invalid superseded record IDs."
        )
    if superseded_comment_ids and kind not in _SUPERSEDING_AUTHORIZATION_KINDS:
        raise AgentLoopError(
            "Only a fresh or continuity managed-CI authorization may supersede "
            "earlier records."
        )
    allowed_fields = {
        "version", "kind", "repository", "issue", "pr", "base", "head", "actor",
        "actor_id", "protection", "waiver", "nonce", "label_event_id",
        "predecessor_head", "predecessor_comment_id", "round_comment_ids",
        "approved_plan_hash", "superseded_comment_ids",
    }
    if set(payload) - allowed_fields:
        raise AgentLoopError("Managed-CI issue authorization record has unknown fields.")
    if kind == "creation" and (
        predecessor_head is not None
        or predecessor_comment_id is not None
        or round_comment_ids
    ):
        raise AgentLoopError(
            "Managed-CI creation authorization cannot contain continuity fields."
        )
    if kind in {"creation", "fresh", "continuity"} and (
        payload["protection"] not in _OVERRIDE_PROTECTION_STATES
        or payload["waiver"] != _waiver_for_protection(payload["protection"])
    ):
        raise AgentLoopError(
            "Managed-CI issue authorization has an invalid protection or waiver context."
        )
    if kind == "fresh" and (
        round_comment_ids
        or (predecessor_head is None) != (predecessor_comment_id is None)
    ):
        raise AgentLoopError(
            "Managed-CI fresh authorization cannot contain round metadata and must "
            "contain both predecessor fields or neither."
        )
    if kind == "continuity" and (
        predecessor_head is None
        or predecessor_comment_id is None
        or not round_comment_ids
    ):
        raise AgentLoopError(
            "Managed-CI continuity authorization requires predecessor and round metadata."
        )
    parsed = ManagedCiIssueAuthorization(
        kind=kind,
        repository=payload["repository"],
        issue_number=payload["issue"],
        pr_number=payload["pr"],
        base_ref=payload["base"],
        head_sha=payload["head"],
        actor_login=payload["actor"],
        actor_id=payload["actor_id"],
        protection=payload["protection"],
        waiver=payload["waiver"],
        nonce=payload["nonce"],
        label_event_id=payload["label_event_id"],
        predecessor_head=predecessor_head,
        predecessor_comment_id=predecessor_comment_id,
        round_comment_ids=tuple(round_comment_ids),
        approved_plan_hash=plan_hash,
        superseded_comment_ids=tuple(superseded_comment_ids),
    )
    # Full-body authentication (#1043): the whole body must equal the canonical
    # rendering of the parsed record, or exactly the historical marker-only
    # rendering that predates the visible label (#878).  Callers remove the
    # known host footer once at their ingestion boundary; this parser never
    # strips, so a doubled, misplaced, or variant footer, or any other prose,
    # is rejected here.
    historical = (
        f"<!-- {ISSUE_AUTHORIZATION_MARKER}: "
        f"{_encode_issue_authorization_payload(parsed.to_payload())} -->"
    )
    if body not in {str(format_issue_created_authorization_comment(parsed)), historical}:
        raise AgentLoopError("Managed-CI issue authorization record is malformed.")
    return parsed


def parse_managed_ci_override_record(
    body: str,
    *,
    surface: str,
    schema: Literal["body", "audit"],
    required: bool = False,
    expected_nonce: str | None = None,
    additional_allowed_tokens: frozenset[str] = frozenset(),
) -> ManagedCiOverrideRecord | None:
    """Parse one canonical managed-CI override record and reject every variant.

    ``TrustedBody`` supplies the registry/surface/canonical-wire checks.  This
    helper owns the inner field schema so body records cannot quietly acquire
    audit provenance fields and audit comments cannot omit their provenance.
    """
    occurrences = scan_reserved_markers(body)
    override = [
        occurrence for occurrence in occurrences
        if occurrence.definition.token == UNPROTECTED_OVERRIDE_TRAILER
    ]
    allowed = set(additional_allowed_tokens)
    allowed.add(UNPROTECTED_OVERRIDE_TRAILER)
    unexpected = [
        occurrence.definition.token for occurrence in occurrences
        if occurrence.definition.token not in allowed
    ]
    if unexpected:
        raise AgentLoopError(
            "Managed-CI override carrier contains unrelated reserved protocol record(s): "
            + ", ".join(sorted(set(unexpected)))
        )
    if not override:
        if required:
            raise AgentLoopError("Managed-CI override record is required for this handoff.")
        # Still canonicalize allowed companion records, including malformed
        # bare-token fallbacks, before returning absence.
        if occurrences:
            TrustedBody.canonical(
                body,
                surface=surface,
                expected_tokens={item.definition.token for item in occurrences},
            )
        return None
    if len(override) != 1:
        raise AgentLoopError("Managed-CI override carrier contains duplicate override records.")
    TrustedBody.canonical(
        body,
        surface=surface,
        expected_tokens={item.definition.token for item in occurrences},
    )
    parts = override[0].text.strip().split()
    if not parts or parts[0] != UNPROTECTED_OVERRIDE_TRAILER:
        raise AgentLoopError("Managed-CI override record has an invalid schema.")
    fields: list[tuple[str, str]] = []
    for token in parts[1:]:
        key, separator, value = token.partition("=")
        if not separator or not key or not value or any(key == old for old, _ in fields):
            raise AgentLoopError("Managed-CI override record has an invalid schema.")
        fields.append((key, value))
    values = dict(fields)
    body_keys = {"nonce"}
    audit_keys = {
        "nonce", "repo", "base", "head", "protection", "active_label_event_id",
        "resume_from", "provenance_head", "generation",
    }
    if schema == "body":
        if set(values) != body_keys:
            raise AgentLoopError("Managed-CI PR-body override record must contain only its nonce.")
    else:
        if not {"nonce", "repo", "base", "head", "protection"}.issubset(values) or set(values) - audit_keys:
            raise AgentLoopError("Managed-CI override audit record has an invalid provenance schema.")
    nonce = values.get("nonce")
    if not nonce:
        raise AgentLoopError("Managed-CI override record has no nonce.")
    if expected_nonce is not None and nonce != expected_nonce:
        raise AgentLoopError("Managed-CI override record does not match this creation handoff.")
    return ManagedCiOverrideRecord(nonce=nonce, fields=tuple(fields))


def _opening_override_nonce(body: str | None) -> str | None:
    """Return the nonce authenticated by the PR-opening body, if present.

    A fresh or continuity authorization has its own nonce.  Revalidation must
    never pass that terminal nonce to the PR-body parser; the body, when it
    carries an override, is bound only to the original opening nonce.
    """
    body = body or ""
    if UNPROTECTED_OVERRIDE_TRAILER not in body:
        return None
    record = parse_managed_ci_override_record(
        body,
        surface=PR_BODY_SURFACE,
        schema="body",
        required=True,
    )
    return record.nonce if record is not None else None

def _issue_created_tuple(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    issue_number: int,
    metadata: PullRequestMetadata,
    expected_branch: str,
    expected_nonce: str | None,
    protection_mode: str,
    lifecycle: Literal["draft-labeled", "draft-unlabeled", "ready-unlabeled"] = "draft-labeled",
) -> AuthenticatedIssueCreatedHandoff:
    """Read and verify the immutable tuple before any handoff publication.

    The GraphQL view is useful to the review loop but cannot attest author,
    same-repository head, labels, and draft state together.  Require it to
    agree with the REST tuple and re-read the REST tuple before returning so a
    transient body/head race cannot escape into a writer.
    """
    if not config.base or metadata.number != pr_number or metadata.repo.casefold() != config.repo.casefold():
        raise AgentLoopError("Managed-CI issue handoff does not match the configured repository/base.")
    body = metadata.body or ""
    record = parse_managed_ci_override_record(
        body,
        surface=PR_BODY_SURFACE,
        schema="body",
        required=expected_nonce is not None,
        expected_nonce=expected_nonce,
    )
    if expected_nonce is None and record is not None:
        raise AgentLoopError("Strict managed-CI issue handoff must not contain an override record.")
    linked = parse_strong_issue_reference_evidence(body, repo=config.repo, issue_number=issue_number)
    if len(linked) != 1:
        raise AgentLoopError("Managed-CI issue handoff requires one canonical closing reference to its issue.")
    who = managed_ci_api._api_json(runner, config, "user", quiet=True)
    actor_login = who.get("login") if isinstance(who.get("login"), str) else None
    actor_id = who.get("id") if isinstance(who.get("id"), int) else None
    advertised = _advertised_managed_actor(runner, config)
    configured = (config.managed_ci_trusted_actor or "").strip()
    if (
        not configured or not actor_login or actor_id is None or not advertised
        or actor_login.casefold() != configured.casefold()
        or actor_login.casefold() != advertised.casefold()
    ):
        trusted_actor = advertised or actor_login or configured
        if trusted_actor:
            remediation_config = replace(config, managed_ci_trusted_actor=trusted_actor)
            remediation = render_managed_ci_resume_command(
                remediation_config,
                pr_number=pr_number,
                issue_number=issue_number,
                managed_ci=True,
            )
        else:
            remediation = render_managed_ci_resume_command(
                config,
                pr_number=pr_number,
                issue_number=issue_number,
                managed_ci=True,
            ) + " --managed-ci-trusted-actor '<trusted-actor>'"
        raise AgentLoopError(
            "Managed-CI issue handoff actor authentication does not match repository settings. "
            f"Observed identities: gh login=`{actor_login or '<missing>'}`, "
            f"--managed-ci-trusted-actor=`{configured or '<missing>'}`, "
            f"AGENT_LOOP_MANAGED_ACTOR=`{advertised or '<missing>'}`; expected all three to match. "
            f"Rerun with the configured trusted actor: `{remediation}`."
        )

    remediation = render_managed_ci_resume_command(
        config,
        pr_number=pr_number,
        issue_number=issue_number,
        managed_ci=True,
    )

    def check_rest(phase: str) -> tuple[str, str]:
        pr = read_opening_pr(runner, config, pr_number, phase=phase)
        return compare_opening_pr(
            pr, metadata=metadata, repo=config.repo, base=config.base,
            branch=expected_branch, actor_login=actor_login, actor_id=actor_id,
            lifecycle=lifecycle, phase=phase, remediation=remediation,
        )

    sha, branch = check_rest("first-validation")
    live_sha, live_branch = check_rest("second-validation")
    if (live_sha, live_branch) != (sha, branch):
        raise AgentLoopError(
            "Managed-CI issue-created opening tuple changed during validation. "
            f"The PR was left unchanged. Resume with `{remediation}`."
        )
    return AuthenticatedIssueCreatedHandoff(
        pr_number=pr_number,
        issue_number=issue_number,
        repository=config.repo,
        base_ref=config.base,
        head_sha=sha,
        branch=branch,
        trusted_actor_login=actor_login,
        trusted_actor_id=actor_id,
        protection_mode=protection_mode,
        override_nonce=record.nonce if record is not None else None,
        opening_override_nonce=record.nonce if record is not None else None,
        lifecycle={
            "draft-labeled": "draft-labeled",
            "draft-unlabeled": "draft-unlabeled-reentry",
            "ready-unlabeled": "ready-unlabeled-reentry",
        }[lifecycle],
    )


def authenticate_issue_created_handoff(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    intent: ManagedCiCreationIntent,
    issue_number: int,
    pr_number: int,
    metadata: PullRequestMetadata,
) -> AuthenticatedIssueCreatedHandoff:
    """Authenticate a just-created issue draft before *any* remote write."""
    return _issue_created_tuple(
        runner,
        config=config,
        pr_number=pr_number,
        issue_number=issue_number,
        metadata=metadata,
        expected_branch=intent.branch,
        expected_nonce=intent.audit_nonce,
        protection_mode=intent.protection_mode,
        lifecycle="draft-labeled",
    )


def _authorization_actor(
    runner: Runner,
    *,
    config: AgentLoopConfig,
) -> tuple[str, int]:
    """Authenticate the CLI actor against both GitHub identity sources."""
    who = managed_ci_api._api_json(runner, config, "user", quiet=True)
    login = who.get("login") if isinstance(who.get("login"), str) else None
    actor_id = who.get("id") if isinstance(who.get("id"), int) else None
    advertised = _advertised_managed_actor(runner, config)
    configured = (config.managed_ci_trusted_actor or "").strip()
    if (
        not configured
        or not isinstance(login, str)
        or not isinstance(actor_id, int)
        or not isinstance(advertised, str)
        or login.casefold() != configured.casefold()
        or login.casefold() != advertised.casefold()
    ):
        raise AgentLoopError(
            "Managed-CI fresh authorization requires the authenticated GitHub user, "
            "--managed-ci-trusted-actor, and AGENT_LOOP_MANAGED_ACTOR to match."
        )
    return login, actor_id


def _authorization_comment_records(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    actor_login: str,
    actor_id: int,
) -> list[tuple[int, ManagedCiIssueAuthorization]]:
    comments, reason = _api_list_detailed(
        runner, config, f"repos/{config.repo}/issues/{pr_number}/comments?per_page=100"
    )
    if comments is None:
        raise AgentLoopError(
            _with_reason("Managed-CI authorization comments could not be inspected.", reason)
        )
    records: list[tuple[int, ManagedCiIssueAuthorization]] = []
    for comment in comments:
        body = _normalized_comment_body(comment, config=config)
        if ISSUE_AUTHORIZATION_MARKER not in body:
            continue
        try:
            parsed = parse_issue_created_authorization_comment(body)
        except AgentLoopError as exc:
            raise AgentLoopError(
                "Managed-CI authorization comment is malformed; refusing to infer authority."
            ) from exc
        if parsed is None:
            continue
        user = comment.get("user") if isinstance(comment.get("user"), dict) else {}
        if user.get("login") != actor_login or user.get("id") != actor_id:
            raise AgentLoopError(
                "Managed-CI authorization comment is not authored by the authenticated actor."
            )
        comment_id = comment.get("id")
        if not isinstance(comment_id, int):
            raise AgentLoopError("Managed-CI authorization comment has no stable identity.")
        records.append((comment_id, parsed))
    superseded = _superseded_authorization_ids(records)
    return [item for item in records if item[0] not in superseded]


def _superseded_authorization_ids(
    records: Iterable[tuple[int, ManagedCiIssueAuthorization]],
) -> frozenset[int]:
    """Return the comment IDs retired by a later superseding record.

    Only a fresh (#1065) or continuity (#1069) record may supersede, only
    records published before it, and only records in its own
    repository/issue/PR scope.  Supersession is
    transitive by construction: a retired record stays retired even after
    the grant that retired it is itself superseded.
    """
    records = list(records)
    by_comment_id = dict(records)
    superseded: set[int] = set()
    for comment_id, record in records:
        if record.kind not in _SUPERSEDING_AUTHORIZATION_KINDS:
            continue
        for retired_id in record.superseded_comment_ids:
            retired = by_comment_id.get(retired_id)
            if (
                retired_id < comment_id
                and retired is not None
                and retired.repository.casefold() == record.repository.casefold()
                and retired.issue_number == record.issue_number
                and retired.pr_number == record.pr_number
            ):
                superseded.add(retired_id)
    return frozenset(superseded)


def _superseded_authorization_comment_ids(
    comments: Iterable[Mapping[str, object]],
    *,
    config: AgentLoopConfig,
    actor_login: str,
    actor_id: int,
) -> frozenset[int]:
    """Return superseded IDs among the actor's parseable authorization comments.

    Malformed and foreign records are ignored here; each caller's own pass
    still fails closed on them.
    """
    records: list[tuple[int, ManagedCiIssueAuthorization]] = []
    for comment in comments:
        body = _normalized_comment_body(comment, config=config)
        user = comment.get("user") if isinstance(comment.get("user"), dict) else {}
        comment_id = comment.get("id")
        if (
            ISSUE_AUTHORIZATION_MARKER not in body
            or user.get("login") != actor_login
            or user.get("id") != actor_id
            or not isinstance(comment_id, int)
        ):
            continue
        try:
            parsed = parse_issue_created_authorization_comment(body)
        except AgentLoopError:
            continue
        if parsed is not None:
            records.append((comment_id, parsed))
    return _superseded_authorization_ids(records)


def _github_proves_descendant(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    predecessor_head: str,
    live_head: str,
) -> bool:
    """Return whether GitHub proves that live_head descends from predecessor_head."""
    comparison = managed_ci_api._api_json(
        runner,
        config,
        (
            f"repos/{config.repo}/compare/"
            f"{quote(predecessor_head, safe='')}...{quote(live_head, safe='')}"
        ),
    )
    base_commit = (
        comparison.get("base_commit")
        if isinstance(comparison.get("base_commit"), dict) else {}
    )
    merge_base = (
        comparison.get("merge_base_commit")
        if isinstance(comparison.get("merge_base_commit"), dict) else {}
    )
    return (
        comparison.get("status") == "ahead"
        and base_commit.get("sha") == predecessor_head
        and merge_base.get("sha") == predecessor_head
    )


def find_actor_round_metadata_comment_ids(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    actor_login: str,
    actor_id: int,
    predecessor_head: str,
    new_head: str,
    round_number: int,
    after_comment_id: int,
    authorization_records: Collection[tuple[int, ManagedCiIssueAuthorization]] | None = None,
    fresh_comment_id: int | None = None,
    chain_context: "ChainAuthContext | None" = None,
    expected_seat_binding: Mapping[str, object] | None = None,
) -> tuple[int, ...]:
    """Return the exact blocking-review and coder records for one head transition.

    The optional keywords open the rejected-coder recovery shape (#1367): when
    the handoff is a fresh grant whose immutable link names the original
    authorization, the blocking review predating that grant is accepted only
    through the exact authenticated chain checked by
    ``_rejected_recovery_round_ids``.  The ``after_comment_id`` cutoff of the
    ordinary shapes is never relaxed.
    """
    comments, reason = _api_list_detailed(
        runner, config, f"repos/{config.repo}/issues/{pr_number}/comments?per_page=100"
    )
    if comments is None:
        raise AgentLoopError(
            _with_reason(
                "Managed-CI continuity could not inspect PR round metadata comments.", reason
            )
        )
    by_index = _continuity_round_records(comments)
    reviewers: list[int] = []
    coders: list[int] = []
    conflict_coders: list[int] = []
    ci_repair_coders: list[int] = []
    for index, comment in enumerate(comments):
        metadata = by_index.get(index)
        user = comment.get("user") if isinstance(comment.get("user"), dict) else {}
        comment_id = comment.get("id")
        if (
            metadata is None
            or user.get("login") != actor_login
            or user.get("id") != actor_id
            or not isinstance(comment_id, int)
            or comment_id <= after_comment_id
        ):
            continue
        if (
            is_complete_code_review(metadata)
            and metadata["state"] == "blocking"
            and metadata["subject"] == predecessor_head
            and metadata["round_number"] == round_number
        ):
            reviewers.append(comment_id)
        elif (
            metadata["role"] == "coder"
            and metadata["subject"] == new_head
            and metadata["round_number"] == round_number + 1
        ):
            coders.append(comment_id)
            if metadata.get("resolves_merge_conflict"):
                conflict_coders.append(comment_id)
            if (predecessor_head, new_head) in metadata["ci_repair_transitions"]:
                ci_repair_coders.append(comment_id)
    # A conflict-resolution round advances the head with no reviewer pair by
    # construction: the orchestrator skips reviewers and routes the round to
    # the coder (#829).  Accept that transition on the tool-owned merge-conflict
    # obligation alone.  Every other guard is unchanged, and reviewers must
    # still approve the exact final head before qualification or merge.
    if not reviewers and len(conflict_coders) == 1 and len(coders) == 1:
        return (conflict_coders[0],)
    # A managed exact-head CI repair round on an approved head has no blocking
    # review either: the failure came from CI, not a reviewer (#1024).  Accept
    # it only when the tool-minted CI obligation, already advanced to the
    # pushed head, binds both the failed predecessor head and the new head.
    if not reviewers and len(ci_repair_coders) == 1 and len(coders) == 1:
        return (ci_repair_coders[0],)
    if (
        not reviewers
        or len(coders) != 1
        or any(comment_id >= coders[0] for comment_id in reviewers)
    ):
        recovered = None
        if (
            authorization_records is not None
            and fresh_comment_id is not None
            and fresh_comment_id == after_comment_id
            and chain_context is not None
            and chain_context.actor_login == actor_login
            and chain_context.actor_id == actor_id
        ):
            recovered = _correlate_rejected_recovery(
                comments,
                by_index,
                records_by_id=dict(authorization_records),
                context=chain_context,
                fresh_comment_id=fresh_comment_id,
                predecessor_head=predecessor_head,
                new_head=new_head,
                round_number=round_number,
                expected_seat_binding=expected_seat_binding,
            )
        if recovered is None:
            raise AgentLoopError(
                "Managed-CI head continuity requires correlated blocking-review and coder round metadata."
            )
        return recovered
    return tuple(sorted(set((*reviewers, coders[0]))))


def _continuity_round_records(
    comments: list[dict[str, object]],
) -> dict[int, dict[str, object]]:
    """Decode only the immutable fields needed by continuity validation."""
    bodies = tuple(
        comment.get("body") if isinstance(comment.get("body"), str) else ""
        for comment in comments
    )
    result: dict[int, dict[str, object]] = {}
    for index, body in enumerate(bodies):
        matches = tuple(ROUND_RESUME_MARKER_RE.finditer(body))
        if not matches:
            continue
        payload, missing = hydrate_mapping(
            decode_mapping(matches[-1].group("payload")), bodies
        )
        if missing:
            raise AgentLoopError("Managed-CI continuity round metadata is incomplete.")
        required = {
            "flow": str,
            "role": str,
            "subject": str,
            "round_number": int,
        }
        if any(not isinstance(payload.get(key), kind) for key, kind in required.items()):
            raise AgentLoopError("Managed-CI continuity round metadata is malformed.")
        if payload["flow"] != "pr":
            continue
        # Payloads are arbitrary JSON from any commenter and are decoded before
        # author filtering, so every raw field is type-checked before it is
        # compared or hashed (#1367).
        raw_phase = payload.get("phase")
        phase = raw_phase if isinstance(raw_phase, str) else None
        seat_binding = payload.get("seat_binding")
        raw_dispatch_round = payload.get("dispatch_round")
        result[index] = {
            "role": payload["role"],
            **({"review_scope": payload["review_scope"]} if "review_scope" in payload else {}),
            "state": payload.get("state"),
            "subject": payload["subject"],
            "round_number": payload["round_number"],
            "phase": phase,
            # The raw dispatch round, kept even when the recovery record fails
            # validation, so a malformed record of the same dispatch is still
            # counted as a recovery candidate (#1367).
            "dispatch_round": (
                raw_dispatch_round
                if isinstance(raw_dispatch_round, int)
                and not isinstance(raw_dispatch_round, bool)
                else None
            ),
            # Strictly a dict or None; any other stored value marks the record
            # unusable as a seat-bound review (#1367).
            "seat_binding": seat_binding if isinstance(seat_binding, dict) else None,
            "seat_binding_valid": seat_binding is None or isinstance(seat_binding, dict),
            # The decoded rejected-follow-up or coder-dispatch record, or None
            # when the record is absent or fails per-phase validation (#1367).
            "recovery": (
                _decode_recovery_round_record(payload)
                if payload["role"] == "summary"
                and phase in _CONTINUITY_RECOVERY_PHASES
                else None
            ),
            # A conflict-resolution round is orchestrator-routed and skips
            # reviewers by construction (#829); the open, live machine-authority
            # obligation on the coder record is the durable evidence of that
            # routing (#1057).
            "resolves_merge_conflict": _records_merge_conflict_obligation(payload),
            # A CI repair round is routed by the orchestrator-minted CI
            # obligation (#1024); the (failed, candidate) head pairs it binds
            # are the durable evidence of that routing.
            "ci_repair_transitions": _records_ci_repair_transitions(payload),
        }
    return result


from .managed_ci_obligations import _records_merge_conflict_obligation, _records_ci_repair_transitions


def _continuity_round_metadata_is_valid(
    comments: list[dict[str, object]],
    *,
    authorization: ManagedCiIssueAuthorization,
    predecessor: ManagedCiIssueAuthorization | None = None,
    records_by_id: Mapping[int, ManagedCiIssueAuthorization] | None = None,
    comments_by_id: Mapping[int, Mapping[str, object]] | None = None,
    chain_context: "ChainAuthContext | None" = None,
    expected_seat_binding: Mapping[str, object] | None = None,
) -> bool:
    """Reauthenticate the exact round records referenced by a continuity grant.

    Without a fresh predecessor this is the ordinary, merge-conflict and
    CI-repair validation, unchanged.  With one, the rejected-coder recovery
    shape (#1367) is re-authenticated in full when the ordinary shapes fail.
    """
    return _continuity_link_shape(
        comments,
        authorization=authorization,
        predecessor=predecessor,
        records_by_id=records_by_id,
        comments_by_id=comments_by_id,
        chain_context=chain_context,
        expected_seat_binding=expected_seat_binding,
    ) is not None


def _continuity_link_shape(
    comments: list[dict[str, object]],
    *,
    authorization: ManagedCiIssueAuthorization,
    predecessor: ManagedCiIssueAuthorization | None = None,
    records_by_id: Mapping[int, ManagedCiIssueAuthorization] | None = None,
    comments_by_id: Mapping[int, Mapping[str, object]] | None = None,
    chain_context: "ChainAuthContext | None" = None,
    expected_seat_binding: Mapping[str, object] | None = None,
) -> Literal["ordinary", "recovery"] | None:
    """Classify a continuity link as ordinary, rejected-coder recovery, or invalid.

    The two shapes are disjoint: the ordinary shapes bind only records newer
    than the predecessor authorization, the recovery shape binds a review
    older than the fresh predecessor grant.
    """
    if _ordinary_continuity_round_metadata_is_valid(comments, authorization=authorization):
        return "ordinary"
    if (
        predecessor is None
        or predecessor.kind != "fresh"
        or records_by_id is None
        or chain_context is None
        or authorization.kind != "continuity"
        or authorization.predecessor_comment_id is None
        or records_by_id.get(authorization.predecessor_comment_id) != predecessor
        or predecessor.head_sha != authorization.predecessor_head
    ):
        return None
    if comments_by_id is None:
        comments_by_id = _comments_by_id(comments)
    if _recovery_continuity_is_valid(
        comments,
        authorization=authorization,
        records_by_id=records_by_id,
        comments_by_id=comments_by_id,
        context=chain_context,
        expected_seat_binding=expected_seat_binding,
    ):
        return "recovery"
    return None


def _ordinary_continuity_round_metadata_is_valid(
    comments: list[dict[str, object]],
    *,
    authorization: ManagedCiIssueAuthorization,
) -> bool:
    """Reauthenticate the exact round records referenced by a continuity grant."""
    if authorization.kind != "continuity" or not authorization.round_comment_ids:
        return False
    try:
        by_index = _continuity_round_records(comments)
    except AgentLoopError:
        return False
    selected: list[tuple[int, dict[str, object]]] = []
    wanted = set(authorization.round_comment_ids)
    for index, comment in enumerate(comments):
        if comment.get("id") not in wanted:
            continue
        user = comment.get("user") if isinstance(comment.get("user"), dict) else {}
        metadata = by_index.get(index)
        if (
            metadata is None
            or user.get("login") != authorization.actor_login
            or user.get("id") != authorization.actor_id
        ):
            return False
        comment_id = comment.get("id")
        if (
            not isinstance(comment_id, int)
            or authorization.predecessor_comment_id is None
            or comment_id <= authorization.predecessor_comment_id
        ):
            return False
        selected.append((comment_id, metadata))
    if len(selected) != len(wanted):
        return False
    # Every referenced ancestor record carries its governing board's binding
    # once the PR has recorded board changes (#1376).
    if _historical_bindings_match(
        comments, [comment_id for comment_id, _metadata in selected], by_index,
        actor_id=authorization.actor_id,
    ) is False:
        return False
    coders = [
        item for item in selected
        if item[1]["role"] == "coder" and item[1]["subject"] == authorization.head_sha
    ]
    if len(coders) != 1:
        return False
    coder_id, coder_metadata = coders[0]
    coder_round = coder_metadata["round_number"]
    reviewers = [
        item for item in selected
        if is_complete_code_review(item[1])
        and item[1]["state"] == "blocking"
        and item[1]["subject"] == authorization.predecessor_head
        and item[1]["round_number"] == coder_round - 1
        and item[0] < coder_id
    ]
    if not reviewers:
        # The conflict-resolution transition is recorded by the coder record
        # alone, so reauthenticate that shape rather than demanding a reviewer
        # pair that never existed (#829).  A CI repair round is likewise
        # recorded by the coder record alone, bound to this exact transition
        # by its advanced CI obligation (#1024).
        return len(selected) == 1 and (
            bool(coder_metadata.get("resolves_merge_conflict"))
            or (authorization.predecessor_head, authorization.head_sha)
            in coder_metadata["ci_repair_transitions"]
        )
    return len(reviewers) + 1 == len(selected)


# Summary-record phases of the rejected-coder recovery shape (#1367).  These
# mirror round_state's CODER_FOLLOWUP_REJECTED_PHASE / CODER_DISPATCH_PHASE.
_CONTINUITY_REJECTED_PHASE = "coder-followup-rejected"
_CONTINUITY_DISPATCH_PHASE = "coder-dispatch"
_CONTINUITY_RECOVERY_PHASES = frozenset(
    {_CONTINUITY_REJECTED_PHASE, _CONTINUITY_DISPATCH_PHASE}
)


class PlanPolicy(Enum):
    """Which approved plans an authorization chain may carry (#1367)."""

    # Every record must name the current approved plan.
    CURRENT_ONLY = "current-only"
    # A verified-retired plan (#993) is also accepted.
    ALLOW_RETIRED = "allow-retired"


@dataclass(frozen=True)
class ChainAuthContext:
    """The authenticated scope every record on an authorization chain must match."""

    config: AgentLoopConfig
    actor_login: str
    actor_id: int
    repository: str
    issue_number: int | None
    pr_number: int
    base_ref: str
    approved_plan_hash: str | None
    retired_plan_hashes: frozenset[str] = frozenset()
    valid_label_event_ids: frozenset[int] | None = None
    check_protection: str | None = None
    handoff: AuthenticatedIssueCreatedHandoff | None = None
    # The nonce of the authenticated PR-opening body, for surfaces without a
    # handoff (plan binding).  A creation root on a recovery chain must carry it.
    opening_override_nonce: str | None = None


def chain_auth_context_for_handoff(
    *,
    config: AgentLoopConfig,
    handoff: AuthenticatedIssueCreatedHandoff,
    valid_label_event_ids: Collection[int] | None,
) -> ChainAuthContext:
    """Build the strict chain context for an authenticated issue-created handoff."""
    return ChainAuthContext(
        config=config,
        actor_login=handoff.trusted_actor_login,
        actor_id=handoff.trusted_actor_id,
        repository=handoff.repository,
        issue_number=handoff.issue_number,
        pr_number=handoff.pr_number,
        base_ref=handoff.base_ref,
        approved_plan_hash=handoff.approved_plan_hash,
        retired_plan_hashes=frozenset(handoff.retired_plan_hashes),
        valid_label_event_ids=(
            frozenset(valid_label_event_ids) if valid_label_event_ids is not None else None
        ),
        check_protection=handoff.protection_mode,
        handoff=handoff,
    )


def load_actor_authorization_records(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    actor_login: str,
    actor_id: int,
) -> tuple[tuple[int, ManagedCiIssueAuthorization], ...]:
    """Return the actor's non-superseded authorization records by comment ID."""
    return tuple(
        _authorization_comment_records(
            runner,
            config=config,
            pr_number=pr_number,
            actor_login=actor_login,
            actor_id=actor_id,
        )
    )


def load_actor_label_event_ids(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    actor_login: str,
    actor_id: int,
) -> frozenset[int] | None:
    """Return the actor-owned managed-label event IDs, or None when unreadable."""
    events = _actor_owned_label_event_ids(
        runner, config=config, pr_number=pr_number,
        actor_login=actor_login, actor_id=actor_id,
    )
    return frozenset(events) if events is not None else None


def rejected_recovery_correlation_inputs(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    handoff: AuthenticatedIssueCreatedHandoff,
) -> dict[str, object]:
    """Keyword inputs that let continuity correlation try the recovery shape (#1367).

    Only a fresh-grant handoff can sit on the stranded head of a rejected
    coder follow-up.  Any other handoff, or records that cannot be read,
    yields no inputs, so correlation keeps its ordinary behaviour and fails
    closed exactly as before.
    """
    if handoff.authorization_kind != "fresh" or handoff.authorization_comment_id is None:
        return {}
    try:
        records = load_actor_authorization_records(
            runner,
            config=config,
            pr_number=handoff.pr_number,
            actor_login=handoff.trusted_actor_login,
            actor_id=handoff.trusted_actor_id,
        )
    except AgentLoopError:
        return {}
    from .reviewer_seats import reviewer_seat_binding

    return {
        "authorization_records": records,
        "fresh_comment_id": handoff.authorization_comment_id,
        "chain_context": chain_auth_context_for_handoff(
            config=config,
            handoff=handoff,
            valid_label_event_ids=load_actor_label_event_ids(
                runner,
                config=config,
                pr_number=handoff.pr_number,
                actor_login=handoff.trusted_actor_login,
                actor_id=handoff.trusted_actor_id,
            ),
        ),
        "expected_seat_binding": reviewer_seat_binding(config),
    }


def _comments_by_id(
    comments: Iterable[Mapping[str, object]],
) -> dict[int, Mapping[str, object]]:
    return {
        comment["id"]: comment
        for comment in comments
        if isinstance(comment.get("id"), int) and not isinstance(comment.get("id"), bool)
    }


def _chain_record_admissible(
    comment: Mapping[str, object] | None,
    record: ManagedCiIssueAuthorization,
    context: ChainAuthContext,
    *,
    plan_policy: PlanPolicy,
    require_provenance: bool,
) -> bool:
    """Shared per-record admission for every authorization-chain surface (#1367).

    The actor and scope checks are the plan-binding walker's.  With a handoff
    or ``require_provenance`` the record must also pass the resume audit's
    predicate (actor, waiver, config-aware protection, label event).  Under
    ``require_provenance`` a missing label history or protection fails closed
    rather than skipping those checks.
    """
    if comment is None:
        return False
    user = comment.get("user") if isinstance(comment.get("user"), dict) else {}
    login = user.get("login")
    if (
        not isinstance(comment.get("id"), int)
        or not isinstance(login, str)
        or login.casefold() != context.actor_login.casefold()
        or user.get("id") != context.actor_id
        or record.actor_login.casefold() != context.actor_login.casefold()
        or record.actor_id != context.actor_id
        or record.repository.casefold() != context.repository.casefold()
        or record.pr_number != context.pr_number
        or (context.issue_number is not None and record.issue_number != context.issue_number)
        or record.base_ref != context.base_ref
    ):
        return False
    if require_provenance:
        if context.valid_label_event_ids is None or context.check_protection is None:
            return False
        if record.protection not in waivable_protection_states(context.config):
            return False
    if context.handoff is not None or require_provenance:
        if not _authorization_record_valid_for_handoff(
            record,
            comment,
            config=context.config,
            handoff=context.handoff,
            actor_login=context.actor_login,
            actor_id=context.actor_id,
            base_ref=context.base_ref,
            pr_number=context.pr_number,
            issue_number=context.issue_number,
            valid_label_event_ids=(
                set(context.valid_label_event_ids)
                if context.valid_label_event_ids is not None else None
            ),
            check_protection=context.check_protection,
            plan_scope_authenticated=True,
        ):
            return False
    if (
        require_provenance
        and context.handoff is None
        and record.kind == "creation"
        and (
            context.opening_override_nonce is None
            or record.nonce != context.opening_override_nonce
        )
    ):
        # Without a handoff the creation root is bound to the PR-opening body
        # directly, as ``_authorization_matches`` does for resume.
        return False
    recorded = record.approved_plan_hash or None
    if recorded == (context.approved_plan_hash or None):
        return True
    return (
        plan_policy is PlanPolicy.ALLOW_RETIRED
        and recorded is not None
        and recorded in context.retired_plan_hashes
    )


def _pending_authorization_fields_admissible(
    authorization: ManagedCiIssueAuthorization,
    context: ChainAuthContext,
    *,
    predecessor: ManagedCiIssueAuthorization,
) -> bool:
    """Check every comment-independent field of an unpublished continuity record.

    The record has no comment yet, so no comment identity or author is
    fabricated; the posted comment's author is verified when it is written
    and every later read authenticates it through ``_chain_record_admissible``.
    """
    handoff_protection = (
        context.handoff.protection_mode if context.handoff is not None
        else context.check_protection
    )
    return (
        authorization.kind == "continuity"
        and authorization.repository.casefold() == context.repository.casefold()
        and (
            context.issue_number is None
            or authorization.issue_number == context.issue_number
        )
        and authorization.pr_number == context.pr_number
        and authorization.base_ref == context.base_ref
        and authorization.actor_login.casefold() == context.actor_login.casefold()
        and authorization.actor_id == context.actor_id
        and (authorization.approved_plan_hash or None) == (context.approved_plan_hash or None)
        and authorization.protection == predecessor.protection
        and authorization.waiver == _waiver_for_protection(authorization.protection)
        and authorization.protection in waivable_protection_states(context.config)
        and (handoff_protection is None or authorization.protection == handoff_protection)
        and context.valid_label_event_ids is not None
        and authorization.label_event_id in context.valid_label_event_ids
    )


def _recovery_context(
    context: ChainAuthContext, fresh: ManagedCiIssueAuthorization
) -> ChainAuthContext:
    """Bind a recovery chain's protection to its fresh grant when none is given."""
    if context.check_protection is None:
        return replace(context, check_protection=fresh.protection)
    return context


def _authenticate_authorization_chain(
    comment_id: int,
    records_by_id: Mapping[int, ManagedCiIssueAuthorization],
    comments_by_id: Mapping[int, Mapping[str, object]],
    comments: list[dict[str, object]],
    context: ChainAuthContext,
    *,
    plan_policy: PlanPolicy | None = None,
    expected_seat_binding: Mapping[str, object] | None = None,
) -> tuple[ManagedCiIssueAuthorization, ...] | None:
    """Walk one terminal to its root and authenticate the whole chain.

    Pass 1 follows each continuity link to a creation or fresh root,
    re-authenticating every link's round metadata.  A rejected-coder recovery
    link (#1367) continues through its fresh grant to the original
    authorization instead of stopping there.  Pass 2 fixes one plan policy for
    the whole chain: when the caller asks for ``CURRENT_ONLY`` or any link is
    recovery-shaped, every record from the terminal to the root must pass
    ``_chain_record_admissible`` on the current plan with full provenance.
    Other chains keep each walker's own per-record checks unchanged.
    """
    current = records_by_id.get(comment_id)
    if current is None:
        return None
    current_id = comment_id
    chain: list[tuple[int, ManagedCiIssueAuthorization]] = []
    seen: set[int] = set()
    recovery_context: ChainAuthContext | None = None
    while current.kind == "continuity":
        if (
            current_id in seen
            or current.predecessor_comment_id is None
            or current.predecessor_head is None
        ):
            return None
        seen.add(current_id)
        predecessor_id = current.predecessor_comment_id
        predecessor = records_by_id.get(predecessor_id)
        if predecessor is None or predecessor.head_sha != current.predecessor_head:
            return None
        shape = _continuity_link_shape(
            comments,
            authorization=current,
            predecessor=predecessor,
            records_by_id=records_by_id,
            comments_by_id=comments_by_id,
            chain_context=context,
            expected_seat_binding=expected_seat_binding,
        )
        if shape is None:
            return None
        chain.append((current_id, current))
        if shape == "recovery":
            if recovery_context is None:
                recovery_context = _recovery_context(context, predecessor)
            if predecessor_id in seen or predecessor.predecessor_comment_id is None:
                return None
            seen.add(predecessor_id)
            chain.append((predecessor_id, predecessor))
            current_id = predecessor.predecessor_comment_id
            next_record = records_by_id.get(current_id)
            if next_record is None or next_record.head_sha != predecessor.predecessor_head:
                return None
            current = next_record
            continue
        current_id, current = predecessor_id, predecessor
    if current.kind not in {"creation", "fresh"} or current_id in seen:
        return None
    chain.append((current_id, current))
    if plan_policy is PlanPolicy.CURRENT_ONLY or recovery_context is not None:
        strict_context = recovery_context or context
        for record_id, record in chain:
            if not _chain_record_admissible(
                comments_by_id.get(record_id),
                record,
                strict_context,
                plan_policy=PlanPolicy.CURRENT_ONLY,
                require_provenance=True,
            ):
                return None
    return tuple(record for _record_id, record in chain)


def _authenticated_recovery_root(
    records_by_id: Mapping[int, ManagedCiIssueAuthorization],
    comments_by_id: Mapping[int, Mapping[str, object]],
    comments: list[dict[str, object]],
    context: ChainAuthContext,
    *,
    fresh_comment_id: int,
    predecessor_head: str,
    expected_seat_binding: Mapping[str, object] | None = None,
) -> tuple[int, ManagedCiIssueAuthorization] | None:
    """Resolve and authenticate the original authorization a fresh grant links to.

    The fresh grant must sit at the stranded head and carry an immutable link
    to an earlier, non-superseded authorization at a different head.  That
    original authorization and its whole ancestry must authenticate on the
    current approved plan with full provenance.
    """
    fresh = records_by_id.get(fresh_comment_id)
    if fresh is None or fresh.kind != "fresh":
        return None
    context = _recovery_context(context, fresh)
    if not _chain_record_admissible(
        comments_by_id.get(fresh_comment_id),
        fresh,
        context,
        plan_policy=PlanPolicy.CURRENT_ONLY,
        require_provenance=True,
    ):
        return None
    if (
        fresh.head_sha != predecessor_head
        or fresh.predecessor_head is None
        or fresh.predecessor_comment_id is None
        or fresh.predecessor_head == fresh.head_sha
    ):
        return None
    root_id = fresh.predecessor_comment_id
    root = records_by_id.get(root_id)
    if (
        root is None
        or root_id >= fresh_comment_id
        or root.head_sha != fresh.predecessor_head
        or root.protection != fresh.protection
        or root.waiver != fresh.waiver
        or (root.approved_plan_hash or None) != (context.approved_plan_hash or None)
    ):
        return None
    if _authenticate_authorization_chain(
        root_id,
        records_by_id,
        comments_by_id,
        comments,
        context,
        plan_policy=PlanPolicy.CURRENT_ONLY,
        expected_seat_binding=expected_seat_binding,
    ) is None:
        return None
    return root_id, root


def _decode_recovery_round_record(payload: Mapping[str, object]) -> object | None:
    """Decode a recovery summary record; None when it fails per-phase validation."""
    from .round_metadata import (
        decode_round_metadata_mapping,
        recovery_record_problems,
    )

    try:
        metadata = decode_round_metadata_mapping(payload)
    except Exception:  # noqa: BLE001 - any decode failure leaves the record unusable
        return None
    if metadata.phase not in _CONTINUITY_RECOVERY_PHASES or recovery_record_problems(metadata):
        return None
    return metadata


def _seat_binding_key(binding: Mapping[str, object] | None) -> str:
    return json.dumps(binding, sort_keys=True, separators=(",", ":"))


def _historical_bindings_match(
    comments: list[dict[str, object]],
    comment_ids: Sequence[int],
    by_index: Mapping[int, Mapping[str, object]],
    *,
    actor_id: int | None,
) -> bool | None:
    """Whether every listed round record carries its governing board's binding.

    ``None`` when the PR carries no board-change record (#1376), so callers
    keep their current-board rule.  Otherwise each record must carry a
    binding its governing lineage element allows; any lineage defect fails.
    """
    from .board_change import lineage_binding_resolver

    resolver = lineage_binding_resolver(comments, actor_id=actor_id)
    if resolver is None:
        return None
    wanted = set(comment_ids)
    for index, comment in enumerate(comments):
        if comment.get("id") not in wanted:
            continue
        metadata = by_index.get(index)
        if metadata is None or not metadata.get("seat_binding_valid", False):
            return False
        allowed = {_seat_binding_key(binding) for binding in resolver(index, metadata["round_number"])}
        if _seat_binding_key(metadata.get("seat_binding")) not in allowed:
            return False
    return True


def _rejected_recovery_round_ids(
    comments: list[dict[str, object]],
    by_index: Mapping[int, Mapping[str, object]],
    *,
    actor_login: str,
    actor_id: int,
    root_comment_id: int,
    root_head: str,
    fresh_comment_id: int,
    fresh_head: str,
    predecessor_head: str,
    new_head: str,
    round_number: int,
    expected_seat_binding: Mapping[str, object] | None,
) -> tuple[int, ...] | None:
    """Return the exact round records of one rejected-coder recovery transition.

    The chain is: seat-bound blocking reviews of the original head for round
    R (newer than the original authorization), one rejected attempt-1
    follow-up from that head to the stranded head (older than the fresh
    grant), one attempt-2 recovery dispatch on the stranded head (newer than
    the fresh grant), and one round R+1 coder record at the new head.  Any
    missing, extra, malformed, foreign or misordered link yields None.
    """
    if (
        fresh_head != predecessor_head
        or root_head == fresh_head
        or new_head in {root_head, fresh_head}
    ):
        return None
    reviewers: list[tuple[int, Mapping[str, object]]] = []
    rejected: list[tuple[int, Mapping[str, object]]] = []
    dispatches: list[tuple[int, Mapping[str, object]]] = []
    original_dispatches: list[int] = []
    coders: list[tuple[int, Mapping[str, object]]] = []
    for index, comment in enumerate(comments):
        metadata = by_index.get(index)
        user = comment.get("user") if isinstance(comment.get("user"), dict) else {}
        comment_id = comment.get("id")
        if (
            metadata is None
            or user.get("login") != actor_login
            or user.get("id") != actor_id
            or not isinstance(comment_id, int)
        ):
            continue
        role = metadata["role"]
        phase = metadata.get("phase")
        if is_complete_code_review(metadata):
            if (
                metadata["state"] == "blocking"
                and metadata["subject"] == root_head
                and metadata["round_number"] == round_number
                and comment_id > root_comment_id
            ):
                reviewers.append((comment_id, metadata))
        elif role == "summary" and phase == _CONTINUITY_REJECTED_PHASE:
            # Candidates are every record of this dispatch (raw dispatch round
            # R) or this round, counted before any field or position check so
            # an extra malformed or misplaced record fails closed.
            if (
                metadata["round_number"] == round_number
                or metadata.get("dispatch_round") == round_number
            ):
                rejected.append((comment_id, metadata))
        elif role == "summary" and phase == _CONTINUITY_DISPATCH_PHASE:
            # Count every dispatch of round R (or for round R+1) wherever it
            # sits and whatever its fields, so the checks below reject a
            # misordered, duplicate or malformed recovery dispatch instead of
            # a filter hiding it.  Only the valid original attempt-1 dispatch
            # of the root head is not a candidate; the ordering check still
            # requires it to precede the rejection.
            if (
                metadata["round_number"] == round_number + 1
                or metadata.get("dispatch_round") == round_number
            ):
                record = metadata.get("recovery")
                if (
                    record is not None
                    and getattr(record, "dispatch_attempt", None) == 1
                    and getattr(record, "recovery_dispatch", None) is False
                    and getattr(record, "dispatch_round", None) == round_number
                    and getattr(record, "round_number", None) == round_number + 1
                    and getattr(record, "dispatch_head", None) == root_head
                ):
                    original_dispatches.append(comment_id)
                else:
                    dispatches.append((comment_id, metadata))
        elif role == "coder":
            if metadata["round_number"] == round_number + 1:
                coders.append((comment_id, metadata))
    if not reviewers or len(rejected) != 1 or len(dispatches) != 1 or len(coders) != 1:
        return None
    rejected_id, rejected_metadata = rejected[0]
    dispatch_id, dispatch_metadata = dispatches[0]
    coder_id, coder_metadata = coders[0]
    rejected_record = rejected_metadata.get("recovery")
    dispatch_record = dispatch_metadata.get("recovery")
    if rejected_record is None or dispatch_record is None:
        return None
    if not (
        getattr(rejected_record, "dispatch_round", None) == round_number
        and getattr(rejected_record, "round_number", None) == round_number
        and getattr(rejected_record, "dispatch_head", None) == root_head
        and getattr(rejected_record, "rejected_coder_followup_from_head", None) == root_head
        and getattr(rejected_record, "subject", None) == fresh_head
        and getattr(rejected_record, "dispatch_attempt", None) == 1
        and getattr(rejected_record, "recovery_dispatch", None) is False
    ):
        return None
    if not (
        getattr(dispatch_record, "dispatch_round", None) == round_number
        and getattr(dispatch_record, "round_number", None) == round_number + 1
        and getattr(dispatch_record, "dispatch_head", None) == fresh_head
        and getattr(dispatch_record, "subject", None) == fresh_head
        and getattr(dispatch_record, "dispatch_attempt", None) == 2
        and getattr(dispatch_record, "recovery_dispatch", None) is True
    ):
        return None
    if coder_metadata["subject"] != new_head:
        return None
    reviewer_ids = [comment_id for comment_id, _metadata in reviewers]
    if not (
        max(reviewer_ids) < rejected_id < fresh_comment_id < dispatch_id < coder_id
    ) or any(comment_id > rejected_id for comment_id in original_dispatches):
        return None
    if any(not metadata.get("seat_binding_valid", False) for _id, metadata in reviewers):
        return None
    bindings = {_seat_binding_key(metadata.get("seat_binding")) for _id, metadata in reviewers}
    # Historical reviews are judged by the board that governed their round and
    # comment position in the validated lineage, never the current one: a
    # recorded board change (#1376) may have rebound the board since.
    historical = _historical_bindings_match(
        comments, [comment_id for comment_id, _metadata in reviewers],
        by_index, actor_id=actor_id,
    )
    if historical is not None:
        if not historical:
            return None
    elif expected_seat_binding is not None:
        if bindings != {_seat_binding_key(expected_seat_binding)}:
            return None
    elif len(bindings) != 1:
        return None
    return tuple(sorted({*reviewer_ids, rejected_id, dispatch_id, coder_id}))


def _correlate_rejected_recovery(
    comments: list[dict[str, object]],
    by_index: Mapping[int, Mapping[str, object]],
    *,
    records_by_id: Mapping[int, ManagedCiIssueAuthorization],
    context: ChainAuthContext,
    fresh_comment_id: int,
    predecessor_head: str,
    new_head: str,
    round_number: int,
    expected_seat_binding: Mapping[str, object] | None,
) -> tuple[int, ...] | None:
    """Correlate a recovery transition against its authenticated original authorization."""
    comments_by_id = _comments_by_id(comments)
    root = _authenticated_recovery_root(
        records_by_id,
        comments_by_id,
        comments,
        context,
        fresh_comment_id=fresh_comment_id,
        predecessor_head=predecessor_head,
        expected_seat_binding=expected_seat_binding,
    )
    if root is None:
        return None
    root_id, root_record = root
    fresh = records_by_id[fresh_comment_id]
    return _rejected_recovery_round_ids(
        comments,
        by_index,
        actor_login=context.actor_login,
        actor_id=context.actor_id,
        root_comment_id=root_id,
        root_head=root_record.head_sha,
        fresh_comment_id=fresh_comment_id,
        fresh_head=fresh.head_sha,
        predecessor_head=predecessor_head,
        new_head=new_head,
        round_number=round_number,
        expected_seat_binding=expected_seat_binding,
    )


def _recovery_continuity_is_valid(
    comments: list[dict[str, object]],
    *,
    authorization: ManagedCiIssueAuthorization,
    records_by_id: Mapping[int, ManagedCiIssueAuthorization],
    comments_by_id: Mapping[int, Mapping[str, object]],
    context: ChainAuthContext,
    expected_seat_binding: Mapping[str, object] | None,
) -> bool:
    """Re-authenticate a published or pending recovery continuity record."""
    fresh_comment_id = authorization.predecessor_comment_id
    if (
        fresh_comment_id is None
        or authorization.predecessor_head is None
        or authorization.actor_login.casefold() != context.actor_login.casefold()
        or authorization.actor_id != context.actor_id
    ):
        return False
    try:
        by_index = _continuity_round_records(comments)
    except AgentLoopError:
        return False
    wanted = set(authorization.round_comment_ids)
    if not wanted or len(wanted) != len(authorization.round_comment_ids):
        return False
    bound = [
        (comment.get("id"), by_index.get(index))
        for index, comment in enumerate(comments)
        if comment.get("id") in wanted
    ]
    if any(metadata is None for _cid, metadata in bound):
        return False
    if any(
        is_complete_code_review(metadata) and metadata["subject"] == authorization.predecessor_head
        for _cid, metadata in bound
    ):
        return False
    coder_rounds = {
        metadata["round_number"]
        for _cid, metadata in bound
        if metadata["role"] == "coder" and metadata["subject"] == authorization.head_sha
    }
    if len(coder_rounds) != 1:
        return False
    round_number = next(iter(coder_rounds)) - 1
    if round_number < 1:
        return False
    expected = _correlate_rejected_recovery(
        comments,
        by_index,
        records_by_id=records_by_id,
        context=context,
        fresh_comment_id=fresh_comment_id,
        predecessor_head=authorization.predecessor_head,
        new_head=authorization.head_sha,
        round_number=round_number,
        expected_seat_binding=expected_seat_binding,
    )
    return expected is not None and expected == tuple(sorted(wanted))


@_surfaces_list_failures
def publish_issue_created_authorization(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    handoff: AuthenticatedIssueCreatedHandoff,
    metadata: PullRequestMetadata,
    approved_plan_hash: str | None = None,
) -> AuthenticatedIssueCreatedHandoff:
    """Persist the creation checkpoint before validating coder test evidence."""
    if handoff.override_nonce is None:
        return handoff
    event = _active_managed_label_event(runner, config=config, pr_number=handoff.pr_number)
    if event is None or event[1].casefold() != handoff.trusted_actor_login.casefold() or event[2] != handoff.trusted_actor_id:
        raise AgentLoopError(
            _with_reason(
                "Managed-CI issue-created authorization requires an actor-owned managed-label event.",
                _last_list_failure(),
            )
        )
    waiver = _waiver_for_protection(handoff.protection_mode)
    if waiver is None:
        raise AgentLoopError(
            "Managed-CI issue-created authorization requires an explicitly waived protection state; "
            f"the handoff carries {handoff.protection_mode}."
        )
    expected = ManagedCiIssueAuthorization(
        kind="creation",
        repository=config.repo,
        issue_number=handoff.issue_number,
        pr_number=handoff.pr_number,
        base_ref=handoff.base_ref,
        head_sha=handoff.head_sha,
        actor_login=handoff.trusted_actor_login,
        actor_id=handoff.trusted_actor_id,
        protection=handoff.protection_mode,
        waiver=waiver,
        nonce=handoff.override_nonce,
        label_event_id=event[0],
        approved_plan_hash=approved_plan_hash,
    )
    # Authentication may have preceded timeline/comment inspection. Re-read
    # the complete opening tuple and label event immediately before either an
    # idempotent success or a publication write so a raced head/base/body or
    # label transition cannot inherit the earlier authorization.
    _issue_created_tuple(
        runner,
        config=config,
        pr_number=handoff.pr_number,
        issue_number=handoff.issue_number,
        metadata=replace(
            metadata,
            head_branch=handoff.branch,
            head_sha=handoff.head_sha,
            base_branch=handoff.base_ref,
        ),
        expected_branch=handoff.branch,
        expected_nonce=handoff.override_nonce,
        protection_mode=handoff.protection_mode,
        lifecycle="draft-labeled",
    )
    current_event = _active_managed_label_event(
        runner, config=config, pr_number=handoff.pr_number
    )
    if current_event != event:
        raise AgentLoopError(
            _with_reason(
                "Managed-CI issue-created authorization label provenance changed before publication.",
                _last_list_failure(),
            )
        )
    records = _authorization_comment_records(
        runner,
        config=config,
        pr_number=handoff.pr_number,
        actor_login=handoff.trusted_actor_login,
        actor_id=handoff.trusted_actor_id,
    )
    for comment_id, record in records:
        if record == expected:
            return replace(
                handoff,
                active_label_event_id=event[0],
                authorization_kind="creation",
                authorization_comment_id=comment_id,
                approved_plan_hash=approved_plan_hash,
            )
        if (
            record.kind == "creation"
            and record.repository.casefold() == config.repo.casefold()
            and record.issue_number == handoff.issue_number
            and record.pr_number == handoff.pr_number
        ):
            raise AgentLoopError(
                "Managed-CI issue-created authorization has conflicting PR-comment provenance."
            )
    body = format_issue_created_authorization_comment(expected)
    comment_id = post_verified_trusted_pr_protocol_comment(
        runner,
        config=config,
        pr_number=handoff.pr_number,
        body=body,
        expected_author_login=handoff.trusted_actor_login,
        expected_author_id=handoff.trusted_actor_id,
    )
    return replace(
        handoff,
        active_label_event_id=event[0],
        authorization_kind="creation",
        authorization_comment_id=comment_id,
        approved_plan_hash=approved_plan_hash,
    )


def _continuity_live_pr_tuple(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    handoff: AuthenticatedIssueCreatedHandoff,
    new_head: str,
) -> tuple[object, ...]:
    """Read the immutable PR tuple used by a continuity publication."""
    live_pr = managed_ci_api._api_json(runner, config, f"repos/{config.repo}/pulls/{handoff.pr_number}")
    live_head = live_pr.get("head") if isinstance(live_pr.get("head"), dict) else {}
    live_base = live_pr.get("base") if isinstance(live_pr.get("base"), dict) else {}
    live_author = live_pr.get("user") if isinstance(live_pr.get("user"), dict) else {}
    head_repo = live_head.get("repo") if isinstance(live_head.get("repo"), dict) else {}
    labels = tuple(sorted(
        item.get("name")
        for item in (live_pr.get("labels") or [])
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    ))
    if (
        live_pr.get("state") not in {None, "open", "OPEN"}
        or live_head.get("sha") != new_head
        or live_head.get("ref") != handoff.branch
        or head_repo.get("full_name", "").casefold() != config.repo.casefold()
        or live_base.get("ref") != handoff.base_ref
        or live_author.get("login", "").casefold() != handoff.trusted_actor_login.casefold()
        or live_author.get("id") != handoff.trusted_actor_id
    ):
        raise AgentLoopError(
            "Managed-CI head continuity observed a changed live PR tuple; no continuity record was written."
        )
    return (
        live_pr.get("state"),
        live_pr.get("draft"),
        live_head.get("repo", {}).get("full_name") if isinstance(live_head.get("repo"), dict) else None,
        live_head.get("ref"),
        live_head.get("sha"),
        live_base.get("ref"),
        live_author.get("login"),
        live_author.get("id"),
        labels,
    )


def _stranded_continuity_comment_ids(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    handoff: AuthenticatedIssueCreatedHandoff,
    records: Collection[tuple[int, ManagedCiIssueAuthorization]],
    predecessor: ManagedCiIssueAuthorization,
    predecessor_comment_ids: Collection[int],
    new_head: str,
) -> tuple[int, ...]:
    """Return this actor's unbound records a continuity record retires (#1069).

    A record is retired when GitHub cannot prove its head is an ancestor of
    the new head, so it can never reach it.  Only records that are otherwise
    compatible are in scope: same repository/issue/PR, base, actor, an
    actor-owned managed-label event, the handoff's approved plan, and the
    protection and waiver of the chain this record extends.  A record whose
    recorded protection differs is not retired here: ordinary recovery and
    the resume audit refuse on it before any continuity round can run, because
    which protection state governs the PR is an operator decision that the
    explicit fresh grant adjudicates (#1065).  The predecessor's own ancestry
    is never retired, and any record that differs in base, actor, label
    provenance, or plan is left in place for the existing readers to refuse on.
    """
    by_comment_id = dict(records)
    chain_ids: set[int] = set(predecessor_comment_ids)
    pending = list(predecessor_comment_ids)
    while pending:
        current = by_comment_id.get(pending.pop())
        parent_id = current.predecessor_comment_id if current is not None else None
        if parent_id is not None and parent_id not in chain_ids:
            chain_ids.add(parent_id)
            pending.append(parent_id)
    candidates = [
        (comment_id, record)
        for comment_id, record in records
        if comment_id not in chain_ids
        and record.repository.casefold() == config.repo.casefold()
        and record.issue_number == handoff.issue_number
        and record.pr_number == handoff.pr_number
        and record.base_ref == handoff.base_ref
        and record.actor_login.casefold() == handoff.trusted_actor_login.casefold()
        and record.actor_id == handoff.trusted_actor_id
        and (record.approved_plan_hash or None) == (handoff.approved_plan_hash or None)
        and record.protection == predecessor.protection
        and record.waiver == predecessor.waiver
        and record.head_sha != new_head
    ]
    if not candidates:
        return ()
    label_events = _managed_label_event_history(
        runner,
        config=config,
        pr_number=handoff.pr_number,
        actor_login=handoff.trusted_actor_login,
        actor_id=handoff.trusted_actor_id,
    )
    if label_events is None:
        # Without label provenance no record can be proven compatible, so
        # none is retired; publication proceeds exactly as before.
        return ()
    valid_label_event_ids = {event_id for event_id, _login, _actor_id in label_events}
    stranded: set[int] = set()
    for comment_id, record in candidates:
        if record.label_event_id not in valid_label_event_ids:
            continue
        try:
            reaches = _github_proves_descendant(
                runner,
                config=config,
                predecessor_head=record.head_sha,
                live_head=new_head,
            )
        except AgentLoopError:
            # A failed comparison proves nothing either way: keep the record
            # rather than retire it or block the continuity write.
            continue
        if not reaches:
            stranded.add(comment_id)
    return tuple(sorted(stranded))


@_surfaces_list_failures
def publish_issue_created_continuity_authorization(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    handoff: AuthenticatedIssueCreatedHandoff,
    predecessor_head: str,
    new_head: str,
    round_comment_ids: tuple[int, ...],
) -> AuthenticatedIssueCreatedHandoff:
    """Authorize one orchestrator-produced fix head through a PR comment.

    A coder push is not authority by itself.  The caller reaches this seam only
    after the dispatched round and live exact head have been validated; this
    function binds that predecessor/new-head transition to the prior durable
    authorization and a non-empty set of round metadata identities.
    """
    if handoff.override_nonce is None:
        return handoff
    if not predecessor_head or not new_head or predecessor_head == new_head:
        raise AgentLoopError("Managed-CI head continuity requires two distinct exact heads.")
    if not round_comment_ids or any(value <= 0 for value in round_comment_ids):
        raise AgentLoopError("Managed-CI head continuity requires durable round metadata identities.")
    event = _active_managed_label_event(runner, config=config, pr_number=handoff.pr_number)
    if (
        event is None
        or event[1].casefold() != handoff.trusted_actor_login.casefold()
        or event[2] != handoff.trusted_actor_id
    ):
        raise AgentLoopError(
            _with_reason(
                "Managed-CI head continuity requires an actor-owned active managed-label event.",
                _last_list_failure(),
            )
        )
    live_tuple = _continuity_live_pr_tuple(
        runner,
        config=config,
        handoff=handoff,
        new_head=new_head,
    )
    records = _authorization_comment_records(
        runner,
        config=config,
        pr_number=handoff.pr_number,
        actor_login=handoff.trusted_actor_login,
        actor_id=handoff.trusted_actor_id,
    )
    predecessor_records = [
        (comment_id, record)
        for comment_id, record in records
        if record.head_sha == predecessor_head
        and record.repository.casefold() == config.repo.casefold()
        and record.issue_number == handoff.issue_number
        and record.pr_number == handoff.pr_number
        and record.base_ref == handoff.base_ref
        and record.kind in {"creation", "fresh", "continuity"}
    ]
    # A signed rebind that kept the head leaves the retired-plan grant next to
    # the live-plan grant at this head (#993).  It is history, not a second
    # predecessor, but only when every non-plan field is still consistent.
    retired_history = [
        record
        for _comment_id, record in predecessor_records
        if _is_retired_plan_history(record, handoff)
    ]
    if any(
        record.actor_login.casefold() != handoff.trusted_actor_login.casefold()
        or record.actor_id != handoff.trusted_actor_id
        or record.protection != handoff.protection_mode
        or record.waiver != _waiver_for_protection(record.protection)
        for record in retired_history
    ):
        raise AgentLoopError(
            "Managed-CI head continuity found conflicting prior authorizations for the predecessor head; refusing to proceed."
        )
    predecessor_records = [
        (comment_id, record)
        for comment_id, record in predecessor_records
        if not _is_retired_plan_history(record, handoff)
    ]
    if not predecessor_records:
        raise AgentLoopError(
            "Managed-CI head continuity has no unique prior authorization for the predecessor head."
        )
    if len({record for _comment_id, record in predecessor_records}) != 1:
        raise AgentLoopError(
            "Managed-CI head continuity found conflicting prior authorizations for the predecessor head; refusing to proceed."
        )
    predecessor_comment_id, predecessor = sorted(predecessor_records, key=lambda item: item[0])[-1]
    predecessor_comment_ids = {
        comment_id
        for comment_id, record in predecessor_records
        if record == predecessor
    }
    # Classify stranded records first (#1069): a child left by an interrupted
    # attempt whose head the branch has since discarded cannot reach the new
    # head, so the new record retires it rather than treating it as a
    # competing fork.  A child that can still reach the new head remains a
    # fork and refuses.
    superseded_comment_ids = _stranded_continuity_comment_ids(
        runner,
        config=config,
        handoff=handoff,
        records=records,
        predecessor=predecessor,
        predecessor_comment_ids=predecessor_comment_ids,
        new_head=new_head,
    )
    predecessor_children = [
        (comment_id, record)
        for comment_id, record in records
        if (
            record.kind == "continuity"
            and record.predecessor_head == predecessor_head
            and record.predecessor_comment_id in predecessor_comment_ids
        )
    ]
    # The exemption holds only when this call will publish the superseding
    # record.  An existing child at the new head is reused or refused below,
    # never re-published, so the fork it forms with a stranded sibling is
    # still on record and must refuse.
    fork_exempt = (
        frozenset(superseded_comment_ids)
        if not any(record.head_sha == new_head for _cid, record in predecessor_children)
        else frozenset()
    )
    if any(
        record.head_sha != new_head
        for comment_id, record in predecessor_children
        if comment_id not in fork_exempt
    ):
        raise AgentLoopError(
            "Managed-CI head continuity found a forked predecessor authorization; refusing to proceed."
        )
    normalized_round_comment_ids = tuple(sorted(set(round_comment_ids)))
    # A fresh predecessor may be the stranded head of a rejected coder
    # follow-up (#1367).  Its recovery shape is authenticated through the
    # original authorization with the same strict context every later reader
    # uses; ordinary shapes never consult it.
    records_by_id = dict(records)
    recovery_context: ChainAuthContext | None = None
    recovery_seat_binding: Mapping[str, object] | None = None
    if predecessor.kind == "fresh":
        from .reviewer_seats import reviewer_seat_binding

        recovery_context = chain_auth_context_for_handoff(
            config=config,
            handoff=handoff,
            valid_label_event_ids=load_actor_label_event_ids(
                runner,
                config=config,
                pr_number=handoff.pr_number,
                actor_login=handoff.trusted_actor_login,
                actor_id=handoff.trusted_actor_id,
            ),
        )
        recovery_seat_binding = reviewer_seat_binding(config)

    def round_metadata_valid(
        round_comments: list[dict[str, object]],
        authorization: ManagedCiIssueAuthorization,
    ) -> bool:
        shape = _continuity_link_shape(
            round_comments,
            authorization=authorization,
            predecessor=predecessor,
            records_by_id=records_by_id,
            chain_context=recovery_context,
            expected_seat_binding=recovery_seat_binding,
        )
        if shape == "recovery":
            return recovery_context is not None and _pending_authorization_fields_admissible(
                authorization, recovery_context, predecessor=predecessor
            )
        return shape is not None

    def revalidate_before_publication(authorization: ManagedCiIssueAuthorization) -> None:
        """Reject a live tuple, label, round, or authorization race before writing."""
        current_event = _active_managed_label_event(
            runner, config=config, pr_number=handoff.pr_number
        )
        event_reason = _last_list_failure()
        current_tuple = _continuity_live_pr_tuple(
            runner,
            config=config,
            handoff=handoff,
            new_head=new_head,
        )
        if current_event != event or current_tuple != live_tuple:
            raise AgentLoopError(
                _with_reason(
                    "Managed-CI head continuity observed a changed live PR tuple or active "
                    "managed-label event; no continuity record was written.",
                    event_reason,
                )
            )
        current_round_comments, list_reason = _api_list_detailed(
            runner, config, f"repos/{config.repo}/issues/{handoff.pr_number}/comments?per_page=100"
        )
        if current_round_comments is None or not round_metadata_valid(
            current_round_comments, authorization
        ):
            raise AgentLoopError(
                _with_reason(
                    "Managed-CI head continuity requires authenticated, correlated round metadata.",
                    list_reason,
                )
            )
        current_records = _authorization_comment_records(
            runner,
            config=config,
            pr_number=handoff.pr_number,
            actor_login=handoff.trusted_actor_login,
            actor_id=handoff.trusted_actor_id,
        )
        if current_records != records:
            raise AgentLoopError(
                "Managed-CI head continuity authorization records changed before publication; "
                "no continuity record was written."
            )

    same_transition = [
        (comment_id, record)
        for comment_id, record in records
        if record.kind == "continuity"
        and record.repository.casefold() == config.repo.casefold()
        and record.issue_number == handoff.issue_number
        and record.pr_number == handoff.pr_number
        and record.base_ref == handoff.base_ref
        and record.head_sha == new_head
        and record.predecessor_head == predecessor_head
        and record.predecessor_comment_id in predecessor_comment_ids
        and record.round_comment_ids == normalized_round_comment_ids
        and record.actor_login.casefold() == handoff.trusted_actor_login.casefold()
        and record.actor_id == handoff.trusted_actor_id
        and record.protection == predecessor.protection
        and record.waiver == predecessor.waiver
        and record.label_event_id == event[0]
        and (record.approved_plan_hash or None) == (handoff.approved_plan_hash or None)
    ]
    if same_transition:
        if len({record for _comment_id, record in same_transition}) != 1:
            raise AgentLoopError(
                "Managed-CI head continuity found a conflicting record; refusing to proceed."
            )
        comment_id, existing_record = sorted(same_transition, key=lambda item: item[0])[-1]
        revalidate_before_publication(existing_record)
        return replace(
            handoff,
            head_sha=new_head,
            active_label_event_id=event[0],
            authorization_kind="continuity",
            authorization_comment_id=comment_id,
            override_nonce=existing_record.nonce,
        )
    authorization = ManagedCiIssueAuthorization(
        kind="continuity",
        repository=config.repo,
        issue_number=handoff.issue_number,
        pr_number=handoff.pr_number,
        base_ref=handoff.base_ref,
        head_sha=new_head,
        actor_login=handoff.trusted_actor_login,
        actor_id=handoff.trusted_actor_id,
        protection=predecessor.protection,
        waiver=predecessor.waiver,
        nonce=secrets.token_urlsafe(24),
        label_event_id=event[0],
        predecessor_head=predecessor_head,
        predecessor_comment_id=predecessor_comment_id,
        round_comment_ids=normalized_round_comment_ids,
        approved_plan_hash=handoff.approved_plan_hash,
        superseded_comment_ids=superseded_comment_ids,
    )
    if superseded_comment_ids:
        log(
            config,
            f"PR #{handoff.pr_number}: continuity authorization supersedes "
            f"{len(superseded_comment_ids)} earlier unbound actor-owned authorization "
            f"record(s) ({', '.join(str(value) for value in superseded_comment_ids)})",
        )
    round_comments, list_reason = _api_list_detailed(
        runner, config, f"repos/{config.repo}/issues/{handoff.pr_number}/comments?per_page=100"
    )
    if round_comments is None or not round_metadata_valid(round_comments, authorization):
        raise AgentLoopError(
            _with_reason(
                "Managed-CI head continuity requires authenticated, correlated round metadata.",
                list_reason,
            )
        )
    revalidate_before_publication(authorization)
    for comment_id, record in records:
        if record == authorization:
            return replace(
                handoff,
                head_sha=new_head,
                active_label_event_id=event[0],
                authorization_kind="continuity",
                authorization_comment_id=comment_id,
                override_nonce=authorization.nonce,
            )
        if (
            record.kind == "continuity"
            and record.repository.casefold() == config.repo.casefold()
            and record.issue_number == handoff.issue_number
            and record.pr_number == handoff.pr_number
            and record.predecessor_head == predecessor_head
            and record.head_sha == new_head
            and record != authorization
        ):
            raise AgentLoopError("Managed-CI head continuity found a conflicting record; refusing to proceed.")
    comment_id = post_verified_trusted_pr_protocol_comment(
        runner,
        config=config,
        pr_number=handoff.pr_number,
        body=format_issue_created_authorization_comment(authorization),
        expected_author_login=handoff.trusted_actor_login,
        expected_author_id=handoff.trusted_actor_id,
    )
    return replace(
        handoff,
        head_sha=new_head,
        active_label_event_id=event[0],
        authorization_kind="continuity",
        authorization_comment_id=comment_id,
        override_nonce=authorization.nonce,
    )


def _timeline_source_repository(source_issue: dict[str, object]) -> str | None:
    """Return the repository identity GitHub attached to a timeline issue."""
    repository = source_issue.get("repository")
    if isinstance(repository, dict):
        full_name = repository.get("full_name")
        if isinstance(full_name, str) and full_name:
            return full_name

    repository_url = source_issue.get("repository_url")
    if not isinstance(repository_url, str) or not repository_url:
        return None
    parts = [part for part in urlparse(repository_url).path.split("/") if part]
    try:
        repos_index = next(index for index, part in enumerate(parts) if part.casefold() == "repos")
    except StopIteration:
        return None
    if repos_index + 2 >= len(parts):
        return None
    owner, name = parts[repos_index + 1:repos_index + 3]
    if not owner or not name:
        return None
    return f"{owner}/{name}"


@_surfaces_list_failures
def authorize_fresh_issue_created_resume(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    issue_number: int,
    metadata: PullRequestMetadata,
    approved_plan_hash: str | None = None,
    retired_plan_hashes: Collection[str] = (),
) -> AuthenticatedIssueCreatedHandoff:
    """Create a new operator grant for a PR whose original checkpoint is absent.

    ``retired_plan_hashes`` names approved plans that a verified signed
    child-plan supersession replaced (#993).  An authorization recorded under
    such a plan is history, not a competing grant, so its plan hash alone does
    not make it conflicting.  Every other field is still compared, and a plan
    divergence with no verified supersession edge still refuses.
    """
    retired = frozenset(retired_plan_hashes)
    if approved_plan_hash is None or approved_plan_hash in retired:
        retired = frozenset()
    if not config.allow_unprotected_managed_ci:
        raise AgentLoopError(
            "Managed-CI fresh authorization requires --allow-unprotected-managed-ci."
        )
    actor_login, actor_id = _authorization_actor(runner, config=config)
    protection = assess_exact_head_protection(
        runner,
        context=ManagedCiProbeContext(
            config.repo, config.gh_cmd, github_api_cwd()
        ),
        base=config.base,
    )
    if protection.state not in waivable_protection_states(config):
        if protection.state in _OVERRIDE_PROTECTION_STATES:
            raise AgentLoopError(
                "Managed-CI fresh authorization for a base whose protection is "
                f"{protection.state} requires {waiver_flags_for_protection(protection.state)}; "
                "no authorization record was written."
            )
        raise AgentLoopError(
            "Managed-CI fresh authorization is only available for an authenticated "
            "unprotected, plan-limited, or explicitly waived unreadable base; the current "
            f"protection assessment is {protection.state}."
        )
    branch = metadata.head_branch or ""
    expected_branch = f"agent-loop/managed-{issue_number}"
    if branch != expected_branch:
        raise AgentLoopError(
            f"Managed-CI fresh authorization requires reserved branch `{expected_branch}`; "
            f"the live PR uses `{branch or '<missing>'}`."
        )
    if not config.base or metadata.base_branch != config.base:
        raise AgentLoopError("Managed-CI fresh authorization does not match the selected base branch.")
    if metadata.head_sha is None or metadata.repo.casefold() != config.repo.casefold():
        raise AgentLoopError("Managed-CI fresh authorization requires a same-repository exact live head.")
    pr = managed_ci_api._api_json(runner, config, f"repos/{config.repo}/pulls/{pr_number}")
    head = pr.get("head") if isinstance(pr.get("head"), dict) else {}
    base = pr.get("base") if isinstance(pr.get("base"), dict) else {}
    head_repo = head.get("repo") if isinstance(head.get("repo"), dict) else {}
    author = pr.get("user") if isinstance(pr.get("user"), dict) else {}
    labels = {
        item.get("name") for item in (pr.get("labels") or [])
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    if (
        pr.get("state") not in {None, "open", "OPEN"}
        or head_repo.get("full_name", "").casefold() != config.repo.casefold()
        or head.get("ref") != expected_branch
        or head.get("sha") != metadata.head_sha
        or base.get("ref") != config.base
        or author.get("login", "").casefold() != actor_login.casefold()
        or author.get("id") != actor_id
    ):
        raise AgentLoopError(
            "Managed-CI fresh authorization could not authenticate the live PR tuple; "
            "the PR was left unchanged."
        )
    label_events = _managed_label_event_history(
        runner,
        config=config,
        pr_number=pr_number,
        actor_login=actor_login,
        actor_id=actor_id,
    )
    if not label_events:
        raise AgentLoopError(
            _with_reason(
                "Managed-CI fresh authorization requires an actor-owned managed-label event.",
                _last_list_failure() if label_events is None else "",
            )
        )
    label_event = label_events[-1]
    valid_label_event_ids = {event_id for event_id, _login, _actor_id in label_events}

    def revalidate_live_authorization_tuple() -> None:
        live_pr = managed_ci_api._api_json(
            runner, config, f"repos/{config.repo}/pulls/{pr_number}", quiet=True
        )
        live_head = live_pr.get("head") if isinstance(live_pr.get("head"), dict) else {}
        live_base = live_pr.get("base") if isinstance(live_pr.get("base"), dict) else {}
        live_repo = live_head.get("repo") if isinstance(live_head.get("repo"), dict) else {}
        live_author = (
            live_pr.get("user") if isinstance(live_pr.get("user"), dict) else {}
        )
        live_labels = {
            item.get("name")
            for item in (live_pr.get("labels") or [])
            if isinstance(item, dict) and isinstance(item.get("name"), str)
        }
        live_event = _historical_managed_label_event(
            runner,
            config=config,
            pr_number=pr_number,
            actor_login=actor_login,
            actor_id=actor_id,
        )
        if (
            live_pr.get("state") not in {None, "open", "OPEN"}
            or live_pr.get("draft") is not pr.get("draft")
            or (MANAGED_LABEL in live_labels) != (MANAGED_LABEL in labels)
            or live_repo.get("full_name", "").casefold() != config.repo.casefold()
            or live_head.get("ref") != expected_branch
            or live_head.get("sha") != metadata.head_sha
            or live_base.get("ref") != config.base
            or live_author.get("login", "").casefold() != actor_login.casefold()
            or live_author.get("id") != actor_id
            or live_event != label_event
        ):
            raise AgentLoopError(
                "Managed-CI fresh authorization observed a changed live PR tuple; "
                "no authorization record was written."
            )

    records = _authorization_comment_records(
        runner,
        config=config,
        pr_number=pr_number,
        actor_login=actor_login,
        actor_id=actor_id,
    )
    existing = [
        (comment_id, record)
        for comment_id, record in records
        if record.kind in {"creation", "fresh", "continuity"}
        and record.repository.casefold() == config.repo.casefold()
        and record.issue_number == issue_number
        and record.pr_number == pr_number
        and record.base_ref == config.base
        and record.head_sha == metadata.head_sha
        and record.actor_login.casefold() == actor_login.casefold()
        and record.actor_id == actor_id
        and record.protection == protection.state
        and record.protection in waivable_protection_states(config)
        and record.waiver == _waiver_for_protection(record.protection)
        and record.label_event_id in valid_label_event_ids
        and (record.approved_plan_hash or None) == (approved_plan_hash or None)
    ]
    scoped_records = [
        (comment_id, record)
        for comment_id, record in records
        if record.repository.casefold() == config.repo.casefold()
        and record.issue_number == issue_number
        and record.pr_number == pr_number
    ]

    def conflicts_beyond_protection(record: ManagedCiIssueAuthorization) -> bool:
        return (
            record.base_ref != config.base
            or record.actor_login.casefold() != actor_login.casefold()
            or record.actor_id != actor_id
            or record.label_event_id not in valid_label_event_ids
            or (
                (record.approved_plan_hash or None) != (approved_plan_hash or None)
                and record.approved_plan_hash not in retired
            )
        )

    def protection_conflicts(record: ManagedCiIssueAuthorization) -> bool:
        return (
            record.protection != protection.state
            or record.protection not in waivable_protection_states(config)
            or record.waiver != _waiver_for_protection(record.protection)
        )

    incompatible_records = [
        record
        for _comment_id, record in scoped_records
        if conflicts_beyond_protection(record) or protection_conflicts(record)
    ]
    # Records that differ from this grant only in their recorded protection
    # are stale history from an earlier host or assessment (#1065).  The
    # explicit fresh grant retires every prior scoped record by publishing a
    # superseding record instead of refusing on the conflict it exists to
    # clear.  Any other divergence (base, actor, label provenance, or an
    # unretired plan) is not supersedable and still refuses.
    supersede = bool(incompatible_records) and not any(
        conflicts_beyond_protection(record) for record in incompatible_records
    )
    if existing and not supersede:
        revalidate_live_authorization_tuple()
        distinct_existing = {record for _comment_id, record in existing}
        if len(distinct_existing) > 1:
            creation_records = {
                record for record in distinct_existing if record.kind == "creation"
            }
            fresh_records = {
                record for record in distinct_existing if record.kind == "fresh"
            }
            if (
                len(creation_records) == 1
                and len(fresh_records) == 1
                and len(distinct_existing) == 2
            ):
                existing = [item for item in existing if item[1].kind == "fresh"]
            else:
                raise AgentLoopError(
                    "Managed-CI fresh authorization found conflicting exact-head "
                    "authorization records; refusing to proceed."
                )
        comment_id, record = sorted(existing, key=lambda item: item[0])[-1]
        return AuthenticatedIssueCreatedHandoff(
            pr_number=pr_number,
            issue_number=issue_number,
            repository=config.repo,
            base_ref=config.base,
            head_sha=metadata.head_sha,
            branch=expected_branch,
            trusted_actor_login=actor_login,
            trusted_actor_id=actor_id,
            protection_mode=record.protection,
            override_nonce=record.nonce,
            opening_override_nonce=_opening_override_nonce(metadata.body),
            active_label_event_id=record.label_event_id,
            lifecycle=(
                "draft-labeled" if MANAGED_LABEL in labels
                else "draft-unlabeled-reentry" if pr.get("draft") is True
                else "ready-unlabeled-reentry"
            ),
            authorization_kind=record.kind,
            authorization_comment_id=comment_id,
            approved_plan_hash=record.approved_plan_hash,
            retired_plan_hashes=retired,
        )
    issue_timeline, timeline_reason = _api_list_detailed(
        runner,
        config,
        f"repos/{config.repo}/issues/{issue_number}/timeline?per_page=100",
    )
    associated = False
    if issue_timeline is not None:
        for timeline_event in issue_timeline:
            source = (
                timeline_event.get("source")
                if isinstance(timeline_event.get("source"), dict) else {}
            )
            source_issue = source.get("issue") if isinstance(source.get("issue"), dict) else {}
            if (
                timeline_event.get("event") == "cross-referenced"
                and source_issue.get("number") == pr_number
                and isinstance(source_issue.get("pull_request"), dict)
                and (
                    source_repository := _timeline_source_repository(source_issue)
                ) is not None
                and source_repository.casefold() == config.repo.casefold()
            ):
                associated = True
                break
    if not associated:
        raise AgentLoopError(
            _with_reason(
                "Managed-CI fresh authorization requires a server-observed issue-to-PR "
                "association for the explicit issue scope.",
                timeline_reason,
            )
        )
    if incompatible_records and not supersede:
        raise AgentLoopError(
            "Managed-CI fresh authorization found a conflicting actor-owned record whose base, "
            "actor, label provenance, or approved plan differs; only records that differ in "
            "recorded protection can be superseded. Refusing to proceed."
        )
    predecessor: tuple[int, ManagedCiIssueAuthorization] | None = None
    for candidate in (
        () if supersede
        else sorted(scoped_records, key=lambda item: item[0], reverse=True)
    ):
        # A signed rebind changes only the plan, not the PR head (#993).  A
        # grant at the live head under a verified retired plan is therefore
        # the predecessor of this plan transition; any other same-head record
        # would have been reused above or refused as incompatible.
        if (
            candidate[1].head_sha == metadata.head_sha
            and candidate[1].approved_plan_hash in retired
        ):
            predecessor = candidate
            break
        if _github_proves_descendant(
            runner,
            config=config,
            predecessor_head=candidate[1].head_sha,
            live_head=metadata.head_sha,
        ):
            predecessor = candidate
            break
    if scoped_records and predecessor is None:
        # Compatible grants that GitHub cannot chain to the live head are
        # unbound history too (#1065): none reaches the live head, so none is
        # evidence for it.  The explicit fresh grant, issued only after the
        # live tuple and server-observed issue association checks above,
        # retires them instead of refusing.
        supersede = True
    superseded_comment_ids = (
        tuple(sorted({comment_id for comment_id, _record in scoped_records}))
        if supersede else ()
    )
    authorization = ManagedCiIssueAuthorization(
        kind="fresh",
        repository=config.repo,
        issue_number=issue_number,
        pr_number=pr_number,
        base_ref=config.base,
        head_sha=metadata.head_sha,
        actor_login=actor_login,
        actor_id=actor_id,
        protection=protection.state,
        waiver=_waiver_for_protection(protection.state),
        nonce=secrets.token_urlsafe(24),
        label_event_id=label_event[0],
        predecessor_head=(predecessor[1].head_sha if predecessor is not None else None),
        predecessor_comment_id=(predecessor[0] if predecessor is not None else None),
        approved_plan_hash=approved_plan_hash,
        superseded_comment_ids=superseded_comment_ids,
    )
    if supersede:
        log(
            config,
            f"PR #{pr_number}: fresh managed-CI authorization supersedes "
            f"{len(superseded_comment_ids)} earlier actor-owned authorization record(s) "
            f"({', '.join(str(value) for value in superseded_comment_ids)})",
        )
    revalidate_live_authorization_tuple()
    latest_records = _authorization_comment_records(
        runner,
        config=config,
        pr_number=pr_number,
        actor_login=actor_login,
        actor_id=actor_id,
    )
    if latest_records != records:
        raise AgentLoopError(
            "Managed-CI fresh authorization records changed before publication; "
            "no competing authorization was written."
        )
    comment_id = post_verified_trusted_pr_protocol_comment(
        runner,
        config=config,
        pr_number=pr_number,
        body=format_issue_created_authorization_comment(authorization),
        expected_author_login=actor_login,
        expected_author_id=actor_id,
    )
    return AuthenticatedIssueCreatedHandoff(
        pr_number=pr_number,
        issue_number=issue_number,
        repository=config.repo,
        base_ref=config.base,
        head_sha=metadata.head_sha,
        branch=expected_branch,
        trusted_actor_login=actor_login,
        trusted_actor_id=actor_id,
        protection_mode=protection.state,
        override_nonce=authorization.nonce,
        opening_override_nonce=_opening_override_nonce(metadata.body),
        active_label_event_id=label_event[0],
        lifecycle=(
            "draft-labeled" if MANAGED_LABEL in labels
            else "draft-unlabeled-reentry" if pr.get("draft") is True
            else "ready-unlabeled-reentry"
        ),
        authorization_kind="fresh",
        authorization_comment_id=comment_id,
        approved_plan_hash=approved_plan_hash,
        retired_plan_hashes=retired,
    )


@_surfaces_list_failures
def revalidate_issue_created_handoff(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    handoff: AuthenticatedIssueCreatedHandoff,
    metadata: PullRequestMetadata,
) -> AuthenticatedIssueCreatedHandoff:
    """Re-read an authenticated handoff at PR-loop entry before any writer."""
    if handoff.repository.casefold() != config.repo.casefold() or handoff.base_ref != config.base:
        raise AgentLoopError("Managed-CI handoff does not belong to this invocation.")
    if handoff.authorization_kind == "fresh":
        refreshed = authorize_fresh_issue_created_resume(
            runner,
            config=config,
            pr_number=handoff.pr_number,
            issue_number=handoff.issue_number,
            metadata=metadata,
            approved_plan_hash=handoff.approved_plan_hash,
            retired_plan_hashes=handoff.retired_plan_hashes,
        )
        # A fresh retry may find a continuity terminal for the live head.  It
        # is still a fresh operator invocation, but the terminal's nonce is
        # not the nonce embedded in the PR-opening body.  Re-enter the normal
        # exact-tuple validator so the terminal is checked with the preserved
        # opening nonce and its trusted label binding before activation.
        if refreshed.authorization_kind == "continuity":
            return revalidate_issue_created_handoff(
                runner,
                config=config,
                handoff=refreshed,
                metadata=metadata,
            )
        return refreshed
    # Creation and fresh grants can advance to a continuity terminal, but the
    # PR body remains bound to the opening authorization nonce.  Revalidate
    # that immutable opening tuple rather than treating the continuity record's
    # nonce as if it had been written into the PR body.  Recover the opening
    # nonce from the body as a defensive fallback for handoffs assembled from
    # a terminal record without the copied field.
    opening_override_nonce = handoff.opening_override_nonce
    if opening_override_nonce is None:
        opening_override_nonce = _opening_override_nonce(metadata.body)
    validated = _issue_created_tuple(
        runner,
        config=config,
        pr_number=handoff.pr_number,
        issue_number=handoff.issue_number,
        metadata=metadata,
        expected_branch=handoff.branch,
        expected_nonce=opening_override_nonce,
        protection_mode=handoff.protection_mode,
        lifecycle=(
            "ready-unlabeled"
            if handoff.lifecycle == "ready-unlabeled-reentry"
            else "draft-unlabeled"
            if handoff.lifecycle == "draft-unlabeled-reentry"
            else "draft-labeled"
        ),
    )
    if handoff.active_label_event_id is not None and validated.lifecycle != "draft-labeled":
        # A failed activation deliberately releases the suppression label, so
        # an unlabeled re-entry has no active label event by design (#997).
        # `_issue_created_tuple` has already proved the PR is unlabeled; the
        # recorded event must still be one of the trusted actor's own
        # historical applications, and activation will re-apply the label.
        history = _managed_label_event_history(
            runner,
            config=config,
            pr_number=handoff.pr_number,
            actor_login=handoff.trusted_actor_login,
            actor_id=handoff.trusted_actor_id,
        )
        if history is None or handoff.active_label_event_id not in {
            event_id for event_id, _login, _actor_id in history
        }:
            raise AgentLoopError(
                _with_reason(
                    "Managed-CI direct-resume label provenance changed before activation.",
                    _last_list_failure() if history is None else "",
                )
            )
        validated = replace(validated, active_label_event_id=handoff.active_label_event_id)
    elif handoff.active_label_event_id is not None:
        event = _active_managed_label_event(runner, config=config, pr_number=handoff.pr_number)
        if (
            event is None
            or event[0] != handoff.active_label_event_id
            or event[1].casefold() != handoff.trusted_actor_login.casefold()
            or event[2] != handoff.trusted_actor_id
        ):
            raise AgentLoopError(
                _with_reason(
                    "Managed-CI direct-resume label provenance changed before activation.",
                    _last_list_failure(),
                )
            )
        validated = replace(validated, active_label_event_id=handoff.active_label_event_id)
    # `_issue_created_tuple` deliberately returns the nonce authenticated by
    # the PR-opening body.  That nonce is only the body-validation binding;
    # retain the handoff's terminal nonce as the managed-resume identity so a
    # continuity retry cannot silently fall back to the opening grant.
    terminal_override_nonce = handoff.override_nonce
    return replace(
        validated,
        authorization_kind=handoff.authorization_kind,
        authorization_comment_id=handoff.authorization_comment_id,
        override_nonce=terminal_override_nonce,
        opening_override_nonce=opening_override_nonce,
        approved_plan_hash=handoff.approved_plan_hash,
        retired_plan_hashes=handoff.retired_plan_hashes,
    )


_MANAGED_ISSUE_BRANCH_RE = re.compile(r"agent-loop/managed-([1-9]\d*)")


def managed_branch_issue_number(branch: str | None) -> int | None:
    """Return the issue named by a whole ``agent-loop/managed-N`` branch (#1513).

    ``N`` must be a positive integer without leading zeros, and any prefix or
    suffix text rejects the branch.  The value is a parse result only and
    grants no authority; callers still authenticate the PR it belongs to.
    """
    if not isinstance(branch, str):
        return None
    match = _MANAGED_ISSUE_BRANCH_RE.fullmatch(branch)
    return int(match.group(1)) if match is not None else None


@_surfaces_list_failures
def recover_issue_created_handoff(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    metadata: PullRequestMetadata,
    issue_number: int | None = None,
) -> AuthenticatedIssueCreatedHandoff | None:
    """Reconstruct a durable issue-created resume without reusing authority.

    Canonical issue reruns and direct PR reruns use the same read-only proof.
    The body nonce, when present, is provenance only; the current lifecycle,
    head, author, base, label, and issue association are reauthenticated.
    """
    if not (config.managed_ci_pr_mode or issue_number is not None):
        return None
    body = metadata.body or ""
    branch = metadata.head_branch or ""
    branch_issue_number = managed_branch_issue_number(branch)
    if branch_issue_number is None:
        return None
    if issue_number is not None and issue_number != branch_issue_number:
        command = render_managed_ci_resume_command(
            config,
            pr_number=pr_number,
            issue_number=issue_number,
            managed_ci=True,
        )
        raise AgentLoopError(
            f"Managed-CI issue association targets issue #{issue_number}, but the authenticated "
            f"managed branch targets issue #{branch_issue_number}. The PR was left unchanged. "
            f"Rerun the canonical issue command after correcting the issue association: `{command}`"
        )
    issue_number = branch_issue_number
    record = None
    if UNPROTECTED_OVERRIDE_TRAILER in body:
        record = parse_managed_ci_override_record(
            body, surface=PR_BODY_SURFACE, schema="body", required=True,
        )
    pr = read_opening_pr(runner, config, pr_number, phase="lifecycle-classification")
    labels = {
        item.get("name") for item in (pr.get("labels") or [])
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    draft = pr.get("draft")
    if draft is True and MANAGED_LABEL in labels:
        lifecycle = "draft-labeled"
    elif draft is True and MANAGED_LABEL not in labels and config.managed_ci:
        # A failed explicit managed-CI invocation deliberately releases its
        # suppression label.  Re-admit that durable draft state only when the
        # operator explicitly asks for managed qualification again.
        lifecycle = "draft-unlabeled"
    elif draft is False and MANAGED_LABEL not in labels:
        lifecycle = "ready-unlabeled"
    else:
        if draft is True:
            state = "draft/labeled" if MANAGED_LABEL in labels else "draft/unlabeled"
        elif draft is False:
            state = "ready/labeled" if MANAGED_LABEL in labels else "ready/unlabeled"
        else:
            state = "unknown/labeled" if MANAGED_LABEL in labels else "unknown/unlabeled"
        command = render_managed_ci_resume_command(
            config,
            pr_number=pr_number,
            issue_number=issue_number,
            managed_ci=True,
        )
        raise AgentLoopError(
            f"Managed-CI issue-created resume for PR #{pr_number} observed mixed lifecycle state "
            f"{state}; expected draft/labeled, draft/unlabeled with explicit managed CI, or "
            f"ready/unlabeled. The PR was left unchanged. Resume with `{command}`."
        )
    # A resumed nonce is provenance only.  Its value is checked against the
    # live body, then activation mints a fresh audit/generation.
    handoff = _issue_created_tuple(
        runner,
        config=config,
        pr_number=pr_number,
        issue_number=issue_number,
        metadata=metadata,
        expected_branch=branch,
        expected_nonce=record.nonce if record is not None else None,
        protection_mode="voluntary" if record is not None else "strict",
        lifecycle=lifecycle,
    )
    if lifecycle == "draft-labeled":
        event = _active_managed_label_event(runner, config=config, pr_number=pr_number)
        if (
            event is None
            or event[1].casefold() != handoff.trusted_actor_login.casefold()
            or event[2] != handoff.trusted_actor_id
        ):
            raise AgentLoopError(
                _with_reason(
                    "Managed-CI issue-created resume requires an actor-owned active managed-label event.",
                    _last_list_failure(),
                )
            )
        handoff = replace(handoff, active_label_event_id=event[0])
    if record is not None and config.effective_managed_ci:
        # Only an invocation that will activate managed CI consumes the
        # protection state; a plain review rerun keeps its historical path.
        handoff = _recover_issue_created_protection(
            runner, config=config, handoff=handoff, issue_number=issue_number,
        )
    return handoff


def _recover_issue_created_protection(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    handoff: AuthenticatedIssueCreatedHandoff,
    issue_number: int,
) -> AuthenticatedIssueCreatedHandoff:
    """Restore an override-bearing PR's persisted protection state, read-only.

    The tuple helper seeds ``voluntary``.  A selection pass validates every
    actor-authored authorization record without comparing protection or plan
    scope, and the creation record(s) supply the state.  A confirmation pass
    then requires every record, and the live assessment, to agree with it.
    Plan-scope checks run later at the activation resume-audit gate, after the
    canonical plan is bound and before any mutation.
    """
    pr_number = handoff.pr_number
    command = render_managed_ci_resume_command(
        config, pr_number=pr_number, issue_number=issue_number, managed_ci=True,
    )

    # A remedy must never be the invocation that just failed (#1063): only a
    # transient read failure offers a retry of it, and only a missing waiver
    # offers a (different) resume command.  Everything else is irreconcilable.
    irreconcilable = (
        "No resume flag reconciles this state, so rerunning this command will "
        "refuse again."
    )
    retry = f"This read may be transient; retry `{command}` once it succeeds."

    def refuse(reason: str, *, remedy: str | None = None) -> AgentLoopError:
        return AgentLoopError(
            f"Managed-CI issue-created recovery for PR #{pr_number} refused: {reason}. "
            f"The PR was left unchanged. " + (remedy or irreconcilable)
        )

    def waiver_remedy(state: str) -> str:
        # The remedy must differ from the invocation that just failed (#1063).
        waived = replace(
            config,
            allow_unprotected_managed_ci=True,
            allow_unreadable_protection=(
                config.allow_unreadable_protection or state == "unreadable"
            ),
        )
        resume = render_managed_ci_resume_command(
            waived, pr_number=pr_number, issue_number=issue_number, managed_ci=True,
        )
        return f"Resume with `{resume}`."

    valid_label_event_ids = _actor_owned_label_event_ids(
        runner, config=config, pr_number=pr_number,
        actor_login=handoff.trusted_actor_login, actor_id=handoff.trusted_actor_id,
    )
    if valid_label_event_ids is None:
        raise refuse(
            _with_reason(
                "the managed-label event history could not be inspected", _last_list_failure()
            ),
            remedy=retry,
        )
    comments, reason = _api_list_detailed(
        runner, config, f"repos/{config.repo}/issues/{pr_number}/comments?per_page=100"
    )
    if comments is None:
        raise refuse(
            _with_reason("the authorization comments could not be inspected", reason),
            remedy=retry,
        )
    records: list[tuple[Mapping[str, object], ManagedCiIssueAuthorization]] = []
    for comment in comments:
        body = _normalized_comment_body(comment, config=config)
        user = comment.get("user") if isinstance(comment.get("user"), dict) else {}
        if (
            ISSUE_AUTHORIZATION_MARKER not in body
            or user.get("login") != handoff.trusted_actor_login
            or user.get("id") != handoff.trusted_actor_id
        ):
            continue
        try:
            parsed = parse_issue_created_authorization_comment(body)
        except AgentLoopError as error:
            raise refuse(f"authorization comment {comment.get('id')} is malformed ({error})") from error
        if parsed is None:
            raise refuse(f"authorization comment {comment.get('id')} is malformed")
        records.append((comment, parsed))
    # Superseded records are retired history (#1065, #1069).
    superseded = _superseded_authorization_ids(
        (comment["id"], parsed)
        for comment, parsed in records
        if isinstance(comment.get("id"), int)
    )
    records = [
        (comment, parsed) for comment, parsed in records
        if comment.get("id") not in superseded
    ]

    def validate(check_protection: str | None, candidate: AuthenticatedIssueCreatedHandoff) -> None:
        for comment, authorization in records:
            if not _authorization_record_valid_for_handoff(
                authorization,
                comment,
                config=config,
                handoff=candidate,
                actor_login=handoff.trusted_actor_login,
                actor_id=handoff.trusted_actor_id,
                base_ref=handoff.base_ref,
                pr_number=pr_number,
                issue_number=handoff.issue_number,
                valid_label_event_ids=valid_label_event_ids,
                check_protection=check_protection,
                plan_scope_authenticated=False,
            ):
                raise refuse(
                    f"authorization comment {comment.get('id')} does not match this PR's "
                    f"authenticated tuple, actor, label provenance, or waiver"
                    + (f" for protection {check_protection}" if check_protection else "")
                )

    unwaived = {
        authorization.protection
        for _comment, authorization in records
        if authorization.protection in _OVERRIDE_PROTECTION_STATES
        and authorization.protection not in waivable_protection_states(config)
    }
    if unwaived:
        needed = "unreadable" if "unreadable" in unwaived else next(iter(unwaived))
        record_states = {authorization.protection for _comment, authorization in records}
        if len(record_states) > 1:
            # Every record must later equal one recovered state, so adding
            # waiver flags would only reach an irreconcilable refusal.
            raise refuse(
                "its authorization records disagree on protection ("
                + ", ".join(sorted(record_states)) + ")"
            )
        raise refuse(
            f"its authorization records carry protection {', '.join(sorted(unwaived))}, "
            f"which requires {waiver_flags_for_protection(needed)}",
            remedy=waiver_remedy(needed),
        )
    # Selection pass: protection and plan scope are deliberately deferred.
    validate(None, handoff)
    creation_states = {
        authorization.protection
        for _comment, authorization in records
        if authorization.kind == "creation"
    }
    if not any(authorization.kind == "creation" for _comment, authorization in records):
        # A superseding fresh grant retired the creation root (#1065); the
        # surviving fresh root(s) record the state that grant authorized.
        creation_states = {
            authorization.protection
            for _comment, authorization in records
            if authorization.kind == "fresh"
        }
    context = ManagedCiProbeContext(config.repo, config.gh_cmd, github_api_cwd())
    live = assess_exact_head_protection(runner, context=context, base=handoff.base_ref)
    if len(creation_states) > 1:
        raise refuse(
            "creation authorization records disagree on protection ("
            + ", ".join(sorted(creation_states)) + ")"
        )
    if creation_states:
        recovered = next(iter(creation_states))
    elif live.state in _OVERRIDE_PROTECTION_STATES:
        # A legacy PR, or a crash before authorization publication.
        recovered = live.state
    else:
        raise refuse(
            f"no creation authorization records the protection state and the live assessment is "
            f"{live.state} ({live.detail})"
        )
    if recovered not in waivable_protection_states(config):
        raise refuse(
            f"its persisted protection is {recovered}, which requires "
            f"{waiver_flags_for_protection(recovered)}",
            remedy=waiver_remedy(recovered),
        )
    # Confirmation pass against the recovered state and the live assessment.
    confirmed = replace(handoff, protection_mode=recovered)
    validate(recovered, confirmed)
    # Any other live/persisted disagreement refuses, including a base that
    # became strict: activation's strict path skips the resume-audit
    # plan-scope gate, so a deferred plan check must never be carried onto it.
    live = reconcile_live_protection(recovered, live, config=config)
    if live.state != recovered:
        raise refuse(
            f"its persisted protection is {recovered}, but the live assessment is "
            f"{live.state} ({live.detail})",
            remedy=f"{irreconcilable} Restore the base's original protection before retrying.",
        )
    return confirmed


def _is_retired_plan_history(
    authorization: ManagedCiIssueAuthorization,
    handoff: AuthenticatedIssueCreatedHandoff | None,
) -> bool:
    """Whether ``authorization`` was granted under a verified retired plan (#993)."""
    return (
        handoff is not None
        and authorization.approved_plan_hash is not None
        and authorization.approved_plan_hash != handoff.approved_plan_hash
        and authorization.approved_plan_hash in handoff.retired_plan_hashes
    )


def _actor_owned_label_event_ids(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    actor_login: str,
    actor_id: int,
) -> set[int] | None:
    """Return every managed-label application by the trusted actor, or None."""
    events = _api_list(
        runner, config, f"repos/{config.repo}/issues/{pr_number}/events?per_page=100"
    )
    if events is None:
        return None
    valid_label_event_ids: set[int] = set()
    for event in events:
        label = event.get("label") if isinstance(event.get("label"), dict) else {}
        event_actor = event.get("actor") if isinstance(event.get("actor"), dict) else {}
        event_id = event.get("id")
        if (
            event.get("event") == "labeled"
            and label.get("name") == MANAGED_LABEL
            and isinstance(event_id, int)
            and event_actor.get("login", "").casefold() == actor_login.casefold()
            and event_actor.get("id") == actor_id
        ):
            valid_label_event_ids.add(event_id)
    return valid_label_event_ids


def _authorization_matches(
    authorization: ManagedCiIssueAuthorization,
    comment_id: int,
    *,
    config: AgentLoopConfig,
    handoff: AuthenticatedIssueCreatedHandoff | None,
    actor_login: str,
    actor_id: int,
    valid_label_event_ids: set[int] | None,
    check_protection: str | None,
    plan_scope_authenticated: bool,
    history: bool = False,
) -> bool:
    if (
        authorization.actor_login.casefold() != actor_login.casefold()
        or authorization.actor_id != actor_id
        or authorization.waiver != _waiver_for_protection(authorization.protection)
        or (
            # ``unreadable`` is honored only when this invocation waives it.
            authorization.protection == "unreadable"
            and "unreadable" not in waivable_protection_states(config)
        )
        or (
            check_protection is not None
            and authorization.protection != check_protection
        )
        or (
            valid_label_event_ids is not None
            and authorization.label_event_id not in valid_label_event_ids
        )
    ):
        return False
    if handoff is None:
        return True
    if (
        authorization.repository.casefold() != handoff.repository.casefold()
        or authorization.pr_number != handoff.pr_number
        or authorization.issue_number != handoff.issue_number
        or authorization.base_ref != handoff.base_ref
        or authorization.actor_login.casefold()
        != handoff.trusted_actor_login.casefold()
        or authorization.actor_id != handoff.trusted_actor_id
        or (
            plan_scope_authenticated
            and (authorization.approved_plan_hash or None)
            != (handoff.approved_plan_hash or None)
        )
    ):
        return False
    # The opening nonce belongs to the authenticated PR body.  Validate it
    # on the creation root even when a later fresh/continuity record is the
    # selected terminal, so a trusted comment cannot launder a mismatched
    # historical root into resume authority.
    if authorization.kind == "creation":
        expected_nonce = handoff.opening_override_nonce
        if (
            expected_nonce is None
            and handoff.authorization_kind == "creation"
            and authorization.head_sha == handoff.head_sha
        ):
            expected_nonce = handoff.override_nonce
        if expected_nonce is not None and authorization.nonce != expected_nonce:
            return False
    if history:
        # A retired-plan record is never the terminal, so the terminal
        # identity checks below do not apply to it (#993).
        return True
    if (
        authorization.kind == "fresh"
        and handoff.authorization_kind == "fresh"
        and authorization.head_sha == handoff.head_sha
        and authorization.nonce != handoff.override_nonce
    ):
        return False
    # The handoff's comment identity is the authenticated terminal record.
    # Historical roots and continuity ancestors remain eligible for chain
    # traversal, but the terminal record may not be replaced by another
    # actor-authored record with the same public tuple.
    if handoff.authorization_comment_id is not None:
        if comment_id == handoff.authorization_comment_id:
            if authorization.kind != handoff.authorization_kind:
                return False
            if authorization.kind == "creation":
                expected_nonce = handoff.opening_override_nonce
                if expected_nonce is None:
                    expected_nonce = handoff.override_nonce
                return expected_nonce is None or authorization.nonce == expected_nonce
            if authorization.kind == "fresh":
                return authorization.nonce == handoff.override_nonce
            # Continuity nonces are minted for the new exact head. The
            # trusted comment identity and validated predecessor chain,
            # rather than the opening-body nonce, bind that terminal.
            return True
        return True
    if authorization.head_sha != handoff.head_sha:
        return True
    # A recovered handoff initially describes the PR-opening body, so its
    # default kind is ``creation`` even when a later continuity record is
    # the terminal authority for the live head. Creation and fresh records
    # must match the nonce for the checkpoint that authenticated them;
    # continuity records carry a new nonce and are validated by their
    # predecessor/round chain instead.
    if authorization.kind == "creation":
        expected_nonce = handoff.opening_override_nonce
        if expected_nonce is None and handoff.authorization_kind == "creation":
            expected_nonce = handoff.override_nonce
        return expected_nonce is None or authorization.nonce == expected_nonce
    if authorization.kind == "fresh":
        if handoff.authorization_kind == "fresh":
            return authorization.nonce == handoff.override_nonce
        return True
    return True


def _authorization_record_valid_for_handoff(
    authorization: ManagedCiIssueAuthorization,
    comment: Mapping[str, object],
    *,
    config: AgentLoopConfig,
    handoff: AuthenticatedIssueCreatedHandoff | None,
    actor_login: str,
    actor_id: int,
    base_ref: str,
    pr_number: int,
    issue_number: int | None,
    valid_label_event_ids: set[int] | None,
    check_protection: str | None,
    plan_scope_authenticated: bool,
) -> bool:
    """Shared per-record validity for resume audits and issue-created recovery.

    The outer comment-author and tuple checks never touch ``handoff``.  With
    no handoff the predicate stops after the actor, waiver, protection and
    label-event checks.  ``plan_scope_authenticated=False`` skips only the
    plan-hash comparisons, for callers that have not yet bound the canonical
    approved plan; those comparisons run later at the activation gate.
    """
    user = comment.get("user") if isinstance(comment.get("user"), dict) else {}
    cid = comment.get("id")
    if (
        user.get("login") != actor_login
        or user.get("id") != actor_id
        or authorization.repository.casefold() != config.repo.casefold()
        or authorization.base_ref != base_ref
        or authorization.pr_number != pr_number
        or (issue_number is not None and authorization.issue_number != issue_number)
        or not isinstance(cid, int)
    ):
        return False
    common = dict(
        config=config,
        actor_login=actor_login,
        actor_id=actor_id,
        valid_label_event_ids=valid_label_event_ids,
        check_protection=check_protection,
        plan_scope_authenticated=plan_scope_authenticated,
    )
    if plan_scope_authenticated and _is_retired_plan_history(authorization, handoff):
        # Only the plan hash is retired; every other field must still match.
        assert handoff is not None
        return _authorization_matches(
            replace(authorization, approved_plan_hash=handoff.approved_plan_hash),
            cid,
            handoff=handoff,
            history=True,
            **common,
        )
    return _authorization_matches(authorization, cid, handoff=handoff, **common)
