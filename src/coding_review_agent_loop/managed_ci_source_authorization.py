"""Source-managed creation authority and round-backed head continuity.

Editable bodies nominate candidates; only actor-authenticated durable comments
can authorize them. This owner deliberately does not import the activation driver.
"""
from __future__ import annotations

import base64
import json
from dataclasses import asdict, dataclass, replace

from . import managed_ci_api as api
from .config import AgentLoopConfig
from .errors import AgentLoopError
from .github import post_verified_trusted_pr_protocol_comment
from .managed_ci_authorization import (
    _ordinary_continuity_round_metadata_is_valid,
    find_actor_round_metadata_comment_ids,
    parse_managed_ci_override_record,
)
from .managed_ci_protection import assess_exact_head_protection, reconcile_live_protection, waivable_protection_states
from .managed_ci_types import ManagedCiProbeContext, ProtectionAssessment, UNPROTECTED_OVERRIDE_TRAILER
from .managed_ci_tuple import MANAGED_LABEL
from .protocol_markers import PR_COMMENT_SURFACE, TrustedBody, protocol_record_label, scan_reserved_markers
from .runner import Runner
from .workdirs import active_workdir

CONTINUITY_MARKER = "AGENT_MANAGED_CI_SOURCE_CONTINUITY_V1"
VOLUNTARY_NOTICE = (
    "Voluntary gate: GitHub cannot prevent manual merges, other automation, "
    "compromised credentials, or an agent-loop defect from bypassing it."
)


def _encode(value):
    return base64.urlsafe_b64encode(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).decode()


def source_recovery_advice(config, pr_number, *, state="unknown", lifecycle=False):
    """Conditional guidance never advertises the unchanged failing command."""
    if "unknown" in state:
        return (f"Measured lifecycle: {state}. Obtain a successful fresh PR state read before "
                "selecting recovery; do not change draft or label state based on this observation.")
    if lifecycle:
        return (f"Measured lifecycle: {state}. With verified source authority and current applicable "
                "waivers, an operator must restore draft/labeled before a managed retry. "
                "This refusal makes no lifecycle writes.")
    return (f"Measured lifecycle: {state}. Resolve unrelated issue association or body blockers first. "
            f"For ordinary recovery, verify `{MANAGED_LABEL}` is absent (only its trusted owner may "
            "remove it), and verify an available unlabeled pull_request workflow route; restore that "
            "route if needed. Explicitly ready the draft if ordinary CI requires it. Only after "
            "those checks, resume ordinary review without managed-CI, waiver, fresh-authorization, "
            f"adoption or auto-merge options: `agent-loop pr {pr_number} --repo {config.repo}`. "
            "A label DELETE acknowledgement alone does not establish that CI resumed; "
            "no issue-created authorization grant was inferred.")


def _state(pr):
    labels = pr.get("labels")
    if not isinstance(labels, list) or any(not isinstance(x, dict) or not isinstance(x.get("name"), str) for x in labels):
        return "unknown/unknown"
    labeled = any(x["name"] == MANAGED_LABEL for x in labels)
    draft = "draft" if pr.get("draft") is True else "ready" if pr.get("draft") is False else "unknown"
    return draft + ("/labeled" if labeled else "/unlabeled")


def source_body_refusal_advice(runner, config, pr_number):
    """Measure lifecycle for early body/identity refusals without any writes."""
    try:
        pr = api._api_json(runner, config, f"repos/{config.repo}/pulls/{pr_number}")
        state = _state(pr)
    except AgentLoopError:
        state = "unknown/unknown"
    return source_recovery_advice(config, pr_number, state=state)


def _comments(runner, config, pr_number):
    values, reason = api._api_list_detailed(runner, config, f"repos/{config.repo}/issues/{pr_number}/comments?per_page=100")
    if values is None:
        raise AgentLoopError(f"Source-managed provenance comments are unavailable: {reason}")
    return values


def _identity(runner, config):
    who = api._api_json(runner, config, "user")
    login, actor_id = who.get("login"), who.get("id")
    advertised = api._advertised_managed_actor(runner, config)
    if (not isinstance(login, str) or not login or type(actor_id) is not int or actor_id <= 0
        or not config.managed_ci_trusted_actor or login.casefold() != config.managed_ci_trusted_actor.casefold()
        or not advertised or login.casefold() != advertised.casefold()):
        raise AgentLoopError("Source-managed provenance requires the configured and advertised trusted actor.")
    return login, actor_id


def _live(runner, config, pr_number, *, source=None, expected_head=None, creation=False):
    # Local import keeps the authorizer below both managed-pr and activation.
    from .managed_pr import recover_managed_pr_origin, validate_managed_pr_body
    pr = api._api_json(runner, config, f"repos/{config.repo}/pulls/{pr_number}")
    actor, actor_id = _identity(runner, config)
    head, base, user = (pr.get(key) for key in ("head", "base", "user"))
    if not all(isinstance(x, dict) for x in (head, base, user)):
        raise AgentLoopError("Source-managed live PR identity is unreadable.")
    repo = head.get("repo")
    branch = head.get("ref")
    if (pr.get("state") != "open" or user.get("login") != actor or type(user.get("id")) is not int
        or user.get("id") != actor_id or not isinstance(repo, dict)
        or str(repo.get("full_name", "")).casefold() != config.repo.casefold()
        or not isinstance(branch, str) or not branch.startswith("agent-loop/managed-direct-")
        or base.get("ref") != config.base or not isinstance(pr.get("body"), str)
        or not isinstance(head.get("sha"), str) or not head["sha"]
        or (expected_head is not None and head.get("sha") != expected_head)):
        raise AgentLoopError("Source-managed live repository, author, branch, base or head does not match.")
    recovered = recover_managed_pr_origin(pr["body"], fetched_head_branch=branch)
    if recovered is None or (source is not None and recovered != source):
        raise AgentLoopError("Source-managed body source record or override nonce is missing or changed.")
    source_branch, source_sha, managed_branch, nonce = recovered
    if config.managed_ci_expected_override_nonce is not None and nonce != config.managed_ci_expected_override_nonce:
        raise AgentLoopError("Source-managed opening nonce does not match this creation handoff.")
    validate_managed_pr_body(pr["body"], source_branch=source_branch, source_sha=source_sha,
        managed_branch=managed_branch, override_nonce=nonce, fetched_head_sha=head["sha"],
        fetched_head_branch=branch, fetched_base_branch=base.get("ref"), expected_base_branch=config.base,
        require_creation_head_sha=creation)
    state = _state(pr)
    if creation and state != "draft/labeled":
        raise AgentLoopError("Source-managed creation root requires draft/labeled.")
    if "unknown" in state:
        raise AgentLoopError(source_recovery_advice(config, pr_number, state=state))
    event = api._active_managed_label_event(runner, config=config, pr_number=pr_number)
    if state.endswith("/labeled"):
        if event is None or event[1:] != (actor, actor_id):
            raise AgentLoopError("Source-managed active label event is not provably actor-owned.")
    else:
        event = api._historical_managed_label_event(runner, config=config, pr_number=pr_number,
            actor_login=actor, actor_id=actor_id)
        if event is None:
            raise AgentLoopError("Source-managed re-entry requires actor-owned label history.")
    if type(event[0]) is not int or event[0] <= 0:
        raise AgentLoopError("Source-managed label event identity is malformed.")
    return pr, recovered, actor, actor_id, event


def _authority_snapshot(live):
    """Compare trust inputs, excluding GitHub's asynchronous computed metadata."""
    pr, source, actor, actor_id, event = live
    return (
        pr.get("state"), pr.get("draft"),
        pr["user"].get("login"), pr["user"].get("id"),
        pr["head"]["repo"].get("full_name"), pr["head"].get("ref"), pr["head"].get("sha"),
        pr["base"].get("ref"), pr.get("body"),
        tuple(sorted(label["name"] for label in pr["labels"])),
        source, actor, actor_id, event,
    )


def _root_body(fields, pr_number):
    required = ("nonce", "repo", "base", "head", "protection")
    line = UNPROTECTED_OVERRIDE_TRAILER + " " + " ".join(f"{key}={fields[key]}" for key in required)
    if "source" in fields:
        line += " source=" + fields["source"]
    label = protocol_record_label("managed_ci_override_audit", pr_number=pr_number, head_sha=fields["head"])
    return TrustedBody.canonical(f"{label}\n\n{line}\n\n{VOLUNTARY_NOTICE}", expected_tokens=(UNPROTECTED_OVERRIDE_TRAILER,))


@dataclass(frozen=True)
class SourceAuthority:
    repository: str
    pr_number: int
    base: str
    source_branch: str
    source_sha: str
    branch: str
    opening_nonce: str
    actor: str
    actor_id: int
    protection: str
    label_event_id: int
    root_id: int
    terminal_id: int
    head: str


@dataclass(frozen=True)
class SourceContinuity:
    version: int
    origin: str
    authority: SourceAuthority
    predecessor_id: int
    predecessor_head: str
    successor_head: str
    round_comment_ids: tuple[int, ...]

    # Structural round-authentication scope: no issue or plan identity is made up.
    @property
    def kind(self):
        return "continuity"

    @property
    def head_sha(self):
        return self.successor_head

    @property
    def actor_login(self):
        return self.authority.actor

    @property
    def actor_id(self):
        return self.authority.actor_id

    @property
    def predecessor_comment_id(self):
        return self.predecessor_id


def format_source_continuity(record):
    encoded = _encode(asdict(record))
    return TrustedBody.canonical(f"Source-managed head continuity audit record\n\n<!-- {CONTINUITY_MARKER}: {encoded} -->",
        surface=PR_COMMENT_SURFACE, expected_tokens=(CONTINUITY_MARKER,))


def parse_source_continuity(body):
    occurrences = [x for x in scan_reserved_markers(body) if x.definition.token == CONTINUITY_MARKER]
    if not occurrences:
        if CONTINUITY_MARKER.casefold() in body.casefold():
            raise AgentLoopError("Malformed source continuity audit record.")
        return None
    if len(occurrences) != 1:
        raise AgentLoopError("Duplicate source continuity records.")
    try:
        match = occurrences[0].definition.pattern.fullmatch(occurrences[0].text)
        payload = json.loads(base64.urlsafe_b64decode(match.group("payload")))
        authority = SourceAuthority(**payload.pop("authority"))
        rounds = payload.pop("round_comment_ids")
        if (not isinstance(rounds, list) or not rounds
            or any(type(x) is not int or x <= 0 for x in rounds) or rounds != sorted(set(rounds))):
            raise ValueError("rounds")
        record = SourceContinuity(authority=authority, round_comment_ids=tuple(rounds), **payload)
        if (type(record.version) is not int or record.version != 1 or record.origin != "source-managed"
            or type(record.predecessor_id) is not int or record.predecessor_id <= 0
            or not isinstance(record.predecessor_head, str) or not record.predecessor_head
            or not isinstance(record.successor_head, str) or not record.successor_head
            or record.predecessor_head == record.successor_head):
            raise ValueError("shape")
        for key, value in asdict(authority).items():
            if key in {"pr_number", "actor_id", "label_event_id", "root_id", "terminal_id"}:
                if type(value) is not int or value <= 0:
                    raise ValueError(key)
            elif not isinstance(value, str) or not value:
                raise ValueError(key)
        if str(format_source_continuity(record)) != body:
            raise ValueError("noncanonical")
        return record
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise AgentLoopError("Malformed source continuity audit record.") from exc


def _roots(comments, *, config, pr_number, source, actor, actor_id, protection, valid_events):
    roots = []
    for comment in comments:
        user = comment.get("user") or {}
        if (not isinstance(user, dict) or user.get("login") != actor
            or type(user.get("id")) is not int or user.get("id") != actor_id):
            continue
        body = api._normalized_comment_body(comment, config=None)
        if not any(x.definition.token == UNPROTECTED_OVERRIDE_TRAILER for x in scan_reserved_markers(body)):
            continue
        record = parse_managed_ci_override_record(body, surface=PR_COMMENT_SURFACE, schema="audit", required=True)
        fields = record.field_map()
        if "resume_from" in fields:
            # A fresh nonce is never opening authority.
            continue
        if fields.get("repo", "").casefold() != config.repo.casefold() or fields.get("base") != config.base:
            raise AgentLoopError("Conflicting source-managed creation audit repository or base.")
        reconciled = reconcile_live_protection(fields.get("protection"),
            ProtectionAssessment(protection, "live", "source authentication"), config=config)
        if (fields.get("nonce") != source[3] or fields.get("head") != source[1]
            or fields.get("protection") != reconciled.state):
            raise AgentLoopError("Conflicting source-managed creation audit nonce, source head or protection.")
        if fields.get("source") is not None:
            try:
                opening = json.loads(base64.urlsafe_b64decode(fields["source"]))
                label_id = opening["label_event_id"]
                expected = {"source": list(source[:3]), "pr": pr_number, "actor": actor,
                            "actor_id": actor_id, "label_event_id": label_id}
                if (opening != expected or type(label_id) is not int or label_id not in valid_events
                    or fields["source"] != _encode(expected)):
                    raise ValueError("opening identity")
            except (ValueError, TypeError, KeyError) as exc:
                raise AgentLoopError("Conflicting source-managed creation identity.") from exc
        if body != str(_root_body(fields, pr_number)):
            # Historical creation audit used the same wire record with no label.
            legacy = str(_root_body(fields, pr_number)).split("\n\n", 1)[1]
            if body != legacy:
                raise AgentLoopError("Malformed source-managed creation audit body.")
        comment_id = comment.get("id")
        if type(comment_id) is not int or comment_id <= 0:
            raise AgentLoopError("Source-managed creation audit has no durable identity.")
        roots.append((comment_id, fields))
    if not roots:
        raise AgentLoopError("No verified source-managed creation audit exists; editable body contents cannot create authority.")
    if any(fields != roots[0][1] for _, fields in roots):
        raise AgentLoopError("Conflicting source-managed creation roots.")
    return min(roots, key=lambda x: x[0])


def _round_valid(comments, record):
    """Reuse ordered round and historical-board correlation without an issue."""
    return _ordinary_continuity_round_metadata_is_valid(comments, authorization=record)


def authenticate_source_authority(runner, *, config, pr_number, expected_head=None, source=None, protection=None):
    pr, source, actor, actor_id, event = _live(runner, config, pr_number, source=source, expected_head=expected_head)
    if protection is None:
        assessment = assess_exact_head_protection(runner, context=ManagedCiProbeContext(config.repo, config.gh_cmd, active_workdir(config)), base=config.base)
        protection = assessment.state
    comments = _comments(runner, config, pr_number)
    # Label identity on continuity ancestors is historical; every event must
    # still authenticate to this actor. The active application is checked above.
    history = api._managed_label_event_history(runner, config=config, pr_number=pr_number, actor_login=actor, actor_id=actor_id)
    if history is None:
        raise AgentLoopError("Source-managed label history is unavailable.")
    valid_events = {x[0] for x in history}
    root_id, fields = _roots(comments, config=config, pr_number=pr_number, source=source,
        actor=actor, actor_id=actor_id, protection=protection, valid_events=valid_events)
    root = SourceAuthority(config.repo, pr_number, config.base, *source[:3], source[3], actor, actor_id,
        fields["protection"], event[0], root_id, root_id, source[1])
    records = []
    for comment in comments:
        user = comment.get("user") or {}
        if (not isinstance(user, dict) or user.get("login") != actor
            or type(user.get("id")) is not int or user.get("id") != actor_id):
            continue
        record = parse_source_continuity(api._normalized_comment_body(comment, config=None))
        if record is not None:
            records.append((comment.get("id"), record))
    current = root
    visited = {root.head}
    consumed = set()
    while True:
        outgoing = [(cid, r) for cid, r in records if r.predecessor_id == current.terminal_id or r.predecessor_head == current.head]
        if not outgoing:
            break
        if len(outgoing) != 1:
            raise AgentLoopError("Forked source-managed continuity history.")
        cid, record = outgoing[0]
        expected = replace(current, label_event_id=record.authority.label_event_id)
        if (record.authority != expected or record.authority.label_event_id not in valid_events
            or record.predecessor_id != current.terminal_id or record.predecessor_head != current.head
            or type(cid) is not int or cid <= max(record.round_comment_ids)
            or record.successor_head in visited or not _round_valid(comments, record)):
            raise AgentLoopError("Source-managed continuity is not an authenticated gap-free round-backed chain.")
        visited.add(record.successor_head)
        consumed.add(cid)
        current = replace(current, terminal_id=cid, head=record.successor_head)
    if len(consumed) != len(records):
        raise AgentLoopError("Source-managed continuity contains an unreachable link.")
    if expected_head is not None and current.head != expected_head:
        raise AgentLoopError("No trusted source-managed continuity authorizes the live head.")
    final_protection = assess_exact_head_protection(runner,
        context=ManagedCiProbeContext(config.repo, config.gh_cmd, active_workdir(config)), base=config.base)
    final_protection = reconcile_live_protection(current.protection, final_protection, config=config)
    if final_protection.state != current.protection or current.protection not in waivable_protection_states(config):
        raise AgentLoopError("Source-managed protection or current explicit waiver changed during authentication.")
    if _authority_snapshot(_live(runner, config, pr_number, source=source, expected_head=expected_head)) != _authority_snapshot((pr, source, actor, actor_id, event)):
        raise AgentLoopError("Source-managed live tuple or label event changed during authentication.")
    return current


def publish_source_creation_root(runner, *, config, pr_number, source, protection):
    before = _live(runner, config, pr_number, source=source, expected_head=source[1], creation=True)
    assessment = assess_exact_head_protection(runner, context=ManagedCiProbeContext(config.repo, config.gh_cmd, active_workdir(config)), base=config.base)
    assessment = reconcile_live_protection(protection, assessment, config=config)
    if assessment.state != protection or protection not in waivable_protection_states(config):
        raise AgentLoopError("Source-managed creation protection or current waiver changed.")
    comments = _comments(runner, config, pr_number)
    relevant = any(UNPROTECTED_OVERRIDE_TRAILER in api._normalized_comment_body(c, config=None) for c in comments)
    if relevant:
        return authenticate_source_authority(runner, config=config, pr_number=pr_number,
            expected_head=source[1], source=source, protection=protection)
    if _authority_snapshot(_live(runner, config, pr_number, source=source, expected_head=source[1], creation=True)) != _authority_snapshot(before):
        raise AgentLoopError("Source-managed creation tuple or label event changed before audit publication.")
    final_protection = assess_exact_head_protection(runner,
        context=ManagedCiProbeContext(config.repo, config.gh_cmd, active_workdir(config)), base=config.base)
    if reconcile_live_protection(protection, final_protection, config=config).state != protection:
        raise AgentLoopError("Source-managed protection changed before creation audit publication.")
    fields = dict(nonce=source[3], repo=config.repo, base=config.base, head=source[1], protection=protection,
        source=_encode({"source": list(source[:3]), "pr": pr_number, "actor": before[2],
                        "actor_id": before[3], "label_event_id": before[4][0]}))
    post_verified_trusted_pr_protocol_comment(runner, config=config, pr_number=pr_number,
        body=_root_body(fields, pr_number), expected_author_login=before[2], expected_author_id=before[3])
    return authenticate_source_authority(runner, config=config, pr_number=pr_number,
        expected_head=source[1], source=source, protection=protection)


def publish_source_continuity(runner, *, config, pr_number, predecessor_head, new_head, round_number):
    # Authenticate the old chain while the live tuple already names the accepted new head.
    authority = authenticate_source_authority(runner, config=config, pr_number=pr_number)
    if authority.head != predecessor_head or new_head == predecessor_head:
        raise AgentLoopError("Source-managed coder transition has no unique predecessor.")
    rounds = find_actor_round_metadata_comment_ids(runner, config=config, pr_number=pr_number,
        actor_login=authority.actor, actor_id=authority.actor_id, predecessor_head=predecessor_head,
        new_head=new_head, round_number=round_number, after_comment_id=authority.terminal_id)
    record = SourceContinuity(1, "source-managed", authority, authority.terminal_id, predecessor_head, new_head, rounds)
    before = _live(runner, config, pr_number, expected_head=new_head)
    comments = _comments(runner, config, pr_number)
    if not _round_valid(comments, record):
        raise AgentLoopError("Source-managed coder transition requires authenticated round evidence.")
    if authenticate_source_authority(runner, config=config, pr_number=pr_number) != authority or _authority_snapshot(_live(runner, config, pr_number, expected_head=new_head)) != _authority_snapshot(before):
        raise AgentLoopError("Source-managed coder tuple or provenance changed before publication.")
    post_verified_trusted_pr_protocol_comment(runner, config=config, pr_number=pr_number,
        body=format_source_continuity(record), expected_author_login=authority.actor, expected_author_id=authority.actor_id)
    return authenticate_source_authority(runner, config=config, pr_number=pr_number, expected_head=new_head)
