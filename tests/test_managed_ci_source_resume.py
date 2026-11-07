"""Source-managed creation/restart integration against a stateful GitHub transport."""
import copy
import json
from dataclasses import replace

import pytest

from agent_loop_helpers import make_config
from coding_review_agent_loop import managed_ci, managed_ci_source_authorization as source_auth
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.managed_pr import create_managed_pr, _compose_body
from coding_review_agent_loop.managed_ci_types import ManagedCiCreationIntent, UNPROTECTED_OVERRIDE_TRAILER
from coding_review_agent_loop.runner import CommandResult
from test_managed_ci import ManualQualificationRunner, SUPPRESSING_V2_WORKFLOW, label_event, metadata, _round_comment

SHA = "6367c48dd193d56ea7b0baad25b19455e529f5ee"
BRANCH = "agent-loop/managed-direct-1234-test"


class SourceRunner(ManualQualificationRunner):
    """Persist real creation and audit writes, then serve them to fresh callers."""
    def __init__(self, **kwargs):
        kwargs.setdefault("authenticated_actor", (kwargs.get("actor_login", "agent-loop"), 1))
        super().__init__(workflow=SUPPRESSING_V2_WORKFLOW, issue_events=[label_event(login=kwargs.get("actor_login", "agent-loop"))], **kwargs)
        self.rest_pr["head"]["ref"] = BRANCH
        self.rest_pr["body"] = str(_compose_body("", source_branch="fix/source", source_sha=SHA, override_nonce="opening-nonce"))
        self.created = False
        self.fail_audit = False
        self.fail_readback = False
        self.cleanup_failure = False
        self.before_publication = None

    def _maybe_advance_pr_head_for_coder_followup(self, cmd):
        super()._maybe_advance_pr_head_for_coder_followup(cmd)
        if '"kind": "coder_followup"' in "\n".join(cmd):
            self.rest_pr["head"]["sha"] = self.pr_payload["headRefOid"]

    def _run_locked(self, args, *, cwd, check, input_text=None):
        cmd = list(args)
        endpoint = next((x for x in cmd if isinstance(x, str) and x.startswith("repos/")), "")
        method = cmd[cmd.index("--method") + 1] if "--method" in cmd else "GET"
        def result(payload, code=0):
            command, path = self._record_command(args, cwd)
            return CommandResult(command, path, json.dumps(payload), "fixture failure" if code else "", code)
        if cmd[:3] == ["gh", "pr", "comment"]:
            posted = super()._run_locked(args, cwd=cwd, check=check, input_text=input_text)
            if posted.returncode == 0:
                comment = self.pr_payload["comments"][-1]
                comment_id = self._next_comment_id()
                comment["id"] = comment_id
                self.audit_comments[comment_id] = {"id": comment_id, "body": comment["body"],
                    "user": {"login": self.actor_login, "id": self.actor_id},
                    "created_at": comment["createdAt"]}
            return posted
        if cmd[:3] == ["gh", "pr", "ready"]:
            self.rest_pr["draft"] = "--undo" in cmd
            return result({})
        if cmd[:3] == ["gh", "pr", "view"]:
            if "--jq" in cmd and ".headRefOid" in cmd:
                command, path = self._record_command(args, cwd)
                return CommandResult(command, path, self.rest_pr["head"]["sha"] + "\n", "", 0)
            payload = dict(self.pr_payload, number=7, headRefName=self.rest_pr["head"]["ref"],
                headRefOid=self.rest_pr["head"]["sha"], baseRefName=self.rest_pr["base"]["ref"],
                body=self.rest_pr["body"], isDraft=self.rest_pr["draft"], state="OPEN",
                labels=self.rest_pr["labels"], comments=self.pr_payload.get("comments", []))
            return result(payload)
        if endpoint.startswith("repos/OWNER/REPO/git/ref/heads/"):
            return result({"ref": "refs/heads/" + self.rest_pr["head"]["ref"], "object": {"sha": self.rest_pr["head"]["sha"]}})
        if endpoint.endswith("/git/refs") and method == "POST":
            return result({})
        if "/git/refs/" in endpoint and method == "DELETE":
            return result({}, int(self.cleanup_failure))
        if endpoint.startswith("repos/OWNER/REPO/branches/") and "/protection" not in endpoint:
            return result({"commit": {"sha": SHA}})
        if endpoint.endswith("/pulls?state=open"):
            return result([{"number": 7, "state": "open"}] if self.created else [])
        if endpoint == "repos/OWNER/REPO/pulls" and method == "POST":
            payload = json.loads(input_text)
            self.created = True
            self.rest_pr["head"].update(ref=payload["head"], sha=SHA)
            self.rest_pr.update(body=payload["body"], draft=True)
            return result({"number": 7})
        if endpoint == "repos/OWNER/REPO/pulls/7" and method == "PATCH":
            self.rest_pr.update(json.loads(input_text))
            return result({}, int(self.cleanup_failure))
        if endpoint.endswith("/labels/agent-loop-managed") and method == "GET":
            return result({"name": "agent-loop-managed"})
        if endpoint.endswith("/issues/7/comments") and method == "POST":
            if self.before_publication:
                self.before_publication()
            if self.fail_audit:
                return result({}, 1)
        if endpoint.startswith("repos/OWNER/REPO/issues/comments/") and method == "GET" and self.fail_readback:
            return result({}, 1)
        if endpoint.startswith("repos/OWNER/REPO/issues/7/comments?"):
            if self.fail_readback and self.audit_comments:
                return result([])
            return result(self.intent_comments + list(self.audit_comments.values()))
        return super()._run_locked(args, cwd=cwd, check=check, input_text=input_text)


def config(tmp_path, **kwargs):
    options = dict(managed_ci=True, managed_ci_pr_mode=True,
        managed_ci_trusted_actor="agent-loop", allow_unprotected_managed_ci=True)
    options.update(kwargs)
    return make_config(tmp_path, **options)


def create(runner, tmp_path):
    return create_managed_pr(runner, config=replace(config(tmp_path), managed_ci_pr_mode=False),
        source_branch="fix/source", title="Prepared change", body="Prepared code")


def root(runner, tmp_path):
    return source_auth.publish_source_creation_root(runner, config=config(tmp_path), pr_number=7,
        source=("fix/source", SHA, BRANCH, "opening-nonce"), protection="voluntary")


def activate(runner, tmp_path, **kwargs):
    md = replace(metadata(), head_branch=runner.rest_pr["head"]["ref"],
        head_sha=runner.rest_pr["head"]["sha"], body=runner.rest_pr["body"])
    return managed_ci.activate_managed_ci(runner, config=config(tmp_path, **kwargs), pr_number=7, metadata=md)


def assert_no_qualification(runner):
    assert runner.dispatch_count == 0
    assert not any(cmd[:3] in (["gh", "pr", "merge"], ["gh", "workflow", "run"]) for cmd, _ in runner.commands)


def test_creation_publishes_verified_root_before_return_and_first_activation_reuses_it(tmp_path):
    runner = SourceRunner()
    handoff = create(runner, tmp_path)
    assert handoff.source_authority.root_id > 0
    roots_before = len(runner.audit_comments)
    contract = activate(runner, tmp_path)
    assert contract and contract.origin == "source-managed"
    roots = [x for x in runner.audit_comments.values() if "resume_from=" not in x["body"]]
    assert len(roots) == roots_before == 1
    assert runner.rest_pr["labels"] == [{"name": "agent-loop-managed"}]
    assert not any("DELETE" in cmd for cmd, _ in runner.commands)


def test_repeated_fresh_resume_uses_opening_root_with_distinct_nonce_and_generation(tmp_path):
    runner = SourceRunner()
    authority = root(runner, tmp_path)
    first = activate(runner, tmp_path)
    second = activate(runner, tmp_path)
    assert first.audit_nonce != second.audit_nonce != authority.opening_nonce
    assert first.intent_generation != second.intent_generation
    assert first.audit_comment_id != second.audit_comment_id
    assert all(f"resume_from={authority.root_id}" in c["body"] for c in runner.audit_comments.values() if "resume_from=" in c["body"])
    assert not any("DELETE" in cmd for cmd, _ in runner.commands)


@pytest.mark.parametrize("failure", ["post", "readback", "tuple", "body-race", "label-race", "head-race", "cleanup"])
def test_creation_audit_failure_never_returns_handoff_and_cleans_partial_draft(tmp_path, failure):
    runner = SourceRunner()
    runner.fail_audit = failure in {"post", "cleanup"}
    runner.fail_readback = failure == "readback"
    runner.cleanup_failure = failure == "cleanup"
    if failure in {"tuple", "body-race", "label-race", "head-race"}:
        original = runner._run_locked
        reads = 0
        def mutate(args, **kwargs):
            nonlocal reads
            if "repos/OWNER/REPO/pulls/7" in args:
                reads += 1
                if reads == 2:
                    if failure == "tuple":
                        runner.rest_pr["base"]["ref"] = "other"
                    elif failure == "body-race":
                        runner.rest_pr["body"] = "changed source identity"
                    elif failure == "head-race":
                        runner.rest_pr["head"]["sha"] = "a" * 40
                    else:
                        runner.issue_events.append(label_event(999, login="other", actor_id=9))
            return original(args, **kwargs)
        runner._run_locked = mutate
    with pytest.raises(AgentLoopError) as error:
        create(runner, tmp_path)
    if failure == "cleanup":
        assert "cleanup was incomplete" in str(error.value)
    else:
        assert runner.rest_pr["state"] == "closed"
    assert any("DELETE" in cmd and any("/git/refs/" in x for x in cmd) for cmd, _ in runner.commands)
    assert_no_qualification(runner)


def test_equivalent_creation_publication_is_idempotent_and_conflicting_root_refuses(tmp_path):
    runner = SourceRunner()
    first = root(runner, tmp_path)
    assert root(runner, tmp_path) == first
    assert len(runner.audit_comments) == 1
    comment = copy.deepcopy(next(iter(runner.audit_comments.values())))
    comment["id"] += 10
    comment["body"] = comment["body"].replace("opening-nonce", "other-nonce")
    runner.intent_comments.append(comment)
    with pytest.raises(AgentLoopError, match="Conflicting"):
        activate(runner, tmp_path)
    assert_no_qualification(runner)


def test_auditless_legacy_draft_refuses_with_conditional_ordinary_recovery(tmp_path):
    runner = SourceRunner()
    with pytest.raises(AgentLoopError) as error:
        activate(runner, tmp_path)
    message = str(error.value)
    assert message.count("Measured lifecycle:") == 1
    assert "No verified source-managed creation audit" in message
    assert "verify" in message and "unlabeled" in message and "ordinary review" in message
    assert "Resume with" not in message
    assert "--managed-ci --" not in message
    assert_no_qualification(runner)


def test_source_coder_continuity_two_round_backed_advances_and_restarts(tmp_path):
    runner = SourceRunner()
    first = root(runner, tmp_path)
    predecessor = SHA
    for index, successor in enumerate(("a" * 40, "b" * 40)):
        offset = 100 + index * 100
        runner.intent_comments.extend([
            _round_comment(offset, role="reviewer", subject=predecessor, round_number=index * 2 + 1, state="blocking"),
            _round_comment(offset + 1, role="coder", subject=successor, round_number=index * 2 + 2),
        ])
        runner.rest_pr["head"]["sha"] = successor
        published = source_auth.publish_source_continuity(runner, config=config(tmp_path), pr_number=7,
            predecessor_head=predecessor, new_head=successor, round_number=index * 2 + 1)
        assert published.root_id == first.root_id and published.head == successor
        assert activate(runner, tmp_path).origin == "source-managed"
        predecessor = successor
    assert not any("DELETE" in cmd for cmd, _ in runner.commands)


@pytest.mark.parametrize("damage", ["unsolicited", "missing-round", "tampered-round", "fork", "gap"])
def test_unproven_continuity_refuses_without_dispatch_or_synthetic_evidence(tmp_path, damage):
    runner = SourceRunner()
    root(runner, tmp_path)
    successor = "a" * 40
    if damage != "unsolicited":
        runner.intent_comments.extend([
            _round_comment(100, role="reviewer", subject=SHA, round_number=1, state="blocking"),
            _round_comment(101, role="coder", subject=successor, round_number=2),
        ])
        runner.rest_pr["head"]["sha"] = successor
        source_auth.publish_source_continuity(runner, config=config(tmp_path), pr_number=7,
            predecessor_head=SHA, new_head=successor, round_number=1)
        if damage == "missing-round":
            runner.intent_comments.pop(0)
        elif damage == "tampered-round":
            runner.intent_comments[0]["user"]["id"] = 999
        elif damage == "gap":
            runner.rest_pr["head"]["sha"] = "b" * 40
        else:
            comment = copy.deepcopy(max(runner.audit_comments.values(), key=lambda x: x["id"]))
            comment["id"] += 100
            runner.intent_comments.append(comment)
    else:
        runner.rest_pr["head"]["sha"] = successor
    count = len(runner.intent_comments) + len(runner.audit_comments)
    with pytest.raises(AgentLoopError):
        activate(runner, tmp_path)
    assert len(runner.intent_comments) + len(runner.audit_comments) == count
    assert_no_qualification(runner)


@pytest.mark.parametrize("field,value", [
    ("user.login", "other"), ("user.id", 99), ("head.repo.full_name", "FORK/REPO"),
    ("head.ref", "agent-loop/managed-elsewhere"), ("base.ref", "other"),
    ("head.sha", "f" * 40), ("body", "editable text"),
])
def test_identity_and_body_damage_cannot_authorize_resume(tmp_path, field, value):
    runner = SourceRunner()
    root(runner, tmp_path)
    target = runner.rest_pr
    parts = field.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value
    with pytest.raises(AgentLoopError):
        source_auth.authenticate_source_authority(runner, config=config(tmp_path), pr_number=7, expected_head=runner.rest_pr["head"]["sha"])
    assert_no_qualification(runner)


@pytest.mark.parametrize("draft,labels,state", [(False, True, "ready/labeled"), (None, True, "unknown/labeled"), (None, False, "unknown/unlabeled")])
@pytest.mark.parametrize("entry", ["recovered", "low-level"])
def test_lifecycle_refusal_complete_output_has_actionable_correction_and_no_writes(tmp_path, draft, labels, state, entry):
    runner = SourceRunner()
    root(runner, tmp_path)
    runner.rest_pr["draft"] = draft
    runner.rest_pr["labels"] = [{"name": "agent-loop-managed"}] if labels else []
    runner.commands.clear()
    with pytest.raises(AgentLoopError) as error:
        if entry == "recovered":
            managed_ci.authenticate_source_managed_resume(runner, config=config(tmp_path), pr_number=7,
                source_branch="fix/source", source_sha=SHA, managed_branch=BRANCH, override_nonce="opening-nonce")
        else:
            activate(runner, tmp_path)
    message = str(error.value)
    assert state in message or "unknown" in message
    assert ("fresh" in message if draft is None else "restore draft/labeled" in message)
    assert message.count("Measured lifecycle:") == 1
    assert "Resume with `Measured" not in message
    assert "after restoring draft/unlabeled" not in message
    assert "agent-loop pr 7" not in message
    assert not any("POST" in cmd or "DELETE" in cmd or cmd[:3] == ["gh", "pr", "ready"] for cmd, _ in runner.commands)
    assert_no_qualification(runner)


@pytest.mark.parametrize("damage", ["foreign-label", "missing-nonce", "wrong-nonce", "duplicate-source", "malformed-source", "wrong-author", "wrong-source"])
def test_activation_identity_refusals_preserve_label_ownership(tmp_path, damage):
    runner = SourceRunner()
    root(runner, tmp_path)
    if damage == "foreign-label":
        runner.issue_events = [label_event(login="other", actor_id=9)]
    elif damage == "missing-nonce":
        runner.rest_pr["body"] = runner.rest_pr["body"].split(UNPROTECTED_OVERRIDE_TRAILER)[0]
    elif damage == "wrong-nonce":
        runner.rest_pr["body"] = runner.rest_pr["body"].replace("opening-nonce", "changed-nonce")
    elif damage in {"duplicate-source", "malformed-source"}:
        marker = runner.rest_pr["body"].split("\n\n")[0]
        runner.rest_pr["body"] += "\n" + (marker if damage == "duplicate-source" else "<!-- AGENT_MANAGED_PR_SOURCE_V1 broken -->")
    elif damage == "wrong-source":
        runner.rest_pr["body"] = str(_compose_body("", source_branch="fix/source", source_sha="f" * 40, override_nonce="opening-nonce"))
    else:
        next(iter(runner.audit_comments.values()))["user"]["id"] = 99
    with pytest.raises(AgentLoopError):
        activate(runner, tmp_path)
    if damage == "foreign-label":
        assert not any("DELETE" in cmd for cmd, _ in runner.commands)
    assert_no_qualification(runner)


@pytest.mark.parametrize("field,value", [("body", "changed"), ("head.sha", "a" * 40),
    ("base.ref", "other"), ("head.repo.full_name", "FORK/REPO"), ("label-event", 999)])
def test_activation_final_live_races_do_not_create_resume_authority(tmp_path, field, value):
    runner = SourceRunner()
    root(runner, tmp_path)
    original = runner._run_locked
    reads = 0
    def race(args, **kwargs):
        nonlocal reads
        if "repos/OWNER/REPO/pulls/7" in args:
            reads += 1
            # Initial activation read, lifecycle read, root read, final live read.
            if reads == 4:
                if field == "label-event":
                    runner.issue_events.append(label_event(value))
                else:
                    target = runner.rest_pr
                    parts = field.split(".")
                    for part in parts[:-1]:
                        target = target[part]
                    target[parts[-1]] = value
        return original(args, **kwargs)
    runner._run_locked = race
    with pytest.raises(AgentLoopError):
        activate(runner, tmp_path)
    assert len(runner.audit_comments) == 1
    if field == "label-event":
        assert not any("DELETE" in cmd for cmd, _ in runner.commands)
    assert_no_qualification(runner)


@pytest.mark.parametrize("failure", ["comments", "timeline", "audit-post", "audit-readback"])
def test_resume_unavailable_probes_or_audit_never_claim_activation(tmp_path, failure):
    runner = SourceRunner()
    root(runner, tmp_path)
    original = runner._run_locked
    def fail(args, **kwargs):
        endpoint = next((x for x in args if isinstance(x, str) and x.startswith("repos/")), "")
        should_fail = ((failure == "comments" and "/comments?" in endpoint)
            or (failure == "timeline" and "/events?" in endpoint))
        if should_fail:
            command, path = runner._record_command(args, kwargs["cwd"])
            return CommandResult(command, path, "", "probe unavailable", 1)
        return original(args, **kwargs)
    runner._run_locked = fail
    runner.fail_audit = failure == "audit-post"
    runner.fail_readback = failure == "audit-readback"
    with pytest.raises(AgentLoopError):
        activate(runner, tmp_path)
    assert_no_qualification(runner)


@pytest.mark.parametrize("draft,labeled", [(True, True), (True, False), (False, False)])
def test_explicit_source_lifecycle_reentry_authenticates_before_mutation(tmp_path, draft, labeled):
    runner = SourceRunner()
    root(runner, tmp_path)
    runner.rest_pr.update(draft=draft, labels=[{"name": "agent-loop-managed"}] if labeled else [])
    contract = activate(runner, tmp_path)
    assert contract and contract.origin == "source-managed"
    assert runner.rest_pr["draft"] is True and runner.rest_pr["labels"]


@pytest.mark.parametrize("draft", [True, False])
def test_unlabeled_authority_failure_has_no_reentry_writes(tmp_path, draft):
    runner = SourceRunner()
    runner.rest_pr.update(draft=draft, labels=[])
    with pytest.raises(AgentLoopError) as error:
        activate(runner, tmp_path)
    assert "ordinary review" in str(error.value)
    assert not any("POST" in cmd or cmd[:3] == ["gh", "pr", "ready"] for cmd, _ in runner.commands)
    assert_no_qualification(runner)


def test_current_waiver_is_required_even_with_a_creation_audit(tmp_path):
    runner = SourceRunner()
    root(runner, tmp_path)
    with pytest.raises(AgentLoopError, match="waiver is absent"):
        activate(runner, tmp_path, allow_unprotected_managed_ci=False)
    assert_no_qualification(runner)


def test_legacy_creation_audit_only_authorizes_the_opening_head(tmp_path):
    runner = SourceRunner()
    fields = dict(nonce="opening-nonce", repo="OWNER/REPO", base="main", head=SHA, protection="voluntary")
    runner.intent_comments.append({"id": 17, "user": {"login": "agent-loop", "id": 1},
        "body": str(source_auth._root_body(fields, 7))})
    assert activate(runner, tmp_path).origin == "source-managed"
    runner.rest_pr["head"]["sha"] = "a" * 40
    with pytest.raises(AgentLoopError, match="continuity"):
        activate(runner, tmp_path)
    assert_no_qualification(runner)


@pytest.mark.parametrize("draft,labeled", [(True, True), (True, False), (False, False), (None, True), (None, False)])
@pytest.mark.parametrize("variant", ["released", "unconfirmed", "manual-undo", "restored", "no-route"])
def test_all_source_activation_report_variants_are_conditional_without_circular_commands(tmp_path, draft, labeled, variant):
    runner = SourceRunner()
    entry = managed_ci._PrStateSnapshot.from_payload(runner.rest_pr)
    runner.rest_pr.update(draft=draft, labels=[{"name": "agent-loop-managed"}] if labeled else [])
    context = managed_ci._FailedActivationContext(entry=entry, source_managed=True,
        label_release_attempted=variant in {"released", "unconfirmed"}, label_released=variant == "released",
        drafted_by_this_run=variant in {"manual-undo", "restored"}, recovery_capable=variant != "no-route")
    runner.commands.clear()
    output = managed_ci._report_failed_activation_state(runner, config=config(tmp_path), pr_number=7, context=context)
    assert "Before activation" in output and "Measured lifecycle" in output
    if variant == "released":
        assert "label DELETE acknowledged" in output
    if variant == "unconfirmed":
        assert "label DELETE not confirmed" in output
    assert "--managed-ci" not in output
    assert "Resume with" not in output and "rerun `" not in output
    assert not any("POST" in cmd or "DELETE" in cmd or cmd[:3] == ["gh", "pr", "ready"] for cmd, _ in runner.commands)


@pytest.mark.parametrize("protection", ["voluntary", "plan_limited", "unreadable"])
def test_source_restart_rechecks_current_protection_and_applicable_waivers(tmp_path, protection):
    options = {}
    if protection != "voluntary":
        detail = "HTTP 403: Upgrade to GitHub Pro or make this repository public" if protection == "plan_limited" else "gh: Resource not accessible by integration (HTTP 403)\n"
        options.update(pr_branch_protection_returncode=1, pr_branch_protection_stderr=detail,
            pr_effective_rules_returncode=1 if protection == "plan_limited" else 0,
            pr_effective_rules_stderr=detail if protection == "plan_limited" else "",
            repo_payload={"private": True})
    runner = SourceRunner(**options)
    cfg = config(tmp_path, allow_unreadable_protection=protection == "unreadable")
    source_auth.publish_source_creation_root(runner, config=cfg, pr_number=7,
        source=("fix/source", SHA, BRANCH, "opening-nonce"), protection=protection)
    assert activate(runner, tmp_path, allow_unreadable_protection=protection == "unreadable").protection_mode == protection
    if protection == "unreadable":
        with pytest.raises(AgentLoopError, match="waiver is absent"):
            activate(runner, tmp_path)
    assert_no_qualification(runner)


def test_implicit_ready_unlabeled_source_resume_requires_explicit_managed_intent(tmp_path):
    runner = SourceRunner()
    root(runner, tmp_path)
    runner.rest_pr.update(draft=False, labels=[])
    runner.commands.clear()
    with pytest.raises(AgentLoopError, match="explicit.*--managed-ci"):
        activate(runner, tmp_path, managed_ci=False, auto_merge=True)
    assert not any("POST" in cmd or "DELETE" in cmd or cmd[:3] == ["gh", "pr", "ready"] for cmd, _ in runner.commands)


def test_rejected_coder_without_accepted_metadata_cannot_publish_continuity(tmp_path):
    runner = SourceRunner()
    root(runner, tmp_path)
    runner.intent_comments.append(_round_comment(100, role="reviewer", subject=SHA, round_number=1, state="blocking"))
    runner.rest_pr["head"]["sha"] = "a" * 40
    with pytest.raises(AgentLoopError, match="correlated"):
        source_auth.publish_source_continuity(runner, config=config(tmp_path), pr_number=7,
            predecessor_head=SHA, new_head="a" * 40, round_number=1)
    assert len(runner.audit_comments) == 1
    with pytest.raises(AgentLoopError, match="continuity"):
        activate(runner, tmp_path)
    assert len(runner.audit_comments) == 1
    assert_no_qualification(runner)


def test_implicit_source_authority_failure_stops_before_ordinary_auto_merge(tmp_path):
    runner = SourceRunner()
    with pytest.raises(AgentLoopError, match="ordinary review"):
        activate(runner, tmp_path, managed_ci=False, auto_merge=True)
    assert_no_qualification(runner)


@pytest.mark.parametrize("operation", ["creation", "authentication", "continuity"])
def test_async_github_metadata_does_not_change_authority(tmp_path, operation):
    runner = SourceRunner()
    if operation != "creation":
        root(runner, tmp_path)
    original = runner._run_locked
    reads = 0
    def changing(args, **kwargs):
        nonlocal reads
        if "repos/OWNER/REPO/pulls/7" in args and "--method" not in args:
            reads += 1
            runner.rest_pr.update(mergeable=None if reads % 2 else True,
                mergeable_state=str(reads), rebaseable=bool(reads % 2),
                updated_at=str(reads), comments=reads, commits=reads)
        return original(args, **kwargs)
    runner._run_locked = changing
    if operation == "creation":
        assert create(runner, tmp_path).source_authority
    elif operation == "authentication":
        assert activate(runner, tmp_path).origin == "source-managed"
    else:
        successor = "a" * 40
        runner.intent_comments.extend([
            _round_comment(100, role="reviewer", subject=SHA, round_number=1, state="blocking"),
            _round_comment(101, role="coder", subject=successor, round_number=2),
        ])
        runner.rest_pr["head"]["sha"] = successor
        assert source_auth.publish_source_continuity(runner, config=config(tmp_path),
            pr_number=7, predecessor_head=SHA, new_head=successor, round_number=1).head == successor
    assert reads >= 2
    assert runner.rest_pr["state"] == "open"
    assert not any("DELETE" in cmd for cmd, _ in runner.commands)
    assert_no_qualification(runner)


@pytest.mark.parametrize("lifecycle,draft,labeled,explicit", [
    ("draft-unlabeled-reentry", False, True, True),
    ("draft-labeled", False, True, True),
    ("draft-unlabeled-reentry", True, False, False),
])
def test_source_late_lifecycle_refusals_render_advice_once(tmp_path, lifecycle, draft, labeled, explicit):
    from coding_review_agent_loop.managed_ci_types import AuthenticatedManagedResume
    runner = SourceRunner()
    root(runner, tmp_path)
    runner.rest_pr.update(draft=draft, labels=[{"name": "agent-loop-managed"}] if labeled else [])
    resume = AuthenticatedManagedResume(origin="source-managed", lifecycle=lifecycle,
        source_branch="fix/source", source_sha=SHA, managed_branch=BRANCH, override_nonce="opening-nonce")
    runner.commands.clear()
    with pytest.raises(AgentLoopError) as error:
        managed_ci.activate_managed_ci(runner, config=config(tmp_path, managed_ci=explicit, auto_merge=True),
            pr_number=7, metadata=replace(metadata(), head_branch=BRANCH, body=runner.rest_pr["body"]),
            managed_resume=resume)
    message = str(error.value)
    assert message.count("Measured lifecycle:") == 1
    assert "Resume with `" not in message
    assert "after restoring draft/unlabeled" not in message
    if explicit:
        assert "restore draft/labeled" in message
    else:
        assert "requires explicit --managed-ci" in message
    assert not any("POST" in cmd or "DELETE" in cmd or cmd[:3] == ["gh", "pr", "ready"] for cmd, _ in runner.commands)
    assert_no_qualification(runner)
