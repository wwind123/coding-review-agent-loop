"""PR-loop tests for architecture_contract: the PR-loop architecture gate (#1542)."""

import dataclasses
from types import SimpleNamespace

import pytest
from coding_review_agent_loop.round_state import (
    PostedRoundMetadata,
    _attach_round_metadata,
    _latest_pr_approved_reviews_for_head,
)
from coding_review_agent_loop.architecture_contract import _latest_pr_architecture_observation
from agent_loop_helpers import (
    FakeRunner,
    make_config,
)
import coding_review_agent_loop.pr_loop_support as pr_loop_support
from pr_loop_test_helpers import (
    _m943_append,
)


def test_latest_pr_architecture_observation_uses_new_checkpoint_once():
    old_identity = {"target_revision": "main", "merge_base_revision": "old"}
    new_identity = {"target_revision": "release", "merge_base_revision": "new"}
    approval = _attach_round_metadata(
        "approved review",
        PostedRoundMetadata(
            flow="pr", role="reviewer", agent="Codex", round_number=1,
            subject="head", state="approved", architecture_identity=old_identity,
            architecture_contract_version=1,
        ),
    )
    checkpoint = _attach_round_metadata(
        "qualification checkpoint",
        PostedRoundMetadata(
            flow="pr", role="summary", agent="Orchestrator", round_number=2,
            subject="head", phase="qualification-checkpoint",
            architecture_identity=new_identity, architecture_contract_version=1,
        ),
    )
    comments = (SimpleNamespace(body=approval), SimpleNamespace(body=checkpoint))
    assert _latest_pr_architecture_observation(comments, head_sha="head") == new_identity


def test_legacy_same_head_approval_is_not_reusable_with_fresh_architecture_gate():
    approval = _attach_round_metadata(
        "approved review",
        PostedRoundMetadata(
            flow="pr", role="reviewer", agent="Codex", round_number=1,
            subject="head", state="approved",
        ),
    )
    comments = (SimpleNamespace(body=approval),)
    reused = _latest_pr_approved_reviews_for_head(
        comments,
        head_sha="head",
        configured_reviewers=("codex",),
        require_architecture_contract=True,
    )
    assert reused == {}


# ---------------------------------------------------------------------------
# #1454: PR revalidation over the architecture set
# ---------------------------------------------------------------------------

import subprocess as _set_subprocess  # noqa: E402

from coding_review_agent_loop.architecture_contract import (  # noqa: E402
    _freeze_prompt_architecture as _set_freeze,
    _revalidate_pr_architecture_identity as _set_revalidate,
)
from coding_review_agent_loop.runner import Runner as _SetRunner  # noqa: E402


def _set_git(cwd, *args):
    return _set_subprocess.run(
        ("git", *args), cwd=cwd, text=True, capture_output=True, check=True
    ).stdout.strip()


def _set_commit(repo, files):
    for name, content in files.items():
        target = repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    _set_git(repo, "add", "-A")
    _set_git(repo, "commit", "-qm", "update")
    return _set_git(repo, "rev-parse", "HEAD")


def _set_pr_repo(tmp_path, files, **overrides):
    config = make_config(tmp_path, **overrides)
    repo = config.claude_dir
    _set_git(repo, "init", "-q")
    _set_git(repo, "config", "user.email", "test@example.com")
    _set_git(repo, "config", "user.name", "Test")
    base = _set_commit(repo, files)
    _set_git(repo, "update-ref", "refs/remotes/origin/main", base)
    _set_git(repo, "checkout", "-qb", "candidate")
    head = _set_commit(repo, {"README.md": "first push\n"})
    return config, repo, head


def _set_identity(config, head):
    frozen = _set_freeze(
        _SetRunner(), config, target_revision="main", candidate_revision=head, pr_pair=True
    )
    return frozen, frozen.architecture_context.identity()


def _set_metadata(head):
    return SimpleNamespace(base_branch="main", head_sha=head)


_SET_ROOT = "# Root\n\nOverview.\n\n- [Billing](docs/architecture/billing.md)\n"


def test_revalidation_keeps_a_legacy_root_only_identity_unchanged(tmp_path):
    config, _repo, head = _set_pr_repo(tmp_path, {"ARCHITECTURE.md": "# Root\n\nOverview.\n"})
    frozen, identity = _set_identity(config, head)
    assert "set_change" not in identity and "members" not in identity["candidate"]
    _fresh, changed = _set_revalidate(
        _SetRunner(), config=frozen, metadata=_set_metadata(head), stored_identity=identity
    )
    assert changed is False


@pytest.mark.parametrize("read_size", [64 * 1024, 32], ids=["material-root", "oversized-root"])
def test_member_blob_change_is_reported_exactly_once(tmp_path, read_size):
    config, repo, head = _set_pr_repo(
        tmp_path,
        {"ARCHITECTURE.md": _SET_ROOT, "docs/architecture/billing.md": "# Billing\n"},
        architecture_read_size=read_size,
    )
    frozen, identity = _set_identity(config, head)
    assert identity["set_change"] == "unchanged"
    edited = _set_commit(repo, {"docs/architecture/billing.md": "# Billing v2\n"})
    fresh, changed = _set_revalidate(
        _SetRunner(), config=frozen, metadata=_set_metadata(edited), stored_identity=identity
    )
    assert changed is True
    assert fresh.set_change == "modified"
    refrozen = dataclasses.replace(frozen, architecture_context=fresh)
    _again, repeated = _set_revalidate(
        _SetRunner(), config=refrozen, metadata=_set_metadata(edited),
        stored_identity=fresh.identity(),
    )
    assert repeated is False


def test_contentless_push_is_not_a_set_change_for_an_oversized_root(tmp_path):
    config, repo, head = _set_pr_repo(
        tmp_path,
        {"ARCHITECTURE.md": _SET_ROOT, "docs/architecture/billing.md": "# Billing\n"},
        architecture_read_size=32,
    )
    frozen, identity = _set_identity(config, head)
    pushed = _set_commit(repo, {"README.md": "second push\n"})
    _fresh, changed = _set_revalidate(
        _SetRunner(), config=frozen, metadata=_set_metadata(pushed), stored_identity=identity
    )
    assert changed is False


def test_contentless_push_with_a_missing_root_is_not_a_change(tmp_path):
    config, repo, head = _set_pr_repo(tmp_path, {"README.md": "# Readme\n"})
    frozen, identity = _set_identity(config, head)
    pushed = _set_commit(repo, {"src.py": "x = 1\n"})
    _fresh, changed = _set_revalidate(
        _SetRunner(), config=frozen, metadata=_set_metadata(pushed), stored_identity=identity
    )
    assert changed is False


class _SetHybridRunner(FakeRunner):
    """Fake GitHub plus real immutable Git object reads for architecture acquisition."""

    def run_binary(self, args, *, cwd, max_bytes=None, check=True, env=None):
        cmd = [str(value) for value in args]
        if cmd[:1] == ["git"] and ("ls-tree" in cmd or "cat-file" in cmd):
            return _SetRunner.run_binary(self, args, cwd=cwd, max_bytes=max_bytes, check=check, env=env)
        return super().run_binary(args, cwd=cwd, max_bytes=max_bytes, check=check, env=env)

    def run(self, args, *, cwd, input_text=None, check=True, env=None):
        cmd = [str(value) for value in args]
        if cmd[:2] == ["git", "merge-base"] or cmd[:3] == ["git", "rev-parse", "--verify"]:
            return _SetRunner.run(self, args, cwd=cwd, input_text=input_text, check=check, env=env)
        return super().run(args, cwd=cwd, input_text=input_text, check=check, env=env)


def _set_persist_approval(runner, head, identity):
    _m943_append(
        runner,
        _attach_round_metadata(
            "approved review",
            PostedRoundMetadata(
                flow="pr", role="reviewer", agent="Codex", round_number=1,
                subject=head, state="approved", architecture_identity=identity,
                architecture_contract_version=1,
            ),
        ),
    )


def _set_qualify(runner, config):
    context, _requirements, _plan, fresh_config = pr_loop_support._fresh_pr_qualification_snapshot(
        runner, config=config, pr_number=77, issue_context=None, parent_issue_context=None,
    )
    return context, fresh_config


@pytest.mark.parametrize("read_size", [64 * 1024, 32], ids=["material-root", "oversized-root"])
def test_qualification_revalidates_a_persisted_set_identity_once(tmp_path, read_size):
    config, repo, head = _set_pr_repo(
        tmp_path,
        {"ARCHITECTURE.md": _SET_ROOT, "docs/architecture/billing.md": "# Billing\n"},
        architecture_read_size=read_size,
    )
    frozen, identity = _set_identity(config, head)
    runner = _SetHybridRunner(pr_payload={"headRefOid": head})
    _set_persist_approval(runner, head, identity)
    context, _config = _set_qualify(runner, frozen)
    assert context.architecture_identity_changed is False

    edited = _set_commit(repo, {"docs/architecture/billing.md": "# Billing v2\n"})
    runner.pr_payload["headRefOid"] = edited
    _set_persist_approval(runner, edited, identity)
    context, fresh_config = _set_qualify(runner, frozen)
    assert context.architecture_identity_changed is True
    assert fresh_config.architecture_context.set_change == "modified"

    # Once the fresh observation is the persisted approval, the gate is quiet.
    runner.pr_payload["comments"].clear()
    _set_persist_approval(runner, edited, fresh_config.architecture_context.identity())
    context, _config = _set_qualify(runner, fresh_config)
    assert context.architecture_identity_changed is False


def test_qualification_keeps_a_legacy_root_only_identity_and_missing_root_quiet(tmp_path):
    config, _repo, head = _set_pr_repo(tmp_path, {"ARCHITECTURE.md": "# Root\n\nOverview.\n"})
    frozen, identity = _set_identity(config, head)
    assert set(identity["candidate"]) == {
        "repository", "path", "revision", "blob_oid", "sha256", "availability", "size",
    }
    runner = _SetHybridRunner(pr_payload={"headRefOid": head})
    _set_persist_approval(runner, head, identity)
    context, _config = _set_qualify(runner, frozen)
    assert context.architecture_identity_changed is False

    missing_tmp = tmp_path / "missing"
    missing_tmp.mkdir()
    config, repo, head = _set_pr_repo(missing_tmp, {"README.md": "# Readme\n"})
    frozen, identity = _set_identity(config, head)
    pushed = _set_commit(repo, {"src.py": "x = 1\n"})
    runner = _SetHybridRunner(pr_payload={"headRefOid": pushed})
    _set_persist_approval(runner, pushed, identity)
    context, _config = _set_qualify(runner, frozen)
    assert context.architecture_identity_changed is False


def test_empty_push_identity_provenance_and_derived_approval_are_stable(tmp_path):
    """Real Git acquisition and public serialization preserve every H2 identity field."""
    from coding_review_agent_loop.architecture_contract import (
        _architecture_metadata_fields, _architecture_identity_differences,
    )
    from coding_review_agent_loop.protocol import ParsedReview, ApprovedFollowups, parse_review
    from coding_review_agent_loop.comment_rendering import render_public_agent_comment
    from coding_review_agent_loop.review_rounds import _normalize_evidence_only_approval
    from pr_loop_test_helpers import _EVIDENCE_TEXT
    from coding_review_agent_loop.unresolved_items import _upsert_evidence_obligation

    config, repo, h1 = _set_pr_repo(tmp_path, {"ARCHITECTURE.md": _SET_ROOT,
                                             "docs/architecture/billing.md": "# Billing\n"})
    old_config, old_identity = _set_identity(config, h1)
    _set_git(repo, "commit", "--allow-empty", "-qm", "empty external push")
    h2 = _set_git(repo, "rev-parse", "HEAD")
    assert _set_git(repo, "rev-parse", f"{h1}^{{tree}}") == _set_git(repo, "rev-parse", f"{h2}^{{tree}}")
    fresh_config, identity = _set_identity(config, h2)
    assert set(_architecture_identity_differences(old_identity, identity)) == {
        "candidate_revision", "candidate.revision",
    }
    runner = _SetHybridRunner(pr_payload={"headRefOid": h2})
    # Reproduce the writer seam: the result subject advances, its prompt stays H1.
    stale = _attach_round_metadata("coder result", PostedRoundMetadata(
        flow="pr", role="coder", agent="Claude", round_number=2, subject=h2,
        **_architecture_metadata_fields(old_config),
    ))
    _m943_append(runner, stale)
    assert _latest_pr_architecture_observation(
        tuple(SimpleNamespace(body=c["body"]) for c in runner.pr_payload["comments"]), head_sha=h2,
    ) is None
    items, _ = _upsert_evidence_obligation([], item_number=1, reviewer="Codex", text=_EVIDENCE_TEXT,
                                          source_round=1, current_head_sha=h2)
    from coding_review_agent_loop.protocol import ReviewItemDisposition
    review = ParsedReview(state="blocking", summary="Evidence remains deferred.", blocking_items=(),
                          followups=ApprovedFollowups(same_pr=(), future=()), dispositions=(
        ReviewItemDisposition(item_id="item-1", reviewer="Codex", disposition="blocking", note="Still needed."),
    ))
    approved = _normalize_evidence_only_approval(review, items)
    body = render_public_agent_comment(kind="pr_review", parsed=approved, agent="codex",
                                      prior_items=items, dispositions=approved.dispositions, config=fresh_config)
    assert parse_review(body, reviewer="Codex").state == "approved"
    _m943_append(runner, _attach_round_metadata(body, PostedRoundMetadata(
        flow="pr", role="reviewer", agent="Codex", round_number=3, subject=h2,
        state="approved", prior_items=tuple(items), **_architecture_metadata_fields(fresh_config),
    )))
    for _ in range(3):
        context, fresh_config = _set_qualify(runner, fresh_config)
        assert context.architecture_identity_changed is False
        assert fresh_config.architecture_context.identity() == identity


@pytest.mark.parametrize("architecture_changed,head_changed", [(True, False), (False, True), (True, True)])
def test_restart_checkpoint_names_real_changes_and_advances_only_new_head(tmp_path, monkeypatch, architecture_changed, head_changed):
    from coding_review_agent_loop.pr_restart import checkpoint_restart
    import coding_review_agent_loop.unresolved_items as ledger
    config = make_config(tmp_path)
    context = SimpleNamespace(metadata=SimpleNamespace(head_sha="h3" if head_changed else "h2", base_branch="main"),
                              architecture_identity_changed=architecture_changed)
    advanced = []
    saved = []
    monkeypatch.setattr(ledger, "_advance_machine_obligations_for_head", lambda items, **kwargs: advanced.append(kwargs) or items)
    monkeypatch.setattr(pr_loop_support, "_machine_obligation_checkpoint", lambda *args, **kwargs: "checkpoint")
    monkeypatch.setattr(pr_loop_support, "_persist_qualification_checkpoint", lambda *args, **kwargs: saved.append(kwargs))
    items = []
    result, checkpoint = checkpoint_restart(
        _SetHybridRunner(), config=config, pr_number=77, round_number=3,
        previous_head="h2", context=context, items=items, checkpoint=None, managed_contract=None,
        allowed_rounds=8, watch_failure_extension_used=False, watch_head_extension_used=False,
        phase="during reviewer reconciliation",
    )
    assert result is items and checkpoint == "checkpoint"
    assert bool(advanced) == head_changed
    assert ("PR head changed" in saved[0]["message"]) == head_changed
    assert ("architecture identity changed" in saved[0]["message"]) == architecture_changed


def test_selective_converted_freeze_empty_push_uses_real_acquisition(tmp_path, monkeypatch):
    """Drive the reported freeze/push/derived-approval sequence over real Git objects."""
    from coding_review_agent_loop.cli import run_pr_loop
    from coding_review_agent_loop.errors import HumanDecisionRequiredError
    import coding_review_agent_loop.pr_loop as loop
    from test_pr_loop_evidence_conversion import (
        _plan_context, _implementation_record, _install_derivation, _review, _coder, _records,
    )
    context = _plan_context("row-a", "row-b")
    config, repo, h1 = _set_pr_repo(
        tmp_path, {"ARCHITECTURE.md": _SET_ROOT, "docs/architecture/billing.md": "# Billing\n"},
        coder="claude", reviewer="codex", pr_review_policy="selective-intermediate",
        pr_convert_evidence_rows=("row-a",), pr_non_convergence_rounds=0,
    )
    runner = _SetHybridRunner(
        pr_payload={"headRefOid": h1, "comments": [_implementation_record(
            context, {"row-a": "limited", "row-b": "verified"}, head=h1,
        )]}, codex_outputs=[_review(kept=["item-1"], state="blocking")],
        advance_pr_head_on_coder_followup=False,
    )
    runner.git_head = h1
    real_qualify = loop._fresh_pr_qualification_snapshot
    snapshots = []

    def qualify(*args, **kwargs):
        plan = kwargs.pop("approved_plan_context", None)  # isolate the unrelated issue-side handoff
        fresh, requirements, _plan, fresh_config = real_qualify(*args, **kwargs)
        snapshots.append((fresh.metadata.head_sha, fresh.architecture_identity_changed))
        return fresh, requirements, plan, fresh_config

    monkeypatch.setattr(loop, "_fresh_pr_qualification_snapshot", qualify)
    with pytest.raises(HumanDecisionRequiredError, match=f"frozen at head {h1}"):
        run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context)
    assert _records(runner)[-1].metadata.architecture_identity["candidate_revision"] == h1
    _set_git(repo, "commit", "--allow-empty", "-qm", "external empty push")
    h2 = _set_git(repo, "rev-parse", "HEAD")
    assert _set_git(repo, "rev-parse", f"{h1}^{{tree}}") == _set_git(repo, "rev-parse", f"{h2}^{{tree}}")
    runner.pr_payload["headRefOid"] = h2
    runner.git_head = h2
    runner.simulate_new_process()
    runner.claude_outputs.append(_coder())
    runner.codex_outputs.extend([_review(kept=["item-1"], state="blocking"), _review(resolved=["item-1"])] * 2)
    _install_derivation(monkeypatch, context, [{"row-a": "verified", "row-b": "verified"}])
    config = dataclasses.replace(config, pr_convert_evidence_rows=())
    assert run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context) == 0
    assert snapshots and all(not changed for head, changed in snapshots if head == h2)
    coder_records = [r for r in _records(runner) if r.metadata.role == "coder" and r.metadata.subject == h2]
    assert len(coder_records) == 1 and coder_records[0].metadata.followup_dispatch_head == h2
    fresh_config, identity = _set_identity(config, h2)
    for _ in range(3):
        fresh, fresh_config = _set_qualify(runner, fresh_config)
        assert not fresh.architecture_identity_changed
        assert fresh_config.architecture_context.identity() == identity
