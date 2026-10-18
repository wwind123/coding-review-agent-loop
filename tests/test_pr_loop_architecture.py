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
