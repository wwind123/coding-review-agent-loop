"""Integration test: invokes demo_loop.py as a subprocess."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

from coding_review_agent_loop.architecture_context import ArchitectureSnapshot
from helpers.prompt_builders import build_review_prompt_for_skill


def test_skill_planning_recovery_refuses_to_invent_fresh_contract():
    from coding_review_agent_loop.errors import FreshContractIntegrityError
    from helpers.skill_runner import _recover_structured_response

    with pytest.raises(FreshContractIntegrityError, match="new planner turn"):
        _recover_structured_response(
            "leading prose before a plan",
            expected_kind="plan_state",
            validate=lambda _text: None,
            require_execution_strategy_contract=True,
        )


def test_demo_loop_dry_run() -> None:
    """
    Run helpers/demo_loop.py and verify it:
    - exits 0
    - prints "validation passed: plan_state"
    - prints "validation passed: plan_review"
    - produces metadata-tagged comments with AGENT_LOOP_META
    - writes a session file with last_completed_step=post_review
    - verifies _resume_plan_round can reconstruct the round from the metadata
    """
    repo = "demo/skill-loop-test"
    issue = 88888

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "helpers.demo_loop",
            "--issue",
            str(issue),
            "--repo",
            repo,
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
        check=False,
    )

    assert result.returncode == 0, (
        f"demo_loop failed (exit {result.returncode}):\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )

    assert "validation passed: plan_state" in result.stdout, result.stdout
    assert "validation passed: plan_review" in result.stdout, result.stdout

    # demo_loop now also verifies _resume_plan_round internally
    assert "_resume_plan_round found round" in result.stdout, result.stdout

    # Verify session state was written with last_completed_step=post_review
    slug = repo.replace("/", "-").replace(":", "-")
    state_home = Path(
        os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state")
    )
    session_path = (
        state_home
        / "coding-review-agent-loop"
        / "skill-sessions"
        / slug
        / f"{issue}.json"
    )
    assert session_path.exists(), f"session file not found: {session_path}"
    data = json.loads(session_path.read_text(encoding="utf-8"))
    assert data.get("last_completed_step") == "post_review", data


def test_skill_review_builder_accepts_frozen_architecture_snapshot(tmp_path) -> None:
    snapshot = ArchitectureSnapshot(
        repository="owner/repo", path="ARCHITECTURE.md", revision="a" * 40,
        blob_oid="b" * 40, sha256="c" * 64, availability="available",
        size=12, content="# Components\n",
    )
    prompt = build_review_prompt_for_skill(
        {"number": 77, "title": "PR", "body": "", "headRefOid": "d" * 40},
        "diff --git a/x b/x", [], 1, "codex", repo="owner/repo", pr_number=77,
        workdir=str(tmp_path), architecture_context=snapshot,
    )
    assert "advisory repository context" in prompt
    assert snapshot.blob_oid in prompt


def test_skill_pr_resume_rejects_stale_architecture_identity() -> None:
    from helpers.skill_runner import _filter_resume_for_architecture

    identity = {"repository": "owner/repo", "path": "ARCHITECTURE.md", "revision": "new"}
    stale = {
        "reviewer_name": "Codex",
        "architecture_identity": {"repository": "owner/repo", "path": "ARCHITECTURE.md", "revision": "old"},
        "architecture_contract_version": 1,
    }
    current = {
        "reviewer_name": "Gemini",
        "architecture_identity": identity,
        "architecture_contract_version": 1,
    }
    resume = {
        "completed_reviewer_data": [stale, current],
        "completed_reviewer_names": ["Codex", "Gemini"],
    }
    filtered = _filter_resume_for_architecture(resume, architecture_identity=identity)
    assert filtered["completed_reviewer_names"] == ["Gemini"]
    assert filtered["completed_reviewer_data"] == [current]


def test_default_pr_round_prepares_auto_checkout_before_architecture_acquisition(
    monkeypatch, tmp_path
) -> None:
    import helpers.skill_runner as skill_runner

    calls: list[str] = []
    monkeypatch.setattr(
        skill_runner,
        "ensure_temp_checkout",
        lambda path, **kwargs: calls.append(f"ensure:{path}"),
    )
    monkeypatch.setattr(
        skill_runner,
        "sync_checkout_to_pr",
        lambda config, runner, **kwargs: calls.append(f"sync:{kwargs['path']}"),
    )
    args = SimpleNamespace(dry_run=False, workdir=None, workdir_gemini=None)
    workdir = str(tmp_path / "skill-runner-gemini")

    assert skill_runner._prepare_skill_pr_architecture_checkout(
        args,
        repo="owner/repo",
        pr=781,
        pr_info={"headRefOid": "h" * 40, "baseRefName": "main"},
        agent="gemini",
        workdir=workdir,
    ) is True
    assert calls == [f"ensure:{workdir}", f"sync:{workdir}"]


def test_explicit_pr_round_checkout_is_synced_before_architecture_acquisition(
    monkeypatch, tmp_path
) -> None:
    import helpers.skill_runner as skill_runner

    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(
        skill_runner,
        "ensure_temp_checkout",
        lambda *args, **kwargs: calls.append(("ensure", kwargs)),
    )

    def fake_sync(config, runner, **kwargs):
        calls.append(("sync", kwargs))

    monkeypatch.setattr(skill_runner, "sync_checkout_to_pr", fake_sync)
    args = SimpleNamespace(
        dry_run=False,
        workdir=str(tmp_path / "explicit"),
        workdir_gemini=None,
    )
    workdir = str(tmp_path / "explicit")

    assert skill_runner._prepare_skill_pr_architecture_checkout(
        args,
        repo="owner/repo",
        pr=781,
        pr_info={"headRefOid": "h" * 40, "baseRefName": "main"},
        agent="gemini",
        workdir=workdir,
    ) is True
    assert [kind for kind, _details in calls] == ["sync"]
    assert calls[0][1]["path"] == Path(workdir)
    assert calls[0][1]["default_owned"] is False


def test_pr_round_reuses_frozen_architecture_for_host_handoff(
    monkeypatch, capsys
) -> None:
    import helpers.prompt_builders as prompt_builders
    import helpers.skill_runner as skill_runner
    pair = object()
    captured: list[object] = []
    args = SimpleNamespace(
        pr=781,
        repo="owner/repo",
        reviewers=["claude"],
        dry_run=True,
        head_sha="h" * 40,
        architecture_context_enabled=True,
        workdir=None,
        workdir_claude=None,
    )
    monkeypatch.setattr(
        skill_runner, "_fetch_pr_json",
        lambda *_args, **_kwargs: {"headRefOid": "h" * 40, "baseRefName": "main"},
    )
    monkeypatch.setattr(
        skill_runner,
        "_recover_skill_pr_review_contexts",
        lambda *_args, **_kwargs: (None, None, None, None, ()),
    )
    monkeypatch.setattr(
        skill_runner,
        "_build_resume",
        lambda *_args, **_kwargs: {
            "current_plan_subject": "h" * 40,
            "round_number": 1,
            "completed_round_number": 1,
            "completed_reviewer_names": [],
            "completed_reviewer_data": [],
            "prior_items": [],
        },
    )
    monkeypatch.setattr(skill_runner, "_reconcile_pending_comment", lambda *args, **kwargs: None)
    monkeypatch.setattr(skill_runner, "_fetch_pr_diff", lambda *args, **kwargs: "diff")
    monkeypatch.setattr(skill_runner, "_authoritative_issue_number_from_pr", lambda *args, **kwargs: 781)
    monkeypatch.setattr(skill_runner, "_prepare_skill_memory", lambda *args, **kwargs: None)
    monkeypatch.setattr(skill_runner, "_prepare_skill_pr_architecture_checkout", lambda *args, **kwargs: True)
    monkeypatch.setattr(prompt_builders, "_acquire_skill_architecture", lambda *args, **kwargs: pair)
    monkeypatch.setattr(
        skill_runner,
        "_write_host_review_request",
        lambda **kwargs: captured.append(kwargs["architecture_context"]) or Path("/tmp/request"),
    )

    skill_runner.cmd_run_pr_round(args)
    capsys.readouterr()
    assert captured == [pair]


def test_pr_round_reacquires_architecture_once_and_reuses_identical_resume(
    monkeypatch, capsys
) -> None:
    import helpers.prompt_builders as prompt_builders
    import helpers.skill_runner as skill_runner

    old = ArchitectureSnapshot(
        repository="owner/repo", path="ARCHITECTURE.md", revision="o" * 40,
        blob_oid="p" * 40, sha256="q" * 64, availability="available",
        size=8, content="# Old\n", heading_index=("Old",),
    )
    current = ArchitectureSnapshot(
        repository="owner/repo", path="ARCHITECTURE.md", revision="n" * 40,
        blob_oid="r" * 40, sha256="s" * 64, availability="available",
        size=12, content="# Current\n", heading_index=("Current",),
    )
    resume = {
        "current_plan_subject": "h" * 40,
        "round_number": 1,
        "completed_round_number": 1,
        "completed_reviewer_names": ["Codex"],
        "completed_reviewer_data": [{
            "reviewer_name": "Codex",
            "state": "approved",
            "new_items": [],
            "blocking_items": [],
            "dispositions": [],
            "architecture_identity": old.identity(),
            "architecture_contract_version": 1,
        }],
        "prior_items": [],
    }
    args = SimpleNamespace(
        pr=781, repo="owner/repo", reviewers=["codex"], dry_run=True,
        head_sha="h" * 40, architecture_context_enabled=True,
        workdir=None, workdir_codex=None,
    )
    preparation_order: list[str] = []
    prompts: list[object] = []
    reviewer_calls = 0

    monkeypatch.setattr(
        skill_runner, "_fetch_pr_json",
        lambda *_args, **_kwargs: {"headRefOid": "h" * 40, "baseRefName": "main"},
    )
    monkeypatch.setattr(
        skill_runner, "_recover_skill_pr_review_contexts",
        lambda *_args, **_kwargs: (None, None, None, None, ()),
    )
    monkeypatch.setattr(skill_runner, "_build_resume", lambda *_args, **_kwargs: resume)
    monkeypatch.setattr(skill_runner, "_reconcile_pending_comment", lambda *args, **kwargs: None)
    monkeypatch.setattr(skill_runner, "_fetch_pr_diff", lambda *args, **kwargs: "diff")
    monkeypatch.setattr(skill_runner, "_authoritative_issue_number_from_pr", lambda *args, **kwargs: 781)
    monkeypatch.setattr(skill_runner, "_prepare_skill_memory", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        skill_runner,
        "_prepare_skill_pr_architecture_checkout",
        lambda *args, **kwargs: preparation_order.append("prepare") or True,
    )
    monkeypatch.setattr(
        prompt_builders,
        "_acquire_skill_architecture",
        lambda *args, **kwargs: preparation_order.append("acquire") or current,
    )
    monkeypatch.setattr(
        prompt_builders,
        "build_review_prompt_for_skill",
        lambda *args, **kwargs: prompts.append(kwargs["architecture_context"]) or "prompt",
    )

    def fake_run_reviewer(**kwargs):
        nonlocal reviewer_calls
        reviewer_calls += 1
        record = {
            "reviewer_name": "Codex", "state": "approved", "new_items": [],
            "blocking_items": [], "dispositions": [],
            "architecture_identity": current.identity(),
            "architecture_contract_version": 1,
        }
        resume["completed_reviewer_data"] = [record]
        resume["completed_reviewer_names"] = ["Codex"]
        return record

    monkeypatch.setattr(skill_runner, "_run_reviewer", fake_run_reviewer)

    skill_runner.cmd_run_pr_round(args)
    capsys.readouterr()
    assert preparation_order == ["prepare", "acquire"]
    assert prompts == [current]
    assert reviewer_calls == 1

    skill_runner.cmd_run_pr_round(args)
    capsys.readouterr()
    assert reviewer_calls == 1
    assert prompts == [current]


def test_build_resume_preserves_architecture_identity_for_resume_filter(
    monkeypatch, capsys
) -> None:
    import helpers.state_manager as state_manager
    from coding_review_agent_loop.round_state import PostedRoundMetadata

    identity = {
        "repository": "owner/repo",
        "path": "ARCHITECTURE.md",
        "revision": "a" * 40,
    }
    metadata = PostedRoundMetadata(
        flow="pr",
        role="reviewer",
        agent="Gemini",
        round_number=2,
        subject="head-7",
        architecture_identity=identity,
        architecture_contract_version=1,
    )
    resumed = type(
        "Resumed",
        (),
        {
            "round_number": 2,
            "prior_items": (),
            "compact_prior_summaries": (),
            "completed_reviews": (type("Record", (), {"metadata": metadata})(),),
            "local_test_evidence": None,
        },
    )()
    monkeypatch.setattr(state_manager, "_fetch_issue_comments", lambda *args, **kwargs: [])
    monkeypatch.setattr(state_manager, "_resume_pr_round", lambda *args, **kwargs: resumed)

    state_manager.cmd_build_resume(
        type(
            "Args",
            (),
            {
                "repo": "owner/repo",
                "issue": 7,
                "reviewers": ["gemini"],
                "flow": "pr",
                "head_sha": "head-7",
                "pr": 7,
                "gh_cmd": "gh",
            },
        )()
    )

    descriptor = json.loads(capsys.readouterr().out)
    record = descriptor["completed_reviewer_data"][0]
    assert record["architecture_identity"] == identity
    assert record["architecture_contract_version"] == 1

    from helpers.skill_runner import _filter_resume_for_architecture

    current = _filter_resume_for_architecture(
        descriptor, architecture_identity=identity
    )
    assert current["completed_reviewer_names"] == ["Gemini"]
    changed = _filter_resume_for_architecture(
        descriptor,
        architecture_identity={**identity, "revision": "b" * 40},
    )
    assert changed["completed_reviewer_names"] == []
