"""Early signed-evidence acceptance through the real fake-runner PR lifecycle."""
import dataclasses

import pytest

from coding_review_agent_loop.cli import run_pr_loop
from coding_review_agent_loop.errors import AgentLoopError, HumanDecisionRequiredError
from coding_review_agent_loop.github import _parse_pr_human_requirements
from coding_review_agent_loop.round_metadata import (
    _attach_round_metadata,
)
import coding_review_agent_loop.pr_loop as pr_loop
import coding_review_agent_loop.pr_loop_support as support
from agent_loop_helpers import FakeRunner, make_config
from pr_loop_test_helpers import (_evidence_review, _signed, _resolve, _EVIDENCE_TEXT,
                                  _pr_records, _freeze_records, _release_records)

HEAD = "6367c48dd193d56ea7b0baad25b19455e529f5ee"


def deferred(tmp_path, *, mode="compact", policy="all-reviewers", two_owners=False, requests=(_EVIDENCE_TEXT,)):
    runner = FakeRunner(codex_outputs=[_evidence_review(evidence=list(requests))],
        gemini_outputs=[_evidence_review(reviewer="Google Gemini", evidence=["Operator acceptance B"])] if two_owners else [],
        pr_check_runs_payload={"check_runs": [{"name": "test", "status": "in_progress", "conclusion": None}]})
    config = make_config(tmp_path, pr_review_context_mode=mode, pr_review_policy=policy,
                         reviewer=("codex", "gemini") if two_owners else "codex")
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert not _freeze_records(runner)
    if mode == "compact":
        # Start from a legitimate deferred-evidence checkpoint in round 2 so
        # the compact-context branch (which begins at round 2) is exercised.
        from coding_review_agent_loop.native_evidence import (
            _append_recorded_evidence_requests,
        )
        records = _pr_records(runner)
        items = []
        _append_recorded_evidence_requests(items, [r for r in records if r.metadata.role == "reviewer"], records=records)
        support._publish_evidence_release(runner, config=config, pr_number=77, round_number=2,
            head_sha=HEAD, items=items, reason="refresh-clean", surfaced_requirement_ids=(),
            allowed_rounds=config.max_rounds, watch_failure_extension_used=False,
            watch_head_extension_used=False)
    runner.pr_check_runs_payload = {"check_runs": [{"name": "test", "status": "completed", "conclusion": "success"}]}
    return runner, config


def acceptance(runner, text, *, citation=True, reemit=False, signed=True, two_owners=False):
    comment = _signed(text, 1)
    if not signed:
        comment["body"] = text
    runner.pr_payload["comments"].append(comment)
    reqs = _parse_pr_human_requirements(runner.pr_payload)
    vote = _resolve()
    if citation:
        vote["evidence_requirement_ids"] = [reqs[-1].requirement_id] if reqs else ["not-live"]
    votes = [vote]
    if two_owners:
        votes.append({"item_id": "item-2", "disposition": "blocking", "note": "Gemini owns the other acceptance."})
        runner.gemini_outputs.append(_evidence_review(state="blocking", reviewer="Google Gemini", hr=True, dispositions=[
            _resolve(), {"item_id": "item-2", "disposition": "blocking", "note": "Still need acceptance B."}]))
    runner.codex_outputs.append(_evidence_review(state="blocking" if two_owners else "approved", hr=bool(reqs), dispositions=votes,
        evidence=[_EVIDENCE_TEXT] if reemit else []))
    return reqs[-1].requirement_id if reqs else None


@pytest.mark.parametrize("mode", ["compact", "full"])
@pytest.mark.parametrize("policy", ["all-reviewers", "selective-intermediate"])
@pytest.mark.parametrize("reference", [HEAD[:7], HEAD[:12].upper(), HEAD])
def test_early_acceptance(tmp_path, mode, policy, reference):
    runner, config = deferred(tmp_path, mode=mode, policy=policy)
    rid = acceptance(runner, f"Screenshot accepted for {reference}; manifest verified.", reemit=True)
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert not _freeze_records(runner)
    releases = _release_records(runner)
    assert releases[-1].metadata.evidence_release.reason == "evidence-cleared"
    assert len([r for r in releases if r.metadata.evidence_release.reason == "evidence-cleared"]) == 1
    assert rid in releases[-1].metadata.evidence_release.signed_requirement_ids_surfaced
    assert not releases[-1].metadata.prior_items
    assert releases[-1].metadata.evidence_clearances
    assert not runner.codex_outputs
    assert len(_parse_pr_human_requirements(runner.pr_payload)) == 1
    assert not any(cmd[:1] == ["claude"] for cmd, _ in runner.commands)
    # An admitted release survives restart without another reviewer or terminal write.
    before = len(_release_records(runner))
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert len(_release_records(runner)) == before


@pytest.mark.parametrize("text,citation,kind", [
    ("Accepted for " + "a" * 40, True, "wrong"),
    ("Screenshots accepted", True, "missing"),
    ("Accepted for " + HEAD[:6], True, "six"),
    ("Accepted digest " + HEAD + "a" * 24, True, "digest"),
    ("Accepted for " + HEAD, False, "bare"),
    ("Accepted for " + HEAD[:7], True, "ambiguous"),
    ("Accepted for " + HEAD[:12], True, "ambiguous-long"),
    ("Accepted for " + HEAD, True, "unsurfaced"),
    ("Accepted for " + HEAD, True, "unsigned"),
])
def test_invalid_binding_freezes(tmp_path, text, citation, kind):
    runner, config = deferred(tmp_path)
    if kind.startswith("ambiguous"):
        record = _pr_records(runner)[0]
        runner.pr_payload["comments"].insert(0, {"author": {"login": "coding-review-agent-loop"}, "_rest_author_id": 424242,
            "body": _attach_round_metadata("Historical review", dataclasses.replace(record.metadata,
                subject=HEAD[:12] + "a" * 28 if kind == "ambiguous-long" else HEAD[:7] + "a" * 33))})
    acceptance(runner, text, citation=citation, signed=kind != "unsigned")
    if kind == "unsurfaced":
        import json
        output = runner.codex_outputs.pop()
        payload = json.loads(output[:output.index("\n<!--")])
        payload["prior_item_dispositions"][0]["evidence_requirement_ids"] = ["not-surfaced"]
        runner.codex_outputs.append(json.dumps(payload) + output[output.index("\n<!--"):])
    with pytest.raises(HumanDecisionRequiredError):
        run_pr_loop(runner, pr_number=77, config=config)
    assert _freeze_records(runner)
    assert not any(r.metadata.evidence_release.reason == "evidence-cleared" for r in _release_records(runner))


def test_partial_acceptance_persists_remaining_owner(tmp_path):
    runner, config = deferred(tmp_path, two_owners=True)
    acceptance(runner, "Accepted for " + HEAD, two_owners=True)
    with pytest.raises(HumanDecisionRequiredError):
        run_pr_loop(runner, pr_number=77, config=config)
    freeze = _freeze_records(runner)[-1].metadata
    assert [i.item_id for i in freeze.prior_items if i.obligation_kind == "human-exact-head-evidence"] == ["item-2"]
    assert freeze.evidence_clearances
    count = len(_freeze_records(runner))
    with pytest.raises(HumanDecisionRequiredError):
        run_pr_loop(runner, pr_number=77, config=config)
    assert len(_freeze_records(runner)) == count


@pytest.mark.parametrize("change", ["edit", "delete", "new", "push"])
@pytest.mark.parametrize("classified", [False, True])
def test_live_gate_races(tmp_path, monkeypatch, change, classified):
    runner, config = deferred(tmp_path)
    acceptance(runner, "Accepted for " + HEAD)
    if classified:
        from pr_loop_test_helpers import _classified_review, _association
        rid = _parse_pr_human_requirements(runner.pr_payload)[-1].requirement_id
        runner.codex_outputs[-1] = _classified_review([_association(rid, "evidence-only", answers=["item-1"])],
            dispositions=[{**_resolve(), "evidence_requirement_ids": [rid]}])
    real_gate = pr_loop._evidence_freeze_gate
    changed = []
    def racing_gate(*args, **kwargs):
        if not changed and any(d.evidence_requirement_ids for r in _pr_records(runner) for d in r.metadata.dispositions):
            changed.append(True)
            comments = runner.pr_payload["comments"]
            signed = next(c for c in comments if c["body"].endswith("-- Human Reviewer"))
            if change == "delete":
                comments.remove(signed)
            elif change == "edit":
                signed["body"] = "Edited acceptance without a head.\n\n-- Human Reviewer"
            elif change == "new":
                comments.append(_signed("Also inspect the status banner.", 2))
            else:
                runner.git_head = runner.pr_payload["headRefOid"] = "a" * 40
        return real_gate(*args, **kwargs)
    monkeypatch.setattr(pr_loop, "_evidence_freeze_gate", racing_gate)
    # A refresh sees edited/new input and keeps the request; deletion also cannot clear.
    if change in {"edit", "new"}:
        runner.codex_outputs.append(_evidence_review(state="blocking", hr=True, dispositions=[{
            "item_id": "item-1", "disposition": "blocking", "note": "Need valid signed acceptance."}]))
    with pytest.raises(AgentLoopError):
        run_pr_loop(runner, pr_number=77, config=config)
    assert changed
    assert not any(r.metadata.evidence_release.reason == "evidence-cleared" for r in _release_records(runner))
    if change == "new":
        assert _freeze_records(runner)[-1].metadata.round_number == 2


@pytest.mark.parametrize("stop", ["before-release", "after-release"])
@pytest.mark.parametrize("classified", [False, True])
def test_early_release_publication_resume(tmp_path, monkeypatch, stop, classified):
    runner, config = deferred(tmp_path)
    acceptance(runner, "Accepted for " + HEAD)
    if classified:
        from pr_loop_test_helpers import _classified_review, _association
        rid = _parse_pr_human_requirements(runner.pr_payload)[-1].requirement_id
        runner.codex_outputs[-1] = _classified_review([_association(rid, "evidence-only", answers=["item-1"])],
            dispositions=[{**_resolve(), "evidence_requirement_ids": [rid]}])
    real_publish = support._publish_evidence_release
    stopped = []
    def interrupt(*args, **kwargs):
        if kwargs["reason"] != "evidence-cleared" or stopped:
            return real_publish(*args, **kwargs)
        stopped.append(True)
        if stop == "after-release":
            real_publish(*args, **kwargs)
        raise AgentLoopError("simulated release stop")
    monkeypatch.setattr(support, "_publish_evidence_release", interrupt)
    with pytest.raises(AgentLoopError, match="simulated release stop"):
        run_pr_loop(runner, pr_number=77, config=config)
    assert any(r.metadata.evidence_release.reason == "evidence-cleared" for r in _release_records(runner)) == (stop == "after-release")
    if stop == "before-release":
        acceptance_vote = _resolve()
        acceptance_vote["evidence_requirement_ids"] = [_parse_pr_human_requirements(runner.pr_payload)[-1].requirement_id]
        runner.codex_outputs.append(_evidence_review(hr=True, dispositions=[acceptance_vote]))
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert len([r for r in _release_records(runner) if r.metadata.evidence_release.reason == "evidence-cleared"]) == 1
    assert not _freeze_records(runner)


def test_push_after_early_release_reopens_native_request(tmp_path):
    runner, config = deferred(tmp_path)
    acceptance(runner, "Accepted for " + HEAD)
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    runner.git_head = runner.pr_payload["headRefOid"] = "a" * 40
    runner.simulate_new_process()
    runner.codex_outputs.append(_evidence_review(state="blocking", hr=True, dispositions=[{
        "item_id": "item-1", "disposition": "blocking", "note": "Need acceptance at the new head."}]))
    with pytest.raises(HumanDecisionRequiredError):
        run_pr_loop(runner, pr_number=77, config=config)
    freeze = _freeze_records(runner)[-1].metadata
    assert freeze.subject == "a" * 40
    assert freeze.prior_items[0].item_id == "item-1"
    assert not any(cmd[:3] == ["gh", "pr", "merge"] for cmd, _ in runner.commands)


@pytest.mark.parametrize("unanimous", [False, True])
def test_removed_requester_board_fallback(tmp_path, unanimous):
    runner, config = deferred(tmp_path)
    comment = _signed("Screenshot accepted for " + HEAD, 1)
    runner.pr_payload["comments"].append(comment)
    rid = _parse_pr_human_requirements(runner.pr_payload)[-1].requirement_id
    vote = {**_resolve(), "evidence_requirement_ids": [rid]}
    runner.gemini_outputs.append(_evidence_review(reviewer="Google Gemini", hr=True, dispositions=[vote]))
    runner.claude_outputs.append(_evidence_review(reviewer="Anthropic Claude", hr=True,
        state="approved" if unanimous else "blocking", dispositions=[vote if unanimous else {
            "item_id": "item-1", "disposition": "blocking", "note": "Acceptance remains insufficient."}]))
    config = dataclasses.replace(config, reviewer=("gemini", "claude"))
    if unanimous:
        assert run_pr_loop(runner, pr_number=77, config=config) == 0
        assert _release_records(runner)[-1].metadata.evidence_release.reason == "evidence-cleared"
        assert not _freeze_records(runner)
    else:
        with pytest.raises(HumanDecisionRequiredError):
            run_pr_loop(runner, pr_number=77, config=config)
        assert _freeze_records(runner)[-1].metadata.prior_items[0].reviewer == "Codex"


def test_foreign_historical_subject_cannot_change_head_binding(tmp_path):
    runner, config = deferred(tmp_path)
    record = _pr_records(runner)[0]
    runner.pr_payload["comments"].insert(0, {"author": {"login": "foreign"}, "_rest_author_id": 999,
        "body": _attach_round_metadata("Untrusted history", dataclasses.replace(record.metadata,
            subject=HEAD[:7] + "a" * 33))})
    acceptance(runner, "Screenshot accepted for " + HEAD[:7])
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert not _freeze_records(runner)
    assert _release_records(runner)[-1].metadata.evidence_release.reason == "evidence-cleared"


@pytest.mark.parametrize("vote_kind", ["keep", "resolve", "conflict"])
def test_alias_dispositions_in_real_loop(tmp_path, vote_kind):
    from coding_review_agent_loop.unresolved_items import _upsert_evidence_obligation
    runner, config = deferred(tmp_path)
    items = list(_release_records(runner)[-1].metadata.prior_items)
    items, _ = _upsert_evidence_obligation(items, item_number=2, reviewer="Codex",
        text="Second operator screenshot acceptance", source_round=2, current_head_sha=HEAD)
    items, consumed = _upsert_evidence_obligation(items, item_number=3, reviewer="Codex",
        text="Consolidate item-1 and item-2 into this existing acceptance request.",
        source_round=2, current_head_sha=HEAD)
    assert not consumed and items[0].alias_item_ids == ("item-2",)
    support._publish_evidence_release(runner, config=config, pr_number=77, round_number=2,
        head_sha=HEAD, items=items, reason="refresh-clean", surfaced_requirement_ids=(),
        allowed_rounds=config.max_rounds, watch_failure_extension_used=False,
        watch_head_extension_used=False)
    runner.pr_payload["comments"].append(_signed("Accepted for " + HEAD, 1))
    rid = _parse_pr_human_requirements(runner.pr_payload)[-1].requirement_id
    votes = [{"item_id": "item-2", "disposition": "resolved" if vote_kind == "resolve" else "blocking",
              "note": "The historical alias still names this acceptance request.",
              "evidence_requirement_ids": [rid]}]
    if vote_kind == "conflict":
        votes.append({**_resolve(), "evidence_requirement_ids": [rid]})
    runner.codex_outputs.append(_evidence_review(hr=True, dispositions=votes,
        state="approved" if vote_kind == "resolve" else "blocking"))
    if vote_kind == "resolve":
        assert run_pr_loop(runner, pr_number=77, config=config) == 0
        assert not _freeze_records(runner)
        assert len(_release_records(runner)[-1].metadata.evidence_clearances) >= 2
    else:
        with pytest.raises(HumanDecisionRequiredError):
            run_pr_loop(runner, pr_number=77, config=config)
        assert [i.item_id for i in _freeze_records(runner)[-1].metadata.prior_items] == ["item-1"]
        assert not _freeze_records(runner)[-1].metadata.evidence_clearances
    assert not any(cmd[:1] == ["claude"] for cmd, _ in runner.commands)


@pytest.mark.parametrize("mode", ["compact", "full"])
@pytest.mark.parametrize("persisted", [False, True])
@pytest.mark.parametrize("frozen", [False, True])
def test_consolidated_clearance_push_restart_preserves_lineage(tmp_path, mode, persisted, frozen):
    from coding_review_agent_loop.unresolved_items import _upsert_evidence_obligation
    from coding_review_agent_loop.native_evidence import (
        _append_recorded_evidence_requests,
    )
    runner, config = deferred(tmp_path, mode=mode)
    records = _pr_records(runner)
    items = []
    _append_recorded_evidence_requests(items, [r for r in records if r.metadata.role == "reviewer"], records=records)
    items, _ = _upsert_evidence_obligation(items, item_number=2, reviewer="Codex",
        text="Second distinct operator acceptance", source_round=2, current_head_sha=HEAD)
    originals = list(items)
    text = "Consolidate item-1 and item-2 into this existing operator acceptance request."
    if persisted:
        items, _ = _upsert_evidence_obligation(items, item_number=3, reviewer="Codex",
            text=text, source_round=2, current_head_sha=HEAD)
    support._publish_evidence_release(runner, config=config, pr_number=77, round_number=2,
        head_sha=HEAD, items=items, reason="refresh-clean", surfaced_requirement_ids=(),
        allowed_rounds=config.max_rounds, watch_failure_extension_used=False,
        watch_head_extension_used=False)
    if frozen:
        with pytest.raises(HumanDecisionRequiredError):
            run_pr_loop(runner, pr_number=77, config=config)
    if frozen:
        assert not runner.codex_outputs
    initial_freezes = len(_freeze_records(runner))
    runner.pr_payload["comments"].append(_signed("Accepted for " + HEAD, 1))
    rid = _parse_pr_human_requirements(runner.pr_payload)[-1].requirement_id
    votes = [{"item_id": item_id, "disposition": "resolved", "note": "Signed acceptance is adequate.",
              "evidence_requirement_ids": [rid]} for item_id in ("item-1", "item-2")]
    runner.codex_outputs.append(_evidence_review(hr=True, dispositions=votes,
        evidence=[] if persisted else [text]))
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert not _release_records(runner)[-1].metadata.prior_items
    assert len(_freeze_records(runner)) == initial_freezes
    assert len([r for r in _release_records(runner)
                if r.metadata.evidence_release.reason == "evidence-cleared"]) == 1
    assert not runner.codex_outputs
    runner.git_head = runner.pr_payload["headRefOid"] = "a" * 40
    runner.simulate_new_process()
    runner.codex_outputs.append(_evidence_review(hr=True, state="blocking", dispositions=[{
        "item_id": "item-2", "disposition": "blocking", "note": "Need acceptance at the new head."}]))
    with pytest.raises(HumanDecisionRequiredError):
        run_pr_loop(runner, pr_number=77, config=config)
    freeze = _freeze_records(runner)[-1].metadata
    assert len(freeze.prior_items) == 1
    survivor = freeze.prior_items[0]
    assert survivor.item_id == "item-1" and survivor.text == text
    assert survivor.alias_item_ids == ("item-2",)
    assert survivor.obligation_identity == originals[0].obligation_identity
    assert originals[1].obligation_identity in survivor.alias_obligation_identities
    assert freeze.subject == "a" * 40
    assert not any(cmd[:3] == ["gh", "pr", "merge"] for cmd, _ in runner.commands)


@pytest.mark.parametrize("mode", ["compact", "full"])
@pytest.mark.parametrize("foreign", [False, True])
def test_push_restart_preserves_distinct_requests_with_foreign_heads(tmp_path, mode, foreign):
    from coding_review_agent_loop.native_evidence import (
        _append_recorded_evidence_requests,
    )
    tokens = ("1234567" + "b" * 33, "abcdef0" + "c" * 33)
    runner, config = deferred(tmp_path, mode=mode,
        requests=tuple("Accept screenshot " + token[:7] for token in tokens))
    records = _pr_records(runner)
    items = []
    _append_recorded_evidence_requests(items, [r for r in records if r.metadata.role == "reviewer"], records=records)
    assert len(items) == 2
    if foreign:
        template = _pr_records(runner)[0].metadata
        for token in tokens:
            runner.pr_payload["comments"].insert(0, {
                "author": {"login": "foreign"}, "_rest_author_id": 999,
                "body": _attach_round_metadata("Foreign historical subject", dataclasses.replace(
                    template, subject=token, prior_items=(), new_items=(), evidence_requests=()))})
    support._publish_evidence_release(runner, config=config, pr_number=77, round_number=2,
        head_sha=HEAD, items=items, reason="refresh-clean", surfaced_requirement_ids=(),
        allowed_rounds=config.max_rounds, watch_failure_extension_used=False,
        watch_head_extension_used=False)
    runner.pr_payload["comments"].append(_signed("Accepted for " + HEAD, 1))
    rid = _parse_pr_human_requirements(runner.pr_payload)[-1].requirement_id
    runner.codex_outputs.append(_evidence_review(hr=True, dispositions=[{
        "item_id": i.item_id, "disposition": "resolved", "note": "Acceptance is adequate.",
        "evidence_requirement_ids": [rid]} for i in items]))
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert not _release_records(runner)[-1].metadata.prior_items
    runner.git_head = runner.pr_payload["headRefOid"] = "d" * 40
    runner.simulate_new_process()
    runner.codex_outputs.append(_evidence_review(hr=True, state="blocking", dispositions=[{
        "item_id": i.item_id, "disposition": "blocking", "note": "Need new-head acceptance."
    } for i in items]))
    with pytest.raises(HumanDecisionRequiredError):
        run_pr_loop(runner, pr_number=77, config=config)
    reopened = _freeze_records(runner)[-1].metadata.prior_items
    assert [(i.item_id, i.text, i.obligation_identity, i.alias_item_ids) for i in reopened] == [
        (i.item_id, i.text, i.obligation_identity, ()) for i in items]
    assert not any(cmd[:3] == ["gh", "pr", "merge"] for cmd, _ in runner.commands)


@pytest.mark.parametrize("mode", ["compact", "full"])
@pytest.mark.parametrize("frozen", [False, True])
def test_classified_evidence_only_needs_no_generic_ack(tmp_path, mode, frozen):
    from pr_loop_test_helpers import _classified_review, _association
    runner, config = deferred(tmp_path, mode=mode)
    if frozen:
        with pytest.raises(HumanDecisionRequiredError):
            run_pr_loop(runner, pr_number=77, config=config)
    before = len(_freeze_records(runner))
    rid = acceptance(runner, "Accepted for " + HEAD)
    runner.codex_outputs[-1] = _classified_review([_association(rid, "evidence-only", answers=["item-1"])],
        dispositions=[{**_resolve(), "evidence_requirement_ids": [rid]}])
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert len(_freeze_records(runner)) == before
    assert _release_records(runner)[-1].metadata.evidence_release.reason == "evidence-cleared"
    assert not any(cmd[0] == "claude" for cmd, _ in runner.commands)
    runner.simulate_new_process()
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert len(_freeze_records(runner)) == before


@pytest.mark.parametrize("valid", [True, False])
def test_classified_partial_answer_preserves_other_owner(tmp_path, valid):
    from pr_loop_test_helpers import _classified_review, _association
    runner, config = deferred(tmp_path, two_owners=True)
    rid = acceptance(runner, "Accepted for " + (HEAD if valid else "a" * 40), two_owners=True)
    runner.codex_outputs[-1] = _classified_review([_association(rid, "evidence-only", answers=["item-1"])],
        state="blocking", dispositions=[{**_resolve(), "evidence_requirement_ids": [rid]},
            {"item_id": "item-2", "disposition": "blocking", "note": "Other owner pending."}])
    with pytest.raises(HumanDecisionRequiredError):
        run_pr_loop(runner, pr_number=77, config=config)
    ids = {i.item_id for i in _freeze_records(runner)[-1].metadata.prior_items}
    assert ids == ({"item-2"} if valid else {"item-1", "item-2"})
    assert not any(cmd[0] == "claude" for cmd, _ in runner.commands)


def test_classified_mixed_answer_keeps_original_id_with_source_and_attributed_leftover(tmp_path):
    from pr_loop_test_helpers import _classified_review, _association
    from agent_loop_helpers import structured_coder_followup
    runner, config = deferred(tmp_path)
    runner.advance_pr_head_on_coder_followup = False
    rid = acceptance(runner, "Accepted for " + HEAD + ". Also fix the retry bound.")
    runner.codex_outputs[-1] = _classified_review([_association(rid, "mixed", answers=["item-1"], leftover="Fix the retry bound.")],
        state="blocking", blocking=["Fix the retry bound."],
        dispositions=[{**_resolve(), "evidence_requirement_ids": [rid]}])
    runner.claude_outputs = [structured_coder_followup(addressed_items=["item-2"], human_requirement_ids=[rid],
        human_requirement_dispositions=[{"requirement_id": rid, "disposition": "addressed", "evidence": "Retry bound fixed; human answer is supplied separately."}])]
    # A follow-up reviewer retains the source classification and confirms code.
    runner.codex_outputs.append(_classified_review([_association(rid, "mixed", answers=["item-1"], leftover="Fix the retry bound.")], hr=True,
        dispositions=[{**_resolve(), "evidence_requirement_ids": [rid]}, _resolve("item-2")]))
    assert run_pr_loop(runner, pr_number=77, config=dataclasses.replace(config, max_rounds=3)) == 0
    prompts = [cmd[-1] for cmd, _ in runner.commands if cmd[0] == "claude"]
    assert len(prompts) == 1
    assert f"Requirement {rid}:" in prompts[0]
    # The requirement block carries the verbatim signed source plus the attributed leftover.
    block = prompts[0].split(f"Requirement {rid}:", 1)[1]
    assert "Accepted for " + HEAD + ". Also fix the retry bound." in block
    assert "- Codex: Fix the retry bound." in block


@pytest.mark.parametrize("fault", ["foreign-owner", "cross-round-conflict"])
def test_invalid_classification_can_reach_fresh_review(tmp_path, monkeypatch, fault):
    from pr_loop_test_helpers import _association, _classified_review
    from coding_review_agent_loop.round_metadata import (
        _strip_round_metadata,
    )
    runner, config = deferred(tmp_path, mode="full", two_owners=True)
    runner.pr_payload["comments"].append(_signed("A human must accept the policy.", 1))
    rid = _parse_pr_human_requirements(runner.pr_payload)[0].requirement_id
    if fault == "cross-round-conflict":
        # Durable latest records from distinct rounds must not poison resume.
        records = [r for r in _pr_records(runner) if r.metadata.role == "reviewer"]
        for index, record in enumerate(records):
            entry = _association(rid, "code") if index else _association(rid, "human-only", request=_EVIDENCE_TEXT)
            metadata = dataclasses.replace(record.metadata, round_number=index + 1,
                requirement_evidence=(entry,))
            runner.pr_payload["comments"].append({"author": {"login": "coding-review-agent-loop"},
                "_rest_author_id": 424242, "body": _attach_round_metadata(_strip_round_metadata(record.body), metadata)})
    projections = []
    real_projection = pr_loop.historical_projection
    def capture_projection(requirements, records):
        result = real_projection(requirements, records)
        projections.append(result)
        return result
    monkeypatch.setattr(pr_loop, "historical_projection", capture_projection)
    votes = [{"item_id": "item-1", "disposition": "blocking", "note": "Pending human acceptance."},
             {"item_id": "item-2", "disposition": "blocking", "note": "Pending other owner."}]
    if fault == "foreign-owner":
        runner.codex_outputs.append(_classified_review([
            _association(rid, "evidence-only", answers=["item-2"])], state="blocking", dispositions=votes))
    runner.codex_outputs.append(_classified_review([
        _association(rid, "human-only", request=_EVIDENCE_TEXT)], state="blocking", dispositions=votes, evidence=[_EVIDENCE_TEXT]))
    runner.gemini_outputs.append(_classified_review([
        _association(rid, "human-only", request="Operator acceptance B")], reviewer="Google Gemini",
        state="blocking", dispositions=votes, evidence=["Operator acceptance B"]))
    with pytest.raises(HumanDecisionRequiredError):
        run_pr_loop(runner, pr_number=77, config=config)
    assert _freeze_records(runner)
    assert not any(cmd[0] == "claude" for cmd, _ in runner.commands)
    assert not runner.codex_outputs
    if fault == "cross-round-conflict":
        assert projections and any(r.requirement_id == rid and r.body == "A human must accept the policy."
                                   for r in projections[0])
    else:
        assert any("Previous response not accepted: required judgement field" in cmd[-1]
                   and "requirement_evidence" in cmd[-1] for cmd, _ in runner.commands if cmd[0] == "codex")



def _carried_requester(tmp_path, *, policy="all-reviewers", citation=True, reference=HEAD):
    """#2019: Codex resolves its evidence item at HEAD while Gemini blocks; the head is unchanged."""
    from agent_loop_helpers import structured_coder_followup
    runner = FakeRunner(codex_outputs=[_evidence_review(evidence=[_EVIDENCE_TEXT])],
        gemini_outputs=[_evidence_review(reviewer="Google Gemini")],
        pr_check_runs_payload={"check_runs": [{"name": "test", "status": "in_progress", "conclusion": None}]})
    config = make_config(tmp_path, pr_review_context_mode="full", pr_review_policy=policy,
                         reviewer=("codex", "gemini"), max_rounds=5, quiet=False)
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert not _freeze_records(runner)
    runner.pr_check_runs_payload = {"check_runs": [{"name": "test", "status": "completed", "conclusion": "success"}]}
    runner.advance_pr_head_on_coder_followup = False
    runner.pr_payload["comments"].append(_signed("Screenshot accepted for " + reference, 1))
    rid = _parse_pr_human_requirements(runner.pr_payload)[-1].requirement_id
    vote = {**_resolve(), "evidence_requirement_ids": [rid]} if citation else _resolve()
    runner.codex_outputs.append(_evidence_review(hr=True, dispositions=[vote]))
    # Gemini is not the requester: its votes on item-1 never count toward clearance.
    runner.gemini_outputs.append(_evidence_review(reviewer="Google Gemini", state="blocking", hr=True,
        blocking=["Fix the retry bound."], dispositions=[_resolve()]))
    runner.claude_outputs = [structured_coder_followup(addressed_items=["item-2"], human_requirement_ids=[rid],
        human_requirement_dispositions=[{"requirement_id": rid, "disposition": "addressed",
                                         "evidence": "The signed acceptance is supplied separately."}])]
    runner.gemini_outputs.append(_evidence_review(reviewer="Google Gemini", hr=True, dispositions=[_resolve(), _resolve("item-2")]))
    return runner, config


def _codex_calls(runner):
    return sum(1 for cmd, _ in runner.commands if cmd[:1] == ["codex"])


@pytest.mark.parametrize("policy", ["all-reviewers", "selective-intermediate"])
def test_carried_same_head_requester_resolution_clears_without_freeze(tmp_path, capsys, policy):
    runner, config = _carried_requester(tmp_path, policy=policy)
    codex_before = _codex_calls(runner)
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    # Round N: Codex resolves at HEAD and Gemini blocks. Round N+1 carries Codex without a turn.
    assert _codex_calls(runner) == codex_before + 1
    assert "skipping Codex; it approved unchanged PR head " + HEAD in capsys.readouterr().err
    assert not _freeze_records(runner)
    cleared = [r for r in _release_records(runner) if r.metadata.evidence_release.reason == "evidence-cleared"]
    assert len(cleared) == 1
    assert cleared[0].metadata.evidence_clearances
    assert all(head == HEAD for _, head in cleared[0].metadata.evidence_clearances)
    assert not cleared[0].metadata.prior_items
    assert not runner.codex_outputs and not runner.gemini_outputs and not runner.claude_outputs
    # Idempotent: a rerun writes no second release and calls no agent.
    commands = len(runner.commands)
    runner.simulate_new_process()
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert len([r for r in _release_records(runner) if r.metadata.evidence_release.reason == "evidence-cleared"]) == 1
    assert not _freeze_records(runner)
    assert not any(cmd[0] in {"codex", "gemini", "claude"} for cmd, _ in runner.commands[commands:])


@pytest.mark.parametrize("kind", ["no-citation", "foreign-head"])
def test_carried_requester_without_valid_citation_freezes_and_rerun_stops(tmp_path, kind):
    runner, config = _carried_requester(tmp_path, citation=kind != "no-citation",
                                        reference="a" * 40 if kind == "foreign-head" else HEAD)
    with pytest.raises(HumanDecisionRequiredError):
        run_pr_loop(runner, pr_number=77, config=config)
    freezes = _freeze_records(runner)
    assert [i.item_id for i in freezes[-1].metadata.prior_items] == ["item-1"]
    assert not any(r.metadata.evidence_release.reason == "evidence-cleared" for r in _release_records(runner))
    # Without new signed input the rerun stops at once with no agent calls.
    commands = len(runner.commands)
    runner.simulate_new_process()
    with pytest.raises(HumanDecisionRequiredError):
        run_pr_loop(runner, pr_number=77, config=config)
    assert len(_freeze_records(runner)) == len(freezes)
    assert not any(cmd[0] in {"codex", "gemini", "claude"} for cmd, _ in runner.commands[commands:])


def test_primary_then_panel_requester_resolution_clears_after_digest_latch(tmp_path, monkeypatch, capsys):
    """#2019 under a selective (primary-then-panel) policy with a scoped secondary blocker.

    Here the evidence item is created in a round that no coder checkpoint
    follows, so the next round's recovery digest check (pr_loop's
    machine-obligation digest check) sees a changed ledger and latches the full
    board; the requester's same-head approval is then carried through the
    unchanged-head skip, and the gate must still clear it. The selective-pause
    branch is covered by test_selective_paused_requester_resolution_clears_without_freeze.
    """
    from agent_loop_helpers import patch_attr_in, structured_coder_followup
    from coding_review_agent_loop.review_scheduling import TransitionClassification
    from pr_loop_test_helpers import _staged_config
    patch_attr_in(monkeypatch, (support, pr_loop), "_observe_pr_transition",
        lambda *a, **k: TransitionClassification("narrow", "scoped fix"))
    runner = FakeRunner(codex_outputs=[_evidence_review()],
        gemini_outputs=[_evidence_review(reviewer="Google Gemini")],
        antigravity_outputs=[_evidence_review(reviewer="Antigravity", evidence=[_EVIDENCE_TEXT])],
        pr_check_runs_payload={"check_runs": [{"name": "test", "status": "in_progress", "conclusion": None}]})
    config = _staged_config(tmp_path, quiet=False)
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    runner.pr_check_runs_payload = {"check_runs": [{"name": "test", "status": "completed", "conclusion": "success"}]}
    runner.advance_pr_head_on_coder_followup = False
    runner.pr_payload["comments"].append(_signed("Screenshot accepted for " + HEAD, 1))
    rid = _parse_pr_human_requirements(runner.pr_payload)[-1].requirement_id
    runner.codex_outputs.append(_evidence_review(hr=True, dispositions=[_resolve()]))
    runner.antigravity_outputs.append(_evidence_review(reviewer="Antigravity", hr=True,
        dispositions=[{**_resolve(), "evidence_requirement_ids": [rid]}]))
    runner.gemini_outputs.append(_evidence_review(reviewer="Google Gemini", state="blocking", hr=True,
        blocking=[{"text": "Fix the retry bound.", "fix_scope": ["src/worker.py"]}], dispositions=[_resolve()]))
    runner.claude_outputs = [structured_coder_followup(addressed_items=["item-2"], human_requirement_ids=[rid],
        human_requirement_dispositions=[{"requirement_id": rid, "disposition": "addressed",
                                         "evidence": "The signed acceptance is supplied separately."}])]
    runner.codex_outputs.append(_evidence_review(hr=True, dispositions=[_resolve(), _resolve("item-2")]))
    runner.gemini_outputs.append(_evidence_review(reviewer="Google Gemini", hr=True, dispositions=[_resolve(), _resolve("item-2")]))
    agy_before = sum(1 for cmd, _ in runner.commands if cmd[:1] == ["agy"])
    capsys.readouterr()
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    err = capsys.readouterr().err
    assert "scheduler obligation digest changed during recovery" in err
    assert "skipping Antigravity; it approved unchanged PR head " + HEAD in err
    # Antigravity resolved once (citation round) and was not invoked again.
    assert sum(1 for cmd, _ in runner.commands if cmd[:1] == ["agy"]) == agy_before + 1
    assert not _freeze_records(runner)
    cleared = [r for r in _release_records(runner) if r.metadata.evidence_release.reason == "evidence-cleared"]
    assert len(cleared) == 1 and all(head == HEAD for _, head in cleared[0].metadata.evidence_clearances)
    assert not cleared[0].metadata.prior_items
    commands = len(runner.commands)
    runner.simulate_new_process()
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert len([r for r in _release_records(runner) if r.metadata.evidence_release.reason == "evidence-cleared"]) == 1
    assert not any(cmd[0] in {"codex", "gemini", "agy", "claude"} for cmd, _ in runner.commands[commands:])


def _restart_after_coder_checkpoint(runner):
    """Interrupt the process at its first PR write after a coder checkpoint.

    Within one invocation the next round reuses a PR snapshot that predates the
    coder publication (pr_loop's post-coder ``prefetched_pr_context``), so the
    recovery digest check cannot see the checkpoint and latches the full board.
    A restarted process reads the checkpoint, which is the consistent scheduler
    state the selective pause needs.
    """
    real_run, armed = runner.run, [False]

    def run(args, **kwargs):
        comment = list(args)[:3] == ["gh", "pr", "comment"]
        if comment and armed[0]:
            armed[0] = False
            raise KeyboardInterrupt("simulated restart after the coder checkpoint")
        result = real_run(args, **kwargs)
        if comment:
            armed[0] = _pr_records(runner)[-1].metadata.role == "coder"
        return result
    runner.run = run


def test_selective_paused_requester_resolution_clears_without_freeze(tmp_path, monkeypatch, capsys):
    """#2019 paused-selective-approval-clears: the requester is paused, not carried.

    The evidence item and the scoped code work are established together before
    the first coder checkpoint, and the signed acceptance exists before any
    coder turn, so every later round starts from a consistent digest and the
    primary-then-panel remediation narrows to the finding owner. Antigravity
    resolves its evidence item with a citation in round 3 while Gemini still
    blocks; round 4 pauses Antigravity at the early selective continue, and the
    gate must count that scheduler-counted same-head approval.
    """
    from agent_loop_helpers import patch_attr_in, structured_coder_followup
    from coding_review_agent_loop.review_scheduling import TransitionClassification
    from pr_loop_test_helpers import _staged_config
    patch_attr_in(monkeypatch, (support, pr_loop), "_observe_pr_transition",
        lambda *a, **k: TransitionClassification("narrow", "scoped fix"))
    runner = FakeRunner(codex_outputs=[_evidence_review(hr=True)],
        gemini_outputs=[_evidence_review(reviewer="Google Gemini", state="blocking", hr=True,
            blocking=[{"text": "Fix the retry bound.", "fix_scope": ["src/worker.py"]}])],
        antigravity_outputs=[_evidence_review(reviewer="Antigravity", state="blocking", hr=True,
            evidence=[_EVIDENCE_TEXT], blocking=[{"text": "Log the retry count.", "fix_scope": ["src/log.py"]}])],
        pr_check_runs_payload={"check_runs": [{"name": "test", "status": "completed", "conclusion": "success"}]})
    runner.advance_pr_head_on_coder_followup = False
    runner.pr_payload["comments"].append(_signed("Screenshot accepted for " + HEAD, 1))
    rid = _parse_pr_human_requirements(runner.pr_payload)[-1].requirement_id
    _restart_after_coder_checkpoint(runner)
    signed = dict(human_requirement_ids=[rid], human_requirement_dispositions=[
        {"requirement_id": rid, "disposition": "addressed", "evidence": "The signed acceptance is supplied separately."}])
    config = _staged_config(tmp_path, quiet=False)
    # Rounds 1-2: Codex (primary) approves; Gemini and Antigravity block on
    # scoped code work and Antigravity requests evidence (item-3).
    runner.claude_outputs = [structured_coder_followup(addressed_items=["item-1", "item-2"], **signed)]
    with pytest.raises(KeyboardInterrupt):
        run_pr_loop(runner, pr_number=77, config=config)
    evidence = [i for i in _pr_records(runner)[-1].metadata.prior_items if i.obligation_kind == "human-exact-head-evidence"]
    assert [(i.item_id, i.reviewer) for i in evidence] == [("item-3", "Antigravity")]
    # Round 3: the owners recheck; Antigravity resolves item-3 citing the signed
    # acceptance and approves, while Gemini still blocks item-1.
    runner.simulate_new_process()
    runner.claude_outputs = [structured_coder_followup(addressed_items=["item-1"], **signed)]
    runner.codex_outputs.append(_evidence_review(hr=True, dispositions=[_resolve("item-1"), _resolve("item-2"), _resolve("item-3")]))
    runner.gemini_outputs += [
        _evidence_review(reviewer="Google Gemini", state="blocking", hr=True, dispositions=[
            {"item_id": "item-1", "disposition": "blocking", "note": "Still unbounded."}, _resolve("item-2"), _resolve("item-3")]),
        _evidence_review(reviewer="Google Gemini", hr=True, dispositions=[_resolve("item-1"), _resolve("item-2"), _resolve("item-3")])]
    runner.antigravity_outputs.append(_evidence_review(reviewer="Antigravity", hr=True, dispositions=[
        _resolve("item-1"), _resolve("item-2"), {**_resolve("item-3"), "evidence_requirement_ids": [rid]}]))
    with pytest.raises(KeyboardInterrupt):
        run_pr_loop(runner, pr_number=77, config=config)
    assert not runner.antigravity_outputs
    # Round 4: a consistent checkpoint narrows to Gemini and pauses Antigravity.
    runner.simulate_new_process()
    agy_calls = sum(1 for cmd, _ in runner.commands if cmd[:1] == ["agy"])
    capsys.readouterr()
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    err = capsys.readouterr().err
    assert "Round 4: pausing Antigravity; its prior approval is historical for this candidate" in err
    assert "skipping Antigravity" not in err
    assert "scheduler obligation digest changed" not in err
    assert sum(1 for cmd, _ in runner.commands if cmd[:1] == ["agy"]) == agy_calls
    assert not runner.gemini_outputs and not runner.claude_outputs
    assert not _freeze_records(runner)
    cleared = [r for r in _release_records(runner) if r.metadata.evidence_release.reason == "evidence-cleared"]
    assert len(cleared) == 1
    assert [identity for identity, _ in cleared[0].metadata.evidence_clearances] == [evidence[0].obligation_identity]
    assert all(head == HEAD for _, head in cleared[0].metadata.evidence_clearances)
    assert not cleared[0].metadata.prior_items
    # Idempotent: a rerun writes no second release and calls no agent.
    commands = len(runner.commands)
    runner.simulate_new_process()
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert len([r for r in _release_records(runner) if r.metadata.evidence_release.reason == "evidence-cleared"]) == 1
    assert not _freeze_records(runner)
    assert not any(cmd[0] in {"codex", "gemini", "agy", "claude"} for cmd, _ in runner.commands[commands:])
