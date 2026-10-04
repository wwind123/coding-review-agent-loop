"""Frozen publication carriers and cross-invocation spool resume (#1258)."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

import coding_review_agent_loop.github as github_module
from coding_review_agent_loop import github_retry
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.github import post_pr_comment, round_publication
from coding_review_agent_loop.github_retry import GitHubTransientExhaustedError
from coding_review_agent_loop.protocol_markers import TrustedBody
from coding_review_agent_loop.publication_resume import (
    PublicationCarrier,
    PublicationResumeStop,
    classify_frozen_sequence,
    context_digest,
    response_digest,
)
from coding_review_agent_loop.review_rounds import _launch_reviewer_turns, _ReviewerTurnResult
from coding_review_agent_loop.review_spool import ReviewRoundSpool
from coding_review_agent_loop.round_state import PostedRoundMetadata, _attach_round_metadata

from agent_loop_helpers import make_config
from test_github import ACTOR, OTHER_ACTOR, FakeGitHub, _iso

REVIEWER = "Codex"
SUBJECT = "head-sha-1"
CONTEXT = context_digest(head=SUBJECT, surfaced=[])
FIELDS = {"text": "approved", "session_id": None, "model_used": "m"}


class _Prepared(str):
    def validate_for_surface(self, _surface):
        return None


def _anchor(summary="Codex review"):
    metadata = PostedRoundMetadata(
        flow="pr", role="reviewer", agent=REVIEWER, round_number=1, subject=SUBJECT,
        phase="publication",
    )
    return str(_attach_round_metadata(summary, metadata))


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setattr(github_module, "log", lambda _config, _message: None)
    monkeypatch.setattr(github_module, "active_workdir", lambda config: None)
    monkeypatch.setattr(github_retry, "_sleep", lambda _seconds: None)


def _spool(tmp_path, *, store=True):
    spool = ReviewRoundSpool(
        root=tmp_path / "spool", repo="OWNER/REPO", surface="pr", number=7,
        round_number=1, subject=SUBJECT,
    )
    if store:
        spool.store(REVIEWER, FIELDS)
    return spool


def _compose(monkeypatch, *bodies):
    prepared = [_Prepared(body) for body in bodies]
    monkeypatch.setattr(github_module, "prepare_round_comment", lambda carrier: prepared)


def _hook(fake, tmp_path, spool, *, context=CONTEXT, config=None):
    return round_publication(
        fake, config=config or make_config(tmp_path), spool=spool, reviewer_name=REVIEWER,
        flow="pr", round_number=1, subject=SUBJECT, surface_kind="pr", number=7,
        validation_context=context,
    )


def _publish(fake, tmp_path, spool, *, context=CONTEXT):
    post_pr_comment(
        fake, config=make_config(tmp_path), pr_number=7, body="ignored",
        publication=_hook(fake, tmp_path, spool, context=context),
    )


def _bodies(fake):
    return [c["body"] for c in fake.comments]


def test_bodies_are_frozen_before_the_first_post(tmp_path, monkeypatch):
    anchor = _anchor()
    _compose(monkeypatch, "sidecar 1", "sidecar 2", anchor)
    spool = _spool(tmp_path)
    fake = FakeGitHub()
    _publish(fake, tmp_path, spool)
    state, raw = spool.publication_state(REVIEWER)
    assert state == "carrier"
    carrier = PublicationCarrier.from_dict(raw)
    assert list(carrier.bodies) == ["sidecar 1", "sidecar 2", anchor] == _bodies(fake)
    assert carrier.actor_id == ACTOR[1] and carrier.baseline_ids == ()
    assert carrier.response_sha256 == response_digest(spool.load(REVIEWER))
    assert carrier.validation_context_digest == CONTEXT


def test_new_records_carry_the_protocol_marker_and_load_is_unchanged(tmp_path):
    spool = _spool(tmp_path)
    assert spool.publication_state(REVIEWER) == ("fresh", None)
    assert spool.load(REVIEWER)["text"] == "approved"


def test_rerun_after_exhaustion_posts_only_the_missing_suffix_verbatim(tmp_path, monkeypatch):
    anchor = _anchor()
    spool = _spool(tmp_path)
    _compose(monkeypatch, "sidecar 1", "sidecar 2", anchor)
    fake = FakeGitHub(["ok", "fail", "fail", "fail"])
    with pytest.raises(GitHubTransientExhaustedError):
        _publish(fake, tmp_path, spool)
    assert _bodies(fake) == ["sidecar 1"]
    assert spool.publication_state(REVIEWER)[0] == "carrier"
    # The rerun composes different bodies (scheduler metadata drifted).
    _compose(monkeypatch, "DRIFTED sidecar 1", "DRIFTED sidecar 2", _anchor("drifted"))
    _publish(fake, tmp_path, spool)
    assert _bodies(fake) == ["sidecar 1", "sidecar 2", anchor]


def test_rerun_with_everything_public_posts_nothing(tmp_path, monkeypatch):
    anchor = _anchor()
    spool = _spool(tmp_path)
    _compose(monkeypatch, "sidecar 1", anchor)
    fake = FakeGitHub()
    _publish(fake, tmp_path, spool)
    writes = len(fake.writes)
    _publish(fake, tmp_path, spool)
    assert len(fake.writes) == writes and len(fake.comments) == 2


def test_resume_reuses_a_prefix_published_long_ago(tmp_path, monkeypatch):
    anchor = _anchor()
    spool = _spool(tmp_path)
    _compose(monkeypatch, "sidecar 1", anchor)
    fake = FakeGitHub(["ok", "fail", "fail", "fail"])
    with pytest.raises(GitHubTransientExhaustedError):
        _publish(fake, tmp_path, spool)
    # Hours later: resume is untimed, so the old prefix is still claimed.
    for comment in fake.comments:
        comment["created_at"] = _iso(-6 * 3600)
    _, raw = spool.publication_state(REVIEWER)
    spool.store_publication(REVIEWER, {**raw, "prepared_at": _iso_aware(-6 * 3600 - 5)})
    _publish(fake, tmp_path, spool)
    assert _bodies(fake) == ["sidecar 1", anchor]


def _iso_aware(delta_seconds):
    import datetime as dt

    return (dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=delta_seconds)).isoformat()


def test_gap_in_the_frozen_sequence_stops_without_posts(tmp_path, monkeypatch):
    anchor = _anchor()
    spool = _spool(tmp_path)
    _compose(monkeypatch, "sidecar 1", "sidecar 2", anchor)
    fake = FakeGitHub()
    _publish(fake, tmp_path, spool)
    fake.comments.pop(0)  # sidecar 1 deleted; sidecar 2 and the anchor remain
    writes = len(fake.writes)
    with pytest.raises(PublicationResumeStop, match="absent while a later body"):
        _publish(fake, tmp_path, spool)
    assert len(fake.writes) == writes
    assert spool.publication_state(REVIEWER)[0] == "carrier"


def test_ambiguous_duplicate_of_a_frozen_body_stops(tmp_path, monkeypatch):
    anchor = _anchor()
    spool = _spool(tmp_path)
    _compose(monkeypatch, "sidecar 1", anchor)
    fake = FakeGitHub()
    _publish(fake, tmp_path, spool)
    fake.add("sidecar 1")
    writes = len(fake.writes)
    with pytest.raises(PublicationResumeStop, match="identical candidates"):
        _publish(fake, tmp_path, spool)
    assert len(fake.writes) == writes


def test_actor_change_stops_before_any_listing_decision(tmp_path, monkeypatch):
    spool = _spool(tmp_path)
    _compose(monkeypatch, "sidecar 1", _anchor())
    fake = FakeGitHub(["ok", "fail", "fail", "fail"])
    with pytest.raises(GitHubTransientExhaustedError):
        _publish(fake, tmp_path, spool)
    other = FakeGitHub(actor=OTHER_ACTOR)
    other.comments = list(fake.comments)
    with pytest.raises(PublicationResumeStop) as excinfo:
        _publish(other, tmp_path, spool)
    assert "id 7" in str(excinfo.value) and "id 99" in str(excinfo.value)
    assert other.writes == [] and other.listings == 0
    assert spool.publication_state(REVIEWER)[0] == "carrier"


def test_changed_validation_context_stops_without_posts(tmp_path, monkeypatch):
    spool = _spool(tmp_path)
    _compose(monkeypatch, "sidecar 1", _anchor())
    fake = FakeGitHub(["ok", "fail", "fail", "fail"])
    with pytest.raises(GitHubTransientExhaustedError):
        _publish(fake, tmp_path, spool)
    writes = len(fake.writes)
    with pytest.raises(PublicationResumeStop, match="validation context"):
        _publish(fake, tmp_path, spool, context=context_digest(head="moved", surfaced=[]))
    assert len(fake.writes) == writes


def test_response_swapped_after_freezing_stops(tmp_path, monkeypatch):
    spool = _spool(tmp_path)
    _compose(monkeypatch, _anchor())
    fake = FakeGitHub(["fail", "fail", "fail"])
    with pytest.raises(GitHubTransientExhaustedError):
        _publish(fake, tmp_path, spool)
    path = spool._path(REVIEWER)
    payload = json.loads(path.read_text())
    payload["response"]["text"] = "a different review"
    path.write_text(json.dumps(payload))
    with pytest.raises(PublicationResumeStop, match="no longer matches the frozen carrier"):
        _publish(fake, tmp_path, spool)
    assert fake.writes and len(fake.comments) == 0


def test_malformed_and_unbound_carriers_fail_closed(tmp_path, monkeypatch):
    spool = _spool(tmp_path)
    _compose(monkeypatch, _anchor())
    spool.store_publication(REVIEWER, {"bodies": []})
    fake = FakeGitHub()
    with pytest.raises(PublicationResumeStop, match="malformed"):
        _publish(fake, tmp_path, spool)
    assert fake.writes == []
    # A carrier frozen without a resolvable actor publishes now but is never resumed.
    unbound_spool = _spool(tmp_path / "unbound")
    unbound = FakeGitHub(["accepted"])
    unbound.fail_user = True
    with pytest.raises(AgentLoopError):
        _publish(unbound, tmp_path, unbound_spool)
    assert unbound_spool.publication_state(REVIEWER)[0] == "carrier"
    again = FakeGitHub()
    with pytest.raises(PublicationResumeStop, match="provable actor"):
        _publish(again, tmp_path, unbound_spool)


def _legacy_spool(tmp_path):
    spool = _spool(tmp_path)
    path = spool._path(REVIEWER)
    payload = json.loads(path.read_text())
    payload.pop("carrier_protocol")
    path.write_text(json.dumps(payload))
    assert spool.publication_state(REVIEWER)[0] == "legacy"
    return spool


def test_legacy_record_with_nothing_published_composes_and_publishes(tmp_path, monkeypatch):
    spool = _legacy_spool(tmp_path)
    _compose(monkeypatch, "sidecar 1", _anchor())
    fake = FakeGitHub()
    _publish(fake, tmp_path, spool)
    assert len(fake.comments) == 2
    assert spool.publication_state(REVIEWER)[0] == "carrier"


@pytest.mark.parametrize("author", [ACTOR, OTHER_ACTOR])
def test_legacy_record_with_possible_prior_publication_by_any_author_stops(
    tmp_path, monkeypatch, author
):
    spool = _legacy_spool(tmp_path)
    _compose(monkeypatch, "sidecar 1", _anchor())
    fake = FakeGitHub()
    fake.add("Agent-loop review attachment 1/1 <!-- AGENT_LOOP_SIDECAR: e30= -->", author=author)
    with pytest.raises(PublicationResumeStop, match="possible earlier publication") as excinfo:
        _publish(fake, tmp_path, spool)
    assert author[0] in str(excinfo.value)
    assert fake.writes == [] and spool.record_exists(REVIEWER)


def test_legacy_record_with_this_reviewers_anchor_stops_and_unknown_listing_stops(
    tmp_path, monkeypatch
):
    spool = _legacy_spool(tmp_path)
    _compose(monkeypatch, _anchor())
    fake = FakeGitHub()
    fake.add(_anchor(), author=OTHER_ACTOR)
    with pytest.raises(PublicationResumeStop, match="possible earlier publication"):
        _publish(fake, tmp_path, spool)
    blind = FakeGitHub()
    blind.list_failure_page = 1
    with pytest.raises(PublicationResumeStop, match="could not be read completely"):
        _publish(blind, tmp_path, spool)
    assert blind.writes == []


def test_classifier_handles_identical_bodies_and_baseline():
    class C:
        def __init__(self, cid, body, created):
            self.comment_id, self.body, self.created = cid, body, created

    import datetime as dt

    now = dt.datetime.now(dt.timezone.utc)
    carrier = PublicationCarrier(
        bodies=("same", "same", "end"),
        body_sha256=tuple(
            __import__("hashlib").sha256(b.encode()).hexdigest() for b in ("same", "same", "end")
        ),
        prepared_at=now.isoformat(), actor_id=7, actor_login="a", baseline_ids=(1,),
        response_sha256="r", validation_context_digest="c",
    )
    comments = [C(1, "same", now), C(2, "same", now), C(3, "same", now), C(4, "end", now)]
    assert classify_frozen_sequence(carrier, comments).kind == "complete"
    assert classify_frozen_sequence(carrier, comments + [C(5, "same", now)]).kind == "ambiguous"
    assert classify_frozen_sequence(carrier, [C(4, "end", now)]).kind == "gap"
    assert classify_frozen_sequence(carrier, [C(2, "same", now), C(3, "same", now)]).kind == "prefix"
    assert classify_frozen_sequence(carrier, [C(9, "end", now), C(2, "same", now)]).kind == "gap"
    # The baseline comment (id 1) is never claimed.
    assert classify_frozen_sequence(carrier, [C(1, "same", now)]).matched_count == 0


def test_launcher_never_invokes_a_reviewer_whose_frozen_carrier_no_longer_validates(tmp_path):
    spool = _spool(tmp_path)
    spool.store_publication(REVIEWER, {"bodies": ["x"]})
    invoked = []
    runner = SimpleNamespace(terminate_active_processes=lambda: None)
    with pytest.raises(PublicationResumeStop, match="no longer validates"):
        _launch_reviewer_turns(
            runner, ["codex"], thread_name_prefix="t",
            run_turn=lambda reviewer: invoked.append(reviewer),
            spool=spool, replay_turn=lambda reviewer, fields: None,
        )
    assert invoked == [] and spool.record_exists(REVIEWER)
