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
from coding_review_agent_loop.github_retry import GitHubAmbiguousWriteError, GitHubTransientExhaustedError
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


def test_freeze_without_an_actor_or_baseline_stops_before_the_first_post(tmp_path, monkeypatch):
    _compose(monkeypatch, "sidecar 1", _anchor())
    spool = _spool(tmp_path)
    blind_actor = FakeGitHub()
    blind_actor.fail_user = True
    with pytest.raises(PublicationResumeStop, match="actor is unavailable"):
        _publish(blind_actor, tmp_path, spool)
    blind_listing = FakeGitHub()
    blind_listing.list_failure_page = 1
    with pytest.raises(PublicationResumeStop, match="baseline listing is incomplete"):
        _publish(blind_listing, tmp_path, spool)
    assert blind_actor.writes == [] and blind_listing.writes == []
    # The validated response stays spooled and, once the reads recover, a rerun publishes.
    assert spool.publication_state(REVIEWER)[0] == "fresh"
    healthy = FakeGitHub()
    _publish(healthy, tmp_path, spool)
    assert len(healthy.comments) == 2 and spool.publication_state(REVIEWER)[0] == "carrier"


def test_doubled_host_footer_is_neither_adopted_nor_counted_public(tmp_path, monkeypatch):
    footer = "\n\n---\n_Generated by [Claude Code](https://claude.ai/code)_"
    anchor = _anchor()
    _compose(monkeypatch, anchor)
    spool = _spool(tmp_path)
    fake = FakeGitHub(["accepted"])
    original = fake._write

    def doubled(body):
        result = original(body)
        fake.comments[-1]["body"] = body + footer + footer
        return result

    fake._write = doubled  # type: ignore[method-assign]
    _publish(fake, tmp_path, spool)
    # Not adopted: the write was replayed, so a second (single-footer-free) comment exists.
    assert len(fake.writes) == 2
    # And the frozen-sequence classifier does not count the doubled-footer comment.
    class C:
        def __init__(self, cid, body, created):
            self.comment_id, self.body, self.created = cid, body, created

    import datetime as dt

    now = dt.datetime.now(dt.timezone.utc)
    carrier = PublicationCarrier(
        bodies=("a",), body_sha256=(__import__("hashlib").sha256(b"a").hexdigest(),),
        prepared_at=now.isoformat(), actor_id=7, actor_login="x", baseline_ids=(),
        response_sha256="r", validation_context_digest="c",
    )
    assert classify_frozen_sequence(carrier, [C(1, "a" + footer, now)]).matched_count == 0
    assert classify_frozen_sequence(carrier, [C(1, "a", now)]).matched_count == 1


def test_listing_comment_without_a_text_body_is_unknown_and_blocks_replay(tmp_path, monkeypatch):
    _compose(monkeypatch, _anchor())
    spool = _spool(tmp_path)
    fake = FakeGitHub(["accepted"])
    original = fake._write

    def nullified(body):
        result = original(body)
        fake.comments[-1]["body"] = None
        return result

    fake._write = nullified  # type: ignore[method-assign]
    with pytest.raises(GitHubAmbiguousWriteError):
        post_pr_comment(fake, config=make_config(tmp_path), pr_number=7, body="plain text")
    assert len(fake.writes) == 1


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


# ---------------------------------------------------------------------------
# Preflight over every spool record (round 2 of the #1261 review)
# ---------------------------------------------------------------------------

from coding_review_agent_loop import review_rounds


def _two_reviewer_spool(tmp_path):
    spool = ReviewRoundSpool(
        root=tmp_path / "spool2", repo="OWNER/REPO", surface="pr", number=7,
        round_number=1, subject=SUBJECT,
    )
    for name in ("Codex", "Gemini"):
        spool.store(name, FIELDS)
    return spool


def _publication_for(fake, tmp_path, spool, *, contexts):
    def build(name):
        return round_publication(
            fake, config=make_config(tmp_path), spool=spool, reviewer_name=name,
            flow="pr", round_number=1, subject=SUBJECT, surface_kind="pr", number=7,
            validation_context=contexts.get(name, CONTEXT),
        )

    return build


def _freeze_with_prefix(fake, tmp_path, spool, name, *, bodies, post_count):
    """Freeze ``name``'s carrier and post only the first ``post_count`` bodies."""
    import datetime as dt

    from coding_review_agent_loop.github import read_complete_comment_listing

    del dt, read_complete_comment_listing
    hook = round_publication(
        fake, config=make_config(tmp_path), spool=spool, reviewer_name=name, flow="pr",
        round_number=1, subject=SUBJECT, surface_kind="pr", number=7, validation_context=CONTEXT,
    )
    hook.prepare([_Prepared(body) for body in bodies])
    for body in bodies[:post_count]:
        fake.add(body)


def _anchor_for(name):
    metadata = PostedRoundMetadata(
        flow="pr", role="reviewer", agent=name, round_number=1, subject=SUBJECT, phase="publication",
    )
    return str(_attach_round_metadata(f"{name} review", metadata))


def _preflight(fake, tmp_path, spool, *, contexts=None, published=()):
    posted = []
    review_rounds._preflight_spooled_publications(
        fake, config=make_config(tmp_path), spool=spool,
        reviewers=["codex", "gemini"],
        validators_for=lambda name: {},
        publication_for=_publication_for(fake, tmp_path, spool, contexts=contexts or {}),
        published_names=published,
        post_frozen=lambda plan: posted.append(plan),
        selected_reviewers=["codex", "gemini"],
    )
    return posted


@pytest.fixture
def replayable(monkeypatch):
    monkeypatch.setattr(review_rounds, "_replay_spooled_review", lambda *a, **k: object())


def test_preflight_posts_nothing_when_a_later_record_refuses(tmp_path, replayable):
    spool = _two_reviewer_spool(tmp_path)
    fake = FakeGitHub()
    _freeze_with_prefix(fake, tmp_path, spool, "Codex", bodies=["c-side", _anchor_for("Codex")], post_count=1)
    _freeze_with_prefix(fake, tmp_path, spool, "Gemini", bodies=[_anchor_for("Gemini")], post_count=0)
    comments_before = len(fake.comments)
    contexts = {"Gemini": context_digest(head="moved", surfaced=[])}
    with pytest.raises(PublicationResumeStop, match="validation context"):
        _preflight(fake, tmp_path, spool, contexts=contexts)
    assert len(fake.comments) == comments_before and fake.writes == []
    # With both valid, the verified suffix of every record is handed to the poster.
    posted = _preflight(fake, tmp_path, spool)
    assert [len(plan.bodies) - len(plan.already_public) for plan in posted] == [1, 1]


def test_preflight_refuses_a_partly_public_carrier_beside_a_fresh_reviewer(tmp_path, replayable):
    spool = _two_reviewer_spool(tmp_path)
    fake = FakeGitHub()
    _freeze_with_prefix(fake, tmp_path, spool, "Codex", bodies=["c-side", _anchor_for("Codex")], post_count=1)
    spool.remove("Gemini")
    writes = len(fake.writes)
    with pytest.raises(PublicationResumeStop, match="partly public"):
        _preflight(fake, tmp_path, spool)
    assert len(fake.writes) == writes
    # No public prefix: a deferral alone is fine (the suffix waits for settlement).
    clean = _two_reviewer_spool(tmp_path / "clean")
    clean.remove("Gemini")  # a fresh turn is still needed, but nothing is public yet
    clean_fake = FakeGitHub()
    _freeze_with_prefix(clean_fake, tmp_path, clean, "Codex", bodies=["c-side", _anchor_for("Codex")], post_count=0)
    assert _preflight(clean_fake, tmp_path, clean) == []


@pytest.mark.parametrize("damage", ["null_carrier", "unreadable_response"])
def test_malformed_or_null_carrier_stops_before_anything(tmp_path, replayable, damage):
    spool = _two_reviewer_spool(tmp_path)
    fake = FakeGitHub()
    _freeze_with_prefix(fake, tmp_path, spool, "Codex", bodies=["c-side", _anchor_for("Codex")], post_count=1)
    path = spool._path("Codex")
    payload = json.loads(path.read_text())
    if damage == "null_carrier":
        payload["publication"] = None
    else:
        payload["response"] = {"text": 5}
    path.write_text(json.dumps(payload))
    assert spool.publication_state("Codex")[0] == "malformed"
    with pytest.raises(PublicationResumeStop, match="malformed or unreadable"):
        _preflight(fake, tmp_path, spool)
    assert fake.writes == [] and path.exists()


def test_legacy_record_is_checked_even_when_its_anchor_is_already_public(tmp_path, replayable):
    spool = _two_reviewer_spool(tmp_path)
    for name in ("Codex", "Gemini"):
        path = spool._path(name)
        payload = json.loads(path.read_text())
        payload.pop("carrier_protocol")
        path.write_text(json.dumps(payload))
    fake = FakeGitHub(actor=OTHER_ACTOR)
    fake.add(_anchor_for("Codex"), author=ACTOR)  # the old invocation's public anchor
    with pytest.raises(PublicationResumeStop, match="possible earlier publication"):
        _preflight(fake, tmp_path, spool, published=("Codex",))
    blind = FakeGitHub()
    blind.list_failure_page = 1
    with pytest.raises(PublicationResumeStop, match="could not be read completely"):
        _preflight(blind, tmp_path, spool, published=("Codex",))
    assert spool.record_exists("Codex") and spool.record_exists("Gemini")


def test_a_readable_but_unvalidatable_peer_counts_as_a_fresh_turn(tmp_path, monkeypatch):
    """Fresh-turn necessity follows validated replayability, not spool readability."""
    spool = _two_reviewer_spool(tmp_path)
    fake = FakeGitHub()
    _freeze_with_prefix(fake, tmp_path, spool, "Codex", bodies=["c-side", _anchor_for("Codex")], post_count=1)

    def replay(runner, *, config, reviewer, fields, validators):
        return None if reviewer == "gemini" else object()

    monkeypatch.setattr(review_rounds, "_replay_spooled_review", replay)
    writes = len(fake.writes)
    # Gemini's record is readable, but the current validator rejects it: it would
    # launch fresh beside Codex's public sidecar, so the run stops with zero posts.
    with pytest.raises(PublicationResumeStop, match="partly public"):
        _preflight(fake, tmp_path, spool)
    assert len(fake.writes) == writes and spool.record_exists("Gemini")
    # A settled failure outcome is replayed as a failure, never relaunched.
    spool.store_failure("Gemini", message="boom", failure_category=None)
    assert [len(p.bodies) - len(p.already_public) for p in _preflight(fake, tmp_path, spool)] == [1]
