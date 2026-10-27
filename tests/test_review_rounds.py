"""Review-round tests: incomplete plan-review detection (#1543)."""

from types import SimpleNamespace

import pytest
from coding_review_agent_loop.panel_evidence import PostedRoundMetadata, _plan_subject
from coding_review_agent_loop.discuss_loop import _attach_round_metadata
from coding_review_agent_loop.protocol import (
    ApprovedFollowup,
    ParsedPlanReview,
    PlanReviewItems,
    ReviewItemDisposition,
    UnresolvedReviewItem,
)
import coding_review_agent_loop.review_rounds as review_rounds
from agent_loop_helpers import structured_plan_review


@pytest.mark.parametrize(
    ("state", "blocking", "same_plan", "summary", "expected"),
    [
        ("blocking", (), (), "Unable to verify the plan without more context.", True),
        (
            "blocking",
            (ApprovedFollowup(reviewer="Codex", text="Add a migration step."),),
            (),
            "Unable to verify the plan without more context.",
            False,
        ),
        ("approved", (), (), "Unable to verify the plan without more context.", False),
    ],
)
def test_is_incomplete_plan_review(state, blocking, same_plan, summary, expected):
    parsed_review = ParsedPlanReview(
        state=state,
        summary=summary,
        items=PlanReviewItems(blocking=blocking, same_plan=same_plan, future=()),
        dispositions=(),
    )
    assert review_rounds._is_incomplete_plan_review(parsed_review) is expected


def test_is_incomplete_plan_review_matches_disposition_note():
    parsed_review = ParsedPlanReview(
        state="blocking",
        summary="",
        items=PlanReviewItems(blocking=(), same_plan=(), future=()),
        dispositions=(
            ReviewItemDisposition(
                item_id="item-1",
                reviewer="Codex",
                disposition="still blocking",
                note="Resolution could not be confirmed; plan and referenced files need review.",
            ),
        ),
    )
    assert review_rounds._is_incomplete_plan_review(parsed_review) is True


def test_is_incomplete_plan_review_allows_active_carried_disposition():
    parsed_review = ParsedPlanReview(
        state="blocking",
        summary="",
        items=PlanReviewItems(blocking=(), same_plan=(), future=()),
        dispositions=(
            ReviewItemDisposition(
                item_id="item-1",
                reviewer="Codex",
                disposition="blocking",
                note="Could not confirm the revised plan covers migration ordering.",
            ),
        ),
    )

    assert review_rounds._is_incomplete_plan_review(parsed_review) is False


def test_round_resolved_history_ids_available_when_plan_ledger_is_complete():
    # #872 follow-up: a lossless semantic patch needs resolved-history proof in
    # every ledger state, so the history is computed for complete rounds too.
    old_plan = "Old plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    mid_plan = "Mid plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    old_item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Old subject item.",
        status="blocking",
        source_status="blocking",
    )
    raised = _attach_round_metadata(
        structured_plan_review(state="blocking", blocking_plan_issues=["Old subject item."]),
        PostedRoundMetadata(
            flow="plan", role="reviewer", agent="Codex", round_number=1,
            subject=_plan_subject(old_plan), new_items=(old_item,), state="blocking",
        ),
    )
    resolved = _attach_round_metadata(
        structured_plan_review(
            state="blocking",
            prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
        ),
        PostedRoundMetadata(
            flow="plan", role="reviewer", agent="Codex", round_number=2,
            subject=_plan_subject(mid_plan), prior_items=(old_item,),
            dispositions=(ReviewItemDisposition("item-1", "OpenAI Codex", "resolved"),),
            state="blocking",
        ),
    )
    comments = [SimpleNamespace(body=raised), SimpleNamespace(body=resolved)]

    assert review_rounds._round_resolved_history_item_ids(
        prior_unresolved_items=(),
        comments=comments,
        flow="plan",
        reconciliation_mode="aggregate",
        same_status="same-plan",
    ) == ("item-1",)


def test_post_round_resolved_history_ids_cover_the_round_still_in_progress():
    # #874: the snapshot a round holds was fetched before its review turns, so
    # it cannot record the dispositions that round just applied. The proof for
    # the revision therefore also reads the round's in-process dispositions.
    plan = "Current plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    blocking = UnresolvedReviewItem(
        item_id="item-1", reviewer="OpenAI Codex", source_round=1,
        text="Migration ordering is unspecified.", status="blocking",
        source_status="blocking",
    )
    cleared = UnresolvedReviewItem(
        item_id="item-2", reviewer="OpenAI Codex", source_round=1,
        text="Rollback path is uncovered.", status="blocking",
        source_status="blocking",
    )
    raised = _attach_round_metadata(
        structured_plan_review(
            state="blocking",
            blocking_plan_issues=[blocking.text, cleared.text],
        ),
        PostedRoundMetadata(
            flow="plan", role="reviewer", agent="Codex", round_number=1,
            subject=_plan_subject(plan), new_items=(blocking, cleared), state="blocking",
        ),
    )
    # Only round 1 is in the snapshot; round 2's reviewer comment is posted
    # after it was fetched, exactly as the live loop sees it.
    comments = [SimpleNamespace(body=raised)]
    dispositions = {
        "item-1": [ReviewItemDisposition("item-1", "OpenAI Codex", "blocking", "Still unspecified.")],
        "item-2": [ReviewItemDisposition("item-2", "OpenAI Codex", "resolved")],
    }

    assert review_rounds._round_resolved_history_item_ids(
        prior_unresolved_items=(blocking,),
        comments=comments,
        flow="plan",
        reconciliation_mode="aggregate",
        same_status="same-plan",
    ) == ()
    assert review_rounds._post_round_resolved_history_item_ids(
        prior_unresolved_items=(blocking, cleared),
        dispositions_by_item=dispositions,
        carried_items=(blocking,),
        comments=comments,
        flow="plan",
        reconciliation_mode="aggregate",
        same_status="same-plan",
    ) == ("item-2",)


def test_post_round_resolved_history_ids_exclude_actively_disputed_and_carried_items():
    # The whitelist stays fail-closed: a mixed verdict, a still-carried item and
    # an undispositioned item never earn resolved-history proof.
    carried = UnresolvedReviewItem(
        item_id="item-1", reviewer="OpenAI Codex", source_round=1,
        text="Still blocking.", status="blocking", source_status="blocking",
    )
    mixed = UnresolvedReviewItem(
        item_id="item-2", reviewer="OpenAI Codex", source_round=1,
        text="One reviewer still blocks.", status="blocking", source_status="blocking",
    )
    untouched = UnresolvedReviewItem(
        item_id="item-3", reviewer="OpenAI Codex", source_round=1,
        text="Nobody dispositioned this.", status="blocking", source_status="blocking",
    )
    dispositions = {
        "item-1": [ReviewItemDisposition("item-1", "OpenAI Codex", "resolved")],
        "item-2": [
            ReviewItemDisposition("item-2", "OpenAI Codex", "resolved"),
            ReviewItemDisposition("item-2", "Anthropic Claude", "blocking", "Not covered."),
        ],
    }

    assert review_rounds._post_round_resolved_history_item_ids(
        prior_unresolved_items=(carried, mixed, untouched),
        dispositions_by_item=dispositions,
        # item-1 survived reconciliation, so it is still carried.
        carried_items=(carried,),
        comments=[],
        flow="plan",
        reconciliation_mode="aggregate",
        same_status="same-plan",
    ) == ()


# --- Review-round spool location, migration and placement (#1560) -----------

import json as _spool_json
import os as _spool_os
import stat as _spool_stat
import tempfile as _spool_tempfile

from coding_review_agent_loop.agent_permissions import configured_checkouts, response_root
from coding_review_agent_loop.errors import AgentLoopError as _SpoolError
from coding_review_agent_loop.review_spool import (
    ReviewRoundSpool,
    legacy_review_spool_root,
    review_spool_root,
)
from agent_loop_helpers import make_config as _make_spool_config

_NON_ROOT_SPOOL = pytest.mark.skipif(
    not hasattr(_spool_os, "geteuid") or _spool_os.geteuid() == 0,
    reason="root bypasses directory read permission",
)
_SPOOL_FIELDS = {"text": "Codex approves.", "model_used": "m"}


@pytest.fixture
def spool_config(tmp_path, monkeypatch):
    private = tmp_path / "private-tmp"
    private.mkdir()
    monkeypatch.setattr(_spool_tempfile, "tempdir", str(private))
    return _make_spool_config(tmp_path)


def _spool_mode(path):
    return _spool_stat.S_IMODE(_spool_os.lstat(path).st_mode)


def _round_spool(config, *, number=7, round_number=1, subject="abc"):
    return review_rounds._review_round_spool(
        config, surface="pr", number=number, round_number=round_number, subject=subject
    )


def _legacy_round(config, kind, *, number=7, round_number=1, subject="abc"):
    """A round spooled by an older version at the legacy sibling-of-memory root."""
    legacy = ReviewRoundSpool(
        root=legacy_review_spool_root(config.agent_memory_dir), repo=config.repo, surface="pr",
        number=number, round_number=round_number, subject=subject,
    )
    legacy.store("Codex", _SPOOL_FIELDS)
    if kind == "carrier":
        legacy.store_publication("Codex", {"frozen": True})
    elif kind == "malformed":
        path = legacy.directory / "Codex.json"
        payload = _spool_json.loads(path.read_text(encoding="utf-8"))
        payload["publication"] = "not-a-mapping"
        path.write_text(_spool_json.dumps(payload), encoding="utf-8")
    return legacy


@pytest.mark.parametrize("mask", [0o022, 0o700, 0o777], ids=["umask-022", "umask-700", "umask-777"])
def test_spool_lands_in_private_root_outside_reachable_trees(spool_config, mask):
    config = spool_config
    previous = _spool_os.umask(mask)
    try:
        spool = _round_spool(config)
        spool.store("Codex", _SPOOL_FIELDS)
    finally:
        _spool_os.umask(previous)

    root = review_spool_root(config.agent_memory_dir)
    assert spool.root == root
    assert root == config.agent_memory_dir.parent / "private" / "review-round-spool"
    assert root.parent != config.agent_memory_dir.parent  # not beside the memory dir
    assert not legacy_review_spool_root(config.agent_memory_dir).exists()
    resolved = _spool_os.path.realpath(root)
    for tree in (str(response_root(config)), *configured_checkouts(config)):
        tree = _spool_os.path.realpath(tree)
        assert not resolved.startswith(tree + _spool_os.sep) and not tree.startswith(resolved + _spool_os.sep)
    assert _spool_mode(root.parent) == 0o300
    assert _spool_mode(root) == 0o300
    assert _spool_mode(spool.directory) == 0o700
    record = spool.directory / "Codex.json"
    assert _spool_mode(record) == 0o600
    assert spool.publication_state("Codex") == ("fresh", None)
    assert spool.load("Codex")["text"] == "Codex approves."


@pytest.mark.parametrize("kind", ["fresh", "carrier", "malformed"])
def test_legacy_round_is_migrated_with_unchanged_classification(spool_config, kind):
    config = spool_config
    legacy = _legacy_round(config, kind)
    before = legacy.publication_state("Codex")
    loaded_before = legacy.load("Codex")
    legacy_directory = legacy.directory

    spool = _round_spool(config)
    assert spool.has_records()  # the rerun's first spool use migrates the round
    assert not legacy_directory.exists()
    assert (spool.root / legacy_directory.name).is_dir()
    assert spool.publication_state("Codex") == before
    assert spool.load("Codex") == loaded_before
    assert before[0] == kind


def test_round_present_in_both_locations_fails_closed(spool_config):
    config = spool_config
    legacy = _legacy_round(config, "carrier")
    spool = _round_spool(config)
    (spool.root / legacy.directory.name).mkdir()
    legacy_dir = legacy.directory
    with pytest.raises(_SpoolError) as info:
        spool.has_records()
    assert str(legacy_dir) in str(info.value)
    assert str(spool.root / legacy_dir.name) in str(info.value)
    with pytest.raises(_SpoolError):
        spool.publication_state("Codex")
    # Neither directory is discarded.
    assert (legacy_dir / "Codex.json").exists()
    assert (spool.root / legacy_dir.name).is_dir()


@_NON_ROOT_SPOOL
@pytest.mark.parametrize("legacy_mode", [0o755, 0o775], ids=["mode-755", "umask-002-mode-775"])
def test_retained_legacy_round_is_hidden_but_still_migrates(spool_config, legacy_mode):
    config = spool_config
    retained = _legacy_round(config, "carrier", number=99, subject="retained")
    legacy_root = retained.root
    # Older versions created the root with the process umask.
    legacy_root.chmod(legacy_mode)
    before = retained.publication_state("Codex")

    # A new, unrelated round starts and hardens the legacy root first.
    _round_spool(config, number=7)
    with pytest.raises(PermissionError):
        _spool_os.listdir(legacy_root)
    assert _spool_mode(legacy_root) == 0o300
    record = legacy_root / retained.digest / "Codex.json"
    assert _spool_json.loads(record.read_text(encoding="utf-8"))["publication"] == {"frozen": True}

    # Later, a rerun of the retained round migrates it with its classification.
    rerun = _round_spool(config, number=99, subject="retained")
    assert rerun.publication_state("Codex") == before == ("carrier", {"frozen": True})
    assert not (legacy_root / retained.digest).exists()


def test_symlinked_legacy_spool_root_is_refused_before_any_reviewer(tmp_path, spool_config):
    target = tmp_path / "elsewhere"
    target.mkdir()
    target.chmod(0o755)
    legacy_review_spool_root(spool_config.agent_memory_dir).parent.mkdir(parents=True)
    legacy_review_spool_root(spool_config.agent_memory_dir).symlink_to(target)
    with pytest.raises(_SpoolError, match="not a real directory"):
        _round_spool(spool_config)
    assert _spool_mode(target) == 0o755
    assert legacy_review_spool_root(spool_config.agent_memory_dir).is_symlink()


def test_legacy_spool_root_is_never_created(spool_config):
    _round_spool(spool_config)
    assert not legacy_review_spool_root(spool_config.agent_memory_dir).exists()


def _assert_refused_without_spool_state(config, fragment):
    with pytest.raises(_SpoolError, match="overlaps") as info:
        _round_spool(config)
    assert fragment in str(info.value)
    assert not (config.agent_memory_dir.parent / "private").exists()


def test_spool_inside_a_named_seat_workdir_is_refused(tmp_path, spool_config):
    from coding_review_agent_loop.reviewer_seats import ReviewerSeat, SeatAgent

    seat_dir = tmp_path / "seat-checkout"
    seat = SeatAgent(ReviewerSeat("seat-a", "codex", ("Model A",)), seat_dir)
    config = _make_spool_config(
        tmp_path, reviewer=(seat,), reviewer_seats=(seat,),
        agent_memory_dir=seat_dir / "cache" / "memory",
    )
    _assert_refused_without_spool_state(config, "reviewer seat seat-a workdir")


def test_raw_reviewer_seat_overlap_label_uses_the_seat_id(tmp_path, spool_config):
    """#1566: a raw ReviewerSeat is labelled by its ID, never its dataclass repr."""
    from coding_review_agent_loop.reviewer_seats import ReviewerSeat

    seat_dir = tmp_path / "raw-seat-checkout"
    seat = ReviewerSeat("seat-b", "codex", ("Model B",), workdir=seat_dir)
    config = _make_spool_config(tmp_path, reviewer_seats=(seat,))
    labels = [label for label, _ in review_rounds._spool_overlap_set(config)]
    assert "reviewer seat seat-b workdir" in labels
    assert not any("ReviewerSeat(" in label for label in labels)

    config = _make_spool_config(
        tmp_path, reviewer_seats=(seat,), agent_memory_dir=seat_dir / "cache" / "memory",
    )
    with pytest.raises(_SpoolError, match="overlaps") as info:
        review_rounds.validate_review_spool_placement(config)
    assert "reviewer seat seat-b workdir" in str(info.value)
    assert "ReviewerSeat(" not in str(info.value)


def test_spool_inside_the_repository_checkout_is_refused(tmp_path, spool_config, monkeypatch):
    repository = tmp_path / "orchestrator-repo"
    (repository / ".git").mkdir(parents=True)
    monkeypatch.chdir(repository)
    config = _make_spool_config(tmp_path, agent_memory_dir=repository / "cache" / "memory")
    _assert_refused_without_spool_state(config, "repository checkout")


def test_spool_inside_a_provider_checkout_is_refused(tmp_path, spool_config):
    config = _make_spool_config(tmp_path, agent_memory_dir=tmp_path / "codex" / ".agent-loop" / "memory")
    _assert_refused_without_spool_state(config, "agent checkout")


def test_spool_inside_the_response_root_is_refused(tmp_path, spool_config):
    memory = response_root(spool_config) / "cache" / "memory"
    config = _make_spool_config(tmp_path, agent_memory_dir=memory)
    _assert_refused_without_spool_state(config, "response root")


def test_legacy_spool_root_overlapping_a_checkout_is_refused(tmp_path, spool_config):
    # The new root avoids the checkout, but the legacy root (beside memory)
    # would be the checkout itself.
    config = _make_spool_config(
        tmp_path, codex_dir=tmp_path / "review-round-spool" / "review-round-spool",
        agent_memory_dir=tmp_path / "review-round-spool" / "memory",
    )
    _assert_refused_without_spool_state(config, "agent checkout")


# --- Startup spool validation and the dotfiles home exemption (#1565) --------


def _dotfiles_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / ".git").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    return home


def test_validate_review_spool_placement_refuses_without_creating_anything(tmp_path, spool_config):
    config = _make_spool_config(tmp_path, agent_memory_dir=tmp_path / "codex" / ".agent-loop" / "memory")
    with pytest.raises(_SpoolError, match="overlaps") as info:
        review_rounds.validate_review_spool_placement(config)
    assert "agent checkout" in str(info.value)
    assert "invocation start" in str(info.value)
    private = config.agent_memory_dir.parent / "private"
    assert not private.exists()
    assert not review_spool_root(config.agent_memory_dir).exists()
    assert not legacy_review_spool_root(config.agent_memory_dir).exists()


def test_validate_review_spool_placement_accepts_without_creating_anything(spool_config):
    review_rounds.validate_review_spool_placement(spool_config)
    assert not (spool_config.agent_memory_dir.parent / "private").exists()
    assert not review_spool_root(spool_config.agent_memory_dir).exists()
    assert not legacy_review_spool_root(spool_config.agent_memory_dir).exists()


def test_round_start_check_uses_the_shared_placement_helper(tmp_path, spool_config, monkeypatch):
    calls = []
    original = review_rounds.validate_review_spool_placement

    def _spy(config):
        calls.append(config)
        return original(config)

    monkeypatch.setattr(review_rounds, "validate_review_spool_placement", _spy)
    config = _make_spool_config(tmp_path, agent_memory_dir=tmp_path / "codex" / ".agent-loop" / "memory")
    with pytest.raises(_SpoolError, match="overlaps"):
        _round_spool(config)
    assert calls == [config]


def test_dotfiles_home_accepts_the_default_spool(tmp_path, spool_config, monkeypatch):
    home = _dotfiles_home(tmp_path, monkeypatch)
    workdir = home / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    config = _make_spool_config(
        tmp_path, agent_memory_dir=home / ".cache" / "coding-review-agent-loop" / "memory"
    )
    labels = [label for label, _path in review_rounds._spool_overlap_set(config)]
    assert "repository checkout" not in labels
    assert "response root" in labels and "agent checkout" in labels
    review_rounds.validate_review_spool_placement(config)
    spool = _round_spool(config)
    assert spool.root == review_spool_root(config.agent_memory_dir)


def test_nested_repository_under_dotfiles_home_is_still_refused(tmp_path, spool_config, monkeypatch):
    home = _dotfiles_home(tmp_path, monkeypatch)
    project = home / "proj"
    (project / ".git").mkdir(parents=True)
    monkeypatch.chdir(project)
    config = _make_spool_config(tmp_path, agent_memory_dir=project / "cache" / "memory")
    with pytest.raises(_SpoolError, match="overlaps") as info:
        review_rounds.validate_review_spool_placement(config)
    assert "repository checkout" in str(info.value)
    _assert_refused_without_spool_state(config, "repository checkout")


def test_dotfiles_home_as_provider_checkout_is_still_refused(tmp_path, spool_config, monkeypatch):
    home = _dotfiles_home(tmp_path, monkeypatch)
    monkeypatch.chdir(home)
    config = _make_spool_config(
        tmp_path, codex_dir=home, agent_memory_dir=home / ".cache" / "agent-loop" / "memory"
    )
    with pytest.raises(_SpoolError, match="overlaps") as info:
        review_rounds.validate_review_spool_placement(config)
    assert "agent checkout" in str(info.value)


def _raise_home():
    raise RuntimeError("no home")


@pytest.mark.parametrize(
    "resolver",
    [lambda: "~", lambda: "/", _raise_home],
    ids=["unexpanded", "filesystem-root", "raises"],
)
def test_unresolvable_home_keeps_the_repository_member(tmp_path, spool_config, monkeypatch, resolver):
    home = _dotfiles_home(tmp_path, monkeypatch)
    monkeypatch.chdir(home)
    monkeypatch.setattr(review_rounds, "_resolved_home_directory", resolver)
    config = _make_spool_config(tmp_path, agent_memory_dir=home / ".cache" / "agent-loop" / "memory")
    with pytest.raises(_SpoolError, match="overlaps") as info:
        review_rounds.validate_review_spool_placement(config)
    assert "repository checkout" in str(info.value)
    assert not (config.agent_memory_dir.parent / "private").exists()


@pytest.mark.parametrize(
    "expanded",
    ["~", "/", RuntimeError("no home")],
    ids=["unexpanded", "filesystem-root", "raises"],
)
def test_unresolvable_expanduser_keeps_the_repository_member(tmp_path, spool_config, monkeypatch, expanded):
    home = _dotfiles_home(tmp_path, monkeypatch)
    monkeypatch.chdir(home)

    def _expanduser(path):
        if isinstance(expanded, Exception):
            raise expanded
        return expanded if path == "~" else path

    config = _make_spool_config(tmp_path, agent_memory_dir=home / ".cache" / "agent-loop" / "memory")
    monkeypatch.setattr(review_rounds.os.path, "expanduser", _expanduser)
    labels = [label for label, _path in review_rounds._spool_overlap_set(config)]
    assert "repository checkout" in labels
    with pytest.raises(_SpoolError, match="repository checkout"):
        review_rounds.validate_review_spool_placement(config)


def test_home_exemption_does_not_cover_ancestors_of_home(tmp_path, spool_config, monkeypatch):
    # A work tree above home (for example at the filesystem root) stays refused.
    outer = tmp_path / "outer"
    (outer / ".git").mkdir(parents=True)
    home = outer / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(home)
    config = _make_spool_config(tmp_path, agent_memory_dir=home / ".cache" / "agent-loop" / "memory")
    with pytest.raises(_SpoolError, match="repository checkout"):
        review_rounds.validate_review_spool_placement(config)


def test_partial_round_refusal_wording_with_and_without_an_unresolved_conversion_boundary():
    """The ordinary refusal is unchanged; an unresolved boundary drops the withheld-outcome diagnosis (#1676)."""
    recovery_calls = []

    def recovery():
        recovery_calls.append(True)
        raise RuntimeError("listing unavailable")

    ordinary = review_rounds._partial_round_refusal(
        surface="pr", number=77, round_number=3, reviewer_name="Codex",
        public_peers=["Antigravity", "Gemini"], cause="Some cause.",
    )
    assert str(ordinary) == (
        "Review round 3 on PR #77 is partially published (Antigravity, Gemini already posted) "
        "but Codex's withheld same-round outcome is missing or no longer validates, so "
        "invoking it now would let it read its peers' findings. Some cause."
    )
    unresolved = review_rounds._partial_round_refusal(
        surface="pr", number=77, round_number=3, reviewer_name="Codex",
        public_peers=["Antigravity"], cause="Ignored cause.", recovery=recovery,
        unresolved_conversion_boundary=True,
    )
    assert type(unresolved) is review_rounds.PartialReviewRoundError
    message = str(unresolved)
    first, listing = message.split("\n", 1)
    assert first == (
        "Review round 3 on PR #77 is partially published (Antigravity already posted), and "
        "this round's evidence-conversion record could not be admitted as a trusted boundary: "
        "either its identity could not be read or it is not a trusted, accepted conversion "
        "record. So the earlier pass's publications cannot be excluded from the peer set, and "
        "invoking Codex now could let it read them; the run stopped fail-closed. Retry once "
        "GitHub comment reads succeed, or inspect the conversion record if reads were healthy."
    )
    assert recovery_calls == [True] and "comment list is unavailable" in listing
    assert "withheld" not in message and "Ignored cause" not in message


def test_sequential_guard_threads_the_unresolved_conversion_boundary():
    spool = SimpleNamespace(surface="pr", number=77, round_number=3)
    with pytest.raises(review_rounds.PartialReviewRoundError) as refused:
        review_rounds._refuse_partial_round_before_sequential_turns(
            spool=spool, fresh_turn_reviewers=["codex"], public_peers=["antigravity"],
            unresolved_conversion_boundary=True,
        )
    assert "evidence-conversion record could not be admitted" in str(refused.value)
    assert "withheld" not in str(refused.value)


@pytest.mark.parametrize("mixed", ["code-finding", "same-pr", "unknown-disposition", "none"])
def test_evidence_approval_normalizer_preserves_real_blockers_and_requests(mixed):
    from coding_review_agent_loop.protocol import ParsedReview, ApprovedFollowups
    from coding_review_agent_loop.unresolved_items import _upsert_evidence_obligation
    items, _ = _upsert_evidence_obligation([], item_number=1, reviewer="Codex", text="Attach the live run.",
                                          source_round=1, current_head_sha="abc123")
    finding = ApprovedFollowup(reviewer="Codex", text="Fix an actual defect.")
    disposition = ReviewItemDisposition(item_id="item-2" if mixed == "unknown-disposition" else "item-1",
                                        reviewer="Codex", disposition="blocking", note="Still needed.")
    parsed = ParsedReview(state="blocking", summary="Review complete.",
                          blocking_items=(finding,) if mixed == "code-finding" else (),
                          followups=ApprovedFollowups(same_pr=(finding,) if mixed == "same-pr" else (), future=()),
                          dispositions=(disposition,), exact_head_evidence_requests=("Attach the second live run.",))
    normalized = review_rounds._normalize_evidence_only_approval(parsed, items)
    if mixed != "none":
        assert normalized is parsed
    else:
        assert normalized.state == "approved" and normalized.dispositions == ()
        assert normalized.exact_head_evidence_requests == parsed.exact_head_evidence_requests
        assert parsed.dispositions == (disposition,)
        assert items[0].status == "blocking" and items[0].lifecycle == "evidence_deferred"
