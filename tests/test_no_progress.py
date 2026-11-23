"""The single record-derived no-progress detector (#2097, stage 1 of #2067)."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from coding_review_agent_loop import no_progress as np
from coding_review_agent_loop import review_step_back as sb
from coding_review_agent_loop.errors import parse_human_decision
from coding_review_agent_loop.no_progress import (
    NO_PROGRESS_PREFIX_TEXT,
    NoProgressObservation as Obs,
    blocking_item_ids,
    defers_to_evidence_stall,
    no_progress_decision,
    pr_no_progress_observations,
    render_no_progress_item_lines,
    render_no_progress_message,
)
from coding_review_agent_loop.protocol import UnresolvedReviewItem
from coding_review_agent_loop.round_closure import (
    ROUND_CLOSED_INCOMPLETE_PHASE,
    build_round_closure_payload,
)
from coding_review_agent_loop.round_metadata import PostedRoundMetadata, PostedRoundRecord
from coding_review_agent_loop.signed_input_record import SIGNED_INPUT_SUPERSESSION_PHASE

H = "a" * 40
H2 = "b" * 40
IDS = frozenset({"item-1", "item-2"})
SRC = Path(__file__).resolve().parents[1] / "src" / "coding_review_agent_loop"


# --- the pure decision --------------------------------------------------------


def _window(*subjects_and_ids, start=1):
    return [
        Obs(start + offset, subject, frozenset(ids))
        for offset, (subject, ids) in enumerate(subjects_and_ids)
    ]


def test_k_identical_completed_rounds_matching_the_live_candidate_stop():
    """`pr-k-unchanged` (unit)."""
    decision = no_progress_decision(
        _window((H, IDS), (H, IDS), (H, IDS)), live_subject=H, live_blocking_ids=IDS, k=3,
    )
    assert decision == np.NoProgressDecision(H, IDS, (1, 2, 3), 3)
    assert no_progress_decision(
        _window((H, IDS), (H, IDS), (H, IDS)), live_subject=H, live_blocking_ids=IDS, k=3,
        through_round=3,
    ) is not None


@pytest.mark.parametrize(
    "case",
    ["ids-gain", "ids-lose", "subject-change", "empty", "too-few", "gap", "through-round"],
)
def test_any_change_inside_the_window_means_no_stop(case):
    """`pr-ids-change`: an ID gained or lost on the same head never stops."""
    window = _window((H, IDS), (H, IDS), (H, IDS))
    live_ids = IDS
    through = None
    if case == "ids-gain":
        window[1] = Obs(2, H, IDS | {"item-3"})
    elif case == "ids-lose":
        window[2] = Obs(3, H, frozenset({"item-1"}))
        live_ids = frozenset({"item-1"})
    elif case == "subject-change":
        window[0] = Obs(1, H2, IDS)
    elif case == "empty":
        window = _window((H, ()), (H, ()), (H, ()))
        live_ids = frozenset()
    elif case == "too-few":
        window = window[:2]
    elif case == "gap":
        window[2] = Obs(4, H, IDS)
    elif case == "through-round":
        through = 4  # round 4 completed but is not an observation
    assert no_progress_decision(
        window, live_subject=H, live_blocking_ids=live_ids, k=3, through_round=through,
    ) is None


@pytest.mark.parametrize(
    "live_subject,live_ids",
    [(H2, IDS), (H, frozenset({"item-1"})), (None, IDS)],
    ids=["moved-head", "changed-ids", "unknown-head"],
)
def test_a_live_candidate_that_differs_from_the_window_never_stops(live_subject, live_ids):
    """`pr-head-moves` (unit): the unreviewed live candidate is compared, never counted."""
    window = _window((H, IDS), (H, IDS), (H, IDS))
    assert no_progress_decision(
        window, live_subject=live_subject, live_blocking_ids=live_ids, k=3,
    ) is None
    # The live candidate is not appended: K-1 rounds plus a matching live
    # candidate are still too few.
    assert no_progress_decision(
        window[:2], live_subject=H, live_blocking_ids=IDS, k=3,
    ) is None


def test_k_one_stops_on_one_completed_round_and_k_below_one_never_does():
    window = _window((H, IDS))
    assert no_progress_decision(window, live_subject=H, live_blocking_ids=IDS, k=1) is not None
    for k in (0, -1, True):
        assert no_progress_decision(window, live_subject=H, live_blocking_ids=IDS, k=k) is None


def test_only_the_last_k_observations_count():
    window = _window((H2, IDS), (H, IDS), (H, IDS), (H, IDS))
    decision = no_progress_decision(window, live_subject=H, live_blocking_ids=IDS, k=3)
    assert decision is not None and decision.rounds == (2, 3, 4)


# --- blocking semantics -------------------------------------------------------


def _item(item_id, status="blocking", **changes):
    return UnresolvedReviewItem(
        item_id=item_id, reviewer="Codex", source_round=1, text=f"text of {item_id}",
        status=status, **changes,
    )


def test_blocking_ids_exclude_future_inert_and_machine_items():
    items = [
        _item("item-1"),
        _item("item-2", status="same-pr"),
        _item("item-3", status="future"),
        _item("item-4", status="resolved"),
        _item("item-5", status="converted", evidence_row_ids=("row-a",), converted_row_ids=("row-a",)),
        _item(
            "item-6", authority="machine", obligation_kind="human-exact-head-evidence",
            lifecycle="evidence_frozen", candidate_head_sha=H, obligation_identity="request-6",
        ),
    ]
    assert blocking_item_ids(items) == frozenset({"item-1", "item-2"})


def test_evidence_deferral_requires_every_item_tagged_with_unsatisfied_rows():
    """`pr-evidence-owned` (unit)."""
    rows = frozenset({"row-a", "row-b"})
    assert defers_to_evidence_stall({"item-1": ("row-a",), "item-2": ("row-b",)}, rows)
    assert not defers_to_evidence_stall({"item-1": ("row-a",), "item-2": ()}, rows)
    assert not defers_to_evidence_stall({"item-1": ("row-c",)}, rows)
    assert not defers_to_evidence_stall({"item-1": ("row-a",)}, None)
    assert not defers_to_evidence_stall({}, rows)


# --- observation derivation from durable records -------------------------------


def _review(index, round_number, subject, *, agent="Codex"):
    return PostedRoundRecord(
        index=index,
        metadata=PostedRoundMetadata(
            flow="pr", role="reviewer", agent=agent, round_number=round_number,
            subject=subject, state="blocking",
        ),
        body="",
    )


def _coder(index, round_number, subject, *, dispatch=H, items=None, **changes):
    return PostedRoundRecord(
        index=index,
        metadata=PostedRoundMetadata(
            flow="pr", role="coder", agent="Claude", round_number=round_number,
            subject=subject, followup_dispatch_head=dispatch,
            prior_items=tuple(items if items is not None else (_item("item-1"), _item("item-2"))),
            **changes,
        ),
        body="",
    )


def _history(rounds=3, *, dispatch=H):
    """Sequential default mode: review N, then the coder record numbered N+1."""
    records = []
    for n in range(1, rounds + 1):
        records.append(_review(10 * n, n, H))
        records.append(_coder(10 * n + 5, n + 1, H, dispatch=dispatch))
    return records


def test_observations_come_from_the_coder_record_after_each_round():
    """`pr-sequential-default-observations` (unit): no reconciliation summary needed."""
    observations = pr_no_progress_observations(_history())
    assert observations == tuple(Obs(n, H, IDS) for n in (1, 2, 3))
    assert no_progress_decision(
        observations, live_subject=H, live_blocking_ids=IDS, k=3, through_round=3,
    ) is not None


def test_historical_records_without_a_dispatch_head_fall_back_to_the_reviewed_head():
    """`pr-sequential-default-observations`: pre-#1034 coder records."""
    observations = pr_no_progress_observations(_history(dispatch=None))
    assert [o.subject for o in observations] == [H, H, H]
    ambiguous = [*_history(1, dispatch=None), _review(12, 1, H2, agent="Gemini")]
    ambiguous.sort(key=lambda record: record.index)
    assert pr_no_progress_observations(ambiguous) == ()


def test_the_dispatch_site_adds_the_just_settled_round_from_memory():
    records = _history(2)
    current = Obs(3, H, IDS)
    observations = pr_no_progress_observations(records, current=current)
    assert observations[-1] == current and len(observations) == 3


def _closed_incomplete(index, round_number):
    payload = build_round_closure_payload(
        flow="pr", round_number=round_number, subject=H, published=[("Codex", None)],
        unreplayable=[("Codex", "published review could not be replayed")],
        item_number_high_water=2,
    )
    return PostedRoundRecord(
        index=index,
        metadata=PostedRoundMetadata(
            flow="pr", role="summary", agent="Orchestrator", round_number=round_number,
            subject=H, phase=ROUND_CLOSED_INCOMPLETE_PHASE, round_closure=payload,
        ),
        body="",
    )


def _unvalidated_closure(index, round_number):
    return PostedRoundRecord(
        index=index,
        metadata=PostedRoundMetadata(
            flow="pr", role="summary", agent="Orchestrator", round_number=round_number,
            subject=H, phase=ROUND_CLOSED_INCOMPLETE_PHASE,
        ),
        body="",
    )


def _legacy_supersession(index, round_number):
    return PostedRoundRecord(
        index=index,
        metadata=PostedRoundMetadata(
            flow="pr", role="summary", agent="Orchestrator", round_number=round_number,
            subject=H, phase=SIGNED_INPUT_SUPERSESSION_PHASE,
            signed_input_supersession={
                "round": round_number, "head": H,
                "superseded": [{"agent": "Codex", "comment_id": None, "route": "re-review"}],
                "previous_requirement_ids": [], "requirement_ids": [],
                "item_number_high_water": 2,
            },
        ),
        body="",
    )


@pytest.mark.parametrize(
    "breaker",
    ["closed-incomplete", "unvalidated-closure", "legacy-supersession", "no-coder-record",
     "dispatch-head-mismatch", "no-review", "non-contiguous"],
)
def test_an_incomplete_closed_or_unmatched_round_breaks_the_window(breaker):
    """`pr-incomplete-round-breaks-window` (unit), never raising on a missing record."""
    records = _history(3)
    if breaker == "closed-incomplete":
        records.append(_closed_incomplete(32, 3))
    elif breaker == "unvalidated-closure":
        records.append(_unvalidated_closure(32, 3))
    elif breaker == "legacy-supersession":
        records.append(_legacy_supersession(32, 3))
    elif breaker == "no-coder-record":
        records = [r for r in records if not (r.metadata.role == "coder" and r.metadata.round_number == 4)]
    elif breaker == "dispatch-head-mismatch":
        records = [
            _coder(r.index, 4, H, dispatch=H2)
            if r.metadata.role == "coder" and r.metadata.round_number == 4 else r
            for r in records
        ]
    elif breaker == "no-review":
        records = [r for r in records if not (r.metadata.role == "reviewer" and r.metadata.round_number == 3)]
    elif breaker == "non-contiguous":
        records = [
            r for r in records if r.metadata.round_number not in {2, 3} or r.metadata.role == "coder"
        ] + [_review(30, 4, H), _coder(35, 5, H)]
        records = [r for r in records if not (r.metadata.role == "coder" and r.metadata.round_number in {3, 4})]
    records.sort(key=lambda record: record.index)
    observations = pr_no_progress_observations(records)
    through = 4 if breaker == "non-contiguous" else 3
    assert no_progress_decision(
        observations, live_subject=H, live_blocking_ids=IDS, k=3, through_round=through,
    ) is None


def test_inert_reviewer_records_of_a_closed_round_never_count_as_its_review():
    records = sorted(
        [*_history(3), _legacy_supersession(32, 3)], key=lambda record: record.index
    )
    # The coder record numbered 4 follows the closed round, but round 3's only
    # review is inert, so round 3 is no observation and the window breaks.
    assert 3 not in {o.round_number for o in pr_no_progress_observations(records)}


def test_a_replacement_review_after_a_closed_round_must_complete_before_it_counts():
    records = [*_history(2), _review(30, 3, H), _closed_incomplete(32, 3)]
    observations = pr_no_progress_observations(records)
    assert [o.round_number for o in observations] == [1, 2]
    # The replacement round 4 completes and the window refills only from it.
    records += [_review(40, 4, H), _coder(45, 5, H), _review(50, 5, H), _coder(55, 6, H),
                _review(60, 6, H), _coder(65, 7, H)]
    decision = no_progress_decision(
        pr_no_progress_observations(records), live_subject=H, live_blocking_ids=IDS, k=3,
        through_round=6,
    )
    assert decision is not None and decision.rounds == (4, 5, 6)


# --- legacy step-back fields are inert -------------------------------------------


_LEGACY_ENTRY = {
    "phase": "pr", "reviewer": "Codex", "trigger_round": 2, "trigger_head": H,
    "anchor": {"path": "src/a.py", "start": 1, "end": 5},
}


def _with_legacy(records, variant):
    if variant == "absent":
        return records
    out = []
    for record in records:
        if record.metadata.role == "coder" and record.metadata.round_number == 3:
            if variant == "valid":
                record = _coder(record.index, 3, H, step_back_entries=(_LEGACY_ENTRY,))
            else:
                record = _coder(record.index, 3, H, step_back_status="invalid")
        out.append(record)
    return out


def test_legacy_entries_absent_valid_or_malformed_yield_identical_decisions():
    """`pr-resume-legacy` (unit): observations, decisions and derived triggers agree."""
    reference = None
    for variant in ("absent", "valid", "invalid"):
        records = _with_legacy(_history(3), variant)
        observations = pr_no_progress_observations(records)
        decision = no_progress_decision(
            observations, live_subject=H, live_blocking_ids=IDS, k=3, through_round=3,
        )
        triggers = sb.derived_pr_trigger_rounds(
            records, "Codex", k=3, window=40,
            mapper=lambda f, t, p, s, e: sb.AnchorMapping(sb.OUTCOME_SHIFTED, p, p, s, e),
        )
        outcome = (observations, decision, sorted(triggers))
        assert reference is None or outcome == reference, variant
        reference = outcome


_DECISION_SOURCES = ("pr_loop.py", "pr_loop_support.py", "no_progress.py")
_FORBIDDEN_ATTRIBUTES = frozenset({"step_back_entries", "step_back_status"})
_FORBIDDEN_HELPERS = frozenset(
    {"pr_step_back_history", "pr_sweep_entries", "pr_step_back_approach_alternative"}
)


def _legacy_decision_reads(source: str) -> list[str]:
    """Attribute loads of legacy step-back fields and uses of the entry readers.

    A ``step_back_entries=`` keyword in a record constructor is an
    ``ast.keyword``, not an attribute load, so the audit writer is allowed.
    """
    found = []
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.ctx, ast.Load)
            and node.attr in _FORBIDDEN_ATTRIBUTES | _FORBIDDEN_HELPERS
        ):
            found.append(f"{node.lineno}:{node.attr}")
        elif isinstance(node, ast.Name) and node.id in _FORBIDDEN_HELPERS:
            found.append(f"{node.lineno}:{node.id}")
        elif isinstance(node, ast.ImportFrom):
            found.extend(
                f"{node.lineno}:{alias.name}" for alias in node.names
                if alias.name in _FORBIDDEN_HELPERS
            )
    return found


@pytest.mark.parametrize("name", _DECISION_SOURCES)
def test_no_pr_decision_path_reads_legacy_step_back_fields(name):
    """`pr-resume-legacy` (static guard): decision reads are forbidden."""
    assert _legacy_decision_reads((SRC / name).read_text(encoding="utf-8")) == []


def test_the_static_guard_catches_a_planted_read_and_allows_the_audit_writer():
    planted = "def decide(metadata):\n    return metadata.step_back_entries\n"
    assert _legacy_decision_reads(planted) == ["2:step_back_entries"]
    helper = "from .review_step_back import pr_sweep_entries\nx = _sb.pr_step_back_history(r, 'a')\n"
    assert len(_legacy_decision_reads(helper)) == 2
    writer = "record = Metadata(step_back_entries=entries)\n"
    assert _legacy_decision_reads(writer) == []
    # The real loop still writes the audit record history export counts.
    loop = (SRC / "pr_loop.py").read_text(encoding="utf-8")
    assert "step_back_entries=step_back_entries" in loop


# --- the approach-alternative excerpt ---------------------------------------------


def _proposal(index, round_number, subject, summary):
    return PostedRoundRecord(
        index=index,
        metadata=PostedRoundMetadata(
            flow="pr", role="coder", agent="Claude", round_number=round_number,
            subject=subject, raw_structured_coder_response=json.dumps({"summary": summary}),
        ),
        body="",
    )


def test_the_excerpt_is_the_newest_proposal_at_the_head_even_after_later_reviews():
    """`pr-approach-alternative-no-stop` (unit)."""
    records = [
        _proposal(10, 4, H2, "Approach alternative: drop the guard."),
        _review(20, 4, H2), _coder(25, 5, H2, dispatch=H2), _review(30, 5, H2),
    ]
    assert sb.pr_approach_alternative_at_head(records, head_sha=H2) == (4, "drop the guard.")
    # No section when no coder record at the head carries the prefix.
    assert sb.pr_approach_alternative_at_head(records, head_sha=H) is None
    plain = [_proposal(10, 4, H2, "Generalization: one rule.")]
    assert sb.pr_approach_alternative_at_head(plain, head_sha=H2) is None


# --- the single message -----------------------------------------------------------


def _message(**sections):
    return render_no_progress_message(
        lead="PR #77: ", subject_label="head", subject=H, k=3, rounds=(2, 3, 4),
        item_lines=render_no_progress_item_lines(IDS, {"item-1": "first\nsecond", "item-2": ""}),
        **sections,
    )


def test_the_message_names_the_items_head_and_k_in_three_parts():
    message = _message()
    assert message.startswith(
        f"PR #77: {NO_PROGRESS_PREFIX_TEXT}3 consecutive completed round(s): rounds 2-4 "
        f"ended with the same head `{H}` and the same blocking items (--no-progress-rounds 3)."
    )
    assert "- [item-1] first\n- [item-2] (no text recorded)" in message
    parts = parse_human_decision(message)
    assert parts is not None and parts.safe_to_automate is False
    assert "--max-rounds" in parts.what and "signed-requirement path" in parts.what


def test_item_lines_are_sorted_bounded_and_truncated():
    ids = [f"item-{n:02d}" for n in range(25)]
    lines = render_no_progress_item_lines(ids, {"item-00": "x" * 500 + "`tick`"})
    assert len(lines) == 21 and lines[-1] == "- and 5 more"
    assert lines[0].startswith("- [item-00] ") and len(lines[0]) < 270 and lines[0].endswith("…")
    assert lines[1] == "- [item-01] (no text recorded)"


@pytest.mark.parametrize(
    "section,present",
    [
        ({"route": "If the blocking finding requires re-planning the child plan, rerun."},
         "If the blocking finding requires re-planning the child plan, rerun."),
        ({"evidence": " The latest local test evidence still holds receipts."},
         "Latest coder turn evidence: The latest local test evidence still holds receipts."),
        ({"citation_diagnosis": np.citation_diagnosis_sentence(SimpleNamespace(
            non_array_drop=True, repair_attempted=False, uncited_receipts=2))},
         "(same-agent re-asks attempted: no); 2 run-tests receipt(s)"),
        ({"approach_alternative": (4, "drop the guard.")},
         "Step-back approach alternative proposed at round 4 (excerpt): drop the guard."),
        ({"pr_size": "3 files changed, 9 insertions(+)"},
         "Total PR size (a scope-growth signal; compare it with the issue's suggested "
         "direction): 3 files changed, 9 insertions(+)"),
    ],
    ids=["route", "evidence", "citation", "approach-alternative", "pr-size"],
)
def test_each_optional_section_appears_only_when_present(section, present):
    """`pr-planning-child-route`, `pr-size-diagnostic` (unit)."""
    assert present in _message(**section)
    assert present not in _message()


def test_absent_sections_leave_no_placeholder():
    message = _message(route="", evidence="", citation_diagnosis="", pr_size=None)
    for fragment in ("unavailable", "Latest coder turn evidence", "Total PR size",
                     "approach alternative", "child plan", "Diff growth"):
        assert fragment not in message
    assert np.citation_diagnosis_sentence(None) == ""
    assert np.citation_diagnosis_sentence(SimpleNamespace(non_array_drop=False)) == ""


# --- PR glue: size measurement and the error the loop raises ----------------------


class _GitRunner:
    def __init__(self, *, merge_base="e" * 40, shortstat=" 2 files changed, 5 insertions(+)\n",
                 returncode=0, error=None):
        self.merge_base, self.shortstat = merge_base, shortstat
        self.returncode, self.error, self.calls = returncode, error, []

    def run(self, args, *, cwd, check=False, **_kwargs):
        self.calls.append(list(args))
        if self.error is not None:
            raise self.error
        stdout = f"{self.merge_base}\n" if args[1] == "merge-base" else self.shortstat
        return SimpleNamespace(returncode=self.returncode, stdout=stdout)


def _support_config(tmp_path):
    from agent_loop_helpers import make_config

    return make_config(tmp_path, reviewer=("codex",))


@pytest.mark.parametrize(
    "runner_kwargs,expected",
    [
        ({}, "2 files changed, 5 insertions(+)"),
        ({"shortstat": "\n"}, "no change"),
        ({"returncode": 1}, None),
        ({"shortstat": "garbage\n"}, None),
        ({"merge_base": "not-a-sha"}, None),
        ({"error": OSError("git missing")}, None),
    ],
    ids=["measured", "empty", "non-zero", "unparseable", "bad-merge-base", "oserror"],
)
def test_the_pr_size_measurement_never_raises(tmp_path, runner_kwargs, expected):
    """`pr-size-diagnostic` (unit)."""
    from coding_review_agent_loop.pr_loop_support import _pr_size_measurement

    config = _support_config(tmp_path)
    runner = _GitRunner(**runner_kwargs)
    assert _pr_size_measurement(runner, config, current_head=H, base_branch="main") == expected
    assert _pr_size_measurement(runner, config, current_head=H, base_branch=None) is None


@pytest.mark.parametrize("route", ["", "If the blocking finding requires re-planning the child plan, post it."])
def test_the_pr_error_carries_the_route_only_when_bound(tmp_path, route):
    """`pr-planning-child-route` (unit): no route text without a binding."""
    from coding_review_agent_loop.errors import HumanDecisionRequiredError
    from coding_review_agent_loop.pr_loop_support import _pr_no_progress_error

    decision = no_progress_decision(
        pr_no_progress_observations(_history(3)), live_subject=H, live_blocking_ids=IDS, k=3,
    )
    error = _pr_no_progress_error(
        _GitRunner(error=OSError("no git")), _support_config(tmp_path), pr_number=77,
        decision=decision, records=_history(3), items=[_item("item-1"), _item("item-2")],
        route=route, base_branch="main",
    )
    assert isinstance(error, HumanDecisionRequiredError) and error.EXIT_CODE == 4
    message = str(error)
    assert "- [item-1] text of item-1" in message and "Total PR size" not in message
    assert ("child plan" in message) == bool(route)


@pytest.mark.parametrize("closure", ["closed-incomplete", "legacy-supersession"])
def test_the_dispatch_round_never_counts_when_its_number_is_closed(closure):
    """A replacement review that reuses a closed round number counts neither
    live (as ``current``) nor durably, so both sites agree."""
    record = _closed_incomplete(32, 3) if closure == "closed-incomplete" else _legacy_supersession(32, 3)
    records = sorted([*_history(2), _review(30, 3, H), record], key=lambda r: r.index)
    observations = pr_no_progress_observations(records, current=Obs(3, H, IDS))
    assert [o.round_number for o in observations] == [1, 2]
    assert no_progress_decision(
        observations, live_subject=H, live_blocking_ids=IDS, k=3, through_round=3,
    ) is None


# --- the citation diagnosis comes from the durable coder record (#1626) ---------


def _bounded(*receipts):
    from coding_review_agent_loop.local_test_evidence import (
        EnvironmentIdentityRegistry, EvidenceScope, LocalTestObservation, TreeAttribution,
        bounded_evidence_for_round, reconcile_test_observations,
    )

    registry = EnvironmentIdentityRegistry()
    rows = [
        LocalTestObservation(
            command=("python3", "-m", "pytest", "tests/test_x.py"), outcome="failed",
            provenance="parent-observed", scope=EvidenceScope("unknown", ()),
            receipt_id=receipt, turn_id=turn, cwd="/checkout", returncode=4,
            attribution=TreeAttribution(state="current-head", head=H, stable=True),
            environment_identity=registry.capture({"PATH": "/usr/bin"}),
        )
        for receipt, turn in receipts
    ]
    return bounded_evidence_for_round(reconcile_test_observations(rows, registry=registry))


def _degradation(rule):
    from coding_review_agent_loop.protocol_common import ParseDegradation
    from coding_review_agent_loop.protocol_risk import CITATION_DROPPED_OUTCOME

    return ParseDegradation.build(
        element_path="coder_followup.test_observations", rule=rule,
        observed="not an array", outcome=CITATION_DROPPED_OUTCOME,
    )


@pytest.mark.parametrize("after_repair", [False, True])
@pytest.mark.parametrize("persisted", [0, 3, None], ids=["zero", "three", "historical"])
def test_the_citation_diagnosis_is_read_back_from_the_newest_coder_record(after_repair, persisted):
    """`pr-planning-child-route` (unit): live and resumed stops read the same record.

    The count is the persisted tool-owned value, never inferred from the
    bounded evidence (which may hold other turns' rows or be evicted).
    """
    from coding_review_agent_loop.pr_loop_support import durable_citation_diagnosis
    from coding_review_agent_loop.protocol_risk import (
        TEST_OBSERVATIONS_NOT_ARRAY_AFTER_REPAIR_RULE,
        TEST_OBSERVATIONS_NOT_ARRAY_RULE,
    )

    rule = TEST_OBSERVATIONS_NOT_ARRAY_AFTER_REPAIR_RULE if after_repair else TEST_OBSERVATIONS_NOT_ARRAY_RULE
    earlier = _coder(15, 2, H, local_test_evidence=_bounded(("old-1", "turn-1")))
    latest = _coder(
        25, 3, H, test_observation_degradations=(_degradation(rule),),
        local_test_evidence=_bounded(("old-1", "turn-1"), ("new-1", "turn-2"), ("repair-1", "turn-3")),
        uncited_acquisition_receipts=persisted,
    )
    diagnosis = durable_citation_diagnosis([earlier, latest])
    assert diagnosis is not None and diagnosis.non_array_drop
    assert diagnosis.repair_attempted is after_repair
    assert diagnosis.uncited_receipts == persisted
    sentence = np.citation_diagnosis_sentence(diagnosis)
    if persisted is None:
        assert "receipts went uncited (count not recorded)" in sentence
    else:
        assert f"; {persisted} run-tests receipt(s) from that turn went uncited" in sentence
    assert durable_citation_diagnosis([earlier]) is None
    assert durable_citation_diagnosis([]) is None


def _followup_with(observations):
    from coding_review_agent_loop.protocol import validate_structured_coder_followup

    payload = {
        "schema_version": 1, "kind": "coder_followup", "state": "blocking", "summary": "Re-ran.",
        "addressed_items": [], "remaining_items": [],
        "human_requirements": {"addressed_ids": [], "checked_discussion_directly": False},
        "human_requirement_dispositions": [],
        "test_observations": observations,
    }
    return validate_structured_coder_followup(
        json.dumps(payload) + "\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        test_observations_mode="degrade-after-repair",
    )


@pytest.mark.parametrize(
    "catalog,expected",
    [((), 0), (("a",), 1), (tuple(f"r-{n}" for n in range(40)), 40)],
    ids=["empty-acquisition", "one", "beyond-evidence-cap"],
)
def test_the_persisted_count_comes_from_the_frozen_acquisition_catalog(catalog, expected):
    """Only the acquisition's own catalog counts: a repair turn's receipt is not in
    it, and the bounded evidence cap never trims it."""
    from coding_review_agent_loop.pr_loop_support import acquisition_uncited_receipts

    dropped = _followup_with("pytest passed locally")
    assert acquisition_uncited_receipts(dropped, catalog) == expected
    # A well-formed (empty) citation list records no drop, so no count is persisted.
    assert acquisition_uncited_receipts(_followup_with([]), catalog) is None
    assert acquisition_uncited_receipts(object(), catalog) is None


def test_the_persisted_count_round_trips_and_is_omitted_when_absent():
    from coding_review_agent_loop.round_metadata import (
        _decode_round_metadata, _encode_round_metadata,
    )

    with_count = _coder(1, 2, H, uncited_acquisition_receipts=40).metadata
    without = _coder(1, 2, H).metadata
    assert _decode_round_metadata(_encode_round_metadata(with_count)).uncited_acquisition_receipts == 40
    assert _decode_round_metadata(_encode_round_metadata(without)).uncited_acquisition_receipts is None
    # Historical records encode unchanged: the field is omitted when None.
    from coding_review_agent_loop.round_transport import decode_mapping

    assert "uncited_acquisition_receipts" not in decode_mapping(_encode_round_metadata(without))
    assert decode_mapping(_encode_round_metadata(with_count))["uncited_acquisition_receipts"] == 40
