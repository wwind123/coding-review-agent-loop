"""Pure-move guards and patch-propagation shim for the orchestrator split (#1181, #1189)."""
import sys
import types
from unittest.mock import patch

import pytest

import orchestrator_split_guard as guard
import coding_review_agent_loop.orchestrator as orchestrator

BASELINE = guard.load_baseline()


def _registered_sources():
    return {
        name: guard.module_path(name).read_text(encoding="utf-8")
        for name in guard.registered_module_names()
    }


# --- Guards on the real package ---------------------------------------------


def test_baseline_surface_is_still_exposed_by_the_facade():
    assert guard.missing_surface_names(orchestrator, BASELINE["surface"]) == []


def test_baseline_definitions_exist_exactly_once_and_unchanged():
    facade_source = guard.module_path("orchestrator").read_text(encoding="utf-8")
    assert guard.definition_problems(BASELINE["definitions"], facade_source, _registered_sources()) == []


def test_registered_modules_bind_the_facades_objects():
    assert guard.incoherent_bindings(orchestrator, guard.registered_modules()) == []


def test_registered_modules_respect_layering():
    assert guard.layering_problems(guard.registered_module_names(), _registered_sources()) == []


def test_every_new_package_module_is_registered():
    problems = guard.unregistered_modules(
        guard.package_module_names(), BASELINE["modules"], guard.registered_module_names()
    )
    assert problems == []


def test_registered_modules_import_alone_in_a_fresh_interpreter():
    problems = [guard.fresh_import_problem(name) for name in guard.registered_module_names()]
    assert [problem for problem in problems if problem] == []


def test_orchestrator_facade_has_the_propagation_shim_installed():
    assert isinstance(orchestrator, guard.PatchPropagatingModule)


def test_facade_patch_reaches_a_registered_module_on_the_real_package(monkeypatch):
    import coding_review_agent_loop.github as github
    import coding_review_agent_loop.round_state as round_state

    monkeypatch.setattr(guard, "EXTRACTED_MODULES", ("github", "round_state"))
    original = github.merge_pr
    assert orchestrator.merge_pr is original
    assert "merge_pr" not in vars(round_state)

    def fake(*args, **kwargs):
        return None

    with patch("coding_review_agent_loop.orchestrator.merge_pr", fake):
        assert github.merge_pr is fake
        assert "merge_pr" not in vars(round_state)
    assert github.merge_pr is original
    assert orchestrator.merge_pr is original


def test_split_source_paths_follow_the_registry(monkeypatch):
    assert guard.split_source_paths() == [
        guard.module_path("orchestrator"),
        *(guard.module_path(name) for name in guard.registered_module_names()),
    ]
    monkeypatch.setattr(guard, "EXTRACTED_MODULES", ("usage",))
    assert guard.module_path("usage") in guard.split_source_paths()


def test_baseline_records_every_top_level_definition_of_the_base():
    names = {name for name, _ in guard.top_level_definitions(
        guard.module_path("orchestrator").read_text(encoding="utf-8")
    )}
    for name in guard.registered_module_names():
        names |= {n for n, _ in guard.top_level_definitions(guard.module_path(name).read_text(encoding="utf-8"))}
    assert set(BASELINE["definitions"]) <= names
    assert {"run_pr_loop", "run_issue_loop", "_run_validated_agent"} <= set(BASELINE["definitions"])
    assert "orchestrator" in BASELINE["modules"]


# --- Each guard fails on a synthetic violation -------------------------------

FACADE_SOURCE = '''"""Synthetic facade."""
import os

LIMIT = 3


@staticmethod
def decorated():
    return LIMIT


def helper():
    return os.sep


class Carrier:
    value = 1
'''


def _definitions(source):
    return dict(guard.top_level_definitions(source))


def test_surface_guard_names_a_dropped_attribute():
    facade = types.ModuleType("synthetic_facade")
    facade.kept = 1
    assert guard.missing_surface_names(facade, ["kept", "dropped"]) == [
        "orchestrator no longer exposes 'dropped'"
    ]


def test_definition_guard_accepts_a_pure_move():
    baseline = _definitions(FACADE_SOURCE)
    facade = 'import os\nfrom .moved import helper\n\nLIMIT = 3\n\n\n@staticmethod\ndef decorated():\n    return LIMIT\n\n\nclass Carrier:\n    value = 1\n'
    moved = "import os\n\n\ndef helper():\n    return os.sep\n"
    assert guard.definition_problems(baseline, facade, {"moved": moved}) == []


def test_definition_guard_covers_decorators():
    baseline = _definitions(FACADE_SOURCE)
    undecorated = FACADE_SOURCE.replace("@staticmethod\n", "")
    assert guard.definition_problems(baseline, undecorated, {}) == [
        "'decorated' in orchestrator differs from its baseline source"
    ]


def test_definition_guard_names_an_altered_body():
    baseline = _definitions(FACADE_SOURCE)
    facade = FACADE_SOURCE.replace("def helper():\n    return os.sep\n", "")
    moved = "import os\n\n\ndef helper():\n    return os.pathsep\n"
    assert guard.definition_problems(baseline, facade, {"moved": moved}) == [
        "'helper' in moved differs from its baseline source"
    ]


def test_definition_guard_names_a_copy_left_behind():
    baseline = _definitions(FACADE_SOURCE)
    moved = "import os\n\n\ndef helper():\n    return os.sep\n"
    problems = guard.definition_problems(baseline, FACADE_SOURCE, {"moved": moved})
    assert problems == ["'helper' is defined 2 times (in ['orchestrator', 'moved']); expected exactly once"]


def test_definition_guard_names_a_missing_definition():
    baseline = _definitions(FACADE_SOURCE)
    facade = FACADE_SOURCE.replace("LIMIT = 3\n", "")
    assert guard.definition_problems(baseline, facade, {}) == [
        "'LIMIT' is defined 0 times (in []); expected exactly once"
    ]


def test_definition_guard_names_a_new_name_in_an_extracted_module():
    baseline = _definitions(FACADE_SOURCE)
    moved = "NEW_CONSTANT = 1\n"
    assert guard.definition_problems(baseline, FACADE_SOURCE, {"moved": moved}) == [
        "moved defines 'NEW_CONSTANT', which is not a baseline orchestrator definition"
    ]


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("from .orchestrator import x\n", "early imports the orchestrator facade"),
        ("def f():\n    from . import orchestrator\n", "early imports the orchestrator facade"),
        ("import coding_review_agent_loop.orchestrator\n", "early imports the orchestrator facade"),
        ("from coding_review_agent_loop.orchestrator import x\n", "early imports the orchestrator facade"),
        ("def f():\n    from .later import y\n", "early imports later-registered later"),
    ],
)
def test_layering_guard_rejects_back_imports(source, expected):
    sources = {"early": source, "later": "from .early import x\nfrom .github import merge_pr\n"}
    assert guard.layering_problems(("early", "later"), sources) == [expected]


def test_layering_guard_accepts_earlier_and_pre_existing_modules():
    sources = {
        "early": "from .github import merge_pr\nfrom .agents.base import AgentResult\n",
        "later": "from .early import merge_pr\n",
    }
    assert guard.layering_problems(("early", "later"), sources) == []


def test_completeness_guard_names_an_unregistered_module():
    assert guard.unregistered_modules(["a", "b", "new_one", "registered"], ["a", "b"], ["registered"]) == [
        "new package module 'new_one' is not in EXTRACTED_MODULES"
    ]


def test_coherence_guard_names_divergent_and_unknown_bindings():
    facade = types.ModuleType("synthetic_facade")
    shared, divergent = object(), object()
    facade.shared = shared
    facade.divergent = object()
    module = types.ModuleType("synthetic_consumer")
    module.shared = shared
    module.divergent = divergent
    module.only_here = 1
    assert guard.incoherent_bindings(facade, [module]) == [
        "synthetic_consumer.divergent is not the facade's object",
        "synthetic_consumer.only_here is not bound on the orchestrator facade",
    ]


def test_fresh_import_guard_accepts_a_standalone_module_and_rejects_a_facade_importer():
    assert guard.fresh_import_problem("usage") is None
    assert guard.fresh_import_problem("cli") == "cli imports the orchestrator facade"
    assert "does not import alone" in guard.fresh_import_problem("no_such_module_for_split_guard")


# --- Patch propagation on synthetic modules ------------------------------------


@pytest.fixture
def synthetic(monkeypatch):
    """A facade, two consumers that bind ``target`` and one that never did."""
    original = lambda: "original"  # noqa: E731
    facade = types.ModuleType("synthetic_split_facade")
    facade.target = original
    owner_a = types.ModuleType("synthetic_split_owner_a")
    owner_a.target = original
    exec("def call():\n    return target()\n", vars(owner_a))
    owner_b = types.ModuleType("synthetic_split_owner_b")
    owner_b.target = original
    bystander = types.ModuleType("synthetic_split_bystander")
    outside = types.ModuleType("synthetic_split_outside")
    outside.target = original
    guard.install_patch_propagation(facade, registry=lambda: [owner_a, owner_b, bystander])
    monkeypatch.setitem(sys.modules, facade.__name__, facade)
    return types.SimpleNamespace(
        original=original, facade=facade, owners=(owner_a, owner_b),
        bystander=bystander, outside=outside,
    )


def _fake():
    return "fake"


def _assert_restored(s):
    assert s.facade.target is s.original
    for owner in s.owners:
        assert owner.target is s.original
    assert "target" not in vars(s.bystander)
    assert s.outside.target is s.original
    assert guard.pending_restores(s.facade) == {}


def test_monkeypatch_setattr_on_facade_reaches_owners_and_undo_restores(synthetic):
    s = synthetic
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(s.facade, "target", _fake)
        assert s.owners[0].call() == "fake"
        assert all(owner.target is _fake for owner in s.owners)
        assert "target" not in vars(s.bystander)
        assert s.outside.target is s.original
    _assert_restored(s)


def test_dotted_string_patch_reaches_owners_and_restores(synthetic):
    s = synthetic
    with patch("synthetic_split_facade.target", _fake):
        assert s.owners[0].call() == "fake"
        assert s.owners[1].target is _fake
        assert "target" not in vars(s.bystander)
    _assert_restored(s)


def test_patch_object_reaches_owners_and_restores(synthetic):
    s = synthetic
    with patch.object(s.facade, "target", _fake):
        assert s.owners[0].call() == "fake"
        assert s.owners[1].target is _fake
    _assert_restored(s)


def test_patch_of_a_name_no_registered_module_owns_stays_on_the_facade(synthetic):
    s = synthetic
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(s.facade, "brand_new", _fake, raising=False)
        assert s.facade.brand_new is _fake
        for module in (*s.owners, s.bystander):
            assert "brand_new" not in vars(module)
    assert "brand_new" not in vars(s.facade)
    _assert_restored(s)


def _assert_deleted(s):
    assert "target" not in vars(s.facade)
    for owner in s.owners:
        assert "target" not in vars(owner)
    assert "target" not in vars(s.bystander)
    assert s.outside.target is s.original
    assert guard.pending_restores(s.facade) == {"target": {owner.__name__ for owner in s.owners}}


def _assert_later_patch_propagates_and_restores(s):
    with patch.object(s.facade, "target", _fake):
        assert all(owner.target is _fake for owner in s.owners)
        assert "target" not in vars(s.bystander)
    _assert_restored(s)


def test_monkeypatch_delattr_then_undo_restores_every_owner(synthetic):
    s = synthetic
    with pytest.MonkeyPatch.context() as mp:
        mp.delattr(s.facade, "target")
        _assert_deleted(s)
    _assert_restored(s)
    _assert_later_patch_propagates_and_restores(s)


def test_plain_delattr_then_setattr_restores_every_owner(synthetic):
    s = synthetic
    delattr(s.facade, "target")
    _assert_deleted(s)
    setattr(s.facade, "target", s.original)
    _assert_restored(s)
    _assert_later_patch_propagates_and_restores(s)


def test_delete_of_a_missing_name_touches_no_owner(synthetic):
    s = synthetic
    with pytest.raises(AttributeError):
        delattr(s.facade, "absent")
    assert guard.pending_restores(s.facade) == {}
    _assert_restored(s)


def test_install_is_idempotent(synthetic):
    s = synthetic
    guard.install_patch_propagation(s.facade, registry=lambda: [])
    with patch.object(s.facade, "target", _fake):
        assert s.owners[0].target is _fake
    _assert_restored(s)
