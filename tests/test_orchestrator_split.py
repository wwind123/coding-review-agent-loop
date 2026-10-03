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


def test_orchestrator_facade_is_only_a_docstring_and_imports():
    facade_source = guard.module_path("orchestrator").read_text(encoding="utf-8")
    assert guard.thin_facade_problems(facade_source) == []


def test_baseline_no_longer_freezes_definition_digests():
    # The per-definition digest freeze was effort-scoped (#1181) and retired
    # in #1204 so the extracted modules can evolve; surface and module list stay.
    assert set(BASELINE) == {"modules", "surface"}
    assert "orchestrator" in BASELINE["modules"]
    assert {"run_pr_loop", "run_issue_loop", "_run_validated_agent"} <= set(BASELINE["surface"])


def test_registered_modules_bind_the_facades_objects():
    # Checked in a fresh interpreter: in this process the shim and the autouse
    # repair stub would overwrite a divergent binding before it is observed.
    registered = [f"{guard.PACKAGE}.{name}" for name in guard.registered_module_names()]
    assert guard.fresh_incoherent_bindings(registered) == []


def test_coherence_guard_rejects_divergent_binding_hidden_by_the_repair_stub(tmp_path, monkeypatch):
    (tmp_path / "split_probe_consumer.py").write_text(
        "def attempt_repair(*args, **kwargs):\n    return None\n", encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    import split_probe_consumer

    monkeypatch.setattr(guard, "registered_modules", lambda: [split_probe_consumer])
    monkeypatch.setattr(orchestrator, "attempt_repair", lambda *args, **kwargs: None)
    # Under a propagated patch the divergent binding looks coherent in-process...
    assert guard.incoherent_bindings(orchestrator, [split_probe_consumer]) == []
    # ...but the pristine fresh-interpreter check still rejects it.
    assert guard.fresh_incoherent_bindings(["split_probe_consumer"], extra_path=tmp_path) == [
        "split_probe_consumer.attempt_repair is not the facade's object"
    ]


def test_fresh_coherence_check_accepts_identical_bindings(tmp_path):
    (tmp_path / "split_probe_coherent.py").write_text(
        "from coding_review_agent_loop.orchestrator import attempt_repair\n", encoding="utf-8"
    )
    assert guard.fresh_incoherent_bindings(["split_probe_coherent"], extra_path=tmp_path) == []


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


def test_autouse_repair_stub_reaches_the_moved_validated_agent():
    # tests/conftest.py stubs attempt_repair on the facade for every test; the
    # moved repair path in validated_agent must see that stub, never the real
    # repair CLI, and a nested facade patch must restore the stub on exit.
    import coding_review_agent_loop.repair as repair
    import coding_review_agent_loop.validated_agent as validated_agent

    stub = orchestrator.attempt_repair
    assert stub is not repair.attempt_repair
    assert validated_agent.attempt_repair is stub
    assert validated_agent._ORIGINAL_ATTEMPT_REPAIR is repair.attempt_repair

    def fake(*args, **kwargs):
        return None

    with patch("coding_review_agent_loop.orchestrator.attempt_repair", fake):
        assert validated_agent.attempt_repair is fake
    assert validated_agent.attempt_repair is stub
    assert orchestrator.attempt_repair is stub


def test_split_source_paths_follow_the_registry(monkeypatch):
    assert guard.split_source_paths() == [
        guard.module_path("orchestrator"),
        *(guard.module_path(name) for name in guard.registered_module_names()),
    ]
    monkeypatch.setattr(guard, "EXTRACTED_MODULES", ("usage",))
    assert guard.module_path("usage") in guard.split_source_paths()


def test_cli_entry_points_are_the_moved_definitions():
    # cli.py still imports its loop entry points from the facade; each must be
    # the identical function object defined in the module that now owns it.
    import coding_review_agent_loop.checks as checks
    import coding_review_agent_loop.cli as cli
    import coding_review_agent_loop.discuss_loop as discuss_loop
    import coding_review_agent_loop.issue_loop as issue_loop
    import coding_review_agent_loop.pr_loop as pr_loop

    owners = {
        "run_issue_loop": issue_loop,
        "run_task_loop": issue_loop,
        "run_pr_loop": pr_loop,
        "run_discuss_loop": discuss_loop,
        "run_optional_tests": checks,
    }
    for name, owner in owners.items():
        assert getattr(cli, name) is getattr(owner, name), name
        assert getattr(orchestrator, name) is getattr(owner, name), name
    for name in ("run_issue_loop", "run_task_loop"):
        assert getattr(issue_loop, name).__module__ == issue_loop.__name__


# --- Each guard fails on a synthetic violation -------------------------------

def test_thin_facade_guard_accepts_docstring_and_imports():
    source = '"""Facade."""\nfrom __future__ import annotations\n\nimport os\nfrom .moved import (\n    helper,\n)\n'
    assert guard.thin_facade_problems(source) == []
    assert guard.thin_facade_problems("import os\n") == []


@pytest.mark.parametrize(
    ("addition", "kind"),
    [
        ("LIMIT = 3\n", "Assign"),
        ("LIMIT: int = 3\n", "AnnAssign"),
        ("def helper():\n    return 1\n", "FunctionDef"),
        ("class Carrier:\n    value = 1\n", "ClassDef"),
        ("if True:\n    import sys\n", "If"),
        ('"""A second string is not the docstring."""\n', "Expr"),
    ],
)
def test_thin_facade_guard_rejects_a_top_level_definition_or_statement(addition, kind):
    source = '"""Facade."""\nimport os\n' + addition
    assert guard.thin_facade_problems(source) == [
        f"orchestrator.py line 3 has a top-level {kind}; the facade may hold only a docstring and imports"
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


def test_coherence_guard_names_divergent_bindings_and_allows_module_only_names():
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
