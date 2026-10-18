"""Pure-move and facade-use guards for the orchestrator split (#1181, #1189, #1186)."""
import types

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
    # Checked in a fresh interpreter: in this process the autouse repair stub
    # and per-test patches rewrite owned names before a divergence is observed.
    registered = [f"{guard.PACKAGE}.{name}" for name in guard.registered_module_names()]
    assert guard.fresh_incoherent_bindings(registered) == []


def test_coherence_guard_rejects_divergent_binding_hidden_by_a_test_patch(tmp_path, monkeypatch):
    (tmp_path / "split_probe_consumer.py").write_text(
        "def attempt_repair(*args, **kwargs):\n    return None\n", encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    import split_probe_consumer

    stub = lambda *args, **kwargs: None  # noqa: E731
    monkeypatch.setattr(orchestrator, "attempt_repair", stub)
    monkeypatch.setattr(split_probe_consumer, "attempt_repair", stub)
    # Under a test patch of both bindings the divergence looks coherent in-process...
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
    # These Git boundary helpers are package modules, not orchestrator moves.
    independent_git_modules = {
        "secure_git", "git_transport", "git_windows_launcher", "inspect_git_subprocess",
        "transport_check",
    }
    problems = guard.unregistered_modules(
        guard.package_module_names(), (*BASELINE["modules"], *independent_git_modules),
        guard.registered_module_names()
    )
    assert problems == []


def test_registered_modules_import_alone_in_a_fresh_interpreter():
    problems = [guard.fresh_import_problem(name) for name in guard.registered_module_names()]
    assert [problem for problem in problems if problem] == []


def test_autouse_repair_stub_is_bound_on_validated_agent():
    # tests/conftest.py stubs attempt_repair on validated_agent, the only
    # module that binds it, so the repair path never reaches the real CLI.
    import coding_review_agent_loop.repair as repair
    import coding_review_agent_loop.validated_agent as validated_agent

    assert validated_agent.attempt_repair is not repair.attempt_repair
    assert validated_agent._ORIGINAL_ATTEMPT_REPAIR is repair.attempt_repair
    assert validated_agent.attempt_repair(raw="x", gemini_cmd="gemini") is None


def test_no_test_file_imports_or_patches_the_orchestrator_facade():
    assert guard.tests_facade_use_problems() == []


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

def test_surface_guard_names_a_dropped_attribute():
    facade = types.ModuleType("synthetic_facade")
    facade.kept = 1
    assert guard.missing_surface_names(facade, ["kept", "dropped"]) == [
        "orchestrator no longer exposes 'dropped'"
    ]


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


# --- Facade-use guard on synthetic sources -------------------------------------


@pytest.mark.parametrize(
    ("source", "expected", "line"),
    [
        ("import coding_review_agent_loop.orchestrator\n", "imports the orchestrator facade", 1),
        ("import coding_review_agent_loop.orchestrator as orch\n", "imports the orchestrator facade", 1),
        ("import os, coding_review_agent_loop.orchestrator.sub\n", "imports the orchestrator facade", 1),
        ("from coding_review_agent_loop import orchestrator\n", "imports the orchestrator facade", 1),
        ("from coding_review_agent_loop import github, orchestrator as o\n", "imports the orchestrator facade", 1),
        ("def f():\n    from coding_review_agent_loop.orchestrator import run_pr_loop\n",
         "imports the orchestrator facade", 2),
        ("patch('coding_review_agent_loop.orchestrator.attempt_repair')\n",
         "names the orchestrator facade in a string such as a patch target", 1),
        ("monkeypatch.setattr('coding_review_agent_loop.orchestrator.log', fake)\n",
         "names the orchestrator facade in a string such as a patch target", 1),
    ],
    ids=["import", "import-as", "import-submodule", "from-package", "from-package-aliased",
         "nested-from-import", "patch-string", "setattr-string"],
)
def test_facade_use_guard_rejects_each_banned_form(source, expected, line):
    assert guard.facade_use_problems(source, "test_synthetic.py") == [f"test_synthetic.py line {line} {expected}"]


@pytest.mark.parametrize(
    "source",
    [
        "importlib.import_module('coding_review_agent_loop.orchestrator')\n",
        "import_module('coding_review_agent_loop.orchestrator')\n",
    ],
    ids=["importlib-attribute", "bare-import-module"],
)
def test_facade_use_guard_rejects_import_module_of_the_facade(source):
    assert guard.facade_use_problems(source, "test_synthetic.py") == [
        "test_synthetic.py line 1 imports the orchestrator facade through import_module",
        "test_synthetic.py line 1 names the orchestrator facade in a string such as a patch target",
    ]


def test_facade_use_guard_rejects_relative_import_module_of_the_facade():
    source = "importlib.import_module('.orchestrator', 'coding_review_agent_loop')\n"
    assert guard.facade_use_problems(source, "test_synthetic.py") == [
        "test_synthetic.py line 1 imports the orchestrator facade through import_module"
    ]


def test_facade_use_guard_accepts_owning_modules_and_lookalike_names():
    source = (
        "import coding_review_agent_loop.validated_agent as validated_agent\n"
        "from coding_review_agent_loop import pr_loop\n"
        "from coding_review_agent_loop.orchestrator_extra import helper\n"
        "import orchestrator_split_guard\n"
        "importlib.import_module('coding_review_agent_loop.issue_loop')\n"
        "patch('coding_review_agent_loop.validated_agent.attempt_repair')\n"
        "note = 'orchestrator.py lines 577-581'\n"
    )
    assert guard.facade_use_problems(source, "test_synthetic.py") == []


def test_facade_use_guard_scans_only_test_files_outside_the_allowlist(tmp_path):
    banned = "import coding_review_agent_loop.orchestrator\n"
    (tmp_path / "test_banned.py").write_text(banned, encoding="utf-8")
    (tmp_path / "conftest.py").write_text("patch('coding_review_agent_loop.orchestrator.x')\n", encoding="utf-8")
    (tmp_path / "test_allowed.py").write_text(banned, encoding="utf-8")
    (tmp_path / "notes.txt").write_text(banned, encoding="utf-8")
    assert guard.tests_facade_use_problems(tmp_path, allowlist={"test_allowed.py"}) == [
        "conftest.py line 1 names the orchestrator facade in a string such as a patch target",
        "test_banned.py line 1 imports the orchestrator facade",
    ]


def test_facade_use_allowlist_is_the_split_guard_and_its_tests():
    assert guard.FACADE_USE_ALLOWLIST == {"test_orchestrator_split.py", "orchestrator_split_guard.py"}
    assert guard.TESTS_DIR == guard.REPO_ROOT / "tests"
