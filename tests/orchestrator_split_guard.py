"""Pure-move safety net for splitting ``orchestrator.py`` (#1181, #1189).

This module is NOT a test file.  It holds:

* ``EXTRACTED_MODULES`` -- the ordered registry of modules extracted from
  ``orchestrator.py``.  Each extraction stage appends exactly one name.
* Baseline helpers that freeze the ``orchestrator`` import surface and the
  package module list at the start of the split.  The per-definition digest
  freeze that kept every extraction a pure move was retired once the split
  finished (#1204), so the extracted modules can now evolve; the facade itself
  must stay a docstring plus import statements (``thin_facade_problems``).
* Pure check functions used by ``tests/test_orchestrator_split.py``; each
  returns a list of human-readable problems so it can be exercised on
  synthetic violations as well as on the real package.
* ``facade_use_problems`` -- a static guard that keeps test files from
  importing or patching the ``orchestrator`` facade.  A facade patch would
  not reach the extracted module that owns the name, so tests patch owning
  modules directly (#1186).
"""
from __future__ import annotations

import ast
import importlib
import json
import os
import subprocess
import sys
import types
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

PACKAGE = "coding_review_agent_loop"
FACADE = f"{PACKAGE}.orchestrator"

# Ordered: an extracted module may import only modules listed before it.
EXTRACTED_MODULES: tuple[str, ...] = ("agent_failure", "managed_ci_bases", "integration_close", "architecture_contract", "reviewer_seats", "validated_agent", "response_validation", "panel_evidence", "review_rounds", "discuss_loop", "execution_policy", "child_plan_binding", "evidence_stall", "evidence_conversion", "pr_loop_support", "pr_loop", "issue_implementation", "plan_issue_scope", "plan_first_loop", "history_export", "triage", "triage_backtest", "issue_loop", "scope_declarations")

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
PACKAGE_DIR = SRC_ROOT / PACKAGE
BASELINE_PATH = Path(__file__).resolve().parent / "fixtures" / "orchestrator_split_baseline.json"


def _is_dunder(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")


# --- Registry and source locations ------------------------------------------


def registered_module_names() -> tuple[str, ...]:
    """Read the registry at call time so tests may monkeypatch it."""
    return tuple(EXTRACTED_MODULES)


def module_path(name: str) -> Path:
    return PACKAGE_DIR / f"{name}.py"


def split_source_paths() -> list[Path]:
    """``orchestrator.py`` followed by every registered extracted module."""
    return [module_path("orchestrator"), *(module_path(name) for name in registered_module_names())]


def registered_modules() -> list[types.ModuleType]:
    return [importlib.import_module(f"{PACKAGE}.{name}") for name in registered_module_names()]


def combined_split_tree() -> ast.Module:
    """One module whose body concatenates every split source's top level."""
    body: list[ast.stmt] = []
    for path in split_source_paths():
        body.extend(ast.parse(path.read_text(encoding="utf-8")).body)
    return ast.Module(body=body, type_ignores=[])


# --- Baseline ---------------------------------------------------------------


def package_module_names(package_dir: Path = PACKAGE_DIR) -> list[str]:
    """Dotted names (relative to the package) of every ``.py`` file it ships."""
    names = set()
    for path in package_dir.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        parts = list(path.relative_to(package_dir).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        if parts:
            names.add(".".join(parts))
    return sorted(names)


def facade_surface(facade: types.ModuleType) -> list[str]:
    return sorted(name for name in dir(facade) if not _is_dunder(name))


def compute_baseline() -> dict:
    facade = importlib.import_module(FACADE)
    return {
        "modules": package_module_names(),
        "surface": facade_surface(facade),
    }


def load_baseline(path: Path = BASELINE_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# --- Checks (each returns a list of problems) --------------------------------


def missing_surface_names(facade: object, surface: Iterable[str]) -> list[str]:
    return [f"orchestrator no longer exposes {name!r}" for name in surface if not hasattr(facade, name)]


def thin_facade_problems(facade_source: str) -> list[str]:
    """The facade's top level is an optional docstring plus import statements."""
    problems: list[str] = []
    for index, node in enumerate(ast.parse(facade_source).body):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if (
            index == 0
            and isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            continue
        problems.append(
            f"orchestrator.py line {node.lineno} has a top-level {type(node).__name__}; "
            "the facade may hold only a docstring and imports"
        )
    return problems


def incoherent_bindings(facade: types.ModuleType, modules: Iterable[types.ModuleType]) -> list[str]:
    """Names bound in both the facade and a registered module are the same object.

    A module-only global is allowed: extracted modules may grow new helpers
    the facade does not re-export.
    """
    problems: list[str] = []
    for module in modules:
        for name, value in vars(module).items():
            if _is_dunder(name) or name not in vars(facade):
                continue
            if vars(facade)[name] is not value:
                problems.append(f"{module.__name__}.{name} is not the facade's object")
    return problems


def fresh_incoherent_bindings(
    module_names: Sequence[str], *, facade: str = FACADE, extra_path: str | os.PathLike | None = None
) -> list[str]:
    """Run ``incoherent_bindings`` in a fresh interpreter with no test patches.

    Inside the test session the suite-wide repair stub and per-test patches
    rewrite owned names in registered modules, which would hide or fake a
    divergent production binding, so the pristine namespaces are checked here.
    """
    code = (
        "import importlib, json, sys\n"
        "import orchestrator_split_guard as guard\n"
        "facade = importlib.import_module(sys.argv[1])\n"
        "modules = [importlib.import_module(name) for name in sys.argv[2:]]\n"
        "sys.stdout.write(json.dumps(guard.incoherent_bindings(facade, modules)))\n"
    )
    env = dict(os.environ)
    paths = [str(SRC_ROOT), str(Path(__file__).resolve().parent)]
    if extra_path is not None:
        paths.append(str(extra_path))
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [*paths, env.get("PYTHONPATH")]))
    result = subprocess.run(
        [sys.executable, "-c", code, facade, *module_names],
        capture_output=True, text=True, env=env, cwd=str(REPO_ROOT), timeout=120,
    )
    if result.returncode != 0:
        return [f"coherence check could not run: {result.stderr.strip().splitlines()[-1:]}"]
    return json.loads(result.stdout)


def _imported_module_names(source: str, module_name: str) -> set[str]:
    """Absolute module names a source imports at any nesting level.

    ``from X import y`` yields both ``X`` and ``X.y`` because ``y`` may be a
    submodule.
    """
    package = module_name.rpartition(".")[0]
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                anchor = package.split(".")
                anchor = anchor[: len(anchor) - (node.level - 1)]
                base = ".".join([*anchor, *([node.module] if node.module else [])])
            else:
                base = node.module or ""
            imported.add(base)
            imported.update(f"{base}.{alias.name}" for alias in node.names)
    return imported


def layering_problems(registry: Sequence[str], module_sources: Mapping[str, str]) -> list[str]:
    """No registered module imports the facade or a module registered after it."""
    problems: list[str] = []
    for index, name in enumerate(registry):
        imported = _imported_module_names(module_sources[name], f"{PACKAGE}.{name}")
        forbidden = {FACADE: "the orchestrator facade"}
        forbidden.update({f"{PACKAGE}.{later}": f"later-registered {later}" for later in registry[index:]})
        for target, label in forbidden.items():
            if target in imported:
                problems.append(f"{name} imports {label}")
    return problems


def unregistered_modules(
    current: Iterable[str], baseline: Iterable[str], registry: Iterable[str]
) -> list[str]:
    known = set(baseline) | set(registry)
    return [f"new package module {name!r} is not in EXTRACTED_MODULES" for name in sorted(set(current) - known)]


def fresh_import_problem(name: str) -> str | None:
    """Import ``PACKAGE.name`` alone in a fresh interpreter, without the facade."""
    code = (
        "import sys, importlib\n"
        f"importlib.import_module({f'{PACKAGE}.{name}'!r})\n"
        f"loaded = {FACADE!r} in sys.modules\n"
        "sys.stdout.write('FACADE-LOADED' if loaded else 'OK')\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(SRC_ROOT), env.get("PYTHONPATH")]))
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=str(REPO_ROOT), timeout=120,
    )
    if result.returncode != 0:
        return f"{name} does not import alone: {result.stderr.strip().splitlines()[-1:]}"
    if result.stdout != "OK":
        return f"{name} imports the orchestrator facade"
    return None


# --- Test use of the facade ------------------------------------------------

# Only the split guard and its tests may touch the facade itself.
FACADE_USE_ALLOWLIST: frozenset[str] = frozenset({"test_orchestrator_split.py", "orchestrator_split_guard.py"})
TESTS_DIR = Path(__file__).resolve().parent


def _names_facade(module: str | None) -> bool:
    return module == FACADE or (module or "").startswith(f"{FACADE}.")


def _call_argument(node: ast.Call, position: int, keyword: str) -> ast.expr | None:
    """The argument bound to ``keyword``, passed positionally or by keyword."""
    if len(node.args) > position:
        return node.args[position]
    return next((item.value for item in node.keywords if item.arg == keyword), None)


def _import_module_target(node: ast.Call) -> str | None:
    """The absolute module an ``import_module(name, package)`` call with constant arguments names."""
    func = node.func
    called = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else None
    if called != "import_module":
        return None
    first = _call_argument(node, 0, "name")
    if not (isinstance(first, ast.Constant) and isinstance(first.value, str)):
        return None
    target = first.value
    if not target.startswith("."):
        return target
    package = _call_argument(node, 1, "package")
    if not (isinstance(package, ast.Constant) and isinstance(package.value, str)):
        return None
    level = len(target) - len(target.lstrip("."))
    anchor = package.value.split(".")
    anchor = anchor[: len(anchor) - (level - 1)]
    return ".".join([*anchor, target[level:]]) if target[level:] else ".".join(anchor)


def facade_use_problems(source: str, filename: str) -> list[str]:
    """Imports, ``import_module`` calls and patch-target strings naming the facade."""
    problems: list[str] = []
    for node in ast.walk(ast.parse(source, filename=filename)):
        where = f"{filename} line {getattr(node, 'lineno', '?')}"
        if isinstance(node, ast.Import):
            problems.extend(
                f"{where} imports the orchestrator facade" for alias in node.names if _names_facade(alias.name)
            )
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and (
                _names_facade(node.module)
                or (node.module == PACKAGE and any(alias.name == "orchestrator" for alias in node.names))
            ):
                problems.append(f"{where} imports the orchestrator facade")
        elif isinstance(node, ast.Call):
            target = _import_module_target(node)
            if target is not None and _names_facade(target):
                problems.append(f"{where} imports the orchestrator facade through import_module")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and _names_facade(node.value):
            problems.append(f"{where} names the orchestrator facade in a string such as a patch target")
    return problems


def tests_facade_use_problems(
    tests_dir: Path = TESTS_DIR, allowlist: Iterable[str] = FACADE_USE_ALLOWLIST
) -> list[str]:
    """``facade_use_problems`` over every ``tests/*.py`` file outside ``allowlist``."""
    allowed = set(allowlist)
    problems: list[str] = []
    for path in sorted(tests_dir.glob("*.py")):
        if path.name not in allowed:
            problems.extend(facade_use_problems(path.read_text(encoding="utf-8"), path.name))
    return problems


if __name__ == "__main__":
    sys.path.insert(0, str(SRC_ROOT))
    BASELINE_PATH.write_text(json.dumps(compute_baseline(), indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {BASELINE_PATH}")
