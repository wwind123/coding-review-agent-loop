"""Pure-move safety net for splitting ``orchestrator.py`` (#1181, #1189).

This module is NOT a test file.  It holds:

* ``EXTRACTED_MODULES`` -- the ordered registry of modules extracted from
  ``orchestrator.py``.  Each extraction stage appends exactly one name.
* Baseline helpers that freeze the ``orchestrator`` import surface and a
  SHA-256 of every top-level definition's exact source segment.  Regenerate
  the fixture with ``python tests/orchestrator_split_guard.py`` (only when a
  change to a definition is intended and visible in the diff).
* Pure check functions used by ``tests/test_orchestrator_split.py``; each
  returns a list of human-readable problems so it can be exercised on
  synthetic violations as well as on the real package.
* ``install_patch_propagation()`` -- a test-only shim that makes attribute
  set/delete on the ``orchestrator`` facade also apply to every registered
  extracted module that owns the name, so existing tests that patch
  ``coding_review_agent_loop.orchestrator.<name>`` keep reaching code that
  has moved out of it.
"""
from __future__ import annotations

import ast
import hashlib
import importlib
import json
import os
import subprocess
import sys
import types
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path

PACKAGE = "coding_review_agent_loop"
FACADE = f"{PACKAGE}.orchestrator"

# Ordered: an extracted module may import only modules listed before it.
EXTRACTED_MODULES: tuple[str, ...] = ("agent_failure", "architecture_contract", "validated_agent")

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


def top_level_definitions(source: str) -> list[tuple[str, str]]:
    """``(name, sha256)`` for each top-level def, class and assignment.

    The digest covers the exact source lines of the statement, decorators
    included, so any edit to a moved body changes it.
    """
    lines = source.splitlines(keepends=True)
    found: list[tuple[str, str]] = []
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names = [node.name]
            start = min([node.lineno, *(decorator.lineno for decorator in node.decorator_list)])
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [
                leaf.id
                for target in targets
                for leaf in ast.walk(target)
                if isinstance(leaf, ast.Name)
            ]
            start = node.lineno
        else:
            continue
        segment = "".join(lines[start - 1 : node.end_lineno])
        digest = hashlib.sha256(segment.encode("utf-8")).hexdigest()
        found.extend((name, digest) for name in names)
    return found


def compute_baseline() -> dict:
    facade = importlib.import_module(FACADE)
    source = module_path("orchestrator").read_text(encoding="utf-8")
    definitions = top_level_definitions(source)
    names = [name for name, _ in definitions]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"orchestrator.py defines these names more than once: {duplicates}")
    return {
        "modules": package_module_names(),
        "surface": facade_surface(facade),
        "definitions": dict(sorted(definitions)),
    }


def load_baseline(path: Path = BASELINE_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# --- Checks (each returns a list of problems) --------------------------------


def missing_surface_names(facade: object, surface: Iterable[str]) -> list[str]:
    return [f"orchestrator no longer exposes {name!r}" for name in surface if not hasattr(facade, name)]


def definition_problems(
    baseline_definitions: Mapping[str, str],
    facade_source: str,
    module_sources: Mapping[str, str],
) -> list[str]:
    """Every baseline definition exists exactly once, byte-identical.

    ``module_sources`` maps each registered module name to its source; those
    modules must not introduce a top-level name absent from the baseline.
    """
    occurrences: dict[str, list[tuple[str, str]]] = {}
    problems: list[str] = []
    for owner, source in (("orchestrator", facade_source), *module_sources.items()):
        for name, digest in top_level_definitions(source):
            if owner != "orchestrator" and name not in baseline_definitions:
                problems.append(f"{owner} defines {name!r}, which is not a baseline orchestrator definition")
                continue
            occurrences.setdefault(name, []).append((owner, digest))
    for name, expected in baseline_definitions.items():
        found = occurrences.get(name, [])
        if len(found) != 1:
            owners = [owner for owner, _ in found]
            problems.append(f"{name!r} is defined {len(found)} times (in {owners}); expected exactly once")
            continue
        owner, digest = found[0]
        if digest != expected:
            problems.append(f"{name!r} in {owner} differs from its baseline source")
    return problems


def incoherent_bindings(facade: types.ModuleType, modules: Iterable[types.ModuleType]) -> list[str]:
    problems: list[str] = []
    for module in modules:
        for name, value in vars(module).items():
            if _is_dunder(name):
                continue
            if name not in vars(facade):
                problems.append(f"{module.__name__}.{name} is not bound on the orchestrator facade")
            elif vars(facade)[name] is not value:
                problems.append(f"{module.__name__}.{name} is not the facade's object")
    return problems


def fresh_incoherent_bindings(
    module_names: Sequence[str], *, facade: str = FACADE, extra_path: str | os.PathLike | None = None
) -> list[str]:
    """Run ``incoherent_bindings`` in a fresh interpreter with no shim installed.

    Inside the test session the propagation shim and the suite-wide repair
    stub rewrite owned names in registered modules, which would hide a
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


# --- Patch propagation shim --------------------------------------------------

_STATE_KEY = "__orchestrator_split_propagation__"


class _PropagationState:
    def __init__(self, registry: Callable[[], Iterable[types.ModuleType]]):
        self.registry = registry
        # name -> names of registered modules a propagated delete removed it from
        self.pending_restore: dict[str, set[str]] = {}


class PatchPropagatingModule(types.ModuleType):
    """Facade module whose attribute set/delete reaches owning extracted modules.

    A registered module owns a name when it currently binds it, or when a
    propagated delete removed it and the name has not been set again since.
    Modules that never owned a name are never touched, so propagation can
    not create attributes anywhere new.
    """

    def __setattr__(self, name: str, value: object) -> None:
        super().__setattr__(name, value)
        state = vars(self).get(_STATE_KEY)
        if state is None or _is_dunder(name):
            return
        pending = state.pending_restore.pop(name, set())
        for module in state.registry():
            if name in vars(module) or module.__name__ in pending:
                setattr(module, name, value)

    def __delattr__(self, name: str) -> None:
        super().__delattr__(name)
        state = vars(self).get(_STATE_KEY)
        if state is None or _is_dunder(name):
            return
        for module in state.registry():
            if name in vars(module):
                delattr(module, name)
                state.pending_restore.setdefault(name, set()).add(module.__name__)


def install_patch_propagation(
    facade: types.ModuleType | None = None,
    registry: Callable[[], Iterable[types.ModuleType]] | None = None,
) -> types.ModuleType:
    """Install the shim on ``facade`` (default: the real orchestrator); idempotent."""
    if facade is None:
        facade = importlib.import_module(FACADE)
    if _STATE_KEY not in vars(facade):
        vars(facade)[_STATE_KEY] = _PropagationState(registry or (lambda: registered_modules()))
    if not isinstance(facade, PatchPropagatingModule):
        facade.__class__ = PatchPropagatingModule
    return facade


def pending_restores(facade: types.ModuleType) -> dict[str, set[str]]:
    state = vars(facade).get(_STATE_KEY)
    return {} if state is None else {name: set(owners) for name, owners in state.pending_restore.items()}


if __name__ == "__main__":
    sys.path.insert(0, str(SRC_ROOT))
    BASELINE_PATH.write_text(json.dumps(compute_baseline(), indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {BASELINE_PATH}")
