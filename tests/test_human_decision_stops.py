"""Every human-decision stop exits 4 with one three-part message (#2069).

The guard is static: it parses every module under ``src/coding_review_agent_loop``
with ``ast`` and builds the exception class graph from source, so the result
never depends on which modules happen to be imported first.

* A plain ``AgentLoopError`` (or any subclass that is not a
  ``HumanDecisionRequiredError``) must never carry "human decision" text.
* Every ``HumanDecisionRequiredError`` construction is a ``.decision(...)``
  call or takes a registered renderer's output, directly or through a local
  bound only from such calls in the same function.  Parameters, attributes
  and concatenations never qualify; error objects are carried across
  functions instead.
* Every registered renderer returns ``render_human_decision`` output.
"""

from __future__ import annotations

import ast
import importlib
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from coding_review_agent_loop import cli
from coding_review_agent_loop.errors import (
    HUMAN_DECISION_RENDERERS,
    AgentLoopError,
    HumanDecisionRequiredError,
    parse_human_decision,
    render_human_decision,
)

SRC = Path(__file__).resolve().parents[1] / "src" / "coding_review_agent_loop"
PHRASE = re.compile(r"human\s+decision", re.IGNORECASE)
ROOT_ERROR = "AgentLoopError"
DECISION_ERROR = "HumanDecisionRequiredError"
DECISION_FACTORY = "decision"
RENDER_FUNCTION = "render_human_decision"


# ---------------------------------------------------------------------------
# Pure scanners over {module name: source text}
# ---------------------------------------------------------------------------


def load_tree_sources() -> dict[str, str]:
    sources: dict[str, str] = {}
    for path in sorted(SRC.rglob("*.py")):
        module = ".".join(path.relative_to(SRC).with_suffix("").parts)
        sources[module] = path.read_text(encoding="utf-8")
    return sources


def _tail(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _resolve_relative(module: str, level: int, target: str | None) -> str:
    package = module.split(".")[:-1]
    if level > 1:
        package = package[: len(package) - (level - 1)]
    return ".".join([*package, *(target.split(".") if target else [])])


@dataclass(frozen=True)
class _Index:
    trees: dict[str, ast.Module]
    constants: dict[str, dict[str, str]]
    # module -> local name -> (source module, source name or None for a module alias)
    imports: dict[str, dict[str, tuple[str, str | None]]]
    flagged: frozenset[str]
    exempt: frozenset[str]


def _string_constant(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _string_constant(node.left), _string_constant(node.right)
        if left is not None and right is not None:
            return left + right
    return None


def _subclasses(graph: dict[str, set[str]], root: str) -> set[str]:
    found = {root}
    changed = True
    while changed:
        changed = False
        for name, bases in graph.items():
            if name not in found and bases & found:
                found.add(name)
                changed = True
    return found


def build_index(sources: dict[str, str]) -> _Index:
    trees = {module: ast.parse(text) for module, text in sources.items()}
    graph: dict[str, set[str]] = {}
    constants: dict[str, dict[str, str]] = {}
    imports: dict[str, dict[str, tuple[str, str | None]]] = {}
    for module, tree in trees.items():
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                graph.setdefault(node.name, set()).update(
                    tail for tail in (_tail(base) for base in node.bases) if tail
                )
        module_constants: dict[str, str] = {}
        module_imports: dict[str, tuple[str, str | None]] = {}
        for node in tree.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                value = _string_constant(node.value) if node.value is not None else None
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if value is not None:
                    for target in targets:
                        if isinstance(target, ast.Name):
                            module_constants[target.id] = value
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level:
                source = _resolve_relative(module, node.level, node.module)
                for alias in node.names:
                    local = alias.asname or alias.name
                    candidate = f"{source}.{alias.name}" if source else alias.name
                    if candidate in sources:
                        module_imports[local] = (candidate, None)
                    else:
                        module_imports[local] = (source, alias.name)
        constants[module] = module_constants
        imports[module] = module_imports
    exempt = _subclasses(graph, DECISION_ERROR)
    flagged = _subclasses(graph, ROOT_ERROR) - exempt
    return _Index(trees, constants, imports, frozenset(flagged), frozenset(exempt))


def _resolve_name(index: _Index, module: str, name: str, depth: int = 0) -> str | None:
    if name in index.constants.get(module, {}):
        return index.constants[module][name]
    imported = index.imports.get(module, {}).get(name)
    if imported is not None and imported[1] is not None and depth < 8:
        return _resolve_name(index, imported[0], imported[1], depth + 1)
    return None


def _call_strings(index: _Index, module: str, call: ast.Call) -> list[str]:
    parts: list[str] = []
    for argument in [*call.args, *(keyword.value for keyword in call.keywords)]:
        for node in ast.walk(argument):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                parts.append(node.value)
            elif isinstance(node, ast.Name):
                resolved = _resolve_name(index, module, node.id)
                if resolved is not None:
                    parts.append(resolved)
    return parts


def scan_plain_human_decision_errors(sources: dict[str, str]) -> list[str]:
    """``module:line`` of every non-exempt AgentLoopError call carrying the phrase."""
    index = build_index(sources)
    violations: list[str] = []
    for module, tree in index.trees.items():
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or _tail(node.func) not in index.flagged:
                continue
            parts = _call_strings(index, module, node)
            if any(PHRASE.search(part) for part in parts) or PHRASE.search("".join(parts)):
                violations.append(f"{module}:{node.lineno}")
    return violations


def _renderer_names(index: _Index, module: str, renderers) -> tuple[set[str], set[str]]:
    """Local function names and module aliases that resolve to registered renderers."""
    by_module: dict[str, set[str]] = {}
    for renderer_module, function in renderers:
        by_module.setdefault(renderer_module, set()).add(function)
    names = set(by_module.get(module, set()))
    aliases: set[str] = set()
    for local, (source, imported) in index.imports.get(module, {}).items():
        if imported is None:
            if source in by_module:
                aliases.add(local)
        elif imported in by_module.get(source, set()):
            names.add(local)
    return names, aliases


def _is_renderer_call(node: ast.AST, renderer_names, renderer_aliases, by_function) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Name):
        return func.id in renderer_names
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        return func.value.id in renderer_aliases and func.attr in by_function
    return False


def _function_parameters(function: ast.AST) -> set[str]:
    args = function.args  # type: ignore[attr-defined]
    names = [*args.posonlyargs, *args.args, *args.kwonlyargs]
    if args.vararg:
        names.append(args.vararg)
    if args.kwarg:
        names.append(args.kwarg)
    return {arg.arg for arg in names}


def _local_bound_only_from(function: ast.AST, name: str, accept) -> bool:
    """``name`` is a non-parameter local whose every binding satisfies ``accept``.

    Only a single-target ``name = <value>`` (or annotated) assignment whose value
    ``accept`` approves counts as a binding.  Any other binding form in the
    function rejects: context managers, loop and comprehension targets, walrus,
    augmented or multi-target assignments, unpacking, ``del``, imports, nested
    definitions, exception names, match captures and global/nonlocal (#2069).
    """
    if function is None or name in _function_parameters(function):
        return False
    accepted: set[int] = set()
    for node in _own_nodes(function):
        target = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            target = node.target
        if isinstance(target, ast.Name) and target.id == name:
            if not accept(node.value):
                return False
            accepted.add(id(target))
    for node in _own_nodes(function):
        if isinstance(node, ast.Name) and node.id == name:
            if isinstance(node.ctx, (ast.Store, ast.Del)) and id(node) not in accepted:
                return False
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == name:
                return False
        elif isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)):
            if node.name == name:
                return False
        elif isinstance(node, ast.MatchMapping) and node.rest == name:
            return False
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            if any((alias.asname or alias.name.split(".")[0]) == name for alias in node.names):
                return False
        elif isinstance(node, (ast.Global, ast.Nonlocal)) and name in node.names:
            return False
    return bool(accepted)


def _own_nodes(function: ast.AST):
    """Nodes of ``function`` excluding nested function and class bodies."""
    stack = list(ast.iter_child_nodes(function))
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        stack.extend(ast.iter_child_nodes(node))


def _calls_with_function(tree: ast.Module):
    """(call, innermost enclosing function or None) for every Call in ``tree``."""
    stack: list[tuple[ast.AST, ast.AST | None]] = [(tree, None)]
    while stack:
        node, function = stack.pop()
        for child in ast.iter_child_nodes(node):
            inner = (
                child if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else function
            )
            if isinstance(child, ast.Call):
                yield child, function
            stack.append((child, inner))


def check_decision_provenance(sources: dict[str, str], renderers) -> list[str]:
    """``module:line`` of every exempt-class construction outside the provenance rule."""
    index = build_index(sources)
    by_function = {function for _module, function in renderers}
    violations: list[str] = []
    for module, tree in index.trees.items():
        names, aliases = _renderer_names(index, module, renderers)

        def is_renderer(node: ast.AST) -> bool:
            return _is_renderer_call(node, names, aliases, by_function)

        for call, function in _calls_with_function(tree):
            if _tail(call.func) not in index.exempt:
                continue
            ok = False
            if len(call.args) == 1 and not call.keywords:
                argument = call.args[0]
                if is_renderer(argument):
                    ok = True
                elif isinstance(argument, ast.Name):
                    ok = _local_bound_only_from(function, argument.id, is_renderer)
            if not ok:
                violations.append(f"{module}:{call.lineno}")
    return violations


def check_renderer_bodies(sources: dict[str, str], renderers) -> list[str]:
    """Each registered renderer exists and returns only ``render_human_decision`` output."""
    problems: list[str] = []
    for module, function_name in renderers:
        tree = ast.parse(sources[module]) if module in sources else None
        function = next(
            (
                node for node in (tree.body if tree else ())
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == function_name
            ),
            None,
        )
        if function is None:
            problems.append(f"{module}.{function_name}: missing")
            continue

        def is_render(node: ast.AST) -> bool:
            return isinstance(node, ast.Call) and _tail(node.func) == RENDER_FUNCTION

        returns = [node for node in _own_nodes(function) if isinstance(node, ast.Return)]
        if not returns:
            problems.append(f"{module}.{function_name}: no return")
        for node in returns:
            value = node.value
            if value is not None and is_render(value):
                continue
            if isinstance(value, ast.Name) and _local_bound_only_from(function, value.id, is_render):
                continue
            problems.append(f"{module}.{function_name}:{node.lineno}")
    return problems


# ---------------------------------------------------------------------------
# The real tree
# ---------------------------------------------------------------------------


def test_no_plain_agent_loop_error_carries_human_decision_text():
    assert scan_plain_human_decision_errors(load_tree_sources()) == []


def test_every_human_decision_construction_follows_the_provenance_rule():
    assert check_decision_provenance(load_tree_sources(), HUMAN_DECISION_RENDERERS) == []


def test_every_registered_renderer_returns_render_human_decision_output():
    assert check_renderer_bodies(load_tree_sources(), HUMAN_DECISION_RENDERERS) == []


def test_class_graph_comes_from_source_and_includes_dual_inheritance():
    index = build_index(load_tree_sources())
    assert "PlanPrePanelHumanDecisionError" in index.exempt
    assert "PlanPrePanelSafetyError" in index.flagged
    assert DECISION_ERROR not in index.flagged


def _stall_items():
    from coding_review_agent_loop.protocol import UnresolvedReviewItem

    return [
        UnresolvedReviewItem(
            item_id="item-1", reviewer="Human requester", text="Run the live suite",
            source_round=1, status="blocking", authority="machine",
            obligation_kind="human-exact-head-evidence", lifecycle="evidence_frozen",
            candidate_head_sha="a" * 40, obligation_identity="request-1",
        )
    ]


def _renderer_fixtures():
    from coding_review_agent_loop import review_step_back as sb

    entry = sb.PrStepBackEntry(
        reviewer="OpenAI Codex", trigger_round=2, trigger_head="b" * 40, path="src/x.py",
        start=1, end=9, coder_round=3, resulting_head="c" * 40, record_index=1,
    )
    non_convergence = sb.PrNonConvergence(
        reviewer="OpenAI Codex", k=3, trigger_round=4, first_round=1,
        first_head="d" * 40, current_head="e" * 40,
        items=(sb.PersistingItem("item-1", "still broken", 3),), coder_rounds=(2, 3, 4),
    )
    return {
        ("review_step_back", "render_pr_non_convergence_decision"): (
            (77, non_convergence), {"diff_growth": "g", "pr_size": "s"},
            "PR #77: human decision required: ",
        ),
        ("review_step_back", "render_pr_step_back_approach_alternative_decision"): (
            (77, sb.PrApproachAlternative(3, "c" * 40, (entry,), "drop it")), {},
            "PR #77: " + sb.PR_STEP_BACK_APPROACH_ALTERNATIVE_PREFIX_TEXT,
        ),
        ("review_step_back", "render_step_back_human_decision"): (
            (), {"phase": "plan", "measurements": "m.", "step_back_round": 2, "blocks": 3,
                 "threshold": 3, "alternative": "simpler"},
            sb.PLAN_STEP_BACK_HUMAN_DECISION_PREFIX,
        ),
        ("panel_evidence", "plan_primary_stall_message"): (
            (), {"streak": 3, "threshold": 3, "plan_chars": 10},
            "the primary plan reviewer has blocked ",
        ),
        ("pr_loop_support", "render_evidence_stall_decision"): (
            (77, "a" * 40, "f" * 64, {"row-a": "x"}, 2), {},
            "PR #77 is stalled on evidence only at head ",
        ),
        ("pr_loop_support", "_evidence_freeze_diagnostic"): (
            (), {"pr_number": 77, "head_sha": "a" * 40, "items": _stall_items()},
            "PR #77 is frozen at head ",
        ),
        ("requirement_evidence", "_role_conflict_message"): (
            ([{"requirement_id": "r1", "role": "code"}],
             [("OpenAI Codex", {"requirement_id": "r1", "role": "human-only"})], "Claude"), {},
            "Signed-source classification remained conflicting",
        ),
    }


def test_every_registered_renderer_renders_the_three_part_shape_with_its_prefix():
    fixtures = _renderer_fixtures()
    assert set(fixtures) == set(HUMAN_DECISION_RENDERERS)
    for (module, function), (args, kwargs, prefix) in fixtures.items():
        renderer = getattr(importlib.import_module(f"coding_review_agent_loop.{module}"), function)
        text = renderer(*args, **kwargs)
        parts = parse_human_decision(text)
        assert parts is not None, (module, function)
        assert text.startswith(prefix), (module, function)
        assert parts.clear and parts.automation_note


# ---------------------------------------------------------------------------
# Message contract
# ---------------------------------------------------------------------------


def test_render_and_parse_round_trip_with_multi_line_what():
    text = render_human_decision(
        what="Human decision required: x.\nDetail line.",
        clear="Remove the record, then rerun.",
        safe_to_automate=False,
        automation_note="signed input.",
    )
    assert text.splitlines()[-2:] == [
        "To clear: Remove the record, then rerun.",
        "Safe to automate: no - signed input.",
    ]
    parts = parse_human_decision(text)
    assert parts is not None
    assert parts.what == "Human decision required: x.\nDetail line."
    assert parts.safe_to_automate is False
    assert parse_human_decision(text + "\ntrailing") is None
    assert parse_human_decision("Human decision required: x.") is None
    with pytest.raises(ValueError):
        render_human_decision(what=" ", clear="c", safe_to_automate=True, automation_note="n")


def test_decision_factory_and_with_context_keep_the_shape_and_class():
    from coding_review_agent_loop.plan_review_scheduling import (
        PlanPrePanelHumanDecisionError,
        PlanPrePanelSafetyError,
    )

    exc = HumanDecisionRequiredError.decision(
        what="Human decision required: x.", clear="Do y.", safe_to_automate=True,
        automation_note="nothing signed changes.",
    )
    assert str(exc) == render_human_decision(
        what="Human decision required: x.", clear="Do y.", safe_to_automate=True,
        automation_note="nothing signed changes.",
    )
    widened = exc.with_context("Post-merge context.")
    parts = parse_human_decision(str(widened))
    assert parts.what == "Human decision required: x.\nPost-merge context."
    assert parts.clear == "Do y." and parts.safe_to_automate is True
    assert str(widened).endswith("Safe to automate: yes - nothing signed changes.")
    dual = PlanPrePanelHumanDecisionError.decision(
        what="w", clear="c", safe_to_automate=False, automation_note="n"
    ).with_context("more")
    assert isinstance(dual, PlanPrePanelHumanDecisionError)
    assert isinstance(dual, PlanPrePanelSafetyError) and isinstance(dual, HumanDecisionRequiredError)


# ---------------------------------------------------------------------------
# Synthetic self-tests of the scanners
# ---------------------------------------------------------------------------

_ERRORS_SOURCE = '''
class AgentLoopError(RuntimeError):
    pass

class HumanDecisionRequiredError(AgentLoopError):
    pass

def render_human_decision(*, what, clear, safe_to_automate, automation_note):
    return what
'''

_RENDERERS = (("render", "registered"),)

_RENDER_SOURCE = '''
from .errors import render_human_decision

PREFIX = "Human decision required: "

def registered(x):
    return render_human_decision(what=x, clear="c", safe_to_automate=False, automation_note="n")

def unregistered(x):
    return x
'''


def _sources(**modules: str) -> dict[str, str]:
    return {"errors": _ERRORS_SOURCE, "render": _RENDER_SOURCE, **modules}


@pytest.mark.parametrize(
    "body",
    [
        "from .errors import AgentLoopError\n"
        "def f():\n    raise AgentLoopError('Human decision required: x')\n",
        "from .errors import AgentLoopError\n"
        "def f(x):\n    raise AgentLoopError(f'human   DECISION needed for {x}')\n",
        "from .errors import AgentLoopError\nfrom .render import PREFIX\n"
        "def f():\n    raise AgentLoopError(PREFIX + 'x')\n",
        "from . import errors\nLOCAL = 'needs a human decision'\n"
        "def f():\n    return errors.AgentLoopError(LOCAL)\n",
    ],
    ids=["literal", "fstring", "imported-constant", "module-constant-attribute-callee"],
)
def test_scanner_flags_plain_human_decision_text(body):
    violations = scan_plain_human_decision_errors(_sources(site=body))
    assert len(violations) == 1 and violations[0].startswith("site:")


def test_scanner_flags_a_cross_module_subclass_violation():
    sources = _sources(
        base="from .errors import AgentLoopError\nclass MidError(AgentLoopError):\n    pass\n",
        sub="from .base import MidError\nclass LeafError(MidError):\n    pass\n",
        site="from .sub import LeafError\ndef f():\n    raise LeafError('Human decision required')\n",
    )
    assert scan_plain_human_decision_errors(sources) == ["site:3"]


def test_scanner_accepts_decision_classes_and_the_hyphenated_protocol_token():
    sources = _sources(
        dual=(
            "from .errors import AgentLoopError, HumanDecisionRequiredError\n"
            "class SafetyError(AgentLoopError):\n    pass\n"
            "class DualError(SafetyError, HumanDecisionRequiredError):\n    pass\n"
            "def f():\n    raise DualError.decision(what='Human decision required: x',"
            " clear='c', safe_to_automate=False, automation_note='n')\n"
        ),
        protocol=(
            "from .errors import AgentLoopError\n"
            "def f():\n    raise AgentLoopError('status must be one of: blocker, human-decision.')\n"
        ),
        prose="HELP = 'stop for a human decision'\n",
    )
    assert scan_plain_human_decision_errors(sources) == []


_PROVENANCE_REJECTED = {
    "literal": "def f():\n    raise HumanDecisionRequiredError('literal')\n",
    "parameter-after-parse": (
        "from .errors import parse_human_decision\n"
        "def f(message):\n    if parse_human_decision(message) is None:\n        return\n"
        "    raise HumanDecisionRequiredError(message)\n"
    ),
    "attribute": "def f(result):\n    raise HumanDecisionRequiredError(result.deferred_stop)\n",
    "unregistered-helper": (
        "from .render import unregistered\n"
        "def f():\n    raise HumanDecisionRequiredError(unregistered('x'))\n"
    ),
    "concatenation": (
        "from .render import registered\n"
        "def f():\n    raise HumanDecisionRequiredError('PR #1: ' + registered('x'))\n"
    ),
    "with-rebinding": (
        "from .render import registered\n"
        "def f(manager):\n    message = registered('x')\n    with manager() as message:\n"
        "        raise HumanDecisionRequiredError(message)\n"
    ),
    "async-with-rebinding": (
        "from .render import registered\n"
        "async def f(manager):\n    message = registered('x')\n"
        "    async with manager() as message:\n        raise HumanDecisionRequiredError(message)\n"
    ),
    "for-rebinding": (
        "from .render import registered\n"
        "def f(items):\n    message = registered('x')\n    for message in items:\n        pass\n"
        "    raise HumanDecisionRequiredError(message)\n"
    ),
    "walrus-rebinding": (
        "from .render import registered\n"
        "def f(other):\n    message = registered('x')\n    if (message := other):\n        pass\n"
        "    raise HumanDecisionRequiredError(message)\n"
    ),
    "unpack-rebinding": (
        "from .render import registered\n"
        "def f(pair):\n    message = registered('x')\n    message, _ = pair\n"
        "    raise HumanDecisionRequiredError(message)\n"
    ),
    "import-rebinding": (
        "from .render import registered\n"
        "def f():\n    message = registered('x')\n    from os import sep as message\n"
        "    raise HumanDecisionRequiredError(message)\n"
    ),
    "mixed-local": (
        "from .render import registered\n"
        "def f(flag):\n    message = registered('x')\n    if flag:\n        message = 'other'\n"
        "    raise HumanDecisionRequiredError(message)\n"
    ),
}


@pytest.mark.parametrize("case", sorted(_PROVENANCE_REJECTED))
def test_provenance_checker_rejects(case):
    site = "from .errors import HumanDecisionRequiredError\n" + _PROVENANCE_REJECTED[case]
    violations = check_decision_provenance(_sources(site=site), _RENDERERS)
    assert len(violations) == 1 and violations[0].startswith("site:")


def test_provenance_checker_accepts_renderer_output_and_carried_objects():
    site = (
        "from dataclasses import dataclass\n"
        "from .errors import AgentLoopError, HumanDecisionRequiredError\n"
        "from .render import registered\n"
        "from . import render as _render\n"
        "class DualError(AgentLoopError, HumanDecisionRequiredError):\n    pass\n"
        "@dataclass\nclass Decision:\n    deferred_stop: HumanDecisionRequiredError | None = None\n"
        "def build():\n    message = registered('x')\n"
        "    return Decision(deferred_stop=HumanDecisionRequiredError(message))\n"
        "def direct():\n    raise DualError(_render.registered('y'))\n"
        "def factory():\n    raise HumanDecisionRequiredError.decision(what='w', clear='c',"
        " safe_to_automate=False, automation_note='n')\n"
        "def stop(exc):\n    raise exc\n"
        "def consume(decision):\n    stop(decision.deferred_stop)\n"
        "def outer():\n    def stop_decision(exc):\n        raise exc\n"
        "    stop_decision(DualError(registered('z')))\n"
    )
    assert check_decision_provenance(_sources(site=site), _RENDERERS) == []


def test_renderer_body_check_flags_a_non_rendering_renderer():
    assert check_renderer_bodies(_sources(), _RENDERERS) == []
    assert check_renderer_bodies(_sources(), (("render", "unregistered"),)) == [
        "render.unregistered:10"
    ]
    assert check_renderer_bodies(_sources(), (("render", "absent"),)) == ["render.absent: missing"]


# ---------------------------------------------------------------------------
# Exit-4 behaviour of converted sites
# ---------------------------------------------------------------------------


def _assert_three_part(exc: BaseException, *, safe: bool | None = False) -> None:
    assert isinstance(exc, HumanDecisionRequiredError)
    parts = parse_human_decision(str(exc))
    assert parts is not None, str(exc)
    if safe is not None:
        assert parts.safe_to_automate is safe


def _human_stage_phase():
    return SimpleNamespace(
        stage_id="manual-stage", position=1, automation="human-action",
        execution_disposition=None,
    )


def test_execution_policy_override_naming_a_human_stage_exits_4():
    """`execution-policy-exit4`: execution_policy.py human-owned stage override."""
    from coding_review_agent_loop.execution_policy import resolve_child_execution_route

    record = SimpleNamespace(
        comment_locator="parent issue #5 comment 2", disposition="direct-implementation",
        digest="d1",
    )
    with pytest.raises(HumanDecisionRequiredError) as caught:
        resolve_child_execution_route(
            _human_stage_phase(), topology_source="approved-plan-v1", overrides=(record,)
        )
    _assert_three_part(caught.value)
    assert str(caught.value).startswith("Human decision required: signed child-disposition override")
    assert "Remove the signed override at parent issue #5 comment 2" in str(caught.value)
    # Without an override the human stage keeps its human route (never agent-pr).
    route = resolve_child_execution_route(_human_stage_phase(), topology_source="approved-plan-v1")
    assert route.disposition == "human"


def test_cli_returns_4_for_a_converted_board_amendment_stop(monkeypatch, capsys):
    """`board-amendment-exit4`: the CLI maps the converted stop to exit 4."""
    from coding_review_agent_loop.board_amendment import reject_misplaced_pr_amendments

    def raise_board_stop(*_args, **_kwargs):
        comment = SimpleNamespace(body=_signed_pr_amendment_body(pr_number=12))
        reject_misplaced_pr_amendments([comment], issue_number=5)
        raise AssertionError("expected a human-decision stop")

    monkeypatch.setattr(cli, "run_pr_loop", raise_board_stop)
    code = cli.main(["pr", "12", "--repo", "owner/repo"])
    err = capsys.readouterr().err
    assert code == 4
    assert "agent-loop: HUMAN DECISION REQUIRED" in err
    assert "To clear: Remove the signed `pr` record from issue #5" in err
    assert err.rstrip().splitlines()[-1].startswith("Safe to automate: no - ")


def _signed_pr_amendment_body(*, pr_number: int) -> str:
    from coding_review_agent_loop.board_amendment import format_reviewer_board_amendment_comment

    return format_reviewer_board_amendment_comment(
        flow="pr", issue=None, pr_number=pr_number,
        original_required_reviewers=("OpenAI Codex", "Anthropic Claude"), policy="all-reviewers",
        primary_reviewer=None, removed_reviewers=("Anthropic Claude",), effective_from_round=2,
        rationale="unreachable",
    )


def test_cli_returns_4_for_post_merge_rewrapped_stop(monkeypatch, capsys):
    """`post-merge-context-rewrap`: context lands in 'what'; the CLI still exits 4."""
    base = HumanDecisionRequiredError.decision(
        what="Human decision required: x.", clear="Do y.", safe_to_automate=False,
        automation_note="signed input.",
    )
    rewrapped = base.with_context("Post-merge obligations: target #9 pending.")

    def raise_stop(*_args, **_kwargs):
        raise rewrapped

    monkeypatch.setattr(cli, "run_pr_loop", raise_stop)
    assert cli.main(["pr", "12", "--repo", "owner/repo"]) == 4
    err = capsys.readouterr().err
    assert err.rstrip().endswith("Safe to automate: no - signed input.")
    parts = parse_human_decision(str(rewrapped))
    assert parts.what.endswith("Post-merge obligations: target #9 pending.")


def test_protocol_discuss_human_decision_token_stays_a_plain_format_error():
    """`protocol-token-stays-plain`: the hyphenated status token is not a stop."""
    import json

    from coding_review_agent_loop.protocol_discuss import parse_structured_discuss_answer

    body = json.dumps({
        "schema_version": 1, "kind": "discuss_answer", "position": "needs-human",
        "rationale": "unclear", "confidence": "low",
        "unresolved_items": [{"status": "blocker", "text": "x"}],
    }) + "\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex"
    with pytest.raises(AgentLoopError) as caught:
        parse_structured_discuss_answer(body, reviewer="OpenAI Codex")
    assert not isinstance(caught.value, HumanDecisionRequiredError)
    assert "human-decision" in str(caught.value)
    assert not PHRASE.search(str(caught.value))


def test_review_images_reduce_history_stops_are_three_part_and_the_fallbacks_are_scoped():
    """`catch-handler-preservation`: only visual-state stops reach the fallbacks."""
    from coding_review_agent_loop.review_images_state import reduce_history

    with pytest.raises(HumanDecisionRequiredError) as caught:
        reduce_history(SimpleNamespace(complete=False, diagnostic="gap"), repo="o/r", pr=1, head="a")
    _assert_three_part(caught.value, safe=True)
    # The swallowing handlers wrap only ``reduce_history``, which never reaches
    # board, decomposition or execution-policy code.
    sources = load_tree_sources()
    for module in ("review_images", "review_images_coverage_history"):
        tree = ast.parse(sources[module])
        for node in ast.walk(tree):
            if isinstance(node, ast.Try) and any(
                _tail(handler.type) == DECISION_ERROR
                and all(isinstance(stmt, ast.Pass) or isinstance(stmt, ast.Return)
                        for stmt in handler.body)
                for handler in node.handlers if handler.type is not None
            ):
                called = {
                    _tail(sub.func) for stmt in node.body for sub in ast.walk(stmt)
                    if isinstance(sub, ast.Call)
                }
                assert called <= {"reduce_history", "reliable_head", "History", "add"}, (module, called)


def test_pr_loop_adds_post_merge_context_through_with_context():
    """`post-merge-context-rewrap`: stop_for_human_decision re-wraps via with_context."""
    tree = ast.parse(load_tree_sources()["pr_loop"])
    (stop,) = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "stop_for_human_decision"
    ]
    calls = [node for node in ast.walk(stop) if isinstance(node, ast.Call)]
    assert any(_tail(call.func) == "with_context" for call in calls)
    assert not any(_tail(call.func) == DECISION_ERROR for call in calls)


# ---------------------------------------------------------------------------
# Re-wrap handlers keep human-decision stops (#2069, review item-5)
# ---------------------------------------------------------------------------

_BROAD_CATCHES = {"AgentLoopError", "Exception", "RuntimeError", "BaseException"}
REWRAP_KEEPER = "reraise_human_decision"


def _reachable_stop_functions(index: _Index) -> set[str]:
    """Uniquely named module-level functions that can raise a human-decision stop.

    The call graph is built from parsed source over module-level functions
    whose names are unique in the package, so a name resolves to one body.
    """
    functions: dict[str, list[ast.AST]] = {}
    for tree in index.trees.values():
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.setdefault(node.name, []).append(node)
    unique = {name: nodes[0] for name, nodes in functions.items() if len(nodes) == 1}

    def constructs_stop(function: ast.AST) -> bool:
        for node in ast.walk(function):
            if not isinstance(node, ast.Call):
                continue
            tail = _tail(node.func)
            if tail in index.exempt:
                return True
            if (
                tail == DECISION_FACTORY and isinstance(node.func, ast.Attribute)
                and _tail(node.func.value) in index.exempt
            ):
                return True
        return False

    calls = {
        name: {_tail(node.func) for node in ast.walk(body) if isinstance(node, ast.Call)}
        for name, body in unique.items()
    }
    reach = {name for name, body in unique.items() if constructs_stop(body)}
    changed = True
    while changed:
        changed = False
        for name, called in calls.items():
            if name not in reach and called & reach:
                reach.add(name)
                changed = True
    return reach


def check_rewrap_handlers(sources: dict[str, str]) -> list[str]:
    """``module:line`` of re-wraps that would turn a reachable stop into a plain error.

    A try block whose body calls a function that can raise a human-decision
    stop must not rebuild the caught error into a non-exempt error unless the
    handler first calls ``reraise_human_decision(<caught>, ...)`` or an earlier
    handler of the same try catches the human-decision class.
    """
    index = build_index(sources)
    reach = _reachable_stop_functions(index)
    violations: list[str] = []
    for module, tree in index.trees.items():
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            body_calls = {
                _tail(call.func) for stmt in node.body for call in ast.walk(stmt)
                if isinstance(call, ast.Call)
            }
            if not body_calls & reach:
                continue
            exempt_handled = False
            for handler in node.handlers:
                types = (
                    handler.type.elts if isinstance(handler.type, ast.Tuple)
                    else [handler.type] if handler.type is not None else []
                )
                names = {_tail(item) for item in types} or {"BaseException"}
                if names & index.exempt:
                    exempt_handled = True
                    continue
                if exempt_handled or handler.name is None or not names & _BROAD_CATCHES:
                    continue
                caught = handler.name
                handler_nodes = [n for stmt in handler.body for n in ast.walk(stmt)]
                kept = any(
                    isinstance(n, ast.Call) and _tail(n.func) == REWRAP_KEEPER
                    and n.args and isinstance(n.args[0], ast.Name) and n.args[0].id == caught
                    for n in handler_nodes
                )
                if kept:
                    continue
                for raised in handler_nodes:
                    if not (isinstance(raised, ast.Raise) and isinstance(raised.exc, ast.Call)):
                        continue
                    arguments = [*raised.exc.args, *(k.value for k in raised.exc.keywords)]
                    if any(
                        isinstance(sub, ast.Name) and sub.id == caught
                        for argument in arguments for sub in ast.walk(argument)
                    ) and _tail(raised.exc.func) not in index.exempt:
                        violations.append(f"{module}:{raised.lineno}")
    return violations


def test_no_handler_rewraps_a_reachable_human_decision_stop_into_a_plain_error():
    assert check_rewrap_handlers(load_tree_sources()) == []


_REWRAP_STOP = (
    "from .errors import AgentLoopError, HumanDecisionRequiredError, reraise_human_decision\n"
    "def stop():\n    raise HumanDecisionRequiredError.decision(what='w', clear='c',"
    " safe_to_automate=False, automation_note='n')\n"
    "def entry():\n    stop()\n"
)


@pytest.mark.parametrize(
    "handler, flagged",
    [
        ("    except AgentLoopError as exc:\n        raise AgentLoopError(f'{exc} more')\n", True),
        ("    except Exception as exc:\n        raise RuntimeError(str(exc))\n", True),
        ("    except AgentLoopError as exc:\n        reraise_human_decision(exc, 'more')\n"
         "        raise AgentLoopError(f'{exc} more')\n", False),
        ("    except HumanDecisionRequiredError:\n        raise\n"
         "    except AgentLoopError as exc:\n        raise AgentLoopError(str(exc))\n", False),
        ("    except AgentLoopError as exc:\n        raise\n", False),
    ],
    ids=["plain-rewrap", "broad-rewrap", "kept", "earlier-exempt-handler", "bare-reraise"],
)
def test_rewrap_guard_self_test(handler, flagged):
    site = "from .stops import entry\nfrom .errors import *\ndef f():\n    try:\n        entry()\n" + handler
    violations = check_rewrap_handlers({"errors": _ERRORS_SOURCE.replace(
        "def render_human_decision", "def reraise_human_decision(exc, context):\n    pass\n\n"
        "def render_human_decision"), "stops": _REWRAP_STOP, "site": site})
    assert bool(violations) is flagged, violations


def test_reraise_human_decision_keeps_class_shape_and_passes_plain_errors_through():
    from coding_review_agent_loop.errors import reraise_human_decision

    stop = HumanDecisionRequiredError.decision(
        what="Human decision required: x.", clear="Do y.", safe_to_automate=False,
        automation_note="signed input.",
    )
    with pytest.raises(HumanDecisionRequiredError) as caught:
        reraise_human_decision(stop, "No qualification or merge is permitted.")
    parts = parse_human_decision(str(caught.value))
    assert parts.what.endswith("No qualification or merge is permitted.")
    assert parts.clear == "Do y."
    # Context already present is not appended twice.
    with pytest.raises(HumanDecisionRequiredError) as again:
        reraise_human_decision(caught.value, "No qualification or merge is permitted.")
    assert again.value is caught.value
    assert reraise_human_decision(AgentLoopError("plain"), "context") is None


def test_qualification_board_check_keeps_a_misplaced_amendment_stop_at_exit_4(monkeypatch):
    """item-5: the qualification gate no longer turns a board stop into exit 1."""
    from coding_review_agent_loop import pr_board_state
    from coding_review_agent_loop.board_amendment import collect_reviewer_board_amendments

    def resolve_with_real_collection(runner, *, config, pr_number, comments, **_kwargs):
        collect_reviewer_board_amendments(comments, flow="pr", pr_number=pr_number)
        raise AssertionError("expected the misplaced amendment to stop")

    monkeypatch.setattr(pr_board_state, "resolve_pr_board_state", resolve_with_real_collection)
    check = pr_board_state.qualification_board_check(
        object(), config=SimpleNamespace(), pr_number=12, expected=None,
    )
    misplaced = SimpleNamespace(body=_signed_pr_amendment_body(pr_number=99))
    with pytest.raises(HumanDecisionRequiredError) as caught:
        check([misplaced])
    parts = parse_human_decision(str(caught.value))
    assert parts is not None and parts.safe_to_automate is False
    assert parts.what.startswith("Human decision required: signed reviewer-board amendment")
    assert parts.what.endswith("No qualification or merge is permitted.")
    assert "names PR #99, but it is posted on PR #12" in parts.what

    def raise_cli(*_args, **_kwargs):
        raise caught.value

    monkeypatch.setattr(cli, "run_pr_loop", raise_cli)
    assert cli.main(["pr", "12", "--repo", "owner/repo"]) == 4


# ---------------------------------------------------------------------------
# Fallback handlers must not transport a stop's text (#2069, items 7-9)
# ---------------------------------------------------------------------------

# Handlers that turn a caught error into a fallback result and that the
# resolved call graph can reach a stop through.  Each stays as it was for a
# stated reason and is tracked as follow-up; a new entry needs a new reason.
KNOWN_TRANSPORT_FALLBACKS = {
    "discuss_loop:_run_discuss_analyzer": "discuss analyzer turn failure keeps its degraded outcome",
    "discuss_loop:_run_discuss_final_analyzer": "discuss final analyzer turn failure keeps its degraded outcome",
    "discuss_loop:_run_discuss_evidence_reconciler": "discuss evidence reconciler turn failure policy",
    "discuss_loop:_debater_worker": "discuss debater failure policy records the failed seat",
    "discuss_loop:_run_discuss_loop": "discuss debater failure policy records the failed seat",
    "plan_first_loop:_plan_reviewer_worker": "plan reviewer turn failure is recorded per seat",
    "pr_loop:_pr_reviewer_worker": "PR reviewer turn failure is recorded per seat",
    "review_images_coverage_run:gate": "visual coverage turn degrades to unavailable evidence",
    "triage:run_issue_triage": "advisory triage turn",
    "execution_policy:_print_dry_run_execution_preview": "dry-run topology preview text (review item-6, deferred)",
    "discuss_loop:_run_discuss_semantic_finalization": "discuss semantic finalization turn failure keeps its degraded outcome",
    "child_plan_binding:_fresh_child_resume_command": "advisory resume-command hint; the real failure is raised elsewhere",
    "pr_loop_support:_pr_amendment_plan_board_hint": "advisory rerun hint text only",
    "review_rounds:_unavailable_reviewer_amendment_advisory": "advisory amendment template; the stop stays authoritative",
}


def _resolved_stop_graph(index: _Index):
    defs = {
        module: {
            node.name: node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for module, tree in index.trees.items()
    }

    def resolve(module, func):
        if isinstance(func, ast.Name):
            if func.id in defs[module]:
                return module, func.id
            imported = index.imports.get(module, {}).get(func.id)
            if imported and imported[1] and func.id and imported[1] in defs.get(imported[0], {}):
                return imported[0], imported[1]
        elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            imported = index.imports.get(module, {}).get(func.value.id)
            if imported and imported[1] is None and func.attr in defs.get(imported[0], {}):
                return imported[0], func.attr
        return None

    def handler_names(handler):
        types = (
            handler.type.elts if isinstance(handler.type, ast.Tuple)
            else [handler.type] if handler.type is not None else []
        )
        return {_tail(item) for item in types} or {"BaseException"}

    def escaping_calls(module, function):
        found = set()

        def visit(node, shielded):
            if isinstance(node, ast.Try):
                catches = any(
                    handler_names(h) & index.exempt and not isinstance(h.body[-1], ast.Raise)
                    for h in node.handlers
                )
                for stmt in node.body:
                    visit(stmt, shielded or catches)
                for handler in node.handlers:
                    for stmt in handler.body:
                        visit(stmt, shielded)
                for stmt in [*node.orelse, *node.finalbody]:
                    visit(stmt, shielded)
                return
            if isinstance(node, ast.Call) and not shielded:
                target = resolve(module, node.func)
                if target:
                    found.add(target)
            for child in ast.iter_child_nodes(node):
                visit(child, shielded)

        visit(function, False)
        return found

    def constructs(function):
        for node in ast.walk(function):
            if isinstance(node, ast.Call):
                tail = _tail(node.func)
                if tail in index.exempt or (
                    tail == DECISION_FACTORY and isinstance(node.func, ast.Attribute)
                    and _tail(node.func.value) in index.exempt
                ):
                    return True
        return False

    graph = {
        (module, name): escaping_calls(module, node)
        for module, functions in defs.items() for name, node in functions.items()
    }
    reach = {key for key in graph if constructs(defs[key[0]][key[1]])}
    changed = True
    while changed:
        changed = False
        for key, called in graph.items():
            if key not in reach and called & reach:
                reach.add(key)
                changed = True
    return resolve, handler_names, reach


def check_transport_fallbacks(sources: dict[str, str]) -> list[str]:
    """``module:line`` of broad handlers that carry a reachable stop's text into a fallback."""
    index = build_index(sources)
    resolve, handler_names, reach = _resolved_stop_graph(index)
    violations: list[str] = []
    for module, tree in index.trees.items():
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            calls = {
                target for stmt in node.body for call in ast.walk(stmt)
                if isinstance(call, ast.Call) for target in [resolve(module, call.func)] if target
            }
            if not calls & reach:
                continue
            for handler in node.handlers:
                names = handler_names(handler)
                if names & index.exempt:
                    break
                if not names & _BROAD_CATCHES:
                    continue
                # Every broad handler counts, named or not, whether it carries
                # the error text forward or just returns a degraded value
                # (#2069 review item-10).
                body = [n for stmt in handler.body for n in ast.walk(stmt)]
                if handler.name is not None and any(
                    isinstance(n, ast.Call) and _tail(n.func) == REWRAP_KEEPER
                    and n.args and isinstance(n.args[0], ast.Name) and n.args[0].id == handler.name
                    for n in body
                ):
                    continue
                last = handler.body[-1]
                if isinstance(last, ast.Raise) and (
                    last.exc is None or (isinstance(last.exc, ast.Name) and last.exc.id == handler.name)
                ):
                    continue
                violations.append(f"{module}:{handler.lineno}")
    return violations


def _enclosing_function(source: str, line: int) -> str | None:
    best = None
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.lineno <= line <= node.end_lineno:
            if best is None or node.lineno > best.lineno:
                best = node
    return best.name if best else None


def test_no_fallback_transports_a_reachable_human_decision_stop():
    sources = load_tree_sources()
    found = set()
    unexpected = []
    for site in check_transport_fallbacks(sources):
        module, line = site.split(":")
        key = f"{module}:{_enclosing_function(sources[module], int(line))}"
        found.add(key)
        if key not in KNOWN_TRANSPORT_FALLBACKS:
            unexpected.append(site)
    assert unexpected == []
    # Every known entry still exists, so the list cannot silently go stale.
    assert set(KNOWN_TRANSPORT_FALLBACKS) <= found


def test_transport_guard_flags_a_result_fallback_and_accepts_a_kept_stop():
    stops = (
        "from .errors import HumanDecisionRequiredError\n"
        "def stop():\n    raise HumanDecisionRequiredError.decision(what='w', clear='c',"
        " safe_to_automate=False, automation_note='n')\n"
    )
    transport = (
        "from .stops import stop\nfrom .errors import AgentLoopError, reraise_human_decision\n"
        "def f():\n    try:\n        stop()\n    except AgentLoopError as exc:\n"
        "        return ('undetermined', str(exc))\n"
    )
    kept = transport.replace(
        "        return ('undetermined', str(exc))\n",
        "        reraise_human_decision(exc, 'ctx')\n        return ('undetermined', str(exc))\n",
    )
    shielded = (
        "from .stops import stop\nfrom .errors import AgentLoopError, HumanDecisionRequiredError\n"
        "def g():\n    try:\n        stop()\n    except HumanDecisionRequiredError:\n        return None\n"
        "def f():\n    try:\n        g()\n    except AgentLoopError as exc:\n        return str(exc)\n"
    )
    unnamed = transport.replace(
        "    except AgentLoopError as exc:\n        return ('undetermined', str(exc))\n",
        "    except (AgentLoopError, ValueError):\n        return None\n",
    )
    base = {"errors": _ERRORS_SOURCE, "stops": stops}
    assert check_transport_fallbacks({**base, "site": transport}) == ["site:6"]
    # An unnamed handler that drops the stop for a degraded value is flagged too.
    assert check_transport_fallbacks({**base, "site": unnamed}) == ["site:6"]
    assert check_transport_fallbacks({**base, "site": kept}) == []
    # A callee that swallows the stop itself does not make its callers reachable.
    assert check_transport_fallbacks({**base, "site": shielded}) == []
