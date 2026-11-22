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
