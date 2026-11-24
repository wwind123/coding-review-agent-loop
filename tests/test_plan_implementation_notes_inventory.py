"""Every PR-facing approved-plan context attaches implementation notes (#2145).

A new recovery or construction site in a PR-facing module cannot silently drop
plan-review implementation notes: each call that builds or recovers an
``ApprovedPlanContext`` must either be wrapped by a notes helper, have its
result (or an alias of it) passed to one later in the same function, or be
listed below with the reason it never reaches a PR reviewer or implementer.
"""
from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent / "src" / "coding_review_agent_loop"
PR_FACING_MODULES = (
    "pr_loop.py",
    "pr_loop_support.py",
    "pr_plan_recovery.py",
    "plan_execution.py",
    "issue_implementation.py",
)
CONTEXT_BUILDERS = frozenset({
    "recover_approved_plan_context",
    "make_approved_plan_context",
    "_recover_managed_ci_approved_plan",
})
NOTES_HELPERS = frozenset({
    "attach_plan_implementation_notes",
    "with_plan_implementation_notes",
    "inherit_parent_plan_implementation_notes",
    "_notes",  # the local alias in pr_loop_support's qualification snapshot
})

# (module, enclosing function, builder) -> (call count, reason).
EXEMPTIONS: dict[tuple[str, str, str], tuple[int, str]] = {
    ("pr_loop.py", "_entry_plan_handoff_board", "recover_approved_plan_context"): (
        1, "identity only: selects the approving plan board, never rendered to a reviewer",
    ),
    ("pr_loop.py", "run_pr_loop", "recover_approved_plan_context"): (
        3,
        "identity only: the fresh-phase parent topology check and the handoff-board "
        "child/parent selection compare plan identity and never replace the reviewer context",
    ),
    ("pr_loop_support.py", "_fresh_pr_qualification_snapshot", "recover_approved_plan_context"): (
        1, "identity only: the same-hash re-recovery check compares hash and subject",
    ),
    ("pr_loop_support.py", "_recover_managed_ci_approved_plan", "recover_approved_plan_context"): (
        2, "the child-then-parent helper returns to callers that attach notes on assignment",
    ),
    ("issue_implementation.py", "_decompose_approved_plan", "make_approved_plan_context"): (
        2, "matrix only: reads the risk-matrix payload for topology validation",
    ),
}


def _name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _functions(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def _direct_function(tree: ast.AST) -> dict[int, str]:
    """Map each call node id to its innermost enclosing function name."""
    owner: dict[int, str] = {}
    for function in _functions(tree):
        for node in ast.walk(function):
            if isinstance(node, ast.Call):
                owner[id(node)] = function.name  # inner functions overwrite outer ones
    return owner


def _helper_args(function: ast.AST) -> list[tuple[int, ast.Call]]:
    return [
        (node.lineno, node)
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and _name(node.func) in NOTES_HELPERS
    ]


def _assigned_names(function: ast.AST, call: ast.Call) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(function):
        value = getattr(node, "value", None)
        if isinstance(node, (ast.Assign, ast.NamedExpr, ast.AnnAssign)) and value is call:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names.update(target.id for target in targets if isinstance(target, ast.Name))
    # Follow simple aliases (``y = x``) so a later attach on the alias counts.
    changed = True
    while changed:
        changed = False
        for node in ast.walk(function):
            if (
                isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Name)
                and node.value.id in names
            ):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id not in names:
                        names.add(target.id)
                        changed = True
    return names


def _covered(function: ast.AST, call: ast.Call) -> bool:
    helpers = _helper_args(function)
    for _line, helper in helpers:
        if any(node is call for node in ast.walk(helper)):
            return True
    names = _assigned_names(function, call)
    return any(
        line >= call.lineno
        and helper.args
        and isinstance(helper.args[0], ast.Name)
        and helper.args[0].id in names
        for line, helper in helpers
    )


def _uncovered_calls() -> Counter:
    uncovered: Counter = Counter()
    for module in PR_FACING_MODULES:
        tree = ast.parse((PACKAGE / module).read_text(encoding="utf-8"))
        owner = _direct_function(tree)
        functions = {function.name: function for function in _functions(tree)}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or _name(node.func) not in CONTEXT_BUILDERS:
                continue
            function_name = owner.get(id(node), "<module>")
            function = functions.get(function_name, tree)
            if not _covered(function, node):
                uncovered[(module, function_name, _name(node.func))] += 1
    return uncovered


def test_every_pr_facing_plan_context_site_attaches_notes_or_is_exempt():
    """Matrix row `pr-reviewer-context-resume` (inventory)."""
    uncovered = _uncovered_calls()
    unexpected = {
        key: count for key, count in uncovered.items()
        if EXEMPTIONS.get(key, (0, ""))[0] != count
    }
    assert not unexpected, (
        "PR-facing ApprovedPlanContext sites must attach plan implementation notes "
        "(attach_plan_implementation_notes / with_plan_implementation_notes) or be "
        f"listed in EXEMPTIONS with a reason: {unexpected}"
    )


def test_every_exemption_is_still_live_and_explained():
    uncovered = _uncovered_calls()
    for key, (count, reason) in EXEMPTIONS.items():
        assert uncovered.get(key) == count, f"stale exemption {key}"
        assert reason.strip()


def test_inventory_detects_an_unattached_recovery_site(tmp_path):
    source = (
        "def resume(comments):\n"
        "    context = recover_approved_plan_context(comments, expected_hash='h')\n"
        "    return context\n"
        "def attached(comments):\n"
        "    context = recover_approved_plan_context(comments, expected_hash='h')\n"
        "    alias = context\n"
        "    return attach_plan_implementation_notes(alias, runner=None, config=None, sources=())\n"
    )
    tree = ast.parse(source)
    functions = {function.name: function for function in _functions(tree)}
    calls = {
        name: next(
            node for node in ast.walk(function)
            if isinstance(node, ast.Call) and _name(node.func) == "recover_approved_plan_context"
        )
        for name, function in functions.items()
    }
    assert not _covered(functions["resume"], calls["resume"])
    assert _covered(functions["attached"], calls["attached"])
