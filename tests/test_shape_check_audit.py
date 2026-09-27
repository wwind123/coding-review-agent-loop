"""Guard for the protocol.py shape-check audit registry (#927).

The collector below reads a module's source with ``ast`` and ``tokenize``.
It enumerates units (module-level functions, ``Class.method`` methods and
qualified nested defs), resolves the exception classes each ``raise`` names,
builds the call and reference graph (bare names, same-module ``Class.method``,
``cls``/``self``/``super()`` calls, constructors and untyped receivers), and
classifies every site as handled or propagating inside its own unit's
``try`` context.  The guard then checks the registry in
``coding_review_agent_loop.shape_check_audit`` against that inventory.
"""

from __future__ import annotations

import ast
import builtins
import io
import re
import tokenize
from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest

from coding_review_agent_loop import shape_check_audit
from coding_review_agent_loop.shape_check_audit import (
    DEGRADING_PARSERS,
    FATAL_CLAUSES,
    IMPORTED_NON_RAISERS,
    IMPORTED_RAISERS,
    SHAPE_CHECK_AUDIT,
    ShapeCheckClassification,
)

PACKAGE_DIR = Path(__file__).resolve().parents[1] / "src" / "coding_review_agent_loop"
FAMILY_ROOT = ("errors", "AgentLoopError")
DEGRADATION_BUILD = "ParseDegradation.build"
ANNOTATION_RE = re.compile(r"shape-check:\s*(\S+)")
CONSTRUCTOR_METHODS = ("__init__", "__post_init__", "__new__")


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------


@dataclass
class ModuleInfo:
    name: str
    tree: ast.Module
    classes: dict[str, ast.ClassDef]
    functions: set[str]
    # alias -> (module, original name) for ``from .module import name``.
    package_imports: dict[str, tuple[str, str]]
    # Names bound to imported module objects (``import json``).
    module_objects: set[str]


def _module_info(name: str, source: str) -> ModuleInfo:
    tree = ast.parse(source)
    classes: dict[str, ast.ClassDef] = {}
    functions: set[str] = set()
    package_imports: dict[str, tuple[str, str]] = {}
    module_objects: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            classes[node.name] = node
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.add(node.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 1 and node.module and "." not in node.module:
                for alias in node.names:
                    package_imports[alias.asname or alias.name] = (node.module, alias.name)
            elif node.level == 0:
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    # ``from dataclasses import dataclass``: outside the package.
                    module_objects.discard(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                module_objects.add((alias.asname or alias.name).split(".")[0])
    return ModuleInfo(name, tree, classes, functions, package_imports, module_objects)


class PackageIndex:
    """Lazily parsed package modules, used to resolve classes and imports."""

    def __init__(self, overrides: dict[str, str] | None = None) -> None:
        self._overrides = dict(overrides or {})
        self._modules: dict[str, ModuleInfo] = {}
        self._ancestors: dict[tuple[str, str], frozenset[tuple[str, str]]] = {}

    def source(self, name: str) -> str | None:
        if name in self._overrides:
            return self._overrides[name]
        path = PACKAGE_DIR / f"{name}.py"
        return path.read_text(encoding="utf-8") if path.exists() else None

    def module(self, name: str) -> ModuleInfo | None:
        if name not in self._modules:
            source = self.source(name)
            if source is None:
                return None
            self._modules[name] = _module_info(name, source)
        return self._modules[name]

    def resolve_class(self, module: str, name: str) -> tuple[str, str] | None:
        """Resolve a bare class name in ``module`` to (module, class) or builtins."""
        info = self.module(module)
        if info is not None:
            if name in info.classes:
                return (module, name)
            if name in info.package_imports:
                source_module, original = info.package_imports[name]
                return self.resolve_class(source_module, original)
        candidate = getattr(builtins, name, None)
        if isinstance(candidate, type) and issubclass(candidate, BaseException):
            return ("builtins", name)
        return None

    def ancestors(self, key: tuple[str, str]) -> frozenset[tuple[str, str]]:
        """Every class ``key`` is a subclass of, including itself."""
        if key in self._ancestors:
            return self._ancestors[key]
        self._ancestors[key] = frozenset({key})
        module, name = key
        result: set[tuple[str, str]] = {key}
        if module == "builtins":
            for base in getattr(builtins, name).__mro__:
                result.add(("builtins", base.__name__))
        else:
            info = self.module(module)
            node = info.classes.get(name) if info is not None else None
            for base in node.bases if node is not None else ():
                base_name = base.id if isinstance(base, ast.Name) else (
                    base.attr if isinstance(base, ast.Attribute) else None
                )
                if base_name is None:
                    continue
                resolved = self.resolve_class(module, base_name)
                if resolved is not None:
                    result |= self.ancestors(resolved)
        self._ancestors[key] = frozenset(result)
        return self._ancestors[key]

    def is_family(self, key: tuple[str, str]) -> bool:
        return FAMILY_ROOT in self.ancestors(key)

    def covers(self, handler: tuple[str, str], raised: tuple[str, str]) -> bool:
        return handler in self.ancestors(raised)


@dataclass
class Handler:
    # ``None`` means a bare ``except:``.
    classes: tuple[tuple[str, str], ...] | None
    contains_raise: bool


@dataclass
class Site:
    unit: str
    line: int
    col: int
    kind: str  # "raise", "call" or "imported"
    targets: tuple[str, ...]
    direct_classes: frozenset[tuple[str, str]]
    tries: tuple[tuple[Handler, ...], ...]
    classes: frozenset[tuple[str, str]] = frozenset()
    handled: bool = False
    builds_degradation: bool = False

    @property
    def label(self) -> str:
        return f"{self.unit}:{self.line} ({self.kind} {', '.join(self.targets) or 'raise'})"


@dataclass
class Unit:
    name: str
    node: ast.AST
    class_name: str | None
    scope_chain: tuple[str, ...]  # enclosing function units, innermost first
    sites: list[Site] = field(default_factory=list)
    edges: set[str] = field(default_factory=set)
    builds_degradation: bool = False


@dataclass
class Inventory:
    module: str
    units: dict[str, Unit]
    sites: list[Site]
    raise_sets: dict[str, frozenset[tuple[str, str]]]
    annotations: dict[int, str]
    comment_only_lines: set[int]
    imported_functions: dict[str, tuple[str, str]]

    @property
    def raise_reaching(self) -> set[str]:
        return {name for name, classes in self.raise_sets.items() if classes}

    def propagating(self, unit: str | None = None) -> list[Site]:
        return [
            site for site in self.sites
            if site.classes and not site.handled and (unit is None or site.unit == unit)
        ]

    def handled(self, unit: str | None = None) -> list[Site]:
        return [
            site for site in self.sites
            if site.classes and site.handled and (unit is None or site.unit == unit)
        ]

    def annotation(self, line: int) -> str | None:
        """The annotation on ``line`` or in the comment block directly above it."""
        if line in self.annotations:
            return self.annotations[line]
        cursor = line - 1
        while cursor in self.comment_only_lines:
            if cursor in self.annotations:
                return self.annotations[cursor]
            cursor -= 1
        return None


def _comment_annotations(source: str) -> tuple[dict[int, str], set[int]]:
    annotations: dict[int, str] = {}
    code_lines: set[int] = set()
    comment_lines: set[int] = set()
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            comment_lines.add(token.start[0])
            match = ANNOTATION_RE.search(token.string)
            if match:
                annotations[token.start[0]] = match.group(1)
        elif token.type not in {
            tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT,
            tokenize.ENDMARKER,
        }:
            for line in range(token.start[0], token.end[0] + 1):
                code_lines.add(line)
    return annotations, comment_lines - code_lines


def _enumerate_units(info: ModuleInfo) -> dict[str, Unit]:
    units: dict[str, Unit] = {}

    def visit_body(
        body: list[ast.stmt],
        *,
        prefix: str,
        class_name: str | None,
        scope_chain: tuple[str, ...],
        in_class: bool,
    ) -> None:
        for node in body:
            walk_defs(node, prefix=prefix, class_name=class_name, scope_chain=scope_chain, in_class=in_class)

    def walk_defs(node, *, prefix, class_name, scope_chain, in_class) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = f"{prefix}{node.name}"
            units[name] = Unit(name, node, class_name, (name, *scope_chain))
            visit_body(
                node.body,
                prefix=f"{name}.",
                class_name=class_name,
                scope_chain=(name, *scope_chain),
                in_class=False,
            )
            return
        if isinstance(node, ast.ClassDef):
            visit_body(
                node.body,
                prefix=f"{prefix}{node.name}.",
                class_name=node.name if not scope_chain else class_name,
                # Class scopes do not enclose their methods' name lookups.
                scope_chain=scope_chain,
                in_class=True,
            )
            return
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.stmt):
                walk_defs(child, prefix=prefix, class_name=class_name, scope_chain=scope_chain, in_class=in_class)

    visit_body(info.tree.body, prefix="", class_name=None, scope_chain=(), in_class=False)
    return units


class _UnitWalker:
    def __init__(self, collector: "Collector", unit: Unit) -> None:
        self.collector = collector
        self.unit = unit

    def run(self) -> None:
        node = self.unit.node
        for statement in node.body:
            self.visit(statement, (), None)

    def visit(self, node: ast.AST, tries: tuple, handler: ast.ExceptHandler | None) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # A nested def's body is its own unit; its decorators and
            # defaults execute here.
            for child in [*node.decorator_list, *node.args.defaults, *node.args.kw_defaults]:
                if child is not None:
                    self.visit(child, tries, handler)
            return
        if isinstance(node, ast.ClassDef):
            for child in [*node.decorator_list, *node.bases, *[kw.value for kw in node.keywords]]:
                self.visit(child, tries, handler)
            return
        if isinstance(node, (ast.Try, getattr(ast, "TryStar", ast.Try))):
            specs = tuple(self.collector.handler_spec(item) for item in node.handlers)
            for statement in node.body:
                self.visit(statement, (specs, *tries), handler)
            for item in node.handlers:
                if item.type is not None:
                    self.visit(item.type, tries, handler)
                for statement in item.body:
                    self.visit(statement, tries, item)
            for statement in [*node.orelse, *node.finalbody]:
                self.visit(statement, tries, handler)
            return
        if isinstance(node, ast.Raise):
            classes = self.collector.raise_classes(node, handler)
            if classes:
                self.add_site(node, "raise", (), classes, tries)
            for child in (node.exc, node.cause):
                if child is not None:
                    self.visit(child, tries, handler)
            return
        if isinstance(node, ast.Call):
            self.visit_call(node, tries, handler)
            return
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            self.visit_reference(node, tries)
            return
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
            targets = self.collector.class_attribute_targets(self.unit, node)
            if targets:
                self.add_edge_site(node, targets, tries)
            self.visit(node.value, tries, handler)
            return
        for child in ast.iter_child_nodes(node):
            self.visit(child, tries, handler)

    def visit_call(self, node: ast.Call, tries: tuple, handler) -> None:
        func = node.func
        if isinstance(func, ast.Name):
            targets, imported = self.collector.name_targets(self.unit, func.id, called=True)
            if targets:
                self.add_edge_site(node, targets, tries)
            if imported is not None:
                self.add_imported_site(node, imported, tries)
        elif isinstance(func, ast.Attribute):
            targets = self.collector.attribute_call_targets(self.unit, func)
            if targets:
                site = self.add_edge_site(node, targets, tries)
                if DEGRADATION_BUILD in targets:
                    site.builds_degradation = True
                    self.unit.builds_degradation = True
            value = func.value
            if not (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id == "super"
            ):
                self.visit(value, tries, handler)
        else:
            self.visit(func, tries, handler)
        for child in [*node.args, *[kw.value for kw in node.keywords]]:
            self.visit(child, tries, handler)

    def visit_reference(self, node: ast.Name, tries: tuple) -> None:
        targets, imported = self.collector.name_targets(self.unit, node.id, called=False)
        if targets:
            self.add_edge_site(node, targets, tries)
        if imported is not None:
            self.add_imported_site(node, imported, tries)

    def add_edge_site(self, node, targets: tuple[str, ...], tries) -> Site:
        self.unit.edges.update(targets)
        return self.add_site(node, "call", targets, frozenset(), tries)

    def add_imported_site(self, node, imported: str, tries) -> None:
        self.add_site(node, "imported", (imported,), self.collector.imported_classes(imported), tries)

    def add_site(self, node, kind, targets, classes, tries) -> Site:
        site = Site(self.unit.name, node.lineno, node.col_offset, kind, tuple(targets), classes, tries)
        self.unit.sites.append(site)
        return site


class Collector:
    """Inventory one module: units, sites, handled/propagating, raise sets."""

    def __init__(
        self,
        module: str,
        source: str,
        *,
        index: PackageIndex | None = None,
        imported_raisers: dict[str, frozenset[str]] | None = None,
        imported_cache: dict[tuple[str, str], frozenset[tuple[str, str]]] | None = None,
    ) -> None:
        self.module = module
        self.source = source
        self.index = index or PackageIndex({module: source})
        self.info = self.index.module(module)
        self.units = _enumerate_units(self.info)
        # Imported function name -> (source module, original name).
        self.imported_functions: dict[str, tuple[str, str]] = {}
        for alias, (source_module, original) in self.info.package_imports.items():
            source_info = self.index.module(source_module)
            if source_info is not None and original in source_info.functions:
                self.imported_functions[alias] = (source_module, original)
        self._imported_raisers = imported_raisers
        self._imported_cache = imported_cache if imported_cache is not None else {}
        self._class_methods: dict[str, dict[str, str]] = {}
        self._methods_by_name: dict[str, list[str]] = {}
        for name in self.units:
            parts = name.split(".")
            if len(parts) == 2 and parts[0] in self.info.classes:
                self._class_methods.setdefault(parts[0], {})[parts[1]] = name
                self._methods_by_name.setdefault(parts[1], []).append(name)

    # -- class resolution -------------------------------------------------

    def resolve_class(self, name: str) -> tuple[str, str] | None:
        return self.index.resolve_class(self.module, name)

    def handler_spec(self, handler: ast.ExceptHandler) -> Handler:
        contains_raise = any(isinstance(node, ast.Raise) for node in _walk_without_defs(handler.body))
        if handler.type is None:
            return Handler(None, contains_raise)
        names = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
        classes: list[tuple[str, str]] = []
        for item in names:
            name = item.id if isinstance(item, ast.Name) else (
                item.attr if isinstance(item, ast.Attribute) else None
            )
            resolved = self.resolve_class(name) if name else None
            if resolved is not None:
                classes.append(resolved)
        return Handler(tuple(classes), contains_raise)

    def raise_classes(self, node: ast.Raise, handler: ast.ExceptHandler | None) -> frozenset:
        if node.exc is None:
            if handler is None:
                return frozenset()
            spec = self.handler_spec(handler)
            if spec.classes is None:
                return frozenset({FAMILY_ROOT})
            caught: set[tuple[str, str]] = set()
            for key in spec.classes:
                if self.index.is_family(key):
                    caught.add(key)
                elif key in self.index.ancestors(FAMILY_ROOT):
                    caught.add(FAMILY_ROOT)
            return frozenset(caught)
        exc = node.exc
        target = exc.func if isinstance(exc, ast.Call) else exc
        if isinstance(target, ast.Name):
            resolved = self.resolve_class(target.id)
            if resolved is not None:
                return frozenset({resolved}) if self.index.is_family(resolved) else frozenset()
        # A bound name, attribute or other expression: conservatively the root.
        return frozenset({FAMILY_ROOT})

    # -- call graph ---------------------------------------------------------

    def _mro_lookup(self, class_name: str, attr: str, *, skip_self: bool = False) -> tuple[str, ...]:
        seen: set[str] = set()
        order: list[str] = []

        def linearize(name: str) -> None:
            if name in seen or name not in self.info.classes:
                return
            seen.add(name)
            order.append(name)
            for base in self.info.classes[name].bases:
                if isinstance(base, ast.Name):
                    linearize(base.id)

        linearize(class_name)
        for name in order[1:] if skip_self else order:
            method = self._class_methods.get(name, {}).get(attr)
            if method is not None:
                return (method,)
        return ()

    def _constructor_targets(self, class_name: str) -> tuple[str, ...]:
        targets: list[str] = []
        for method in CONSTRUCTOR_METHODS:
            targets.extend(self._mro_lookup(class_name, method))
        return tuple(dict.fromkeys(targets))

    def name_targets(self, unit: Unit, name: str, *, called: bool) -> tuple[tuple[str, ...], str | None]:
        for scope in unit.scope_chain:
            candidate = f"{scope}.{name}"
            if candidate in self.units:
                return (candidate,), None
        if name in self.units:
            return (name,), None
        if name in self.info.classes:
            return (self._constructor_targets(name) if called else ()), None
        if name in self.imported_functions:
            return (), name
        return (), None

    def class_attribute_targets(self, unit: Unit, node: ast.Attribute) -> tuple[str, ...]:
        value = node.value
        if isinstance(value, ast.Name) and value.id in self.info.classes:
            return self._mro_lookup(value.id, node.attr)
        return ()

    def attribute_call_targets(self, unit: Unit, func: ast.Attribute) -> tuple[str, ...]:
        value = func.value
        if isinstance(value, ast.Name):
            if value.id in self.info.classes:
                return self._mro_lookup(value.id, func.attr)
            if value.id in {"cls", "self"} and unit.class_name:
                return self._mro_lookup(unit.class_name, func.attr)
            if value.id in self.info.module_objects:
                # Calls through other modules' attributes are outside the
                # inventory (a recorded caveat).
                return ()
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "super"
            and unit.class_name
        ):
            return self._mro_lookup(unit.class_name, func.attr, skip_self=True)
        # Untyped receiver: an edge to every same-module method of that name.
        return tuple(self._methods_by_name.get(func.attr, ()))

    # -- imported raisers -------------------------------------------------

    def imported_classes(self, alias: str) -> frozenset[tuple[str, str]]:
        source_module, original = self.imported_functions[alias]
        if self._imported_raisers is not None:
            recorded = self._imported_raisers.get(alias, frozenset())
            resolved = set()
            for name in recorded:
                key = self.resolve_class(name) or self.index.resolve_class(source_module, name)
                resolved.add(key if key is not None else ("errors", name))
            return frozenset(resolved)
        return computed_imported_classes(self.index, source_module, original, self._imported_cache)

    # -- inventory ----------------------------------------------------------

    def site_handled(self, site: Site) -> bool:
        for handlers in site.tries:
            covered = True
            for raised in site.classes:
                covering = [
                    handler for handler in handlers
                    if handler.classes is None
                    or any(self.index.covers(caught, raised) for caught in handler.classes)
                ]
                if not covering or any(handler.contains_raise for handler in covering):
                    covered = False
                    break
            if covered:
                return True
        return False

    def collect(self) -> Inventory:
        for unit in self.units.values():
            _UnitWalker(self, unit).run()
        sites = [site for unit in self.units.values() for site in unit.sites]
        raise_sets: dict[str, frozenset] = {name: frozenset() for name in self.units}
        changed = True
        while changed:
            changed = False
            for site in sites:
                if site.kind == "call":
                    classes = frozenset().union(*(raise_sets.get(target, frozenset()) for target in site.targets))
                else:
                    classes = site.direct_classes
                site.classes = classes
                site.handled = bool(classes) and self.site_handled(site)
            for name, unit in self.units.items():
                total = frozenset().union(
                    *(site.classes for site in unit.sites if site.classes and not site.handled)
                )
                if total != raise_sets[name]:
                    raise_sets[name] = total
                    changed = True
        annotations, comment_only = _comment_annotations(self.source)
        return Inventory(
            self.module, self.units, [site for site in sites if site.classes],
            raise_sets, annotations, comment_only, dict(self.imported_functions),
        )


def _walk_without_defs(body):
    stack = list(body)
    while stack:
        node = stack.pop()
        yield node
        for child in ast.iter_child_nodes(node):
            if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                stack.append(child)


def computed_imported_classes(index, module, name, cache) -> frozenset[tuple[str, str]]:
    """Transitive raise classes of ``module.name``, recursing into its imports."""
    key = (module, name)
    if key in cache:
        return cache[key]
    cache[key] = frozenset()  # cycle guard
    inventory = Collector(module, index.source(module), index=index, imported_cache=cache).collect()
    cache[key] = inventory.raise_sets.get(name, frozenset())
    return cache[key]


def collect_module(module: str, source: str, **kwargs) -> Inventory:
    return Collector(module, source, **kwargs).collect()


def class_label(key: tuple[str, str]) -> str:
    return key[1]


# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------


def audit_problems(
    inventory: Inventory,
    *,
    registry: dict[str, ShapeCheckClassification],
    degrading_parsers: tuple[str, ...],
    imported_raisers: dict[str, frozenset[str]],
    imported_non_raisers: frozenset[str],
    index: PackageIndex | None = None,
) -> list[str]:
    """Every way the registry and annotations disagree with the inventory."""
    problems: list[str] = []
    reaching = inventory.raise_reaching
    for name in sorted(reaching - set(registry)):
        problems.append(f"unregistered raise-reaching unit: {name}")
    for name in sorted(set(registry) - reaching):
        problems.append(f"stale registry entry (not raise-reaching): {name}")
    for name, entry in registry.items():
        if entry.disposition not in {"fatal", "mixed", "delegated"}:
            problems.append(f"{name}: unknown disposition {entry.disposition!r}")
        if not entry.justification.strip():
            problems.append(f"{name}: blank justification")
        for clause in entry.clauses:
            if clause not in FATAL_CLAUSES:
                problems.append(f"{name}: clause {clause!r} is not in FATAL_CLAUSES")
    mixed_units = {name for name, entry in registry.items() if entry.disposition == "mixed"}
    for site in inventory.handled():
        value = inventory.annotation(site.line)
        if value != "handled":
            problems.append(f"handled site without `handled` annotation: {site.label} (found {value!r})")
    for name, entry in registry.items():
        if name not in inventory.units:
            continue
        propagating = inventory.propagating(name)
        annotated: set[str] = set()
        for site in propagating:
            value = inventory.annotation(site.line)
            if entry.disposition == "delegated":
                if value != "delegated":
                    problems.append(f"delegated unit site not annotated `delegated`: {site.label} (found {value!r})")
                continue
            if value is None or not value.startswith("fatal:"):
                problems.append(f"propagating site without a fatal clause: {site.label} (found {value!r})")
                continue
            clause = value.split(":", 1)[1]
            if clause not in FATAL_CLAUSES:
                problems.append(f"{site.label}: clause {clause!r} is not in FATAL_CLAUSES")
            annotated.add(clause)
        unit = inventory.units[name]
        degradation_path = (
            bool(inventory.handled(name))
            or bool(unit.edges & set(degrading_parsers))
            or bool(unit.edges & (mixed_units - {name}))
            or unit.builds_degradation
        )
        if entry.disposition == "fatal":
            if set(entry.clauses) != annotated:
                problems.append(
                    f"{name}: fatal clauses {sorted(entry.clauses)} differ from annotated {sorted(annotated)}"
                )
            if degradation_path:
                problems.append(f"{name}: registered fatal but has a degradation path")
        if entry.disposition == "mixed":
            if not annotated:
                problems.append(f"{name}: registered mixed without a propagating fatal site")
            if not degradation_path:
                problems.append(f"{name}: registered mixed without a degradation path")
            if set(entry.clauses) != annotated:
                problems.append(
                    f"{name}: mixed clauses {sorted(entry.clauses)} differ from annotated {sorted(annotated)}"
                )
    # A handled or fatal annotation on the wrong kind of site.
    site_lines_handled = {site.line for site in inventory.handled()}
    for site in inventory.propagating():
        if inventory.annotation(site.line) == "handled" and site.line not in site_lines_handled:
            problems.append(f"`handled` annotation on a propagating site: {site.label}")
    for site in inventory.handled():
        value = inventory.annotation(site.line)
        if value is not None and value.startswith("fatal:"):
            problems.append(f"`fatal:` annotation on a handled site: {site.label}")
    # Delegated units are reached only through classified units.
    delegated = {name for name, entry in registry.items() if entry.disposition == "delegated"}
    for site in inventory.propagating():
        if set(site.targets) & delegated and site.unit not in registry:
            problems.append(f"delegated helper reached from unclassified unit: {site.label}")
    for name in degrading_parsers:
        if name not in inventory.units:
            problems.append(f"DEGRADING_PARSERS entry does not exist: {name}")
            continue
        if inventory.propagating(name):
            problems.append(f"DEGRADING_PARSERS entry has a propagating site: {name}")
        if inventory.units[name].builds_degradation or DEGRADATION_BUILD in inventory.units[name].edges:
            problems.append(f"DEGRADING_PARSERS entry builds ParseDegradation records: {name}")
        if name in registry:
            problems.append(f"{name} is both registered and a degrading parser")
    for site in inventory.propagating():
        if site.builds_degradation and inventory.annotation(site.line) != "fatal:authentication-or-forgery":
            problems.append(f"record construction not annotated authentication-or-forgery: {site.label}")
    for site in inventory.handled():
        if site.builds_degradation:
            problems.append(f"record construction is swallowed by a handler: {site.label}")
    # Imported tables.
    imported = set(inventory.imported_functions)
    for name in sorted(imported - set(imported_raisers) - set(imported_non_raisers)):
        problems.append(f"imported function in neither imported table: {name}")
    for name in sorted(set(imported_raisers) & set(imported_non_raisers)):
        problems.append(f"imported function in both imported tables: {name}")
    index = index or PackageIndex()
    cache: dict = {}
    for name in sorted(imported & (set(imported_raisers) | set(imported_non_raisers))):
        source_module, original = inventory.imported_functions[name]
        computed = {class_label(key) for key in computed_imported_classes(index, source_module, original, cache)}
        recorded = set(imported_raisers.get(name, frozenset()))
        if computed != recorded:
            problems.append(
                f"imported function {name}: recorded {sorted(recorded)} but source computes {sorted(computed)}"
            )
    return problems


def _protocol_source() -> str:
    return (PACKAGE_DIR / "protocol.py").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def protocol_inventory() -> Inventory:
    return collect_module("protocol", _protocol_source(), imported_raisers=IMPORTED_RAISERS)


def _real_problems(inventory, **overrides) -> list[str]:
    kwargs = dict(
        registry=SHAPE_CHECK_AUDIT,
        degrading_parsers=DEGRADING_PARSERS,
        imported_raisers=IMPORTED_RAISERS,
        imported_non_raisers=IMPORTED_NON_RAISERS,
    )
    kwargs.update(overrides)
    return audit_problems(inventory, **kwargs)


def test_protocol_registry_matches_the_transitive_inventory(protocol_inventory):
    problems = _real_problems(protocol_inventory)
    assert problems == [], "\n".join(problems[:60])


def test_registry_keys_equal_raise_reaching_set(protocol_inventory):
    assert set(SHAPE_CHECK_AUDIT) == protocol_inventory.raise_reaching


def test_no_name_in_both_collections():
    assert not set(SHAPE_CHECK_AUDIT) & set(DEGRADING_PARSERS)
    assert DEGRADING_PARSERS == ("_test_observation_item_defect",)


def test_fatal_clause_set_is_closed():
    assert FATAL_CLAUSES == frozenset({
        "unparseable-envelope", "kind-or-version-mismatch", "authentication-or-forgery",
        "payload-bound", "authority-decision", "orchestrator-authored", "no-conservative-reading",
    })


def test_strict_canonical_citation_path_is_pinned_fatal():
    for name in ("_parse_risk_evidence_citations", "_expect_test_observations"):
        entry = SHAPE_CHECK_AUDIT[name]
        assert entry.disposition == "fatal"
        assert "authentication-or-forgery" in entry.clauses


def test_named_units_are_registered_under_qualified_keys(protocol_inventory):
    for name in (
        "_normalize_architecture_status",
        "_expect_typed_plan_stages.parse_child_stages",
        "_expect_typed_plan_stages.recorded_stages",
    ):
        assert name in SHAPE_CHECK_AUDIT
        assert protocol_inventory.propagating(name)
    raises = [
        site for site in protocol_inventory.propagating("_normalize_architecture_status")
        if site.kind == "raise"
    ]
    assert len(raises) == 2


def test_claim_row_id_helper_is_mixed_with_a_handled_row_id_call(protocol_inventory):
    assert SHAPE_CHECK_AUDIT["_claim_row_id_or_degradation"].disposition == "mixed"
    [handled] = protocol_inventory.handled("_claim_row_id_or_degradation")
    assert handled.targets == ("_validate_risk_row_id",)
    raises = [
        site for site in protocol_inventory.propagating("_claim_row_id_or_degradation")
        if site.kind == "raise"
    ]
    assert raises
    assert all(protocol_inventory.annotation(site.line) == "fatal:payload-bound" for site in raises)


def test_approved_followups_rewrapped_fix_scope_propagates(protocol_inventory):
    sites = protocol_inventory.propagating("_expect_review_finding_list")
    imported = [site for site in sites if site.kind == "imported" and site.targets == ("normalize_fix_scope",)]
    reraises = [site for site in sites if site.kind == "raise"]
    assert imported and reraises
    for site in [*imported, *reraises]:
        assert protocol_inventory.annotation(site.line) == "fatal:no-conservative-reading"


def test_managed_test_command_call_is_handled(protocol_inventory):
    assert IMPORTED_RAISERS["parse_managed_test_command"] == frozenset({"TestRuntimeConfigurationError"})
    [site] = protocol_inventory.handled("_managed_test_wrapper_inner_command")
    assert site.targets == ("parse_managed_test_command",)
    assert "_managed_test_wrapper_inner_command" not in SHAPE_CHECK_AUDIT


def test_post_init_value_errors_are_not_sites(protocol_inventory):
    post_init = [name for name in protocol_inventory.units if name.endswith(".__post_init__")]
    for name in post_init:
        assert not [site for site in protocol_inventory.propagating(name) if site.kind == "raise"]


def test_sanitize_historical_text_is_an_imported_raiser(protocol_inventory):
    assert IMPORTED_RAISERS["sanitize_historical_text"] == frozenset({"AgentLoopError"})
    calls = [
        site for site in protocol_inventory.sites
        if site.kind == "imported" and site.targets == ("sanitize_historical_text",)
    ]
    assert len(calls) >= 40
    for site in calls:
        assert not site.handled
        assert protocol_inventory.annotation(site.line) == "fatal:authentication-or-forgery"
    assert "ParseDegradation.to_payload" in SHAPE_CHECK_AUDIT


def test_every_degradation_builder_is_registered_and_annotated(protocol_inventory):
    builders = sorted({site.unit for site in protocol_inventory.sites if site.builds_degradation})
    for name in (
        "classify_architecture_status_near_miss",
        "_parse_semantic_risk_coverage_claims",
        "_degradable_test_observations",
    ):
        assert name in builders
    for name in builders:
        assert SHAPE_CHECK_AUDIT[name].disposition == "mixed"
    for name in ("ParseDegradation.build", "_bounded_single_line"):
        entry = SHAPE_CHECK_AUDIT[name]
        assert entry.disposition == "fatal"
        assert entry.clauses == frozenset({"authentication-or-forgery"})


def test_degradable_test_observations_sites_are_records_and_the_drop_bound(protocol_inventory):
    entry = SHAPE_CHECK_AUDIT["_degradable_test_observations"]
    assert entry.disposition == "mixed"
    clauses = {
        protocol_inventory.annotation(site.line)
        for site in protocol_inventory.propagating("_degradable_test_observations")
    }
    assert clauses == {"fatal:authentication-or-forgery", "fatal:payload-bound"}
    assert "_test_observation_item_defect" in protocol_inventory.units["_degradable_test_observations"].edges


def test_dropped_value_bound_precedes_every_record_build(protocol_inventory):
    for name in ("_parse_semantic_risk_coverage_claims", "_degradable_test_observations"):
        sites = sorted(protocol_inventory.propagating(name), key=lambda site: site.line)
        bound_lines = [site.line for site in sites if "_check_dropped_value_bound" in site.targets]
        build_lines = [site.line for site in sites if site.builds_degradation]
        assert build_lines
        for line in build_lines:
            assert any(bound < line for bound in bound_lines), (name, line)


def test_architecture_degradation_chain_is_mixed():
    for name in (
        "classify_architecture_status_near_miss",
        "_parse_architecture_impact_degradable",
        "parse_architecture_impact_degradable",
        "_degradable_response_impact",
        "parse_structured_pr_review",
        "parse_structured_plan_review",
        "validate_structured_coder_followup",
        "validate_structured_issue_implementation",
        "validate_structured_task_result",
        "validate_structured_plan_revision",
        "validate_structured_plan_state",
        "_parse_semantic_risk_coverage_claims",
    ):
        assert SHAPE_CHECK_AUDIT[name].disposition == "mixed", name
    for name in (
        "_parse_architecture_impact",
        "_parse_architecture_impact_payload",
        "_normalize_architecture_status",
    ):
        assert SHAPE_CHECK_AUDIT[name].disposition == "fatal", name


@pytest.mark.parametrize("name", ["classify_architecture_status_near_miss", "parse_structured_pr_review"])
def test_registering_a_mixed_architecture_unit_fatal_fails(protocol_inventory, name):
    clauses = {
        protocol_inventory.annotation(site.line).split(":", 1)[1]
        for site in protocol_inventory.propagating(name)
    }
    registry = dict(SHAPE_CHECK_AUDIT)
    registry[name] = ShapeCheckClassification("fatal", frozenset(clauses), "forced fatal")
    problems = _real_problems(protocol_inventory, registry=registry)
    assert f"{name}: registered fatal but has a degradation path" in problems


def test_render_limit_is_shared_and_bounds_citation_drops():
    from coding_review_agent_loop import comment_rendering, protocol

    assert comment_rendering.PARSE_DEGRADATION_RENDER_LIMIT is protocol.PARSE_DEGRADATION_RENDER_LIMIT
    assert protocol.CITATION_DEGRADATION_MAX_DROPS <= protocol.PARSE_DEGRADATION_RENDER_LIMIT


def test_reserved_authority_keys_are_exact():
    from coding_review_agent_loop.protocol import CLAIM_RESERVED_AUTHORITY_KEYS

    assert CLAIM_RESERVED_AUTHORITY_KEYS == frozenset(
        {"status", "evidence_citations", "receipt_id", "command", "claim"}
    )


def test_import_order_has_no_cycle():
    import subprocess
    import sys

    for order in (
        "import coding_review_agent_loop.comment_rendering, coding_review_agent_loop.protocol",
        "import coding_review_agent_loop.protocol, coding_review_agent_loop.comment_rendering",
    ):
        subprocess.run(
            [sys.executable, "-c", order + "; import coding_review_agent_loop.comment_rendering as c; c.PARSE_DEGRADATION_RENDER_LIMIT"],
            check=True,
            env={"PYTHONPATH": str(PACKAGE_DIR.parent)},
        )


# ---------------------------------------------------------------------------
# In-memory guard cases: the guard must detect each defect.
# ---------------------------------------------------------------------------

_HEADER = '''
from .errors import AgentLoopError, NonRepairableEvidenceRejection
from .protocol_markers import sanitize_historical_text


class ParseDegradation:
    @classmethod
    def build(cls, *, text):
        return cls(sanitize_historical_text(text))  # shape-check: fatal:authentication-or-forgery


def helper(value):
    if not value:
        raise AgentLoopError("empty")  # shape-check: delegated
    return value
'''

_BASE_REGISTRY = {
    "ParseDegradation.build": ShapeCheckClassification(
        "fatal", frozenset({"authentication-or-forgery"}), "Marker neutralization of the preview."
    ),
    "helper": ShapeCheckClassification("delegated", frozenset(), "Generic helper."),
}
_TABLES = {"sanitize_historical_text": frozenset({"AgentLoopError"})}


def _synthetic_problems(body: str, registry: dict | None = None, *, degrading=(), raisers=None, non_raisers=frozenset()):
    source = _HEADER + body
    index = PackageIndex({"synthetic": source})
    tables = _TABLES if raisers is None else raisers
    inventory = collect_module("synthetic", source, index=index, imported_raisers=tables)
    full = dict(_BASE_REGISTRY)
    full.update(registry or {})
    return inventory, audit_problems(
        inventory,
        registry=full,
        degrading_parsers=tuple(degrading),
        imported_raisers=tables,
        imported_non_raisers=frozenset(non_raisers),
        index=index,
    )


def _fatal(*clauses):
    return ShapeCheckClassification("fatal", frozenset(clauses), "Test justification.")


def _mixed(*clauses):
    return ShapeCheckClassification("mixed", frozenset(clauses), "Test justification.")


def test_synthetic_baseline_passes():
    _inventory, problems = _synthetic_problems("")
    assert problems == []


def test_guard_detects_a_new_raise_in_an_unregistered_function():
    _inv, problems = _synthetic_problems('''
def parse(value):
    raise AgentLoopError("bad")
''')
    assert "unregistered raise-reaching unit: parse" in problems


def test_guard_detects_an_unregistered_pure_delegator():
    _inv, problems = _synthetic_problems('''
def parse(value):
    return helper(value)
''')
    assert "unregistered raise-reaching unit: parse" in problems


def test_guard_detects_a_stale_registry_entry():
    _inv, problems = _synthetic_problems('''
def parse(value):
    return value
''', {"parse": _fatal("payload-bound")})
    assert "stale registry entry (not raise-reaching): parse" in problems


def test_guard_detects_a_degrading_parser_that_gains_a_propagating_call():
    _inv, problems = _synthetic_problems('''
def classify(value):
    return helper(value)
''', {"classify": _fatal("payload-bound")}, degrading=("classify",))
    assert "DEGRADING_PARSERS entry has a propagating site: classify" in problems


def test_guard_detects_an_unannotated_helper_call_in_a_mixed_function():
    _inv, problems = _synthetic_problems('''
def classify(value):
    return value is None


def parse(value):
    classify(value)
    helper(value)
    raise AgentLoopError("x")  # shape-check: fatal:payload-bound
''', {"parse": _mixed("payload-bound")}, degrading=("classify",))
    assert any("propagating site without a fatal clause: parse:" in item for item in problems)


def test_guard_detects_an_unannotated_raise_in_a_fatal_function():
    _inv, problems = _synthetic_problems('''
def parse(value):
    raise AgentLoopError("bad")
''', {"parse": _fatal("payload-bound")})
    assert any("propagating site without a fatal clause: parse:" in item for item in problems)


def test_guard_detects_a_clause_outside_the_closed_set():
    _inv, problems = _synthetic_problems('''
def parse(value):
    raise AgentLoopError("bad")  # shape-check: fatal:made-up
''', {"parse": _fatal("made-up")})
    assert any("is not in FATAL_CLAUSES" in item for item in problems)


@pytest.mark.parametrize("clauses", [(), ("payload-bound", "unparseable-envelope")])
def test_guard_detects_a_fatal_clause_set_mismatch(clauses):
    _inv, problems = _synthetic_problems('''
def parse(value):
    raise AgentLoopError("bad")  # shape-check: fatal:payload-bound
''', {"parse": _fatal(*clauses)})
    assert any("fatal clauses" in item and "differ from annotated" in item for item in problems)


def test_guard_detects_an_unannotated_site_in_a_delegated_helper():
    _inv, problems = _synthetic_problems('''
def other(value):
    raise AgentLoopError("bad")
''', {"other": ShapeCheckClassification("delegated", frozenset(), "Generic.")})
    assert any("delegated unit site not annotated" in item for item in problems)


def test_guard_detects_a_caught_call_annotated_fatal():
    _inv, problems = _synthetic_problems('''
def parse(value):
    try:
        helper(value)  # shape-check: fatal:payload-bound
    except AgentLoopError:
        return None
    return value
''')
    assert any("handled site without `handled` annotation" in item for item in problems)
    assert any("`fatal:` annotation on a handled site" in item for item in problems)


def test_guard_rejects_handled_annotation_when_the_handler_reraises():
    inventory, problems = _synthetic_problems('''
def parse(value):
    try:
        helper(value)  # shape-check: handled
    except AgentLoopError:
        raise
''', {"parse": _fatal("no-conservative-reading")})
    assert any("`handled` annotation on a propagating site" in item for item in problems)


def test_guard_rejects_handled_annotation_under_an_unrelated_handler():
    _inv, problems = _synthetic_problems('''
def parse(value):
    try:
        helper(value)  # shape-check: handled
    except ValueError:
        return None
''', {"parse": _fatal("no-conservative-reading")})
    assert any("`handled` annotation on a propagating site" in item for item in problems)


def test_guard_rejects_handled_annotation_for_a_narrower_subclass_handler():
    _inv, problems = _synthetic_problems('''
def parse(value):
    try:
        sanitize_historical_text(value)  # shape-check: handled
    except NonRepairableEvidenceRejection:
        return None
''', {"parse": _fatal("authentication-or-forgery")})
    assert any("`handled` annotation on a propagating site" in item for item in problems)


def test_guard_detects_an_unannotated_handled_site():
    _inv, problems = _synthetic_problems('''
def parse(value):
    try:
        helper(value)
    except AgentLoopError:
        return None
    return value
''')
    assert any("handled site without `handled` annotation" in item for item in problems)


def test_a_function_whose_only_raise_reaching_call_is_handled_is_not_a_key():
    inventory, problems = _synthetic_problems('''
def parse(value):
    try:
        helper(value)  # shape-check: handled
    except AgentLoopError:
        return None
    return value
''')
    assert problems == []
    assert "parse" not in inventory.raise_reaching


def test_a_bound_name_raise_is_registered():
    inventory, problems = _synthetic_problems('''
def parse(value):
    error = AgentLoopError("bad")
    raise error
''')
    assert "unregistered raise-reaching unit: parse" in problems
    assert inventory.raise_sets["parse"] == frozenset({FAMILY_ROOT})


def test_a_bare_reraise_counts_as_a_site():
    inventory, _problems = _synthetic_problems('''
def parse(value):
    try:
        return int(value)
    except Exception:
        raise
''')
    assert inventory.raise_sets["parse"] == frozenset({FAMILY_ROOT})


def test_a_nested_def_raise_registers_the_qualified_key():
    inventory, problems = _synthetic_problems('''
def outer(value):
    def inner(item):
        raise AgentLoopError("bad")
    return value
''')
    assert "unregistered raise-reaching unit: outer.inner" in problems
    assert "outer" not in inventory.raise_reaching


def test_a_nested_def_inside_an_outer_try_still_propagates():
    inventory, _problems = _synthetic_problems('''
def outer(value):
    try:
        def inner(item):
            raise AgentLoopError("bad")
    except AgentLoopError:
        return None
    return inner(value)
''')
    [site] = [site for site in inventory.sites if site.unit == "outer.inner"]
    assert not site.handled
    assert "outer" in inventory.raise_reaching


def test_guard_detects_an_import_in_neither_table():
    _inv, problems = _synthetic_problems("", raisers={})
    assert "imported function in neither imported table: sanitize_historical_text" in problems


def test_guard_detects_a_raising_import_listed_as_non_raiser():
    _inv, problems = _synthetic_problems("", raisers={}, non_raisers={"sanitize_historical_text"})
    assert any(
        "imported function sanitize_historical_text: recorded [] but source computes ['AgentLoopError']" in item
        for item in problems
    )


def test_guard_detects_a_wrong_imported_class_set():
    source = "from .test_runtime import parse_managed_test_command\n"
    index = PackageIndex({"synthetic": source})
    tables = {"parse_managed_test_command": frozenset({"AgentLoopError"})}
    inventory = collect_module("synthetic", source, index=index, imported_raisers=tables)
    problems = audit_problems(
        inventory, registry={}, degrading_parsers=(), imported_raisers=tables,
        imported_non_raisers=frozenset(), index=index,
    )
    assert any("parse_managed_test_command" in item and "TestRuntimeConfigurationError" in item for item in problems)


@pytest.mark.parametrize(
    ("handler", "handled"),
    [("NonRepairableEvidenceRejection", True), ("AgentLoopError", True)],
)
def test_named_subclass_raise_is_covered_by_subclass(handler, handled):
    inventory, _problems = _synthetic_problems(f'''
def parse(value):
    try:
        raise NonRepairableEvidenceRejection("bad")
    except {handler}:
        return None
''')
    [site] = [site for site in inventory.sites if site.unit == "parse"]
    assert site.handled is handled
    assert site.classes == frozenset({("errors", "NonRepairableEvidenceRejection")})


def test_a_bound_name_raise_under_a_subclass_handler_propagates():
    inventory, _problems = _synthetic_problems('''
def parse(value):
    error = AgentLoopError("bad")
    try:
        raise error
    except NonRepairableEvidenceRejection:
        return None
''')
    [site] = [site for site in inventory.sites if site.unit == "parse"]
    assert not site.handled


def test_builtin_raises_are_not_sites():
    inventory, _problems = _synthetic_problems('''
def parse(value):
    if value:
        raise ValueError("bad")
    try:
        return int(value)
    except ValueError:
        raise
    raise OSError("x")
''')
    assert "parse" not in inventory.raise_reaching


@pytest.mark.parametrize(
    "call",
    ["Serializer.render(value)", "cls.render(value)", "self.render(value)", "super().render(value)", "value.render()"],
)
def test_same_module_attribute_calls_are_edges(call):
    inventory, _problems = _synthetic_problems(f'''
class Base:
    def render(self, value=None):
        return sanitize_historical_text(str(value))  # shape-check: fatal:authentication-or-forgery


class Serializer(Base):
    @classmethod
    def call_it(cls, value):
        self = cls
        return {call}
''')
    assert "Serializer.call_it" in inventory.raise_reaching


def test_a_degrading_parser_that_builds_records_fails():
    _inv, problems = _synthetic_problems('''
def classify(value):
    try:
        return ParseDegradation.build(text=value)  # shape-check: handled
    except AgentLoopError:
        return None
''', degrading=("classify",))
    assert "DEGRADING_PARSERS entry builds ParseDegradation records: classify" in problems
    assert any("record construction is swallowed by a handler" in item for item in problems)


def test_mixed_unit_with_direct_record_construction_passes():
    _inv, problems = _synthetic_problems('''
def parse(value):
    if value is None:
        return ParseDegradation.build(text="none")  # shape-check: fatal:authentication-or-forgery
    return value
''', {"parse": _mixed("authentication-or-forgery")})
    assert problems == []


def test_mixed_unit_whose_only_path_is_another_mixed_unit_passes():
    _inv, problems = _synthetic_problems('''
def parse(value):
    if value is None:
        return ParseDegradation.build(text="none")  # shape-check: fatal:authentication-or-forgery
    return value


def outer(value):
    return parse(value)  # shape-check: fatal:authentication-or-forgery
''', {"parse": _mixed("authentication-or-forgery"), "outer": _mixed("authentication-or-forgery")})
    assert problems == []


def test_mixed_unit_without_a_degradation_path_fails():
    _inv, problems = _synthetic_problems('''
def parse(value):
    raise AgentLoopError("bad")  # shape-check: fatal:payload-bound
''', {"parse": _mixed("payload-bound")})
    assert "parse: registered mixed without a degradation path" in problems


def test_fatal_unit_with_a_degradation_path_fails():
    _inv, problems = _synthetic_problems('''
def parse(value):
    return ParseDegradation.build(text="x")  # shape-check: fatal:authentication-or-forgery
''', {"parse": _fatal("authentication-or-forgery")})
    assert "parse: registered fatal but has a degradation path" in problems
