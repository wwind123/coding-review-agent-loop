"""Repo-local advisory memory for repeated agent loop runs."""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

from .logging import log
from .runner import Runner
from .workdirs import active_workdir

if TYPE_CHECKING:
    from .config import AgentLoopConfig


ADVISORY_RULE = (
    "Use cached repo memory and execution memory only for orientation. They may be stale. "
    "For correctness, security, and behavior claims, inspect the actual source files and PR "
    "diff directly. If you run tests, prefer verified test commands from the execution profile. "
    "Do not search the whole filesystem for test tools."
)


@dataclass(frozen=True)
class AgentMemoryContext:
    memory_dir: Path
    current_commit: str | None
    last_analyzed_commit: str | None
    changed_files: tuple[str, ...]
    repo_summary: str | None
    architecture_map: str | None
    test_profile: str | None
    toolchain: str | None


def prepare_agent_memory(runner: Runner, config: AgentLoopConfig) -> AgentMemoryContext | None:
    if not config.agent_memory:
        return None

    workdir = active_workdir(config)
    memory_dir = config.agent_memory_dir
    memory_dir.mkdir(parents=True, exist_ok=True)
    (memory_dir / "file-summaries").mkdir(exist_ok=True)

    current_commit = _git_output(runner, workdir, ("rev-parse", "HEAD"))
    last_analyzed_commit = _read_optional(memory_dir / "last-analyzed-commit")
    changed_files = _changed_files(
        runner,
        workdir,
        last_analyzed_commit=last_analyzed_commit,
        current_commit=current_commit,
    )
    tracked_files = _git_lines(runner, workdir, ("ls-files",))

    if config.refresh_agent_memory or not (memory_dir / "repo-summary.md").exists():
        _write_repo_summary(memory_dir / "repo-summary.md", config, current_commit, tracked_files)
    if config.refresh_agent_memory or not (memory_dir / "architecture-map.md").exists():
        _write_architecture_map(memory_dir / "architecture-map.md", tracked_files)
    _write_module_index(memory_dir / "module-index.json", tracked_files, changed_files, current_commit)

    if (
        config.refresh_agent_memory
        or config.refresh_test_profile
        or config.test_command
        or not (memory_dir / "test-profile.md").exists()
    ):
        _write_test_profile(memory_dir / "test-profile.md", config, current_commit, tracked_files)
    if config.refresh_agent_memory or not (memory_dir / "toolchain.json").exists():
        _write_toolchain(memory_dir / "toolchain.json", tracked_files)

    for rel_path in changed_files[:25]:
        if rel_path in tracked_files:
            _write_file_summary(memory_dir / "file-summaries", workdir, rel_path)

    context = load_agent_memory(config)
    if current_commit:
        (memory_dir / "last-analyzed-commit").write_text(f"{current_commit}\n", encoding="utf-8")
    log(config, f"Prepared agent memory at {memory_dir}")
    return context


def load_agent_memory(config: AgentLoopConfig) -> AgentMemoryContext | None:
    if not config.agent_memory:
        return None

    memory_dir = config.agent_memory_dir
    if not memory_dir.exists():
        return None

    module_index = _read_optional(memory_dir / "module-index.json")
    changed_files: tuple[str, ...] = ()
    current_commit: str | None = None
    if module_index:
        try:
            data = json.loads(module_index)
        except json.JSONDecodeError:
            data = {}
        changed_files = tuple(data.get("changed_files") or ())
        current_commit = data.get("current_commit")

    return AgentMemoryContext(
        memory_dir=memory_dir,
        current_commit=current_commit,
        last_analyzed_commit=_read_optional(memory_dir / "last-analyzed-commit"),
        changed_files=changed_files,
        repo_summary=_read_optional(memory_dir / "repo-summary.md"),
        architecture_map=_read_optional(memory_dir / "architecture-map.md"),
        test_profile=_read_optional(memory_dir / "test-profile.md"),
        toolchain=_read_optional(memory_dir / "toolchain.json"),
    )


def format_agent_memory_context(memory: AgentMemoryContext | None) -> str:
    if memory is None:
        return ""

    parts = [
        "Cached repo memory is available for this checkout.",
        "",
        ADVISORY_RULE,
        "",
        f"Memory directory: {memory.memory_dir}",
    ]
    if memory.current_commit:
        parts.append(f"Current memory commit: {memory.current_commit}")
    if memory.last_analyzed_commit:
        parts.append(f"Last analyzed commit: {memory.last_analyzed_commit}")
    if memory.changed_files:
        parts.extend(["", "Changed files since previous memory commit:"])
        parts.extend(f"- {path}" for path in memory.changed_files[:50])
    else:
        parts.extend(["", "Changed files since previous memory commit: none detected."])

    for heading, text in (
        ("Repo Summary", memory.repo_summary),
        ("Architecture Map", memory.architecture_map),
        ("Execution/Test Profile", memory.test_profile),
        ("Toolchain", memory.toolchain),
    ):
        if text:
            parts.extend(["", f"## {heading}", _trim_text(text)])

    return "\n".join(parts).strip()


def _git_output(runner: Runner, cwd: Path, args: tuple[str, ...]) -> str | None:
    result = runner.run(("git", *args), cwd=cwd, check=False)
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None


def _git_lines(runner: Runner, cwd: Path, args: tuple[str, ...]) -> list[str]:
    output = _git_output(runner, cwd, args)
    if not output:
        return []
    return [line for line in output.splitlines() if line.strip()]


def _read_optional(path: Path) -> str | None:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    return value or None


def _changed_files(
    runner: Runner,
    cwd: Path,
    *,
    last_analyzed_commit: str | None,
    current_commit: str | None,
) -> tuple[str, ...]:
    if not current_commit:
        return ()
    if not last_analyzed_commit:
        return tuple(_git_lines(runner, cwd, ("ls-files",)))
    result = runner.run(
        ("git", "diff", "--name-only", f"{last_analyzed_commit}..{current_commit}"),
        cwd=cwd,
        check=False,
    )
    if result.returncode != 0:
        return tuple(_git_lines(runner, cwd, ("ls-files",)))
    return tuple(line for line in result.stdout.splitlines() if line.strip())


def _write_repo_summary(
    path: Path,
    config: AgentLoopConfig,
    current_commit: str | None,
    tracked_files: list[str],
) -> None:
    top_level = sorted({file.split("/", 1)[0] for file in tracked_files})
    text = [
        "# Repo Summary",
        "",
        f"- Repo: {config.repo}",
        f"- Base branch: {config.base}",
        f"- Last generated: {date.today().isoformat()}",
        f"- Commit: {current_commit or 'unknown'}",
        f"- Tracked files: {len(tracked_files)}",
        f"- Top-level paths: {', '.join(top_level) if top_level else '(none detected)'}",
    ]
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def _write_architecture_map(path: Path, tracked_files: list[str]) -> None:
    buckets: dict[str, list[str]] = {}
    for file in tracked_files:
        head = file.split("/", 1)[0]
        buckets.setdefault(head, []).append(file)
    lines = ["# Architecture Map", ""]
    for head in sorted(buckets):
        examples = ", ".join(buckets[head][:8])
        lines.append(f"- `{head}`: {len(buckets[head])} tracked file(s); examples: {examples}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_module_index(
    path: Path,
    tracked_files: list[str],
    changed_files: tuple[str, ...],
    current_commit: str | None,
) -> None:
    suffix_counts: dict[str, int] = {}
    for file in tracked_files:
        suffix = Path(file).suffix or "(none)"
        suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1
    data = {
        "tracked_file_count": len(tracked_files),
        "top_level_paths": sorted({file.split("/", 1)[0] for file in tracked_files}),
        "suffix_counts": dict(sorted(suffix_counts.items())),
        "changed_files": list(changed_files),
        "current_commit": current_commit,
    }
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_test_profile(
    path: Path,
    config: AgentLoopConfig,
    current_commit: str | None,
    tracked_files: list[str],
) -> None:
    lines = [
        "# Test Profile",
        "",
        "Verified test commands:",
    ]
    if config.test_command:
        lines.append(f"- `{shlex.join(config.test_command)}`")
    else:
        lines.append("- None recorded yet.")
    lines.extend(["", "Suggested commands to verify when needed:"])
    if "pyproject.toml" in tracked_files:
        lines.append("- `python -m pytest`")
    if "package.json" in tracked_files:
        lines.append("- `npm test`")
    lines.extend(
        [
            "",
            "Do not use by default:",
            "- Broad filesystem searches such as `find / -name pytest`.",
            "- Unverified tool-discovery commands that scan outside the checkout.",
            "",
            "Last verified:",
            f"- Commit: {current_commit or 'unknown'}",
            f"- Date: {date.today().isoformat()}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_toolchain(path: Path, tracked_files: list[str]) -> None:
    data = {
        "python": "pyproject.toml" in tracked_files or any(file.endswith(".py") for file in tracked_files),
        "node": "package.json" in tracked_files,
        "github_actions": any(file.startswith(".github/workflows/") for file in tracked_files),
    }
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_file_summary(file_summary_dir: Path, workdir: Path, rel_path: str) -> None:
    source = workdir / rel_path
    if not source.is_file():
        return
    try:
        size = source.stat().st_size
    except OSError:
        return
    safe_name = rel_path.replace("/", "__") + ".md"
    text = [
        f"# {rel_path}",
        "",
        f"- Size: {size} bytes",
        "- Summary: Refresh needed. Inspect the source file directly before relying on this note.",
    ]
    (file_summary_dir / safe_name).write_text("\n".join(text) + "\n", encoding="utf-8")


def _trim_text(text: str, *, max_chars: int = 4000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n... (truncated)"
