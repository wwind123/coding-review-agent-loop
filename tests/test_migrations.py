from agent_loop_helpers import *  # noqa: F403


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

def _make_migration(revision: str, down_revision: str | tuple[str, ...] | None) -> str:
    return (
        f'revision = "{revision}"\n'
        f"down_revision = {repr(down_revision)}\n"
        "branch_labels = None\n"
        "depends_on = None\n"
    )

def _init_git_checkout_with_origin(tmp_path: Path) -> Path:
    origin = tmp_path / "origin.git"
    worktree = tmp_path / "repo"
    _git(tmp_path, "init", "--bare", str(origin))
    _git(tmp_path, "clone", str(origin), str(worktree))
    _git(worktree, "config", "user.email", "test@example.com")
    _git(worktree, "config", "user.name", "Test User")
    _git(worktree, "switch", "-c", "main")
    return worktree

def _commit_all(worktree: Path, message: str) -> None:
    _git(worktree, "add", ".")
    _git(worktree, "commit", "-m", message)

def _push_main(worktree: Path) -> None:
    _git(worktree, "push", "-u", "origin", "main")

def test_validate_pr_migration_topology_blocks_wrong_down_revision(tmp_path):
    worktree = _init_git_checkout_with_origin(tmp_path)
    _write(
        worktree / "alembic" / "versions" / "5d5f0e1a2b3c_base.py",
        _make_migration("5d5f0e1a2b3c", None),
    )
    _write(
        worktree / "alembic" / "versions" / "a6b7c8d9e0f1_add_feature.py",
        _make_migration("a6b7c8d9e0f1", "5d5f0e1a2b3c"),
    )
    _write(
        worktree / "alembic" / "versions" / "402b9e8af79b_latest.py",
        _make_migration("402b9e8af79b", "a6b7c8d9e0f1"),
    )
    _commit_all(worktree, "Base migrations")
    _push_main(worktree)

    _git(worktree, "switch", "-c", "feature/wrong-parent")
    _write(
        worktree / "alembic" / "versions" / "e4f5a6b7c8d9_add_gemini_3_5_flash_pricing.py",
        _make_migration("e4f5a6b7c8d9", "5d5f0e1a2b3c"),
    )
    _commit_all(worktree, "Add migration with stale down_revision")

    result = validate_pr_migration_topology(
        Runner(),
        config=make_config(tmp_path / "config"),
        checkout=worktree,
        pr_metadata=PullRequestMetadata(
            number=77,
            repo="OWNER/REPO",
            title="Add migration",
            head_branch="feature/wrong-parent",
            base_branch="main",
            head_sha=None,
            url=None,
        ),
    )

    assert result.ok is False
    assert result.message is not None
    assert "e4f5a6b7c8d9_add_gemini_3_5_flash_pricing.py" in result.message
    assert "`down_revision = '5d5f0e1a2b3c'`" in result.message
    assert "`402b9e8af79b`" in result.message
    assert "`e4f5a6b7c8d9`" in result.message

def test_validate_pr_migration_topology_allows_linear_head_extension(tmp_path):
    worktree = _init_git_checkout_with_origin(tmp_path)
    _write(
        worktree / "alembic" / "versions" / "5d5f0e1a2b3c_base.py",
        _make_migration("5d5f0e1a2b3c", None),
    )
    _write(
        worktree / "alembic" / "versions" / "402b9e8af79b_latest.py",
        _make_migration("402b9e8af79b", "5d5f0e1a2b3c"),
    )
    _commit_all(worktree, "Base migrations")
    _push_main(worktree)

    _git(worktree, "switch", "-c", "feature/right-parent")
    _write(
        worktree / "alembic" / "versions" / "e4f5a6b7c8d9_add_pricing.py",
        _make_migration("e4f5a6b7c8d9", "402b9e8af79b"),
    )
    _commit_all(worktree, "Add linear migration")

    result = validate_pr_migration_topology(
        Runner(),
        config=make_config(tmp_path / "config"),
        checkout=worktree,
        pr_metadata=PullRequestMetadata(
            number=77,
            repo="OWNER/REPO",
            title="Add migration",
            head_branch="feature/right-parent",
            base_branch="main",
            head_sha=None,
            url=None,
        ),
    )

    assert result.ok is True
    assert result.message is None

def test_validate_pr_migration_topology_skips_block_when_base_already_has_multiple_heads(tmp_path):
    worktree = _init_git_checkout_with_origin(tmp_path)
    _write(
        worktree / "alembic" / "versions" / "111111111111_first_head.py",
        _make_migration("111111111111", None),
    )
    _write(
        worktree / "alembic" / "versions" / "222222222222_second_head.py",
        _make_migration("222222222222", None),
    )
    _commit_all(worktree, "Base has multiple heads")
    _push_main(worktree)

    _git(worktree, "switch", "-c", "feature/merge-heads")
    _write(
        worktree / "alembic" / "versions" / "333333333333_merge_heads.py",
        _make_migration("333333333333", ("111111111111", "222222222222")),
    )
    _commit_all(worktree, "Merge existing heads")

    result = validate_pr_migration_topology(
        Runner(),
        config=make_config(tmp_path / "config"),
        checkout=worktree,
        pr_metadata=PullRequestMetadata(
            number=77,
            repo="OWNER/REPO",
            title="Merge heads",
            head_branch="feature/merge-heads",
            base_branch="main",
            head_sha=None,
            url=None,
        ),
    )

    assert result.ok is True
    assert result.message is None

def test_validate_pr_migration_topology_blocks_non_literal_changed_metadata(tmp_path):
    worktree = _init_git_checkout_with_origin(tmp_path)
    _write(
        worktree / "alembic" / "versions" / "402b9e8af79b_latest.py",
        _make_migration("402b9e8af79b", None),
    )
    _commit_all(worktree, "Base migrations")
    _push_main(worktree)

    _git(worktree, "switch", "-c", "feature/non-literal-migration")
    _write(
        worktree / "alembic" / "versions" / "e4f5a6b7c8d9_non_literal.py",
        'revision = "e4f5a6b7c8d9"\n'
        "PREVIOUS = '402b9e8af79b'\n"
        "down_revision = PREVIOUS\n"
        "branch_labels = None\n"
        "depends_on = None\n",
    )
    _commit_all(worktree, "Add non-literal migration metadata")

    result = validate_pr_migration_topology(
        Runner(),
        config=make_config(tmp_path / "config"),
        checkout=worktree,
        pr_metadata=PullRequestMetadata(
            number=77,
            repo="OWNER/REPO",
            title="Bad migration metadata",
            head_branch="feature/non-literal-migration",
            base_branch="main",
            head_sha=None,
            url=None,
        ),
    )

    assert result.ok is False
    assert result.message is not None
    assert "Could not validate Alembic revision metadata" in result.message
    assert "e4f5a6b7c8d9_non_literal.py" in result.message
