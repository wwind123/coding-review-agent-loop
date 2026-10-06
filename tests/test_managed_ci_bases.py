import pytest

from coding_review_agent_loop.managed_ci_bases import base_is_trusted, trusted_base_patterns


@pytest.mark.parametrize(
    ("base", "expected"),
    [
        ("refactor/1181", True),
        ("refactor/a/b", True),
        ("refactor", False),
        ("refactorx/1", False),
        ("refactor/", False),
        ("main", True),
    ],
)
def test_prefix_pattern_matches(base, expected):
    assert base_is_trusted(base, "main", "refactor/*") is expected


@pytest.mark.parametrize("variable", ["*", "a/**", "x*/y", "refactor/*  bad name*", "a..b", "/a", "a/*/b"])
def test_invalid_variable_never_widens_trust(variable):
    assert trusted_base_patterns(variable) == ()
    assert base_is_trusted("a/b", "main", variable) is False
    assert base_is_trusted("main", "main", variable) is True


def test_exact_names_commas_and_whitespace():
    variable = "release/1,  hotfix/*\nint/x"
    assert base_is_trusted("release/1", "main", variable)
    assert base_is_trusted("hotfix/9", "main", variable)
    assert base_is_trusted("int/x", "main", variable)
    assert not base_is_trusted("release/2", "main", variable)
    assert not base_is_trusted("int/x/y", "main", variable)


@pytest.mark.parametrize("variable", [None, "", "  \n"])
def test_unset_trusts_default_only(variable):
    assert base_is_trusted("refactor/1", "main", variable) is False
    assert base_is_trusted("main", "main", variable) is True
