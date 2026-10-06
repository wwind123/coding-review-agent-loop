"""Collected test ids must be identical on every CI runner (#1294)."""

import pytest

from conftest import host_specific_node_ids


@pytest.mark.parametrize(
    "node_id,unstable",
    [
        ("tests/test_x.py::test_a[/opt/py/3.12.14/bin/python -m pkg run-tests]", True),
        ("tests/test_x.py::test_a[python-module]", False),
        ("tests/test_x.py::test_a[/usr/bin/env -u X]", False),
    ],
    ids=["interpreter-path-in-id", "explicit-id", "unrelated-path"],
)
def test_host_specific_node_ids_flags_only_the_interpreter_path(node_id, unstable):
    flagged = host_specific_node_ids([node_id], host_paths=("/opt/py/3.12.14/bin/python",))
    assert flagged == ([node_id] if unstable else [])


def test_collected_ids_do_not_embed_this_interpreter(request):
    ids = [item.nodeid for item in request.session.items]
    assert ids
    assert host_specific_node_ids(ids) == []
