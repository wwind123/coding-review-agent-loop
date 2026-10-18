"""The explicit multi-owner patch helpers in agent_loop_helpers (#1186)."""
import types

import pytest

from agent_loop_helpers import patch_attr_in, patch_object_in


def _modules():
    first = types.ModuleType("owner_one")
    second = types.ModuleType("owner_two")
    stray = types.ModuleType("not_an_owner")
    first.target = second.target = "real"
    return first, second, stray


def test_patch_attr_in_sets_every_listed_owner_and_restores_them():
    first, second, _stray = _modules()
    monkeypatch = pytest.MonkeyPatch()
    patch_attr_in(monkeypatch, (first, second), "target", "fake")
    assert (first.target, second.target) == ("fake", "fake")
    monkeypatch.undo()
    assert (first.target, second.target) == ("real", "real")


def test_patch_attr_in_rejects_a_module_that_does_not_bind_the_name(monkeypatch):
    first, second, stray = _modules()
    with pytest.raises(AttributeError, match="not_an_owner does not bind 'target'"):
        patch_attr_in(monkeypatch, (first, stray, second), "target", "fake")
    assert (first.target, second.target) == ("real", "real")
    assert "target" not in vars(stray)


def test_patch_object_in_shares_one_replacement_and_restores_every_owner():
    first, second, _stray = _modules()
    with patch_object_in((first, second), "target", return_value=7) as replacement:
        assert first.target is replacement and second.target is replacement
        assert second.target() == 7
    assert (first.target, second.target) == ("real", "real")
    with patch_object_in((first, second), "target", "explicit") as replacement:
        assert replacement == "explicit" and (first.target, second.target) == ("explicit", "explicit")
    assert (first.target, second.target) == ("real", "real")


def test_patch_object_in_rejects_a_module_that_does_not_bind_the_name():
    first, _second, stray = _modules()
    with pytest.raises(AttributeError, match="not_an_owner does not bind 'target'"):
        patch_object_in((first, stray), "target")
    assert first.target == "real" and "target" not in vars(stray)
