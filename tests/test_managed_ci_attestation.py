"""Attestation writer and fail-closed verifier tests (#1312, parent #1308)."""

from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
CALLEE = ROOT / "ci" / "managed"
ACTION = ROOT / ".github" / "actions" / "managed-ci-attest" / "action.yml"

sys.path.insert(0, str(CALLEE))
import attestation  # noqa: E402

sys.path.remove(str(CALLEE))

TARGET = "a" * 40
OTHER = "b" * 40
REPO = "owner/repo"
RUN_ID = 777
ATTEMPT = 1
EXPECTED = [
    {"attestation_id": "unit-a", "job_name": "unit (a)", "needs_key": "unit-a"},
    {"attestation_id": "svc", "job_name": "services", "needs_key": "svc"},
]


def record(entry, **over):
    base = attestation.build_record(
        attestation_id=entry["attestation_id"], job_name=entry["job_name"],
        target_sha=TARGET, head_sha=TARGET, run_id=RUN_ID, run_attempt=ATTEMPT,
        repository=REPO, job_status="success",
    )
    base.update(over)
    return base


def art(entry, attempt=ATTEMPT, **over):
    rec = record(entry, **over)
    return (
        attestation.artifact_name(entry["attestation_id"], attempt),
        {"attestation.json": json.dumps(rec).encode()},
    )


def good_artifacts():
    return [art(e) for e in EXPECTED]


def jobs(**conclusions):
    out = []
    for e in EXPECTED:
        out.append({"name": e["job_name"], "conclusion": conclusions.get(e["attestation_id"], "success")})
    return out


NEEDS = {"unit-a": {"result": "success"}, "svc": {"result": "success"}, "validate": {"result": "success"}}


def run(expected=EXPECTED, artifacts=None, api_jobs="default", needs=NEEDS, attempt=ATTEMPT):
    return attestation.verify(
        expected,
        good_artifacts() if artifacts is None else artifacts,
        jobs() if api_jobs == "default" else api_jobs,
        needs, RUN_ID, attempt, TARGET, REPO,
    )


def test_success():
    assert run() == []


def test_missing_attestation():
    assert any("missing" in e for e in run(artifacts=good_artifacts()[:1]))


def test_duplicate_attestation_id():
    arts = good_artifacts() + [art(EXPECTED[0])]
    assert any("duplicate" in e for e in run(artifacts=arts))


def test_duplicate_with_invalid_first_still_fails():
    bad = (attestation.artifact_name("unit-a", 1), {"attestation.json": b"nope"})
    assert run(artifacts=[bad] + good_artifacts())


def test_extra_attestation():
    extra = {"attestation_id": "rogue", "job_name": "rogue", "needs_key": "rogue"}
    assert any("unexpected" in e for e in run(artifacts=good_artifacts() + [art(extra)]))


@pytest.mark.parametrize("over", [
    {"target_sha": OTHER}, {"head_sha": OTHER}, {"run_id": RUN_ID + 1},
    {"run_attempt": 2}, {"repository": "evil/repo"},
])
def test_foreign_binding(over):
    arts = [art(EXPECTED[0], **over), art(EXPECTED[1])]
    assert run(artifacts=arts)


def test_foreign_attempt_artifact_name():
    arts = [art(EXPECTED[0], attempt=2), art(EXPECTED[1])]
    assert run(artifacts=arts)


@pytest.mark.parametrize("mutate", [
    lambda r: r.pop("schema"),
    lambda r: r.__setitem__("schema", "managed-ci-attestation/2"),
    lambda r: r.__setitem__("extra", 1),
    lambda r: r.pop("job_status"),
    lambda r: r.__setitem__("run_id", "777"),
    lambda r: r.__setitem__("run_attempt", True),
    lambda r: r.__setitem__("target_sha", TARGET.upper()),
])
def test_record_schema_invalid(mutate):
    rec = record(EXPECTED[0])
    mutate(rec)
    arts = [(attestation.artifact_name("unit-a", 1), {"attestation.json": json.dumps(rec).encode()}), art(EXPECTED[1])]
    assert run(artifacts=arts)


@pytest.mark.parametrize("content", [b"not json", b"[1]", b'"x"', b" " * (attestation.MAX_RECORD_BYTES + 1)])
def test_record_content_invalid(content):
    arts = [(attestation.artifact_name("unit-a", 1), {"attestation.json": content}), art(EXPECTED[1])]
    assert run(artifacts=arts)


def test_artifact_must_hold_exactly_one_record_file():
    name, files = art(EXPECTED[0])
    files["other.json"] = b"{}"
    assert run(artifacts=[(name, files), art(EXPECTED[1])])


@pytest.mark.parametrize("name", [
    "managed-ci-attest-unit-a", "managed-ci-attest-unit-a-attempt-0",
    "managed-ci-attest-Unit-attempt-1", "managed-ci-attest--attempt-1",
    "managed-ci-attest-unit-a-attempt-1x",
])
def test_malformed_artifact_names(name):
    assert any("malformed" in e for e in run(artifacts=good_artifacts() + [(name, {})]))


def test_name_and_record_id_disagree():
    rec = record(EXPECTED[0])
    arts = [(attestation.artifact_name("svc", 1), {"attestation.json": json.dumps(rec).encode()}), art(EXPECTED[0])]
    assert run(artifacts=arts)


def test_retained_earlier_attempt_plus_complete_current():
    old = [art(e, attempt=1) for e in EXPECTED]
    new = [art(e, attempt=2, run_attempt=2) for e in EXPECTED]
    assert run(artifacts=old + new, attempt=2) == []


def test_carried_over_jobs_fail_as_missing():
    old = [art(e, attempt=1) for e in EXPECTED]
    new = [art(EXPECTED[0], attempt=2, run_attempt=2)]
    errors = run(artifacts=old + new, attempt=2)
    assert any("missing attestation for 'svc'" in e for e in errors)


def test_record_claiming_another_jobs_name():
    arts = [art(EXPECTED[0], job_name="services"), art(EXPECTED[1])]
    assert any("differs from expected" in e for e in run(artifacts=arts))


@pytest.mark.parametrize("conclusion", ["failure", "skipped", "cancelled", None, "timed_out"])
def test_api_conclusion_overrides_self_report(conclusion):
    assert run(api_jobs=jobs(svc=conclusion))


def test_api_job_missing_or_ambiguous():
    assert run(api_jobs=jobs()[:1])
    assert run(api_jobs=jobs() + [{"name": "services", "conclusion": "success"}])


def test_api_lookup_uses_expected_name_not_record_name():
    arts = [art(EXPECTED[0]), art(EXPECTED[1], job_name="unit (a)")]
    assert run(artifacts=arts)


@pytest.mark.parametrize("api", [None, "x", [1], [{"conclusion": "success"}]])
def test_api_unavailable_or_malformed(api):
    assert run(api_jobs=api)


@pytest.mark.parametrize("needs", [
    {}, [], None, {"unit-a": {"result": "success"}},
    {"unit-a": {"result": "success"}, "svc": {"result": "failure"}},
    {"unit-a": {"result": "success"}, "svc": {"result": "skipped"}},
    {"unit-a": {"result": "success"}, "svc": {"result": "cancelled"}},
    {"unit-a": {"result": "success"}, "svc": {}},
    {"unit-a": {"result": "success"}, "svc": "success"},
])
def test_needs_invalid(needs):
    assert run(needs=needs)


@pytest.mark.parametrize("expected", [
    [], None, "x", {}, [{"attestation_id": "a"}],
    [dict(EXPECTED[0], extra=1)],
    [EXPECTED[0], dict(EXPECTED[1], attestation_id="unit-a")],
    [EXPECTED[0], dict(EXPECTED[1], job_name="unit (a)")],
    [dict(EXPECTED[0], attestation_id="Bad Id")],
    [dict(EXPECTED[0], needs_key="bad key")],
    [{"attestation_id": f"id{i}", "job_name": f"j{i}", "needs_key": "k"} for i in range(257)],
])
def test_expected_set_invalid(expected):
    assert run(expected=expected)


def test_verify_does_not_mutate_inputs():
    exp = copy.deepcopy(EXPECTED)
    run(expected=exp)
    assert exp == EXPECTED


# --- writer -----------------------------------------------------------------

def _git(project, *args):
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        cwd=project, check=True, capture_output=True, text=True,
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    project = tmp_path / "proj"
    project.mkdir()
    _git(project, "init", "-q")
    (project / "f").write_text("1")
    _git(project, "add", "f")
    _git(project, "commit", "-q", "-m", "one")
    return project


def _write(project, sha, out, **env):
    e = {k: v for k, v in os.environ.items() if not k.startswith("GITHUB_")}
    e.update(GITHUB_RUN_ID="9001", GITHUB_RUN_ATTEMPT="2", GITHUB_REPOSITORY=REPO)
    e.update(env)
    return subprocess.run(
        [sys.executable, str(CALLEE / "attestation.py"), "write", "--target-sha", sha,
         "--attestation-id", "unit-a", "--job-name", "unit (a)", "--output-dir", str(out),
         "--workdir", str(project)],
        capture_output=True, text=True, env=e,
    )


def test_write_emits_valid_record(repo, tmp_path):
    head = _git(repo, "rev-parse", "HEAD")
    out = tmp_path / "out"
    result = _write(repo, head, out)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "managed-ci-attest-unit-a-attempt-2"
    rec = json.loads((out / "attestation.json").read_text())
    assert attestation.validate_record(rec) == []
    assert rec["target_sha"] == rec["head_sha"] == head
    assert (rec["run_id"], rec["run_attempt"]) == (9001, 2)


def test_write_refuses_head_mismatch_and_writes_nothing(repo, tmp_path):
    out = tmp_path / "out"
    result = _write(repo, OTHER, out)
    assert result.returncode != 0
    assert not out.exists()


@pytest.mark.parametrize("sha", ["abc", "A" * 40, ""])
def test_write_refuses_malformed_sha(repo, tmp_path, sha):
    out = tmp_path / "out"
    assert _write(repo, sha, out).returncode != 0
    assert not out.exists()


def test_write_refuses_without_run_context(repo, tmp_path):
    head = _git(repo, "rev-parse", "HEAD")
    out = tmp_path / "out"
    assert _write(repo, head, out, GITHUB_RUN_ID="").returncode != 0
    assert not out.exists()


# --- composite action --------------------------------------------------------

def test_action_pins_and_uploads_fail_closed():
    text = ACTION.read_text()
    assert "using: composite" in text
    assert "if-no-files-found: error" in text
    assert "[0-9a-f]{40}" in text
    assert "github.action_repository" in text and "github.action_ref" in text
    assert re.search(r"test -n \"\$ACTION_REPOSITORY\"", text)
    for name in ("target_sha", "attestation_id", "job_name"):
        assert f"  {name}:" in text


def test_action_ref_check_rejects_non_sha():
    pattern = re.compile(r"^[0-9a-f]{40}$")
    assert pattern.match("a" * 40)
    assert not pattern.match("managed-ci-split-v1")
    assert not pattern.match("main")
    assert not pattern.match("")


def test_action_checkout_uses_captured_identity_not_nested_context():
    text = ACTION.read_text()
    checkout = text[text.index("actions/checkout"):text.index("Write attestation")]
    assert "steps.pin.outputs.repository" in checkout and "steps.pin.outputs.ref" in checkout
    assert "github.action_" not in checkout
    assert 'id: pin' in text


@pytest.mark.parametrize("name", [
    "managed-ci-attest-unit-a-attempt-1\n",
    "managed-ci-attest-unit-a\n-attempt-1",
])
def test_artifact_name_trailing_newline_rejected(name):
    assert attestation.parse_artifact_name(name) is None
    assert run(artifacts=good_artifacts() + [(name, {})])


@pytest.mark.parametrize("field,value", [
    ("attestation_id", "unit-a\n"), ("target_sha", TARGET + "\n"),
    ("head_sha", TARGET + "\n"), ("repository", REPO + "\n"),
])
def test_record_fields_reject_trailing_newline(field, value):
    assert attestation.validate_record(record(EXPECTED[0], **{field: value}))


def test_expected_and_publisher_inputs_reject_trailing_newline():
    assert run(expected=[dict(EXPECTED[0], attestation_id="unit-a\n"), EXPECTED[1]])
    assert run(expected=[dict(EXPECTED[0], needs_key="unit-a\n"), EXPECTED[1]])
    assert attestation.verify(EXPECTED, good_artifacts(), jobs(), NEEDS, RUN_ID, ATTEMPT, TARGET + "\n", REPO)
    assert attestation.verify(EXPECTED, good_artifacts(), jobs(), NEEDS, RUN_ID, ATTEMPT, TARGET, REPO + "\n")


def test_oversized_record_refused_before_writing(repo, tmp_path):
    head = _git(repo, "rev-parse", "HEAD")
    out = tmp_path / "out"
    e = {k: v for k, v in os.environ.items() if not k.startswith("GITHUB_")}
    e.update(GITHUB_RUN_ID="9001", GITHUB_RUN_ATTEMPT="2", GITHUB_REPOSITORY=REPO)
    result = subprocess.run(
        [sys.executable, str(CALLEE / "attestation.py"), "write", "--target-sha", head,
         "--attestation-id", "unit-a", "--job-name", "unit (a)", "--output-dir", str(out),
         "--workdir", str(repo), "--job-status", "x" * attestation.MAX_RECORD_BYTES],
        capture_output=True, text=True, env=e,
    )
    assert result.returncode != 0
    assert not out.exists()


# --- verify CLI loader --------------------------------------------------------

def _cli_verify(tmp_path, layout):
    arts = tmp_path / "arts"
    for rel, content in layout.items():
        path = arts / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    jobs_file = tmp_path / "jobs.json"
    jobs_file.write_text(json.dumps(jobs()))
    return subprocess.run(
        [sys.executable, str(CALLEE / "attestation.py"), "verify",
         "--expected", json.dumps(EXPECTED), "--needs", json.dumps(NEEDS),
         "--artifacts-dir", str(arts), "--jobs-file", str(jobs_file),
         "--run-id", str(RUN_ID), "--run-attempt", str(ATTEMPT),
         "--target-sha", TARGET, "--repository", REPO],
        capture_output=True, text=True,
    )


def _good_layout():
    return {f"{n}/attestation.json": f for n, d in good_artifacts() for f in [d["attestation.json"]]}


def test_cli_verify_accepts_clean_layout(tmp_path):
    result = _cli_verify(tmp_path, _good_layout())
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("extra", [
    {"managed-ci-attest-unit-a-attempt-1/nested/attestation.json": b"{}"},
    {"managed-ci-attest-unit-a-attempt-1/other.json": b"{}"},
    {"managed-ci-attest-stray-attempt-1": b"x"},
])
def test_cli_verify_rejects_extra_or_stray_contents(tmp_path, extra):
    layout = _good_layout()
    layout.update(extra)
    assert _cli_verify(tmp_path, layout).returncode != 0


def test_cli_verify_same_basename_cannot_be_overwritten_by_valid_record(tmp_path):
    layout = _good_layout()
    name = "managed-ci-attest-unit-a-attempt-1"
    good = layout.pop(f"{name}/attestation.json")
    layout[f"{name}/a/attestation.json"] = b"not json"
    layout[f"{name}/b/attestation.json"] = good
    assert _cli_verify(tmp_path, layout).returncode != 0


def test_cli_verify_nested_lone_record_not_normalised_to_root(tmp_path):
    layout = _good_layout()
    name = "managed-ci-attest-unit-a-attempt-1"
    layout[f"{name}/sub/attestation.json"] = layout.pop(f"{name}/attestation.json")
    assert _cli_verify(tmp_path, layout).returncode != 0
