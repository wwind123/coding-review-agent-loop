"""Scoped, fail-closed observations for issue-created opening tuples.

No authority is stored here: expected values always come from independently
loaded metadata, operator configuration, and authenticated actor identity.
"""
from __future__ import annotations

import hashlib
import json

from .errors import AgentLoopError
from .github_retry import run_gh_read
from .workdirs import github_api_cwd

MANAGED_LABEL = "agent-loop-managed"


def _field(payload, path):
    value = payload
    for key in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def read_opening_pr(runner, config, pr_number, *, phase):
    """Require a successful object and structurally complete observations.

    Never echo API output or stderr: either can contain body nonces or secrets.
    Only the shared read policy decides whether a failed request is retried.
    """
    endpoint = f"repos/{config.repo}/pulls/{pr_number}"
    prefix = f"Managed-CI {phase} REST {endpoint}"
    result = run_gh_read(
        runner, [config.gh_cmd, "api", endpoint], cwd=github_api_cwd(), check=False,
    )
    if result.returncode != 0:
        raise AgentLoopError(f"{prefix}: unavailable state (exit {result.returncode}); the PR was left unchanged. Recheck GitHub availability before retrying.")
    try:
        payload = json.loads(result.stdout)
    except (ValueError, TypeError):
        raise AgentLoopError(f"{prefix}: malformed state (invalid JSON); the PR was left unchanged.") from None
    if not isinstance(payload, dict):
        raise AgentLoopError(f"{prefix}: malformed state (expected object); the PR was left unchanged.")
    malformed = []
    if type(payload.get("draft")) is not bool:
        malformed.append("draft")
    labels = payload.get("labels")
    if not isinstance(labels, list) or not all(
        isinstance(label, dict) and isinstance(label.get("name"), str) for label in labels
    ):
        malformed.append("labels")
    if phase != "lifecycle-classification":
        for field in ("state", "head.repo.full_name", "head.sha", "head.ref", "base.ref", "user.login", "body"):
            value = _field(payload, field)
            if not isinstance(value, str) or (field != "body" and not value):
                malformed.append(field)
        actor_id = _field(payload, "user.id")
        if type(actor_id) is not int or actor_id <= 0:
            malformed.append("user.id")
    if malformed:
        raise AgentLoopError(f"{prefix}: malformed state (missing/invalid fields: {', '.join(malformed)}); the PR was left unchanged.")
    return payload


def _safe(value):
    """Summaries cannot disclose arbitrary API strings, even in ref fields."""
    if type(value) is bool:
        return str(value)
    if type(value) is int:
        rendered = str(value)
        return rendered if len(rendered) <= 20 else _safe(rendered)
    if isinstance(value, str):
        digest = hashlib.sha256(value.encode("utf-8", errors="surrogatepass")).hexdigest()[:16]
        return f"length={len(value)},sha256={digest}"
    return type(value).__name__


def compare_opening_pr(pr, *, metadata, repo, base, branch, actor_login, actor_id, lifecycle, phase, remediation):
    """Retain each tuple predicate and report every differing field/source."""
    labels = {label["name"] for label in pr["labels"]}
    comparisons = [
        ("state", "open", ("open" if pr["state"] in {"open", "OPEN"} else pr["state"]), "open PR contract"),
        ("draft", lifecycle != "ready-unlabeled", pr["draft"], "admitted lifecycle"),
        ("labels", lifecycle == "draft-labeled", MANAGED_LABEL in labels, "admitted lifecycle"),
        ("head.repo.full_name", repo.casefold(), pr["head"]["repo"]["full_name"].casefold(), "configured repository"),
        ("head.ref", branch, pr["head"]["ref"], "configured opening branch"),
        ("base.ref", base, pr["base"]["ref"], "configured base"),
        ("user.login", actor_login.casefold(), pr["user"]["login"].casefold(), "authenticated gh actor"),
        ("user.id", actor_id, pr["user"]["id"], "authenticated gh actor"),
        ("head.sha", metadata.head_sha, pr["head"]["sha"], "GraphQL metadata"),
        ("head.ref", metadata.head_branch, pr["head"]["ref"], "GraphQL metadata"),
        ("base.ref", metadata.base_branch, pr["base"]["ref"], "GraphQL metadata"),
        ("body", metadata.body or "", pr["body"], "raw GraphQL metadata"),
    ]
    differences = [
        f"{field}: expected ({source}) {_safe(expected)}; observed (REST) {_safe(observed)}"
        for field, expected, observed, source in comparisons if expected != observed
    ]
    if differences:
        raise AgentLoopError(
            f"Managed-CI issue-created opening tuple is missing or changed during {phase}: "
            + "; ".join(differences)
            + f". The PR was left unchanged. Resolve or recheck the discrepancy before retrying. Resume with `{remediation}`."
        )
    return pr["head"]["sha"], pr["head"]["ref"]
