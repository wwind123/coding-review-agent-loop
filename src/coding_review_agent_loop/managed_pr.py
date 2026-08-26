"""Create a managed-CI draft from code already pushed to a repository branch."""

from __future__ import annotations

import base64
import json
import secrets
import time
from dataclasses import dataclass, replace
from urllib.parse import quote

from .config import AgentLoopConfig
from .errors import AgentLoopError
from .logging import log
from .managed_ci import (
    MANAGED_LABEL,
    UNPROTECTED_OVERRIDE_TRAILER,
    ensure_managed_label,
    preflight_managed_ci_creation,
)
from .runner import CommandResult, Runner
from .workdirs import active_workdir
from .protocol_markers import PR_BODY_SURFACE, TrustedBody, scan_reserved_markers


SOURCE_MARKER = "AGENT_MANAGED_PR_SOURCE_V1"


@dataclass(frozen=True)
class ManagedPrHandoff:
    pr_number: int
    config: AgentLoopConfig
    source_sha: str
    managed_branch: str
    source_branch: str | None = None


def _api(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    method: str,
    endpoint: str,
    payload: dict[str, object] | None = None,
    check: bool = True,
) -> CommandResult:
    args = [config.gh_cmd, "api", "--method", method, endpoint]
    input_text = None
    if payload is not None:
        args.extend(("--input", "-"))
        input_text = json.dumps(payload)
    result = runner.run(
        args,
        cwd=active_workdir(config),
        input_text=input_text,
        check=False,
    )
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        suffix = f": {detail}" if detail else ""
        raise AgentLoopError(f"GitHub API request failed ({method} {endpoint}){suffix}")
    return result


def _json_payload(result: CommandResult, *, operation: str) -> object:
    try:
        return json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise AgentLoopError(f"GitHub returned invalid JSON while {operation}.") from exc


def _source_marker(*, source_branch: str, source_sha: str) -> TrustedBody:
    encoded = base64.urlsafe_b64encode(
        json.dumps(
            {"source_branch": source_branch, "source_sha": source_sha},
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).decode("ascii")
    return TrustedBody.canonical(
        f"<!-- {SOURCE_MARKER} {encoded} -->",
        expected_tokens=(SOURCE_MARKER,),
    )


def _compose_body(
    body: str,
    *,
    source_branch: str,
    source_sha: str,
    override_nonce: str | None,
) -> TrustedBody:
    sections: list[TrustedBody | str] = []
    if body.strip():
        sections.append(TrustedBody.current_untrusted_visible(body.rstrip()))
    if sections:
        sections.append("\n\n")
    sections.append(_source_marker(source_branch=source_branch, source_sha=source_sha))
    if override_nonce:
        sections.extend(
            [
                "\n\n",
                TrustedBody.marker(
                    UNPROTECTED_OVERRIDE_TRAILER,
                    f"{UNPROTECTED_OVERRIDE_TRAILER} nonce={override_nonce}",
                ),
            ]
        )
    return TrustedBody.join(*sections).append("\n")


def validate_managed_pr_body(
    body: str,
    *,
    source_branch: str,
    source_sha: str,
    managed_branch: str,
    override_nonce: str | None,
    fetched_head_sha: str | None,
    fetched_head_branch: str | None,
    fetched_base_branch: str | None,
    expected_base_branch: str | None,
) -> None:
    """Validate the one managed-PR body that this invocation created."""
    expected_tokens = {SOURCE_MARKER}
    if override_nonce is not None:
        expected_tokens.add(UNPROTECTED_OVERRIDE_TRAILER)
    carrier = TrustedBody.canonical(
        body,
        surface=PR_BODY_SURFACE,
        expected_tokens=expected_tokens,
    )
    carrier.validate_for_surface(PR_BODY_SURFACE)
    occurrences = scan_reserved_markers(str(carrier))
    source_occurrence = next(
        item for item in occurrences if item.definition.token == SOURCE_MARKER
    )
    source_match = source_occurrence.definition.pattern.search(source_occurrence.text)
    if source_match is None:
        raise AgentLoopError("Managed PR source record is malformed.")
    encoded = source_match.group("payload")
    try:
        decoded = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
        payload = json.loads(decoded.decode("utf-8"))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AgentLoopError("Managed PR source record could not be decoded.") from exc
    if not isinstance(payload, dict) or set(payload) != {"source_branch", "source_sha"}:
        raise AgentLoopError("Managed PR source record has an invalid schema.")
    if payload.get("source_branch") != source_branch or payload.get("source_sha") != source_sha:
        raise AgentLoopError("Managed PR source record does not match this creation handoff.")
    if fetched_head_sha != source_sha:
        raise AgentLoopError("Managed PR head SHA does not match its creation handoff.")
    if fetched_head_branch != managed_branch:
        raise AgentLoopError("Managed PR head branch does not match its creation handoff.")
    if expected_base_branch is not None and fetched_base_branch != expected_base_branch:
        raise AgentLoopError("Managed PR base branch does not match its creation handoff.")
    if override_nonce is not None:
        override_occurrence = next(
            item for item in occurrences if item.definition.token == UNPROTECTED_OVERRIDE_TRAILER
        )
        if override_occurrence.text.strip().split() != [
            UNPROTECTED_OVERRIDE_TRAILER,
            f"nonce={override_nonce}",
        ]:
            raise AgentLoopError("Managed PR override record does not match this creation handoff.")


def _close_partial_handoff(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int | None,
    managed_branch: str,
) -> tuple[str, ...]:
    failures: list[str] = []
    if pr_number is not None:
        close_result = _api(
            runner,
            config=config,
            method="PATCH",
            endpoint=f"repos/{config.repo}/pulls/{pr_number}",
            payload={"state": "closed"},
            check=False,
        )
        if close_result.returncode != 0:
            failures.append(f"close PR #{pr_number}")
    encoded_ref = quote(f"heads/{managed_branch}", safe="")
    delete_result = _api(
        runner,
        config=config,
        method="DELETE",
        endpoint=f"repos/{config.repo}/git/refs/{encoded_ref}",
        check=False,
    )
    if delete_result.returncode != 0:
        failures.append(f"delete branch {managed_branch}")
    return tuple(failures)


def create_managed_pr(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    source_branch: str,
    title: str,
    body: str,
) -> ManagedPrHandoff:
    """Create a trusted draft and return a config correlated to its waiver."""
    source_branch = source_branch.strip()
    title = title.strip()
    if not source_branch or source_branch.startswith("refs/") or ":" in source_branch:
        raise AgentLoopError("--head must name a same-repository branch, without `refs/` or an owner prefix.")
    if not title:
        raise AgentLoopError("--title cannot be empty.")
    try:
        TrustedBody.current_untrusted_visible(body)
    except AgentLoopError as exc:
        raise AgentLoopError(
            "The supplied PR body contains a reserved managed-PR protocol marker."
        ) from exc
    if not config.base:
        raise AgentLoopError("Managed PR creation requires a resolved base branch.")
    if source_branch == config.base:
        raise AgentLoopError("--head must differ from the PR base branch.")

    branch_result = _api(
        runner,
        config=config,
        method="GET",
        endpoint=f"repos/{config.repo}/branches/{quote(source_branch, safe='')}",
    )
    branch_payload = _json_payload(branch_result, operation="resolving the source branch")
    commit = branch_payload.get("commit") if isinstance(branch_payload, dict) else None
    source_sha = commit.get("sha") if isinstance(commit, dict) else None
    if not isinstance(source_sha, str) or not source_sha:
        raise AgentLoopError(f"Could not resolve the head SHA for branch {source_branch!r}.")

    pulls_result = _api(
        runner,
        config=config,
        method="GET",
        endpoint=f"repos/{config.repo}/commits/{source_sha}/pulls?state=open",
    )
    pulls = _json_payload(pulls_result, operation="checking for an existing PR")
    open_prs = (
        [item for item in pulls if isinstance(item, dict) and item.get("state") == "open"]
        if isinstance(pulls, list)
        else []
    )
    if open_prs:
        numbers = ", ".join(f"#{item['number']}" for item in open_prs if isinstance(item.get("number"), int))
        raise AgentLoopError(
            f"Source head {source_sha} already has an open PR{f' ({numbers})' if numbers else ''}. "
            "Continue it with `agent-loop pr <number>`; managed-pr never retroactively adopts a PR."
        )

    managed_branch = (
        f"agent-loop/managed-direct-{int(time.time())}-{secrets.token_hex(4)}"
    )
    intent = preflight_managed_ci_creation(
        runner,
        config=config,
        branch=managed_branch,
    )
    if intent is None:
        raise AgentLoopError(
            "Managed PR creation is not ready for this repository. Run `agent-loop managed-ci preflight` "
            "and correct its reported prerequisites; managed-pr does not fall back to ordinary CI."
        )
    if intent.audit_nonce:
        log(
            config,
            "WARNING: managed-pr is using the explicit unprotected managed-CI waiver for this invocation. "
            "GitHub cannot prevent a manual merge, other automation, a compromised credential, or an "
            "agent-loop defect from bypassing the voluntary final-ci/exact-head gate.",
        )

    pr_number: int | None = None
    ref_created = False
    try:
        _api(
            runner,
            config=config,
            method="POST",
            endpoint=f"repos/{config.repo}/git/refs",
            payload={"ref": f"refs/heads/{intent.branch}", "sha": source_sha},
        )
        ref_created = True
        rendered_body = _compose_body(
            body,
            source_branch=source_branch,
            source_sha=source_sha,
            override_nonce=intent.audit_nonce,
        )
        rendered_body.validate_for_surface(PR_BODY_SURFACE)
        create_result = _api(
            runner,
            config=config,
            method="POST",
            endpoint=f"repos/{config.repo}/pulls",
            payload={
                "title": title,
                "head": intent.branch,
                "base": config.base,
                "body": rendered_body,
                "draft": True,
            },
        )
        created = _json_payload(create_result, operation="creating the managed draft PR")
        pr_number = created.get("number") if isinstance(created, dict) else None
        if not isinstance(pr_number, int) or pr_number <= 0:
            raise AgentLoopError("GitHub created a managed draft without returning a valid PR number.")
        if not ensure_managed_label(runner, config=config):
            raise AgentLoopError(f"Unable to create the `{MANAGED_LABEL}` label.")
        _api(
            runner,
            config=config,
            method="POST",
            endpoint=f"repos/{config.repo}/issues/{pr_number}/labels",
            payload={"labels": [MANAGED_LABEL]},
        )

        current_result = _api(
            runner,
            config=config,
            method="GET",
            endpoint=f"repos/{config.repo}/branches/{quote(source_branch, safe='')}",
        )
        current = _json_payload(current_result, operation="revalidating the source branch")
        current_commit = current.get("commit") if isinstance(current, dict) else None
        current_sha = current_commit.get("sha") if isinstance(current_commit, dict) else None
        if current_sha != source_sha:
            raise AgentLoopError(
                f"Source branch {source_branch!r} moved during managed PR creation; the partial draft was closed."
            )
        post_create_pulls_result = _api(
            runner,
            config=config,
            method="GET",
            endpoint=f"repos/{config.repo}/commits/{source_sha}/pulls?state=open",
        )
        post_create_pulls = _json_payload(
            post_create_pulls_result,
            operation="rechecking for a concurrent PR",
        )
        competing_prs = (
            [
                item
                for item in post_create_pulls
                if isinstance(item, dict)
                and item.get("state") == "open"
                and item.get("number") != pr_number
            ]
            if isinstance(post_create_pulls, list)
            else []
        )
        if competing_prs:
            raise AgentLoopError(
                "Another PR was opened for the source commit during managed PR creation; "
                "the partial managed draft was closed."
            )
    except Exception as exc:
        if ref_created:
            cleanup_failures = _close_partial_handoff(
                runner,
                config=config,
                pr_number=pr_number,
                managed_branch=managed_branch,
            )
            if cleanup_failures:
                raise AgentLoopError(
                    "Managed PR creation failed and automatic cleanup was incomplete: "
                    + ", ".join(cleanup_failures)
                    + ". Inspect the partial draft and reserved branch before retrying."
                ) from exc
        raise

    correlated_config = replace(
        config,
        managed_ci_expected_override_nonce=intent.audit_nonce,
        pr_origin_flow="managed-pr",
        # Replaying managed-pr would hit the duplicate-PR guard for the draft
        # just created above. Leave no original argv so a CI-watch timeout
        # renders run_pr_loop's deterministic `agent-loop pr <n>` recovery.
        invocation_argv=(),
    )
    log(
        config,
        f"Created managed draft PR #{pr_number} from {source_branch}@{source_sha[:12]} via {managed_branch}",
    )
    return ManagedPrHandoff(pr_number, correlated_config, source_sha, managed_branch, source_branch)
