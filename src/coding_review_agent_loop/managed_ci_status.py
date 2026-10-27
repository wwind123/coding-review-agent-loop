"""Read-only correlation predicates shared by managed CI observers."""
from __future__ import annotations
import json
from datetime import datetime as _datetime
from urllib.parse import urlparse
from .config import AgentLoopConfig
from .workdirs import active_workdir
from .runner import Runner
from .github import PullRequestCheck
FINAL_CONTEXT = "final-ci/exact-head"

def _v2_correlated_status(
    runner: Runner, *, config: AgentLoopConfig, expected_head: str, contract: ManagedCiContract
) -> PullRequestCheck | None:
    """Return only the status published by the attached, current run.

    A context name is deliberately insufficient: an older duplicate or a
    contributor-created status must remain pending, including if it is red.
    """
    if (
        contract.attached_run_id is None
        or not contract.nonce
        or not _v2_attempt_is_correlatable(contract.run_attempt)
    ):
        return None
    result = runner.run(
        [config.gh_cmd, "api", "--paginate", f"repos/{config.repo}/commits/{expected_head}/statuses?per_page=100"],
        cwd=active_workdir(config), check=False,
    )
    if result.returncode != 0:
        return None
    try:
        pages = _decode_paginated_json(result.stdout)
    except (json.JSONDecodeError, TypeError):
        return None
    statuses = [item for page in pages for item in page]
    required = {f"nonce={contract.nonce}", f"run_id={contract.attached_run_id}"}
    if isinstance(contract.run_attempt, int):
        required.add(f"attempt={contract.run_attempt}")
    excluded = _v2_terminal_attempt_excluded(
        contract.attached_run_id, contract.run_attempt, contract.terminal_attempts
    )
    matches: list[tuple[int, str | None, dict[str, object]]] = []
    for position, raw in enumerate(statuses):
        if not isinstance(raw, dict) or raw.get("context") != FINAL_CONTEXT:
            continue
        description = raw.get("description") if isinstance(raw.get("description"), str) else ""
        target = raw.get("target_url") if isinstance(raw.get("target_url"), str) else ""
        creator = raw.get("creator") if isinstance(raw.get("creator"), dict) else {}
        login = creator.get("login") if isinstance(creator.get("login"), str) else None
        creator_id = creator.get("id") if isinstance(creator.get("id"), int) else None
        allowed_publisher = (
            (login or "").casefold() == "github-actions[bot]"
            or (login == contract.trusted_actor_login and creator_id == contract.trusted_actor_id)
        )
        description_tokens = {token.strip() for token in description.split(";") if token.strip()}
        if not allowed_publisher or not required.issubset(description_tokens):
            continue
        target_path = urlparse(target).path.rstrip("/")
        target_segments = target_path.split("/")
        if (
            len(target_segments) < 3
            or target_segments[-3:-1] != ["actions", "runs"]
            or target_segments[-1] != str(contract.attached_run_id)
        ):
            continue
        if excluded:
            continue
        created_value = raw.get("created_at")
        created = created_value if isinstance(created_value, str) else None
        if created is not None:
            try:
                _datetime.fromisoformat(created.replace("Z", "+00:00"))
            except ValueError:
                continue
        matches.append((position, created, raw))
    if not matches:
        return None
    def sort_key(item: tuple[int, str | None, dict[str, object]]) -> tuple[int, str, int]:
        pos, created, _ = item
        return (1 if created else 0, created or "", -pos)
    raw = sorted(matches, key=sort_key)[-1][2]
    description = raw.get("description") if isinstance(raw.get("description"), str) else ""
    target = raw.get("target_url") if isinstance(raw.get("target_url"), str) else ""
    creator = raw.get("creator") if isinstance(raw.get("creator"), dict) else {}
    login = creator.get("login") if isinstance(creator.get("login"), str) else None
    creator_id = creator.get("id") if isinstance(creator.get("id"), int) else None
    return PullRequestCheck(
        name=FINAL_CONTEXT,
        kind="status_context",
        status=str(raw.get("state") or "pending"),
        url=target or None,
        run_id=str(contract.attached_run_id),
        created_at=raw.get("created_at") if isinstance(raw.get("created_at"), str) else None,
        completed_at=raw.get("updated_at") if isinstance(raw.get("updated_at"), str) else None,
        creator_login=login,
        creator_id=creator_id,
        description=description,
    )


def _decode_paginated_json(text: str) -> list[list[object]]:
    """Decode gh --paginate output, including concatenated JSON arrays."""
    decoder = json.JSONDecoder()
    pages: list[list[object]] = []
    offset = 0
    while offset < len(text):
        while offset < len(text) and text[offset].isspace():
            offset += 1
        if offset >= len(text):
            break
        value, end = decoder.raw_decode(text, offset)
        if not isinstance(value, list):
            raise json.JSONDecodeError("page is not an array", text, offset)
        pages.append(value)
        offset = end
    return pages


def _v2_terminal_attempt_excluded(
    run_id: object, attempt: object, exclusions: object
) -> bool:
    if not isinstance(run_id, int) or not isinstance(exclusions, (set, tuple, list)):
        return False
    for entry in exclusions:
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            continue
        rid, stored_attempt = entry
        if rid != run_id or not isinstance(rid, int):
            continue
        # An omitted terminal attempt cannot prove that any later attempt of
        # the same run is a legitimate rerun. Keep the whole run excluded;
        # only a different fresh run ID can be correlated safely.
        if stored_attempt is None:
            return True
        if isinstance(stored_attempt, int) and stored_attempt == attempt:
            return True
    return False


def _v2_attempt_is_correlatable(attempt: object) -> bool:
    return type(attempt) is int and attempt > 0


def complete_managed_board(board) -> bool:
    """The counted-board gate shared by fresh and historical qualification."""
    return (
        board.check_query_status == "ok"
        and board.branch_protection_status in {"configured", "not_found", "forbidden"}
        and board.state == "passing"
        and bool(board.passing)
        and not board.pending
        and not board.missing_required
    )
