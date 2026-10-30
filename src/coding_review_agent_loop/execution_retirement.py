"""Read-only terminal-work assessment and invocation-scoped history routing.

The view removes individual verified historical records, never comments or
remote history. It is rebuilt from complete inventories on each invocation.
"""
from __future__ import annotations

import json
import re
import urllib.parse
from dataclasses import dataclass, replace

from . import decomposition as d
from . import split_materialization as s
from .errors import AgentLoopError
from .workdirs import active_workdir


@dataclass(frozen=True)
class ExecutionHistory:
    parent_issue: int
    retired_hashes: frozenset[str]
    child_numbers: frozenset[int]
    pr_numbers: frozenset[int]
    removed_records: tuple[tuple[tuple, int, int], ...]

    def view(self, context):
        if context.number != self.parent_issue:
            return context
        comments = []
        for comment in context.comments:
            body = comment.body or ""
            identity = (comment.comment_id, comment.created_at, comment.body)
            spans = [(start, end) for owner, start, end in self.removed_records if owner == identity]
            for start, end in sorted(set(spans), reverse=True):
                body = body[:start] + body[end:]
            comments.append(replace(comment, body=body))
        return replace(context, comments=tuple(comments))


def execution_view(config, context):
    history = getattr(config, "execution_history", None)
    return history.view(context) if history is not None else context


def _number(value, label):
    if type(value) is not int or value <= 0:
        raise AgentLoopError(f"Cannot verify {label}: invalid number {value!r}.")
    return value


def _reference(number, url, repo):
    if number is not None:
        number = _number(number, "historical child reference")
        if url and url != f"https://github.com/{repo}/issues/{number}":
            raise AgentLoopError("Cannot verify historical child URL association.")
        return number
    match = re.fullmatch(r"https://github.com/" + re.escape(repo) + r"/issues/(\d+)", url or "")
    if match is None:
        raise AgentLoopError("Cannot verify unresolved historical child reference.")
    return int(match[1])


def _records(comments, parent, repo):
    from .issue_pr_handoff import AGENT_ISSUE_PR_HANDOFF_RE, decode_issue_pr_handoff_record
    codecs = (
        (d.DECOMPOSITION_MARKER_RE, d._decode_metadata, "summary"),
        (d.TOPOLOGY_CHECKPOINT_MARKER_RE, d._decode_checkpoint, "checkpoint"),
        (d.PHASE_IMPLEMENTATION_MARKER_RE, d._decode_phase_implementation_handoff_metadata, "phase"),
        (d.ONE_SHOT_IMPL_HANDOFF_MARKER_RE, d._decode_one_shot_impl_handoff_metadata, "pr"),
        (s.DISCUSS_SPLIT_MARKER_RE, s._decode_split_metadata, "split"),
        (s.SPLIT_STAGE_HANDOFF_MARKER_RE, s._decode_stage_handoff_metadata, "phase"),
        (AGENT_ISSUE_PR_HANDOFF_RE, decode_issue_pr_handoff_record, "pr"),
    )
    result = []
    for index, comment in enumerate(comments):
        for pattern, decode, kind in codecs:
            for match in pattern.finditer(comment.body or ""):
                # Legacy decoders sometimes coerce numeric strings or booleans.
                # Retirement needs the original, strictly typed references.
                payload = d._decode_json_payload(match["payload"], marker_name="execution history")
                for key in ("parent_issue", "issue_number", "child_issue_number", "pr_number"):
                    if key in payload:
                        _number(payload[key], f"historical {key}")
                children_payload = payload.get("children", [])
                if not isinstance(children_payload, list):
                    raise AgentLoopError("Cannot verify historical child mapping.")
                for child in children_payload:
                    if not isinstance(child, dict):
                        raise AgentLoopError("Cannot verify historical child mapping.")
                    if child.get("number") is not None:
                        _number(child["number"], "historical child reference")
                    if child.get("url") is not None and not isinstance(child["url"], str):
                        raise AgentLoopError("Cannot verify historical child URL association.")
                metadata = decode(match["payload"])
                if getattr(metadata, "parent_issue", getattr(metadata, "issue_number", None)) != parent:
                    continue
                children, prs = set(), set()
                if kind == "summary":
                    children.update(_reference(n, u, repo) for _, u, n in metadata.children)
                elif kind == "split":
                    children.update(_reference(c.number, c.url, repo) for c in metadata.children)
                elif kind == "phase":
                    children.add(_reference(metadata.child_issue_number, metadata.child_issue_url, repo))
                elif kind == "pr":
                    if hasattr(metadata, "pr_url"):
                        from .issue_pr_handoff import _validate_issue_pr_handoff_url
                        _validate_issue_pr_handoff_url(metadata.pr_url, repo=repo, pr_number=metadata.pr_number)
                    prs.add(_number(metadata.pr_number, "PR reference"))
                result.append(((index, match.start()), match[0], getattr(metadata, "plan_hash", None), children, prs))
    return result


def assess_execution_history(runner, *, config, issue_number, comments, plan_hash, newly_retired):
    """Partition attributed current work before checking historical completion.

    Hash retirement applies to records before the retiring decision, so an
    A/B/A recurrence never suppresses the latest A. Hash-less records need a
    retirement boundary AND verified terminal references.
    """
    from .execution_policy import _strict_gh_listing, _REALIZATION_LISTING_LIMIT
    from .child_topology import parent_child_search_queries
    from .github import parse_strong_issue_reference_evidence
    from .issue_pr_handoff import authenticate_canonical_issue_pr
    from .github import IssueContext

    decisions = []
    for index, comment in enumerate(comments):
        for match in d.EXECUTION_DECISION_MARKER_RE.finditer(comment.body or ""):
            decision = d._decode_execution_decision(match["payload"])
            if decision.parent_issue == issue_number:
                decisions.append(((index, match.start()), decision))
    retired = set(newly_retired)
    for _, decision in decisions:
        retired.update(decision.retires_plan_hashes)
    if not retired or config.dry_run:
        return None
    # Authenticate the canonical contract before making any record invisible.
    authenticate_canonical_issue_pr(
        runner, config=config, issue_number=issue_number,
        issue_context=IssueContext(issue_number, config.repo, None, None, None, tuple(comments)),
    )
    records = _records(comments, issue_number, config.repo)
    boundary = max((i for i, dec in decisions if dec.retires_plan_hashes), default=(-1, -1))
    if newly_retired:
        boundary = (len(comments), 0)
    current_children, current_prs, old_children, old_prs = set(), set(), set(), set()
    historical = []
    # Current hash-bearing records establish ownership before hash-less
    # materialization records are considered, regardless of thread order.
    for index, raw, record_hash, children, prs in records:
        if record_hash == plan_hash and not any(
            index < i and record_hash in dec.retires_plan_hashes for i, dec in decisions
        ):
            current_children.update(children)
            current_prs.update(prs)
    for index, raw, record_hash, children, prs in records:
        old = record_hash in newly_retired or any(
            index < i and record_hash in dec.retires_plan_hashes for i, dec in decisions
        ) or (record_hash is None and index < boundary)
        if old:
            historical.append((index, raw))
            old_children.update(children)
            old_prs.update(prs)
        elif record_hash == plan_hash or (record_hash is None and (children or prs)
                and children <= current_children and prs <= current_prs):
            current_children.update(children)
            current_prs.update(prs)
        else:
            raise AgentLoopError("Cannot verify historical topology ownership for the approved plan.")
    if old_children & current_children or old_prs & current_prs:
        raise AgentLoopError("Ambiguous ownership: historical and current execution records claim the same work.")

    blockers = []
    child_items, pr_items = {}, {}
    def inventory(args, label):
        items = _strict_gh_listing(runner, config=config, args=args)
        if items is None:
            raise AgentLoopError(f"Cannot verify {label}: listing unreadable or possibly truncated.")
        return items

    def accept(item, target, kind):
        number = _number(item.get("number"), kind)
        state = item.get("state")
        valid = {"OPEN", "CLOSED"} if kind == "child issue" else {"OPEN", "CLOSED", "MERGED"}
        if not isinstance(state, str) or state not in valid:
            raise AgentLoopError(f"Cannot verify {kind} #{number}: unknown state {state!r}.")
        if number in target and target[number]["state"] != state:
            raise AgentLoopError(f"Cannot verify {kind} #{number}: inconsistent inventory states.")
        target[number] = item

    for query in parent_child_search_queries(issue_number):
        for item in inventory([
            "issue", "list", "--repo", config.repo, "--search", query, "--state", "all",
            "--limit", str(_REALIZATION_LISTING_LIMIT), "--json", "number,title,body,state",
        ], f"child issues that could not be listed ({query})"):
            if not isinstance(item.get("title"), str) or not isinstance(item.get("body"), str):
                raise AgentLoopError("Cannot verify child listing association fields.")
            accept(item, child_items, "child issue")
    branch_name = f"agent-loop/managed-{issue_number}"
    for item in inventory([
        "pr", "list", "--repo", config.repo, "--state", "all", "--json",
        "number,body,headRefName,state", "--limit", str(_REALIZATION_LISTING_LIMIT),
    ], "pull requests"):
        if not isinstance(item.get("body"), str) or not isinstance(item.get("headRefName"), str):
            raise AgentLoopError("Cannot verify PR listing association fields.")
        _number(item.get("number"), "PR listing")
        if item["headRefName"] == branch_name or parse_strong_issue_reference_evidence(
            item["body"], repo=config.repo, issue_number=issue_number
        ) or item.get("number") in old_prs | current_prs:
            accept(item, pr_items, "PR")

    for kind, target, numbers in (("issue", child_items, old_children | current_children), ("pr", pr_items, old_prs | current_prs)):
        for number in sorted(numbers - target.keys()):
            read = runner.run([
                config.gh_cmd, kind, "view", str(number), "--repo", config.repo, "--json", "number,state",
            ], cwd=active_workdir(config), check=False)
            try:
                item = json.loads(read.stdout)
                if read.returncode or not isinstance(item, dict) or item.get("number") != number:
                    raise ValueError("unreadable reference")
                accept(item, target, "child issue" if kind == "issue" else "PR")
            except (ValueError, TypeError) as exc:
                raise AgentLoopError(f"Cannot verify referenced {kind} #{number}.") from exc
    if not newly_retired:
        # Create-before-summary children are attributed by the current
        # checkpoint's recomputed phase identity, not by their display title.
        checkpoints = []
        for index, comment in enumerate(comments):
            for marker in d.TOPOLOGY_CHECKPOINT_MARKER_RE.finditer(comment.body or ""):
                if ((index, marker.start()), marker[0]) not in historical:
                    checkpoint = d._decode_checkpoint(marker["payload"])
                    if checkpoint.parent_issue == issue_number and checkpoint.plan_hash == plan_hash:
                        checkpoints.append(checkpoint)
        # Fresh topology uses approved-plan sidecars instead of a checkpoint.
        active = [dec for dec in d.live_execution_decisions(comments, parent_issue=issue_number)
                  if dec.plan_hash == plan_hash and dec.strategy == "staged"]
        if active:
            from .round_state import recover_approved_plan_context
            context = recover_approved_plan_context(
                comments, expected_hash=plan_hash, expected_subject=active[-1].plan_subject,
            )
            if not context.is_available or not context.canonical_text:
                raise AgentLoopError("Cannot verify current child ownership: approved plan is unavailable.")
            recommendation = d.recover_execution_recommendation(
                comments, expected_digest=active[-1].recommendation_digest,
            )
            topology, _ = d.normalize_execution_recommendation(
                recommendation, approved_plan=context.canonical_text,
                plan_subject=active[-1].plan_subject,
            )
            checkpoints.append(d.TopologyCheckpoint(
                issue_number, plan_hash, "implement-by-phase", d.EXECUTION_TOPOLOGY_SOURCE,
                topology.phases, execution_strategy_contract_version=1,
            ))
        for checkpoint in checkpoints:
            for ordinal, phase in enumerate(checkpoint.phases, 1):
                identity = d.phase_identity(
                    parent_issue=issue_number, plan_hash=plan_hash,
                    topology_source=checkpoint.topology_source, phase_index=ordinal,
                    phase=phase, stage_id=phase.stage_id,
                    execution_strategy_contract_version=checkpoint.execution_strategy_contract_version,
                )
                for number, item in child_items.items():
                    for marker in d.PHASE_IDENTITY_MARKER_RE.finditer(item.get("body", "")):
                        payload = d._decode_json_payload(marker["payload"], marker_name="phase identity")
                        if (payload.get("identity") == identity and payload.get("plan_hash") == plan_hash
                                and payload.get("parent_issue") == issue_number):
                            current_children.add(number)
        # PRs created before their handoff require the existing exact approved
        # provenance contract; an unrelated or unreadable OPEN PR still blocks.
        from .github import validate_pull_request_provenance
        from .issue_pr_provenance import IssuePrProvenanceScope
        for number, item in pr_items.items():
            if item["state"] == "OPEN" and number not in old_prs | current_prs:
                try:
                    validate_pull_request_provenance(
                        runner, config=config, pr_number=number,
                        expected_scope=IssuePrProvenanceScope(config.repo, issue_number, "approved", plan_hash),
                    )
                except AgentLoopError:
                    continue
                current_prs.add(number)
    if old_children & current_children or old_prs & current_prs:
        raise AgentLoopError("Ambiguous ownership: historical and current execution records claim the same work.")
    for target, current, old, label in ((child_items, current_children, old_children, "child issue"), (pr_items, current_prs, old_prs, "PR")):
        for number, item in target.items():
            if number in current:
                continue
            if item["state"] == "OPEN":
                blockers.append(f"{label} #{number} is OPEN")
            else:
                old.add(number)
    # A bare split marker must map to a child; a checkpoint may be unrealized
    # only after the complete inventories and explicit references above agree.
    for index, comment in enumerate(comments):
        for marker in s.SPLIT_CHILD_MARKER_RE.finditer(comment.body or ""):
            if int(marker["parent"]) != issue_number:
                continue
            matches = [n for n, item in child_items.items() if marker[0] in item.get("body", "")]
            if matches and all(n in current_children for n in matches):
                continue
            if not matches or (index, marker.start()) >= boundary or any(n not in old_children for n in matches):
                blockers.append("an unresolved split child record")
            else:
                historical.append(((index, marker.start()), marker[0]))
    branch = runner.run([
        config.gh_cmd, "api", f"repos/{config.repo}/branches/{urllib.parse.quote(branch_name, safe='')}",
    ], cwd=active_workdir(config), check=False)
    current_branch = any(n in current_prs and item["state"] == "OPEN" and item.get("headRefName") == branch_name for n, item in pr_items.items())
    if branch.returncode == 0 and not current_branch:
        blockers.append(f"branch `{branch_name}` exists; verify its work is finished, then delete the branch manually")
    elif branch.returncode != 0 and not re.search(r"(?:HTTP 404|\b404\b.*Not Found|Not Found.*\b404\b)", branch.stderr or ""):
        blockers.append(f"branch `{branch_name}` could not be checked; verify access and retry")
    if blockers:
        raise AgentLoopError("Conflicting execution decision exists under plan hash " + (", ".join(newly_retired) or plan_hash) + "; live or unverifiable work: " + "; ".join(blockers) + ". Resume live work or verify uncertain evidence before re-approval.")
    return ExecutionHistory(issue_number, frozenset(retired), frozenset(old_children), frozenset(old_prs), tuple(
        ((comments[i[0]].comment_id, comments[i[0]].created_at, comments[i[0]].body), i[1], i[1] + len(raw))
        for i, raw in historical
    ))
