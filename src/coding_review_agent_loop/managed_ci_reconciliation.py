"""Bounded, read-only observation of a repair obligation's historical run.

A historical result is evidence, never a dispatch contract. No renewal, run
attachment, dispatch or rerun is performed by this module.
"""
from __future__ import annotations

from .review_scope import is_complete_code_review

from dataclasses import replace
import json
from types import SimpleNamespace
from urllib.parse import urlparse

from .errors import AgentLoopError
from .managed_ci_result import FailedManagedRun, VerifiedManagedResult
from .managed_ci_intent_validation import IntentBinding, classify_intent_page, match_intent_envelope


def capture_failed_run(contract, checkpoint, source_round):
    if contract is None or checkpoint is None or checkpoint.obligation_kind != 'managed-exact-head-ci':
        return None
    if contract.expected_head_sha != checkpoint.failed_head_sha or contract.base_ref != checkpoint.base_branch:
        return None
    try:
        return FailedManagedRun(
            repository=contract.repository, pr_number=contract.pr_number,
            head_sha=checkpoint.failed_head_sha, base_ref=checkpoint.base_branch,
            obligation_identity=checkpoint.obligation_identity, source_round=source_round,
            audit_comment_id=contract.intent_comment_id, nonce=contract.nonce,
            run_id=contract.attached_run_id, failed_attempt=contract.run_attempt,
            generation=contract.intent_generation, workflow_revision=contract.workflow_revision,
        )
    except (TypeError, ValueError):
        return None


def _record(page, contract, ref):
    verdict = classify_intent_page(
        page, requested_nonce=ref.nonce, trusted_login=contract.trusted_actor_login,
        trusted_id=contract.trusted_actor_id, visible_capable=contract.visible_intent_capable,
        host_footer_capable=contract.host_footer_capable,
        binding=IntentBinding(ref.repository, ref.pr_number, ref.head_sha, ref.base_ref,
                              ref.workflow_revision, ref.nonce),
    )
    record = verdict.record
    if (verdict.outcome != 'authorized' or verdict.comment_id != ref.audit_comment_id
        or record is None or record['generation'] != ref.generation
        or record['run_id'] != ref.run_id or record['run_attempt'] != ref.failed_attempt
        or record['state'] != 'completed'):
        return None
    return record


def resolve_legacy_reference(page, contract, checkpoint):
    """Require a unique completed intent inside the authenticated failure round."""
    from .round_state import _extract_round_metadata_records
    # GitHub user objects carry extra fields; compare only the authenticated pair.
    trusted = [c for c in page if isinstance(c, dict) and isinstance(c.get('user'), dict)
               and c['user'].get('login') == contract.trusted_actor_login
               and c['user'].get('id') == contract.trusted_actor_id]
    records = _extract_round_metadata_records([SimpleNamespace(body=c.get('body')) for c in trusted], flow='pr')
    anchors = [(trusted[r.index], r.metadata) for r in records
               if r.metadata.qualification_checkpoint is not None
               and r.metadata.qualification_checkpoint.obligation_identity == checkpoint.obligation_identity
               and r.metadata.qualification_checkpoint.failed_head_sha == checkpoint.failed_head_sha
               and r.metadata.qualification_checkpoint.lifecycle == 'repair_required']
    if not anchors:
        return None
    anchor, meta = min(anchors, key=lambda pair: pair[0].get('id', 0))
    item = next((i for i in meta.prior_items if i.obligation_identity == checkpoint.obligation_identity), None)
    if item is None:
        return None
    lower = [trusted[r.index].get('id', 0) for r in records
             if r.metadata.round_number == item.source_round and r.metadata.subject == checkpoint.failed_head_sha
             and is_complete_code_review(r.metadata) and trusted[r.index].get('id', 0) < anchor.get('id', 0)]
    if not lower:
        return None
    candidates = []
    for c in trusted:
        if not (max(lower) < c.get('id', 0) < anchor.get('id', 0)):
            continue
        match = match_intent_envelope(c.get('body'), visible_capable=contract.visible_intent_capable,
                                      host_footer_capable=contract.host_footer_capable)
        if match is None:
            continue
        try:
            raw = json.loads(match.group('payload'))
            ref = FailedManagedRun(
                contract.repository, contract.pr_number, checkpoint.failed_head_sha,
                checkpoint.base_branch, checkpoint.obligation_identity, item.source_round,
                c['id'], raw['nonce'], raw['run_id'], raw['run_attempt'], raw['generation'], raw['workflow_revision'])
        except (KeyError, TypeError, ValueError):
            continue
        if _record(page, contract, ref) is not None:
            candidates.append(ref)
    return candidates[0] if len(candidates) == 1 else None


def observe_repair_result(runner, *, config, pr_number, metadata, contract, checkpoint):
    """One live observation, returning evidence only for a later passing attempt."""
    from . import managed_ci as ci
    if (checkpoint is None or not checkpoint.valid or contract is None
        or contract.protocol_version != 2 or contract.activation_path != 'managed'
        or checkpoint.obligation_kind != 'managed-exact-head-ci'
        or checkpoint.lifecycle != 'repair_required'
        or metadata.head_sha != checkpoint.failed_head_sha
        or (metadata.base_branch or config.base) != checkpoint.base_branch):
        return None
    try:
        page = ci._read_intent_comment_page(runner, config, pr_number)
        ref = checkpoint.failed_managed_run or resolve_legacy_reference(
            page, replace(contract, repository=config.repo, pr_number=pr_number), checkpoint)
        if ref is None:
            return None
        if (ref.repository, ref.pr_number, ref.head_sha, ref.base_ref, ref.obligation_identity) != (
            config.repo, pr_number, metadata.head_sha, metadata.base_branch or config.base, checkpoint.obligation_identity):
            return None
        record = _record(page, contract, ref)
        if record is None:
            return None
        # A newer authenticated intent for this target supersedes this result.
        for comment in page:
            if (not isinstance(comment, dict) or type(comment.get('id')) is not int
                or comment['id'] <= ref.audit_comment_id
                or not isinstance(comment.get('user'), dict)
                or comment['user'].get('login') != contract.trusted_actor_login
                or comment['user'].get('id') != contract.trusted_actor_id):
                continue
            match = match_intent_envelope(comment.get('body'), visible_capable=contract.visible_intent_capable,
                                          host_footer_capable=contract.host_footer_capable)
            if match is not None:
                other = json.loads(match.group('payload'))
                if (other.get('repository'), other.get('pr'), other.get('expected_head_sha'), other.get('base_ref')) == (
                    ref.repository, ref.pr_number, ref.head_sha, ref.base_ref):
                    return None
        if (ci._advertised_managed_actor(runner, config) != contract.trusted_actor_login
            or ci.managed_label_present(runner, config=config, pr_number=pr_number) is not True):
            return None
        if not ci.revalidate_adopted_managed_ci(runner, config=config, pr_number=pr_number,
                                                metadata=metadata, contract=contract):
            return None
        ci.require_recorded_base(runner, config=config, pr_number=pr_number, base_ref=ref.base_ref,
                                 repository=ref.repository, head_sha=ref.head_sha, action='repair reconciliation')
        # A private view supplies the original workflow identity to shared
        # predicates. It is never returned or passed to a dispatch consumer.
        view = replace(contract, expected_head_sha=ref.head_sha, nonce=ref.nonce,
                       attached_run_id=ref.run_id, run_attempt=ref.failed_attempt,
                       workflow_revision=ref.workflow_revision,
                       terminal_attempts=tuple((x['run_id'], x['run_attempt']) for x in record['terminal_attempts']))
        snapshot = ci._refresh_v2_attached_run(runner, config=config, contract=view, strict_provenance=True)
        if (snapshot is None or type(snapshot.run_attempt) is not int
            or snapshot.run_attempt <= ref.failed_attempt or snapshot.status != 'completed'
            or snapshot.conclusion != 'success'):
            return None
        view.run_attempt = snapshot.run_attempt
        final = ci._v2_correlated_status(runner, config=config, expected_head=ref.head_sha, contract=view)
        if (final is None or final.status != 'success' or not final.url
            or urlparse(final.url)._replace(query='', fragment='').geturl().rstrip('/')
            != f'https://github.com/{ref.repository}/actions/runs/{ref.run_id}'):
            return None
        board = ci.get_pr_checks(runner, config=config, metadata=metadata)
        if not (ci.complete_managed_board(board) and not board.check_query_errors and not board.failing):
            return None
        mergeability = ci.get_pr_mergeability(runner, config=config, pr_number=pr_number)
        if mergeability.state == 'conflicted' or ci.get_pr_head_sha(runner, config, pr_number) != ref.head_sha:
            return None
        return VerifiedManagedResult(ref, snapshot.run_attempt)
    except (AgentLoopError, KeyError, TypeError, ValueError):
        return None


def require_verified_result(runner, *, config, pr_number, expected_head_sha, contract, result):
    """Re-observe persisted provenance before any finalization side effect."""
    from .qualification_checkpoint import QualificationCheckpoint
    from .github import PullRequestMetadata
    ref = result.failed_run
    try:
        result.require_binding(repository=config.repo, pr_number=pr_number,
                               head_sha=expected_head_sha, base_ref=contract.base_ref)
    except ValueError as exc:
        raise AgentLoopError(str(exc)) from exc
    checkpoint = QualificationCheckpoint('managed-exact-head-ci', ref.obligation_identity,
                                         'repair_required', ref.head_sha, None, base_branch=ref.base_ref,
                                         failed_managed_run=ref, verified_managed_result=result)
    metadata = PullRequestMetadata(number=pr_number, repo=config.repo, title=None, head_branch=None,
                                   head_sha=expected_head_sha, base_branch=ref.base_ref, url=None)
    if observe_repair_result(runner, config=config, pr_number=pr_number, metadata=metadata,
                             contract=contract, checkpoint=checkpoint) != result:
        raise AgentLoopError('Historical managed result no longer qualifies the live head.')


def historical_qualification_fields(result):
    ref = result.failed_run
    return (f'nonce={ref.nonce} run_id={ref.run_id} attempt={result.successful_attempt} '
            f'generation={ref.generation} source_audit_record={ref.audit_comment_id}')


def reconcile_repair_checkpoint(runner, *, config, pr_number, metadata, contract, checkpoint, items):
    """Clear only the bound obligation, or restore it if settlement went stale."""
    if (checkpoint is not None and checkpoint.valid and contract is not None
        and checkpoint.obligation_kind == 'managed-exact-head-ci'
        and checkpoint.lifecycle == 'repair_required' and checkpoint.failed_managed_run is None
        and checkpoint.failed_head_sha == metadata.head_sha
        and checkpoint.base_branch == (metadata.base_branch or config.base)):
        from . import managed_ci as ci
        try:
            page = ci._read_intent_comment_page(runner, config, pr_number)
            reference = resolve_legacy_reference(
                page, replace(contract, repository=config.repo, pr_number=pr_number), checkpoint)
            if reference is not None:
                checkpoint = replace(checkpoint, failed_managed_run=reference)
        except (AgentLoopError, KeyError, TypeError, ValueError):
            pass
    result = observe_repair_result(runner, config=config, pr_number=pr_number,
                                   metadata=metadata, contract=contract, checkpoint=checkpoint)
    if result is not None and not any(
        i.obligation_kind == 'managed-exact-head-ci'
        and i.obligation_identity == result.failed_run.obligation_identity
        and i.source_round == result.failed_run.source_round for i in items
    ):
        result = None
    if checkpoint is None:
        return None, checkpoint, items
    if result is not None:
        checkpoint = replace(checkpoint, failed_managed_run=result.failed_run, verified_managed_result=result)
        items = [replace(i, status='resolved', lifecycle='cleared')
                 if i.obligation_kind == 'managed-exact-head-ci'
                 and i.obligation_identity == checkpoint.obligation_identity else i for i in items]
    elif checkpoint.verified_managed_result is not None:
        # Durable clearance is a cache. Losing live evidence restores the
        # same obligation and its spent budget rather than allowing dispatch.
        checkpoint = replace(checkpoint, verified_managed_result=None)
        items = [replace(i, status='blocking', lifecycle='repair_required')
                 if i.obligation_kind == 'managed-exact-head-ci'
                 and i.obligation_identity == checkpoint.obligation_identity else i for i in items]
    return result, checkpoint, items


def historical_outcome(runner, *, config, pr_number, metadata, contract, result):
    from . import managed_ci as ci
    require_verified_result(runner, config=config, pr_number=pr_number,
                            expected_head_sha=metadata.head_sha, contract=contract, result=result)
    return ci.ManagedCiOutcome(status='passed', head_sha=metadata.head_sha,
                               checks=ci.get_pr_checks(runner, config=config, metadata=metadata))


def require_repair_handoff_failure(runner, *, config, pr_number, metadata, contract, checkpoint):
    """Stop boundedly when a recorded run cannot currently justify repair.

    A pending/unreadable rerun or a green run with an incomplete counted board
    retains its checkpoint. It must not consume coder rounds while waiting.
    Old checkpoints without provable correlation keep their existing recovery.
    """
    from . import managed_ci as ci
    if (contract is None or checkpoint is None or not checkpoint.valid
        or checkpoint.lifecycle != 'repair_required' or checkpoint.failed_managed_run is None):
        return
    ref = checkpoint.failed_managed_run
    if metadata.head_sha != ref.head_sha:
        return  # another head follows existing review/qualification recovery
    message = (f'PR #{pr_number}: recorded managed run {ref.run_id} is pending, unreadable, '
               'or lacks authoritative failure evidence; repair remains unresolved. Resume after '
               'the current attempt and complete check board are inspectable.')
    from .managed_ci_failure import require_live_failure
    if require_live_failure(runner, config=config, pr_number=pr_number, metadata=metadata,
                            contract=contract, checkpoint=checkpoint):
        return
    try:
        page = ci._read_intent_comment_page(runner, config, pr_number)
        record = _record(page, contract, ref)
        if record is None or ci._advertised_managed_actor(runner, config) != contract.trusted_actor_login:
            raise AgentLoopError(message)
        # Completed historical baselines retain their existing authentication;
        # a newer live intent does not revoke their repair obligation.
        view = replace(contract, nonce=ref.nonce, attached_run_id=ref.run_id,
                       run_attempt=ref.failed_attempt, workflow_revision=ref.workflow_revision,
                       terminal_attempts=tuple((x['run_id'], x['run_attempt']) for x in record['terminal_attempts']))
        snapshot = ci._refresh_v2_attached_run(runner, config=config, contract=view, strict_provenance=True)
        if (snapshot is not None and snapshot.live_run_attempt == ref.failed_attempt
            and snapshot.run_attempt == ref.failed_attempt and snapshot.status == 'in_progress'
            and snapshot.conclusion is None):
            final = ci._v2_correlated_status(runner, config=config, expected_head=ref.head_sha, contract=view)
            if (final is not None and final.status in {'failure', 'error', 'timed_out', 'cancelled'}
                and final.url and urlparse(final.url)._replace(query='', fragment='').geturl().rstrip('/')
                == f'https://github.com/{ref.repository}/actions/runs/{ref.run_id}'):
                return  # publication can precede completion of the baseline run
        if (snapshot is not None and type(snapshot.run_attempt) is int
            and snapshot.run_attempt >= ref.failed_attempt and snapshot.status == 'completed'
            and snapshot.conclusion in {'failure', 'timed_out', 'cancelled'}):
            if snapshot.run_attempt == ref.failed_attempt:
                return  # durable failure baseline, including terminal-without-status
            view.run_attempt = snapshot.run_attempt
            final = ci._v2_correlated_status(runner, config=config, expected_head=ref.head_sha, contract=view)
            if final is not None and final.status in {'failure', 'error', 'timed_out', 'cancelled'}:
                return
    except (KeyError, TypeError, ValueError) as exc:
        raise AgentLoopError(message) from exc
    raise AgentLoopError(message)
