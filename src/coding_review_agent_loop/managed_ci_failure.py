"""Attempt-bound live failure evidence, separate from historical qualification."""
from __future__ import annotations

from dataclasses import replace
import json
from urllib.parse import urlparse

from .ci_health import check_name_matches, decode_concatenated_json_objects
from .errors import AgentLoopError
from .managed_ci_intent_validation import IntentBinding, classify_intent_page, match_intent_envelope
from .workdirs import active_workdir


def _is_v2_intent_run(
    run: dict[str, object], *, contract: ManagedCiContract, run_name: str
) -> bool:
    """Validate the API fields available for a dispatched base workflow."""
    from . import managed_ci as ci
    if (
        run_name not in (run.get("name"), run.get("display_title"))
        or run.get("event") != "workflow_dispatch"
    ):
        return False
    path = run.get("path")
    if isinstance(path, str) and f".github/workflows/{ci.WORKFLOW_FILE}" not in path:
        return False
    run_ref = contract.dispatch_ref or contract.base_ref
    if isinstance(run.get("head_branch"), str) and run_ref and run["head_branch"] != run_ref:
        return False
    if isinstance(run.get("head_sha"), str) and contract.workflow_revision and run["head_sha"] != contract.workflow_revision:
        return False
    for field in ("actor", "triggering_actor"):
        identity = run.get(field)
        if not isinstance(identity, dict):
            continue
        login = identity.get("login")
        actor_id = identity.get("id")
        if (
            login != contract.trusted_actor_login
            or actor_id != contract.trusted_actor_id
        ):
            return False
    return True

def original_view(page, contract, ref):
    """Authenticate an original attachment without adopting it for dispatch."""
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
        or record['state'] not in {'attached', 'completed'}):
        return None
    for comment in page:
        if (comment.get('id', 0) <= ref.audit_comment_id
            or comment.get('user', {}).get('login') != contract.trusted_actor_login
            or comment.get('user', {}).get('id') != contract.trusted_actor_id):
            continue
        match = match_intent_envelope(comment.get('body'), visible_capable=contract.visible_intent_capable,
                                      host_footer_capable=contract.host_footer_capable)
        if match:
            other = json.loads(match.group('payload'))
            if (other.get('repository'), other.get('pr'), other.get('expected_head_sha'), other.get('base_ref')) == (
                ref.repository, ref.pr_number, ref.head_sha, ref.base_ref):
                return None
    return replace(
        contract, repository=ref.repository, pr_number=ref.pr_number,
        expected_head_sha=ref.head_sha, base_ref=ref.base_ref, nonce=ref.nonce,
        attached_run_id=ref.run_id, run_attempt=ref.failed_attempt, workflow_revision=ref.workflow_revision,
        intent_comment_id=ref.audit_comment_id, intent_generation=ref.generation,
        created_at=record['created_at'], intent_state=record['state'],
        terminal_run_id=record['terminal_run_id'], terminal_run_attempt=record['terminal_run_attempt'],
        terminal_outcome=record['terminal_outcome'],
        terminal_attempts=tuple((x['run_id'], x['run_attempt']) for x in record['terminal_attempts']),
    )


def read_attempt_jobs(runner, config, run_id, attempt):
    """Require complete consistent pages and exact job identities for one attempt."""
    result = runner.run(
        [config.gh_cmd, 'api', '--paginate',
         f'repos/{config.repo}/actions/runs/{run_id}/attempts/{attempt}/jobs?per_page=100'],
        cwd=active_workdir(config), check=False,
    )
    if result.returncode:
        return None
    pages = decode_concatenated_json_objects(result.stdout or '')
    if not pages:
        return None
    jobs, ids, total = [], set(), None
    for page in pages:
        count, values = page.get('total_count'), page.get('jobs')
        if type(count) is not int or count < 0 or not isinstance(values, list):
            return None
        if total is not None and count != total:
            return None
        total = count
        for job in values:
            if (not isinstance(job, dict) or type(job.get('id')) is not int or job['id'] <= 0
                or job['id'] in ids or type(job.get('run_id')) is not int or job['run_id'] != run_id
                or type(job.get('run_attempt')) is not int or job['run_attempt'] != attempt
                or not isinstance(job.get('name'), str) or not job['name'].strip()
                or job.get('status') not in {'queued', 'waiting', 'pending', 'in_progress', 'completed'}):
                return None
            ids.add(job['id'])
            conclusion = job.get('conclusion')
            if job['status'] == 'completed' and not isinstance(conclusion, str):
                return None
            if job['status'] != 'completed' and conclusion is not None:
                return None
            url = job.get('html_url')
            if url is not None and not isinstance(url, str):
                return None
            jobs.append((job['name'], conclusion or job['status'], url or ''))
    return tuple(jobs) if len(jobs) == total else None


def live_authority(runner, *, config, pr_number, metadata, contract):
    from . import managed_ci as ci
    if contract.adopted_existing_pr and contract.guard_head_sha != metadata.head_sha:
        return False  # observation must not publish an adoption guard
    if (ci._advertised_managed_actor(runner, config) != contract.trusted_actor_login
        or ci.managed_label_present(runner, config=config, pr_number=pr_number) is not True
        or not ci.revalidate_adopted_managed_ci(runner, config=config, pr_number=pr_number,
                                               metadata=metadata, contract=contract)):
        return False
    if contract.active_label_event_id is not None:
        event = ci._active_managed_label_event(runner, config=config, pr_number=pr_number)
        if event != (contract.active_label_event_id, contract.trusted_actor_login, contract.trusted_actor_id):
            return False
    resume = contract.authenticated_resume
    if resume is not None and resume.issue_created_handoff is not None:
        ci.revalidate_issue_created_handoff(runner, config=config, metadata=metadata,
                                            handoff=resume.issue_created_handoff)
    ci.require_recorded_base(runner, config=config, pr_number=pr_number, base_ref=contract.base_ref,
                             repository=config.repo, head_sha=metadata.head_sha, action='live failure observation')
    # Mergeability determines coder routing, not failure-evidence authority.
    # The watcher handles conflicts before observing jobs; handoff must allow
    # the existing conflict-resolution turn with an authenticated failure.
    return True


def observe_active_failure(runner, *, config, pr_number, metadata, contract, minimum=None):
    """Freshly bracket a narrow failed-job observation with strict live identity."""
    from . import managed_ci as ci
    from .managed_ci_result import FailedManagedRun
    if (contract is None or contract.protocol_version != 2 or contract.attached_run_id is None
        or type(contract.run_attempt) is not int or contract.run_attempt <= 0
        or (minimum is not None and contract.run_attempt < minimum)):
        return None
    try:
        ref = FailedManagedRun(config.repo, pr_number, metadata.head_sha, contract.base_ref,
                               'live-observation', 1, contract.intent_comment_id, contract.nonce,
                               contract.attached_run_id, contract.run_attempt, contract.intent_generation,
                               contract.workflow_revision)
        page = ci._read_intent_comment_page(runner, config, pr_number)
        view = original_view(page, contract, ref)
        if view is None or ci._v2_attachment_is_excluded(view):
            return None
        def snapshot():
            value = ci._refresh_v2_attached_run(runner, config=config, contract=view, strict_provenance=True)
            return value if (value is not None and value.live_run_attempt == ref.failed_attempt
                             and value.status == 'in_progress' and value.conclusion is None) else None
        before = snapshot()
        if before is None or not live_authority(runner, config=config, pr_number=pr_number,
                                                metadata=metadata, contract=view):
            return None
        jobs = read_attempt_jobs(runner, config, ref.run_id, ref.failed_attempt)
        if jobs is None or not any(c in {'failure', 'timed_out'} and not check_name_matches(n, ci.plumbing_check_names(config))
                                   for n, c, _u in jobs):
            return None
        if snapshot() is None or not live_authority(runner, config=config, pr_number=pr_number,
                                                    metadata=metadata, contract=view):
            return None
        if original_view(ci._read_intent_comment_page(runner, config, pr_number), contract, ref) != view:
            return None
        return before, jobs
    except (AgentLoopError, KeyError, TypeError, ValueError, AttributeError):
        return None


def require_live_failure(runner, *, config, pr_number, metadata, contract, checkpoint):
    """Revalidate an original failure for handoff; settle only its terminal baseline."""
    from . import managed_ci as ci
    ref = checkpoint.failed_managed_run
    if (contract.protocol_version != 2 or contract.activation_path != 'managed'
        or (ref.repository, ref.pr_number, ref.head_sha, ref.base_ref, ref.obligation_identity) != (
            config.repo, pr_number, metadata.head_sha, metadata.base_branch or config.base,
            checkpoint.obligation_identity)):
        return False
    try:
        view = original_view(ci._read_intent_comment_page(runner, config, pr_number), contract, ref)
        if view is None or ci._v2_attachment_is_excluded(view):
            return False
        original_binding = replace(view)
        early = observe_active_failure(runner, config=config, pr_number=pr_number,
                                       metadata=metadata, contract=view)
        if early is not None:
            return True
        # Only an attached original baseline needs settlement. Historical
        # completed-result handling remains in reconciliation, unchanged.
        if view.intent_state != 'attached' or not live_authority(
            runner, config=config, pr_number=pr_number, metadata=metadata, contract=view
        ):
            return False
        def terminal():
            snapshot = ci._refresh_v2_attached_run(runner, config=config, contract=view, strict_provenance=True)
            return snapshot if (snapshot is not None and snapshot.live_run_attempt == ref.failed_attempt
                                and snapshot.status == 'completed'
                                and snapshot.conclusion in {'failure', 'timed_out', 'cancelled'}) else None
        if terminal() is None:
            return False
        final = ci._v2_correlated_status(runner, config=config, expected_head=ref.head_sha, contract=view)
        if final is not None and final.status in ci._TERMINAL_CI_STATUSES:
            if final.status == 'success':
                return False
        else:
            # The existing terminal-without-status rule requires a reread and
            # a second poll, leaving the active publisher's attachment intact.
            final = ci._v2_correlated_status(runner, config=config, expected_head=ref.head_sha, contract=view)
            if final is None or final.status not in ci._TERMINAL_CI_STATUSES:
                runner.run(['sleep', str(config.ci_poll_interval_seconds)], cwd=active_workdir(config))
                if terminal() is None:
                    return False
                final = ci._v2_correlated_status(runner, config=config, expected_head=ref.head_sha, contract=view)
                if final is None or final.status not in ci._TERMINAL_CI_STATUSES:
                    view.terminal_run_id, view.terminal_run_attempt = ref.run_id, ref.failed_attempt
                    view.terminal_attempts += ((ref.run_id, ref.failed_attempt),)
                    view.terminal_outcome = 'no-status'
            if final is not None and final.status == 'success':
                return False
        if terminal() is None or not live_authority(
            runner, config=config, pr_number=pr_number, metadata=metadata, contract=view
        ):
            return False
        original = original_view(ci._read_intent_comment_page(runner, config, pr_number), contract, ref)
        if original != original_binding:
            return False
        if final is not None and final.status in ci._TERMINAL_CI_STATUSES and (
            not final.url or urlparse(final.url)._replace(query='', fragment='').geturl().rstrip('/')
            != f'https://github.com/{ref.repository}/actions/runs/{ref.run_id}'
        ):
            return False
        ci._patch_intent(runner, config=config, contract=view, state='completed')
        return True
    except (AgentLoopError, KeyError, TypeError, ValueError, AttributeError):
        return False
