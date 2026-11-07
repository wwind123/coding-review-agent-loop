"""Durable and invocation-local visual review lifecycle; image evidence has no code authority."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import subprocess
import time

from .agents.registry import agent_display_name
from .errors import AgentLoopError, HumanDecisionRequiredError
from .review_images_config import CaptureConfig, delivery_limits, eligible
from .review_images_capabilities import preflight_images, reviewer_view
from .review_images_capture import capture
from .review_images_prompts import build_visual_prompt
from .review_scope import is_visual_item, is_visual_review, visual_candidate, lifecycle_present


def settings(config):
    # The disabled path must not probe runtimes, reviewers or changed files.
    if not config.review_images_paths:
        return None
    limits = replace(delivery_limits(), entries=config.review_images_max_images,
                     total_bytes=config.review_images_max_bytes,
                     total_pixels=config.review_images_max_pixels)
    limits = delivery_limits(limits)
    result = CaptureConfig(paths=config.review_images_paths, command=config.review_images_cmd,
                           runtime_grants=config.review_images_runtime_grant,
                           capture_timeout=config.review_images_timeout, limits=limits)
    if result.capture_timeout > 1800:
        raise ValueError('capture timeout exceeds 1800 seconds')
    if not result.runtime_grants:
        raise ValueError('enabled image review requires operator runtime grants')
    from .review_images_isolation import validate_grants
    # Shape/trust admission at startup; sandbox/runtime coverage at capture.
    validate_grants(result.runtime_grants)
    return result


def startup(config):
    if not config.review_images_paths or config.dry_run:
        return
    try:
        if settings(config) is not None:
            preflight_images(config, config.review_images_seat)
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        raise AgentLoopError('Image review startup refused: ' + str(exc)[:1024]) from exc


def ordinary_items(items):
    return tuple(item for item in items if not is_visual_item(item))


def assigned_items(items, reviewer):
    return tuple(item for item in items if is_visual_item(item)
                 and reviewer in (item.resolution_owners or (item.reviewer,)))


def validate_visual(text, *, reviewer, items):
    from .unresolved_items import _validate_review_response
    from .protocol import parse_pr_review
    # Explicit dispositions are mandatory; prose inference is not proof.
    raw = parse_pr_review(text, reviewer=reviewer, architecture_status_mode='strict')
    ids = [d.item_id for d in raw.dispositions]
    if len(ids) != len(set(ids)) or set(ids) != {item.item_id for item in items}:
        raise AgentLoopError('Visual dispositions must name exactly the assigned findings')
    parsed = _validate_review_response(text, reviewer=reviewer, unresolved_items=items,
                                      architecture_status_mode='strict', include_visual=True)
    if parsed.exact_head_evidence_requests or parsed.screenshot_requests or parsed.screenshot_coverage_acceptances or any(
            getattr(item, 'evidence_row_ids', ()) for item in
            (*parsed.blocking_items, *parsed.followups.same_pr, *parsed.followups.future)):
        raise AgentLoopError('Visual assessment cannot request screenshot or matrix coverage')
    return parsed


class StaleDispatchInputs(ValueError):
    """The observed tuple moved before capture; refresh eligibility first."""


def pinned_diff(runner, config, metadata, checkout):
    """Read the live immutable base/head tuple, import missing closures, diff SHAs."""
    from .github_retry import run_gh_read
    from .workdirs import github_api_cwd
    from .git_transport import import_ref
    from .run_worktrees import _transport_options
    from .review_images_worktree import git
    result = run_gh_read(runner, [config.gh_cmd, 'api',
                         f'repos/{config.repo}/pulls/{metadata.number}'],
                         cwd=github_api_cwd(), check=False)
    if result.returncode:
        raise ValueError('pinned PR tuple unavailable')
    value = json.loads(result.stdout)
    head = value['head']['sha']
    base = value['base']['sha']
    if head != metadata.head_sha or value['base']['ref'] != metadata.base_branch:
        raise StaleDispatchInputs('PR tuple changed before capture')
    for sha in (head, base):
        if not isinstance(sha, str) or len(sha) != 40 or any(c not in '0123456789abcdef' for c in sha):
            raise ValueError('invalid pinned commit')
    for sha, ref, target in ((head, f'refs/pull/{metadata.number}/head', 'head'),
                             (base, f'refs/heads/{metadata.base_branch}', 'base')):
        try:
            git(checkout, ['rev-list', '--objects', '--missing=error', sha], cap=1, truncate=True)
        except ValueError:
            imported = import_ref(checkout, ref, f'refs/agent-loop/images/{metadata.number}/{target}',
                                  repo=config.repo, runner=runner, gh_cmd=config.gh_cmd,
                                  local_origin=config.trusted_local_origin, **_transport_options(config))
            if imported != sha:
                raise StaleDispatchInputs('PR ref moved during import')
    ancestor = git(checkout, ['merge-base', base, head], cap=128).decode().strip()
    if len(ancestor) != 40 or any(c not in '0123456789abcdef' for c in ancestor):
        raise ValueError('invalid merge base')
    return git(checkout, ['diff', '--no-ext-diff', '--no-textconv', '--ignore-submodules=all',
                          ancestor, head, '--'], cap=32*1024, truncate=True).decode(errors='replace')


def observe_ci(runner, config, metadata, *, managed, diagnostic):
    from .github import get_pr_checks
    deadline = time.monotonic() + config.ci_startup_timeout_seconds
    while True:
        try:
            checks = get_pr_checks(runner, config=config, metadata=metadata)
        except (AgentLoopError, OSError, ValueError, subprocess.SubprocessError):
            diagnostic('Image review: CI observations unavailable; proceeding')
            return
        if checks.check_query_status != 'ok' or checks.state == 'unavailable':
            diagnostic('Image review: CI observations unavailable; proceeding')
            return
        if checks.passing or checks.pending or checks.failing:
            return
        if managed:
            diagnostic('Image review: final CI is approval-gated; proceeding without dispatch')
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            diagnostic('Image review: no CI appeared within startup bound; proceeding')
            return
        time.sleep(min(config.ci_poll_interval_seconds, remaining))


@dataclass(frozen=True)
class VisualOutcome:
    response: object | None = None
    parsed: object | None = None
    scope: dict | None = None
    diagnostic: str = ''
    stale: bool = False
    lifecycle: dict | None = None
    fresh: bool = False
    coverage: dict | None = None


class VisualRun:
    def __init__(self, runner=None):
        if runner is not None:
            runner._visual_boundary_paths = {}
        self.attempted = set()
        self.coder_heads = set()
        self.reasons = {}
        self.outcomes = {}

    @staticmethod
    def rebind_items(items, ledger_view):
        if not any(is_visual_item(item) for item in items):
            return items
        derived = {item.item_id: item for item in ledger_view(items)}
        return [derived.get(item.item_id, item) if is_visual_item(item) else item for item in items]

    def blocker_stop_reason(self, blockers, head, outcome):
        """Success offers an opportunity only while the selected owner is pending."""
        if outcome is not None and outcome.response is not None and outcome.fresh:
            seat = outcome.scope['seat']
            assigned_ids = {item.item_id for item in assigned_items(blockers, seat)
                            if dict(item.owner_states).get(seat, 'pending') != 'cleared'}
            unassigned = [item for item in blockers if item.item_id not in assigned_ids]
            if unassigned:
                owners = '; '.join(
                    f"{item.item_id} owners {', '.join(owner for owner in
                        (item.resolution_owners or (item.reviewer,))
                        if dict(item.owner_states).get(owner, 'pending') != 'cleared')} (pending)"
                    for item in unassigned)
                return f"visual seat {seat} cannot disposition {owners}"
            if head not in self.coder_heads:
                return None
        return self.reasons.get(head, 'the durable visual opportunity is spent; use a new head with changed images and an explicit assessment disposition, or an explicit human decision')

    def attempt(self, runner, *, config, metadata, checkout, items, managed, current_head, diagnostic, round_number=1, usage_context=None):
        if not config.review_images_paths or config.dry_run:
            return VisualOutcome(diagnostic='disabled')
        head = metadata.head_sha
        if head in self.attempted:
            cached = self.outcomes.get(head)
            if cached is not None and not cached.stale:
                return cached
            return VisualOutcome(diagnostic=self.reasons.get(head, 'visual attempt already spent'))
        self.attempted.add(head)
        history = load_history(runner, config, metadata.number)
        try:
            from .github_changed_paths import get_pr_changed_paths
            settings_ = settings(config)
            paths = get_pr_changed_paths(runner, config=config, pr_number=metadata.number, head_sha=head)
            _cache_paths(runner, config, metadata, paths)
            if not paths.available:
                if current_head() != head:
                    return VisualOutcome(stale=True)
                raise ValueError(paths.diagnostic)
            changes = [path.as_record() for path in paths.changes]
            if not eligible(settings_, changes):
                self.reasons[head] = 'no matching changed paths'
                return VisualOutcome(diagnostic=self.reasons[head])
            from .review_images_state import reduce_history
            admission = reduce_history(history, repo=config.repo, pr=metadata.number, head=head)
            if admission.spent:
                reason = admission.diagnostic or 'recovered durable visual result; no coder opportunity'
                self.reasons[head] = reason
                return VisualOutcome(diagnostic=reason)
            observe_ci(runner, config, metadata, managed=managed, diagnostic=diagnostic)
            diff = pinned_diff(runner, config, metadata, checkout)
            if current_head() != head:
                return VisualOutcome(stale=True)
            result = capture(checkout, head, settings_, changes)
            if result.state != 'success' or result.snapshots is None:
                raise ValueError('capture ' + result.state + ': ' + result.diagnostic)
            with result.snapshots as snapshots:
                if current_head() != head:
                    return VisualOutcome(stale=True)
                # Admission is rebound to the final board/config for this dispatch.
                capability = preflight_images(config, config.review_images_seat)
                selected, _, _, _ = reviewer_view(config, config.review_images_seat)
                name = agent_display_name(selected)
                assigned = assigned_items(items, name)
                from .review_images_state import new_binding, can_reuse, owner_shares
                binding = new_binding(config, metadata, snapshots, settings_, capability, name)
                if can_reuse(admission, binding, assigned):
                    binding.update(kind='skip', source_id=admission.latest[0], original_id=admission.original[0],
                                   assigned=[list(share) for share in owner_shares(assigned, name)])
                    if current_head() != head:
                        return VisualOutcome(stale=True)
                    coverage = coverage_attachment(snapshots, binding, config, metadata.number,
                        proof=admission.original[1]["proof"], source_id=admission.original[0], diagnostic=diagnostic)
                    persist_lifecycle(runner, config, metadata, binding, round_number=round_number, coverage=coverage)
                    if current_head() != head:
                        return VisualOutcome(stale=True)
                    self.reasons[head] = 'unchanged images reused from a durable assessment; no coder opportunity'
                    return VisualOutcome(diagnostic=self.reasons[head], lifecycle=binding)
                if current_head() != head:
                    return VisualOutcome(stale=True)
                intent = persist_lifecycle(runner, config, metadata, binding, round_number=round_number)
                if current_head() != head:
                    return VisualOutcome(stale=True)
                from .review_images_transport import acquire_images
                with acquire_images(snapshots, capability, head_sha=head,
                                    config_digest=snapshots.config_digest,
                                    snapshot_digest=snapshots.digest, limits=settings_.limits) as images:
                    prompt = build_visual_prompt(images, diff) + visual_response_instructions(assigned)
                    if current_head() != head:
                        return VisualOutcome(stale=True)
                    from .validated_agent import _run_validated_agent
                    response = _run_validated_agent(runner, agent=selected, config=config,
                        prompt=prompt, role='reviewer', session_id=None, timeout_seconds=300,
                        label='visible-defects', image_inputs=images,
                        validate=lambda text: validate_visual(text, reviewer=name, items=assigned),
                        use_repair=False, usage_context=usage_context)
                if current_head() != head:
                    return VisualOutcome(stale=True)
                if response.acquisition_outcome != 'success':
                    raise ValueError('visual reviewer acquisition did not succeed')
                parsed = validate_visual(response.text, reviewer=name, items=assigned)
                from .review_images_state import digest
                binding.update(kind='result', intent_id=intent.comment_id,
                               proof=digest(asdict(response.image_delivery_proof)))
                coverage = coverage_attachment(snapshots, binding, config, metadata.number,
                    proof=binding['proof'], diagnostic=diagnostic)
                return VisualOutcome(response, parsed, {'version': 1, 'kind': 'visible-defects',
                                                       'head': head, 'seat': name},
                                     lifecycle=binding, coverage=coverage)
        except HumanDecisionRequiredError:
            raise
        except StaleDispatchInputs:
            # No capture was launched. The refreshed tuple must pass listing,
            # CI observation and pin verification before consuming its attempt.
            self.attempted.discard(head)
            getattr(runner, "_visual_boundary_paths", {}).pop((config.repo, metadata.number, head), None)
            return VisualOutcome(stale=True)
        except (AgentLoopError, ValueError, OSError, TimeoutError, subprocess.SubprocessError, KeyError) as exc:
            # Cleanup uncertainty is a security stop, never failed evidence continuation.
            if 'cleanup' in str(exc).lower() and ('unconfirmed' in str(exc).lower()
                    or 'could not be verified' in str(exc).lower() or 'retry is blocked' in str(exc).lower()):
                raise
            reason = 'Image review failed evidence: ' + str(exc)[:1024]
            self.reasons[head] = reason
            diagnostic(reason)
            return VisualOutcome(diagnostic=reason)


def visual_response_instructions(items):
    ledger = json.dumps([asdict(item) for item in items]).replace('<', '\\u003c').replace('>', '\\u003e')
    schema = {"schema_version": 1, "kind": "pr_review", "state": "approved",
              "summary": "Visible assessment", "blocking_items": [], "same_pr_followups": [],
              "future_followups": [], "prior_item_dispositions": [], "exact_head_evidence_requests": [],
              "architecture_impact": {"status": "unchanged", "rationale": "Visual assessment only.",
                  "affected_components": [], "dependencies": [], "execution_data_flows": [],
                  "persistence": [], "public_contracts": [], "security_boundaries": [],
                  "canonical_document_action": "no-change", "canonical_document_path": None,
                  "canonical_document_rationale": ""}}
    return ('\nSchema: ' + json.dumps(schema) + '\nReturn one JSON object with schema_version=1, kind="pr_review", state="approved" or '
            '"blocking", summary, blocking_items, same_pr_followups, future_followups '
            '(all finding lists), prior_item_dispositions (one explicit item_id/disposition per assigned item), '
            'and exact_head_evidence_requests=[]. Only visible defects are in scope. '
            'No code approval, evidence clearance, panel or sweep authority. '
            'Do not assess missing screenshot or requester coverage. '
            'Keep blocking and same-PR findings distinct; future work is deferred. '
            'End with the ordinary review state footer and your signature.\n'
            '<untrusted-assigned-visual-findings>\n' + ledger + '\n</untrusted-assigned-visual-findings>\n')


def admitted_visual_history(runner, config, pr_number, comments):
    """Hydrate authorship before any visual finding replay or ID allocation."""
    from .pr_loop_support import _authenticated_pr_comments_status
    from .round_state import _recovery_record_admitted, _extract_round_metadata_records
    from .review_images_state import History
    from .review_scope import coverage_candidate
    comments, actor, complete = _authenticated_pr_comments_status(
        runner, config, pr_number=pr_number, comments=comments, resume=False)
    records = _extract_round_metadata_records(comments, flow="pr")
    if not complete and any(visual_candidate(r.metadata) and (r.metadata.prior_items or r.metadata.new_items)
                            for r in records):
        raise AgentLoopError('Visual finding history cannot be authenticated; complete identity data is required.')
    return History((record for record in records if not (visual_candidate(record.metadata) or coverage_candidate(record.metadata)
                            or record.metadata.screenshot_requests or record.metadata.screenshot_coverage_acceptances
                            or record.metadata.screenshot_receipt_refs
                            or any(item.screenshot_request for item in (*record.metadata.prior_items, *record.metadata.new_items)))
                    or _recovery_record_admitted(record, comments, actor)),
                   identities={record.index: getattr(comments[record.index], 'comment_id', None)
                               for record in records}, complete=complete,
                   diagnostic='complete authenticated PR comment history is required', repo=config.repo, pr=pr_number)


def recover_visual_findings(records, *, reconciliation_mode):
    """Replay only author-admitted verified visual publications, independent of authority boundaries."""
    from .unresolved_items import _apply_unresolved_item_dispositions
    from .review_images_state import reduce_history, reliable_head, History
    from .errors import HumanDecisionRequiredError
    allowed = set()
    conservative = set()
    if isinstance(records, History):
        for position, record in enumerate(records):
            if not lifecycle_present(record.metadata):
                continue
            # Conservative blocker replay needs only this head's intent/result
            # lineage. Unrelated corruption must never erase authenticated
            # obligations, even when it prevents full-history clearance.
            head = reliable_head(record.metadata)
            prefix = History((r for r in records[:position+1]
                              if head is not None and reliable_head(r.metadata) == head),
                             identities=records.identities,
                             complete=records.complete, repo=records.repo, pr=records.pr)
            try:
                admitted = reduce_history(prefix, repo=records.repo, pr=records.pr,
                                          head=reliable_head(record.metadata))
                if admitted.terminal is not record:
                    continue
                conservative.add(record.index)
                admitted = reduce_history(records, repo=records.repo, pr=records.pr,
                                          head=reliable_head(record.metadata))
                if admitted.terminal is record:
                    allowed.add(record.index)
            except HumanDecisionRequiredError:
                pass
    ledger = []
    for record in records:
        m = record.metadata
        if not is_visual_review(m) or m.acquisition_outcome != 'success':
            continue
        if lifecycle_present(m) and record.index not in allowed:
            # Later corruption cannot delete an earlier verified blocker. It
            # cannot supply clearance, ownership transfer or new acceptance.
            if record.index in conservative:
                known = {item.item_id for item in ledger}
                ledger.extend(item for item in m.new_items if is_visual_item(item) and item.item_id not in known)
            continue
        # A tool-written assignment carries derived board-amendment ownership.
        # Transfer only ownership of the identical original visual claim; never
        # accept a new claim, status, or omission as an implicit disposition.
        prior = {item.item_id: item for item in m.prior_items}
        if len(prior) != len(m.prior_items):
            continue
        candidate = []
        for item in ledger:
            stored = prior.get(item.item_id)
            if (stored is not None and is_visual_item(stored)
                    and (stored.reviewer, stored.source_round, stored.text, stored.source_status)
                        == (item.reviewer, item.source_round, item.text, item.source_status)
                    and m.agent in (stored.resolution_owners or (stored.reviewer,))):
                item = replace(item, resolution_owners=stored.resolution_owners,
                    owner_states=stored.owner_states, owner_evidence=stored.owner_evidence,
                    owner_dispositions=stored.owner_dispositions)
            candidate.append(item)
        assigned = assigned_items(candidate, m.agent)
        ids = [d.item_id for d in m.dispositions]
        if len(ids) != len(set(ids)) or set(ids) != {item.item_id for item in assigned}:
            continue
        dispositions = {d.item_id: [d] for d in m.dispositions}
        ledger, _ = _apply_unresolved_item_dispositions(candidate, dispositions,
                        reconciliation_mode=reconciliation_mode)
        known = {item.item_id for item in ledger}
        ledger.extend(item for item in m.new_items if is_visual_item(item) and item.item_id not in known)
    return tuple(ledger)


def load_history(runner, config, pr_number):
    from .pr_head_sync import get_observed_pr_review_context
    context = get_observed_pr_review_context(runner, config=config, pr_number=pr_number)
    return admitted_visual_history(runner, config, pr_number, context.comments)


def persist_lifecycle(runner, config, metadata, binding, *, round_number=1, coverage=None):
    from .github import resolve_authenticated_github_actor, post_verified_trusted_pr_protocol_comment_observed
    from .round_metadata import (
        PostedRoundMetadata,
        _attach_round_metadata,
    )
    actor = resolve_authenticated_github_actor(runner, config=config)
    from .reviewer_seats import reviewer_seat_binding
    scope = {'version': 1, 'kind': 'visible-defects', 'head': metadata.head_sha, 'seat': binding['seat']}
    m = PostedRoundMetadata(flow='pr', role='visual-' + binding['kind'], agent=binding['seat'],
        round_number=round_number, subject=metadata.head_sha, review_scope=scope, visual_lifecycle=binding,
        seat_binding=reviewer_seat_binding(config), screenshot_coverage=coverage)
    from .review_images_state import validate
    from .round_metadata import (
        PostedRoundRecord,
    )
    validate(PostedRoundRecord(0, m, ''), repo=config.repo, pr=metadata.number)
    body = _attach_round_metadata('Visual lifecycle: ' + binding['kind'], m)
    return post_verified_trusted_pr_protocol_comment_observed(runner, config=config,
        pr_number=metadata.number, body=body, expected_author_login=actor[0], expected_author_id=actor[1])


def publish_assessment(runner, *, config, metadata, outcome, items, next_number,
                       round_number, seat_binding, reconciliation_mode, current_head, diagnostic):
    """Only a verified final anchor may mutate the live finding ledger."""
    if outcome.response is None or outcome.fresh:
        return outcome, items, next_number
    from .github import post_verified_trusted_pr_round_comment, resolve_authenticated_github_actor
    from .round_metadata import (
        PostedRoundMetadata,
        _attach_round_metadata,
    )
    from .unresolved_items import _next_unresolved_item, _apply_unresolved_item_dispositions
    from .agent_failure import _metadata_identity_fields
    parsed = outcome.parsed
    name = outcome.scope['seat']
    prior = assigned_items(items, name)
    new = []
    for status, findings in (('blocking', parsed.blocking_items), ('same-pr', parsed.followups.same_pr),
                            ('future', parsed.followups.future if config.approved_followups != 'ignore' else ())):
        for finding in findings:
            item = _next_unresolved_item(item_number=next_number + len(new), reviewer=name,
                source_round=round_number, text=finding.text, status=status,
                fix_scope=getattr(finding, 'fix_scope', None), sub_items=getattr(finding, 'sub_items', ()))
            new.append(replace(item, origin_scope='visible-defects'))
    if current_head() != metadata.head_sha:
        return VisualOutcome(stale=True), items, next_number
    try:
        actor = resolve_authenticated_github_actor(runner, config=config)
        post_verified_trusted_pr_round_comment(runner, config=config, pr_number=metadata.number,
            expected_author_login=actor[0], expected_author_id=actor[1],
            body=_attach_round_metadata(outcome.response.text,
                PostedRoundMetadata(flow='pr', role='reviewer', agent=name, round_number=round_number,
                    subject=metadata.head_sha, prior_items=tuple(prior), dispositions=parsed.dispositions,
                    new_items=tuple(new), state=parsed.state, review_scope=outcome.scope,
                    visual_lifecycle=outcome.lifecycle, screenshot_coverage=outcome.coverage, phase='visual', seat_binding=seat_binding,
                    **_metadata_identity_fields(outcome.response), acquisition_outcome='success', acquisition_returncode=0)))
    except (AgentLoopError, OSError, ValueError, subprocess.SubprocessError) as exc:
        diagnostic('Visual publication unknown; durable intent remains spent: ' + str(exc)[:1024])
        return VisualOutcome(diagnostic='visual publication unknown; no ledger mutation'), items, next_number
    if current_head() != metadata.head_sha:
        # The verified anchor has already made these findings durable. Carry
        # their obligations and reserve every published ID across the refresh,
        # but do not apply stale dispositions or grant a coder opportunity.
        return VisualOutcome(stale=True), [*items, *new], next_number + len(new)
    updated, _ = _apply_unresolved_item_dispositions(items, {d.item_id: [d] for d in parsed.dispositions},
                                                   reconciliation_mode=reconciliation_mode)
    return replace(outcome, fresh=True), [*updated, *new], next_number + len(new)


def validate_boundary(runner, *, config, metadata, comments=None, diagnostic=lambda _: None):
    """Read-only lifecycle validation never opens a recovery or freeze gate."""
    if not config.review_images_paths or config.dry_run:
        return
    from .github_changed_paths import get_pr_changed_paths
    paths = getattr(runner, '_visual_boundary_paths', {}).get((config.repo, metadata.number, metadata.head_sha))
    if paths is None:
        paths = get_pr_changed_paths(runner, config=config, pr_number=metadata.number, head_sha=metadata.head_sha)
        _cache_paths(runner, config, metadata, paths)
    if not paths.available or not eligible(CaptureConfig(paths=config.review_images_paths, command=config.review_images_cmd), [path.as_record() for path in paths.changes]):
        return
    from .review_images_state import reduce_history
    history = (admitted_visual_history(runner, config, metadata.number, comments)
               if comments is not None else load_history(runner, config, metadata.number))
    admission = reduce_history(history, repo=config.repo, pr=metadata.number, head=metadata.head_sha)
    diagnostic(admission.diagnostic or ('recovered visual state; no coder opportunity' if admission.spent
                                       else 'no durable visual acceptance; execution gate retained'))


def _cache_paths(runner, config, metadata, paths):
    if not hasattr(runner, '_visual_boundary_paths'):
        runner._visual_boundary_paths = {}
    runner._visual_boundary_paths[(config.repo, metadata.number, metadata.head_sha)] = paths


def coverage_attachment(snapshots, binding, config, pr, *, proof, source_id=None, diagnostic=lambda _: None):
    from .review_images_coverage_cache import persist
    from .review_images_coverage_history import envelope
    try:
        attachment = persist(snapshots, binding, proof=proof, source_id=source_id)
        return envelope('attachment', config, pr, snapshots.head_sha, {'attachment': attachment})
    except (AgentLoopError, OSError, ValueError) as exc:
        diagnostic('Screenshot coverage attachment unavailable: ' + str(exc)[:512])
        return None
