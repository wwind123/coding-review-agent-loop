"""Invocation-local visual review lifecycle; image evidence has no code authority."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import subprocess
import time

from .agents.registry import agent_display_name
from .errors import AgentLoopError
from .review_images_config import CaptureConfig, delivery_limits, eligible
from .review_images_capabilities import preflight_images, reviewer_view
from .review_images_capture import capture
from .review_images_prompts import build_visual_prompt
from .review_scope import is_visual_item, is_visual_review


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
    if parsed.exact_head_evidence_requests or any(
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


class VisualRun:
    def __init__(self):
        self.attempted = set()
        self.coder_heads = set()
        self.reasons = {}

    @staticmethod
    def rebind_items(items, ledger_view):
        if not any(is_visual_item(item) for item in items):
            return items
        derived = {item.item_id: item for item in ledger_view(items)}
        return [derived.get(item.item_id, item) if is_visual_item(item) else item for item in items]

    def attempt(self, runner, *, config, metadata, checkout, items, managed, current_head, diagnostic):
        if not config.review_images_paths or config.dry_run:
            return VisualOutcome(diagnostic='disabled')
        head = metadata.head_sha
        if head in self.attempted:
            return VisualOutcome(diagnostic=self.reasons.get(head, 'visual attempt already spent'))
        self.attempted.add(head)
        try:
            from .github_changed_paths import get_pr_changed_paths
            settings_ = settings(config)
            paths = get_pr_changed_paths(runner, config=config, pr_number=metadata.number, head_sha=head)
            if not paths.available:
                if current_head() != head:
                    return VisualOutcome(stale=True)
                raise ValueError(paths.diagnostic)
            changes = [path.as_record() for path in paths.changes]
            if not eligible(settings_, changes):
                self.reasons[head] = 'no matching changed paths'
                return VisualOutcome(diagnostic=self.reasons[head])
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
                from .review_images_transport import acquire_images
                with acquire_images(snapshots, capability, head_sha=head,
                                    config_digest=snapshots.config_digest,
                                    snapshot_digest=snapshots.digest, limits=settings_.limits) as images:
                    prompt = build_visual_prompt(images, diff) + visual_response_instructions(assigned)
                    from .validated_agent import _run_validated_agent
                    response = _run_validated_agent(runner, agent=selected, config=config,
                        prompt=prompt, role='reviewer', session_id=None, timeout_seconds=300,
                        label='visible-defects', image_inputs=images,
                        validate=lambda text: validate_visual(text, reviewer=name, items=assigned),
                        use_repair=False)
                if current_head() != head:
                    return VisualOutcome(stale=True)
                if response.acquisition_outcome != 'success':
                    raise ValueError('visual reviewer acquisition did not succeed')
                parsed = validate_visual(response.text, reviewer=name, items=assigned)
                return VisualOutcome(response, parsed, {'version': 1, 'kind': 'visible-defects',
                                                       'head': head, 'seat': name})
        except StaleDispatchInputs:
            # No capture was launched. The refreshed tuple must pass listing,
            # CI observation and pin verification before consuming its attempt.
            self.attempted.discard(head)
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
    comments, actor, complete = _authenticated_pr_comments_status(
        runner, config, pr_number=pr_number, comments=comments, resume=True)
    records = _extract_round_metadata_records(comments, flow="pr")
    if not complete and any(is_visual_review(record.metadata) for record in records):
        raise AgentLoopError("Visual finding history cannot be authenticated; retry with readable identity data.")
    return tuple(record for record in records if not is_visual_review(record.metadata)
                 or _recovery_record_admitted(record, comments, actor))


def recover_visual_findings(records, *, reconciliation_mode):
    """Replay only author-admitted verified visual publications, independent of authority boundaries."""
    from .unresolved_items import _apply_unresolved_item_dispositions
    ledger = []
    for record in records:
        m = record.metadata
        if not is_visual_review(m) or m.acquisition_outcome != 'success':
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
