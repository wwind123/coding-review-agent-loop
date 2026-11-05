"""Verified admission, durable registry transport and completion-only publication."""
from __future__ import annotations

from dataclasses import replace
import json

from .errors import AgentLoopError
from .post_merge_obligations import (
    active_registry, authorize, diagnostic, digest, fail, filter_items, make_admission,
    operator_directive, parse_request, render_context, retained_diagnostic, source_for_item,
    validate_registry, validate_tracker,
)


def read_api(runner, config, path):
    from .github import run_gh_read, github_api_cwd
    result = run_gh_read(runner, [config.gh_cmd, 'api', f'repos/{config.repo}/{path}'],
                         cwd=github_api_cwd(), check=False)
    if result.returncode:
        raise AgentLoopError(f'Unable to read {path}: {result.stderr}')
    try:
        return json.loads(result.stdout)
    except ValueError as exc:
        raise AgentLoopError(f'Invalid GitHub response for {path}') from exc


def restore(runner, config, pr, comments, *, admitted_before=None):
    from .pr_loop_support import _authenticated_pr_comments_status
    from .github import read_authenticated_protocol_comments, PR_THREAD_SURFACE
    if admitted_before is not None:
        view = read_authenticated_protocol_comments(runner, config=config, surface_kind=PR_THREAD_SURFACE, number=pr)
        comments, actor, complete = view.authored, (view.actor_login, view.actor_id), True
    from .round_state import _extract_round_metadata_records, _recovery_record_admitted
    raw = _extract_round_metadata_records(comments, flow='pr')
    if not any(r.metadata.post_merge_registry for r in raw):
        return (), ()
    if admitted_before is None:
        comments, actor, complete = _authenticated_pr_comments_status(
            runner, config, pr_number=pr, comments=comments, resume=True)
    if not complete:
        fail('complete authenticated admission history is required')
    registry = ()
    diagnostics = ()
    for record in _extract_round_metadata_records(comments, flow='pr'):
        if not record.metadata.post_merge_registry:
            continue
        if not _recovery_record_admitted(record, comments, actor):
            continue
        candidate = validate_registry(list(record.metadata.post_merge_registry), repo=config.repo, pr=pr)
        previous = {a['identity']: a for a in registry}
        if admitted_before is not None and any(a['identity'] not in previous for a in candidate):
            from .github import parse_comment_timestamp
            comment = comments[record.index]
            created = comment.created_at
            updated = getattr(comment, 'updated_at', None) or created
            if not created or not updated or max(parse_comment_timestamp(created), parse_comment_timestamp(updated)) >= parse_comment_timestamp(admitted_before):
                fail('recovery cannot admit obligations first recorded after merge')
        current = {a['identity']: a for a in candidate}
        if not previous.keys() <= current.keys():
            fail('cumulative admission history lost an obligation')
        for identity, old in previous.items():
            new = current[identity]
            if any(old[key] != new[key] for key in old if key not in {'state', 'receipt'}):
                fail('admission provenance or payload mutated')
        registry = candidate
        diagnostics = record.metadata.post_merge_diagnostics
    return registry, diagnostics


def persist(runner, config, pr, head, registry, diagnostics=(), *, items=(), round_number=1):
    from .github import post_verified_trusted_pr_round_comment, resolve_authenticated_github_actor
    from .round_state import PostedRoundMetadata, _attach_round_metadata
    if config.dry_run:
        return
    actor = resolve_authenticated_github_actor(runner, config=config)
    metadata = PostedRoundMetadata(flow='pr', role='summary', agent='Orchestrator',
        round_number=round_number, subject=head, phase='post-merge-obligations',
        prior_items=tuple(items), post_merge_registry=tuple(registry),
        post_merge_diagnostics=tuple(diagnostics))
    from .protocol_markers import sanitize_untrusted_prose
    visible = render_context(registry, diagnostics)
    if len(visible) > 20000:
        visible = 'Required post-merge obligations; complete requirements and provenance are retained in authenticated round metadata.\n' + '\n'.join(
            f"- {a['identity']}: owner={a['request']['owner'][:120]}; tracker={a['request']['tracker_url']}; status={a['state']}" for a in active_registry(registry))
    body = _attach_round_metadata(sanitize_untrusted_prose(visible + '\nAfter confirmed merge, run `' + recovery_command(pr) + '`. Tracking is not fulfillment.'), metadata)
    post_verified_trusted_pr_round_comment(runner, config=config, pr_number=pr,
        body=body, expected_author_login=actor[0], expected_author_id=actor[1])


class ObligationRun:
    """Invocation state; no admission affects eligibility before verified publication."""
    def __init__(self, runner, config, pr):
        self.runner, self.config, self.pr = runner, config, pr
        self.registry, self.diagnostics = (), ()

    def restore(self, comments):
        registry, diagnostics = restore(self.runner, self.config, self.pr, comments)
        # In-process admissions were already read back and may precede the next
        # context snapshot. Never replace them with an older projection.
        known = {a['identity'] for a in registry}
        self.registry = registry + tuple(a for a in self.registry if a['identity'] not in known)
        self.diagnostics = diagnostics or self.diagnostics

    def context(self):
        return render_context(self.registry, self.diagnostics)

    def apply(self, items):
        return filter_items(items, self.registry)

    def admit(self, proposals, *, sources, closing, head, items, round_number):
        """Preflight the complete batch before publishing or releasing any source."""
        candidate = list(self.registry)
        replay_diagnostics = []
        for request, source, provenance in proposals:
            # An authenticated directive remains consumed even after its payload
            # is superseded or its tracker stops accepting new admissions.
            replay = any(a['source'] == source and a['request'] == request.payload()
                         and a['provenance'] == provenance for a in candidate)
            previous = next((a for a in active_registry(candidate) if a['source'] == source), None)
            if replay or (provenance.get('kind') == 'reviewer' and previous
                          and previous['request'] == request.payload()):
                try:
                    number, state = validate_tracker(request, repo=self.config.repo, closing_issues=(),
                        read_issue=lambda n: read_api(self.runner, self.config, f'issues/{n}'), allow_closed=True)
                    if state == 'closed':
                        fail(f'tracker #{number} is closed')
                except AgentLoopError as exc:
                    replay_diagnostics.append(retained_diagnostic(request.target_id or request.requirement, str(exc)))
                continue
            validate_tracker(request, repo=self.config.repo, closing_issues=closing,
                read_issue=lambda n: read_api(self.runner, self.config, f'issues/{n}'))
            admission = make_admission(request, repo=self.config.repo, pr=self.pr,
                source=source, provenance=provenance,
                supersedes=previous['identity'] if previous else None)
            if any(a['identity'] == admission['identity'] for a in candidate):
                continue
            candidate.append(admission)
        validate_registry(candidate, repo=self.config.repo, pr=self.pr)
        resolved_targets = {request.target_id or request.requirement for request, _, _ in proposals}
        resolved_targets.update(provenance['source_id'] for _, _, provenance in proposals
                                if provenance.get('kind') == 'operator')
        diagnostics = tuple(d for d in self.diagnostics if not d.startswith('Post-merge diagnostic refresh failed: ')
            and not any(
            d.startswith((f'Post-merge request {target} rejected: ',
                          f'Admitted post-merge obligation {target} retained: ')) for target in resolved_targets))
        diagnostics = tuple(dict.fromkeys((*diagnostics, *replay_diagnostics)))
        if candidate != list(self.registry) or diagnostics != self.diagnostics:
            if self.config.dry_run:
                return
            try:
                persist(self.runner, self.config, self.pr, head, candidate, diagnostics,
                        items=items, round_number=round_number)
            except AgentLoopError as exc:
                if candidate != list(self.registry):
                    raise
                # Diagnostic transport cannot revoke an authenticated admission
                # or turn its consumed control input back into a requirement.
                problem = (f'Post-merge diagnostic refresh failed: {exc}. '
                           'Existing admissions remain required after merge and do not block merge; '
                           'rerun to retry diagnostic publication.')
                self.diagnostics = (*diagnostics, problem)
                print(problem)
                return
            self.registry, self.diagnostics = tuple(candidate), diagnostics

    def visible_requirements(self, requirements):
        result = []
        known = {a['provenance'].get('source_id'): a['provenance'].get('body')
                 for a in self.registry if a['provenance'].get('kind') == 'operator'}
        scheduled = {a['source']['target_id']: a['source']['requirement']
                     for a in active_registry(self.registry) if a['source']['target_kind'] == 'signed-requirement'}
        for requirement in requirements:
            if scheduled.get(requirement.requirement_id) == requirement.body:
                continue
            if known.get(requirement.requirement_id) == requirement.body:
                _, prose = operator_directive(requirement)
                if prose:
                    result.append(replace(requirement, body=prose))
            else:
                result.append(requirement)
        return tuple(result)

    def operator_inputs(self, requirements, *, items, closing, head, round_number):
        sources = {(s['target_kind'], s['target_id']): s for item in items
                   if item.obligation_kind == 'human-exact-head-evidence'
                   for s in [source_for_item(item)]}
        sources.update({('signed-requirement', r.requirement_id): {
            'target_kind': 'signed-requirement', 'target_id': r.requirement_id,
            'requirement': r.body} for r in requirements if r.author_verification == 'verified'})
        # Original sources remain available after eligibility filtering.
        sources.update({(a['source']['target_kind'], a['source']['target_id']): a['source']
                        for a in self.registry})
        result = []
        for requirement in requirements:
            if '```post-merge-schedule' not in requirement.body:
                result.append(requirement)
                continue
            try:
                requests, prose = operator_directive(requirement)
                proposals = []
                for request in requests:
                    source = authorize(request, sources=sources, operator=requirement)
                    provenance = {'kind': 'operator', 'locator': requirement.url,
                        'actor': requirement.author, 'actor_id': requirement.author_id,
                        'body': requirement.body, 'source_id': requirement.requirement_id}
                    proposals.append((request, source, provenance))
                self.admit(proposals, sources=sources, closing=closing, head=head,
                           items=items, round_number=round_number)
                if prose:
                    result.append(replace(requirement, body=prose))
            except AgentLoopError as exc:
                self.diagnostics = (*self.diagnostics, diagnostic(requirement.requirement_id, str(exc)))
                result.append(requirement)
        scheduled = {a['source']['target_id']: a['source']['requirement']
                     for a in active_registry(self.registry) if a['source']['target_kind'] == 'signed-requirement'}
        return tuple(r for r in result if scheduled.get(r.requirement_id) != r.body)

    def reviewer_inputs(self, reviews, *, items, closing, head, round_number, board):
        sources = {(s['target_kind'], s['target_id']): s for item in items
                   if item.obligation_kind == 'human-exact-head-evidence' and not item.screenshot_request
                   for s in [source_for_item(item)]}
        proposals = []
        for reviewer, review in reviews.items():
            for request in review.post_merge_obligations:
                try:
                    voters = [name for name, other in reviews.items()
                              if request in other.post_merge_obligations]
                    source = authorize(request, sources=sources, reviewer=reviewer, voters=voters, board=board)
                    proposals.append((request, source, {'kind': 'reviewer', 'reviewers': voters,
                        'round': round_number, 'head': head}))
                except AgentLoopError as exc:
                    self.diagnostics = (*self.diagnostics, diagnostic(request.target_id or request.requirement, str(exc)))
        # Independent reviewer requests can be rejected independently, but each
        # operator block above is all-or-nothing.
        for proposal in proposals:
            try:
                self.admit([proposal], sources=sources, closing=closing, head=head,
                           items=items, round_number=round_number)
            except AgentLoopError as exc:
                self.diagnostics = (*self.diagnostics, diagnostic(proposal[0].target_id or proposal[0].requirement, str(exc)))
        return self.apply(items)


def recovery_command(pr):
    return f'agent-loop pr {pr} --reconcile-post-merge-obligations'


def tracker_record(admission, merged):
    return {'version': 1, 'identity': admission['identity'], 'repo': admission['repo'],
        'pr': admission['pr'], 'request': admission['request'], 'source': admission['source'],
        'provenance': admission['provenance'], 'merged_source_head': merged['head']['sha'],
        'merge_commit': merged.get('merge_commit_sha')}


def record_body(record):
    from .round_state import PostedRoundMetadata, _attach_round_metadata
    r = record['request']
    from .protocol_markers import sanitize_untrusted_prose
    return _attach_round_metadata(sanitize_untrusted_prose(
        f"Required post-merge obligation from https://github.com/{record['repo']}/pull/{record['pr']}\n\n"
        f"Identity: {record['identity']}\n\nRequirement: {r['requirement']}\n\n"
        f"Owner: {r['owner']}\n\nRationale: {r['rationale']}\n\n"
        'Tracked publication only; this does not certify fulfillment.'),
        PostedRoundMetadata(flow='pr', role='summary', agent='Orchestrator', round_number=1,
            subject=record['merged_source_head'], post_merge_publication=record))


def find_publication(runner, config, number, expected):
    from .github import read_authenticated_protocol_comments, ISSUE_THREAD_SURFACE
    from .round_state import _extract_round_metadata_records
    view = read_authenticated_protocol_comments(runner, config=config,
        surface_kind=ISSUE_THREAD_SURFACE, number=number)
    for record in _extract_round_metadata_records(view.authored, flow='pr'):
        if record.metadata.post_merge_publication == expected:
            return view.authored[record.index].comment_id
    return None


def reconcile(runner, config, pr, *, registry=None, required_identities=()):
    """Read plain merged context; never touch a branch, worktree, agent or CI."""
    from .github import get_pr_review_context, post_verified_trusted_issue_round_comment, resolve_authenticated_github_actor
    read_config = replace(config, dry_run=False)
    merged = read_api(runner, read_config, f'pulls/{pr}')
    if not merged.get('merged') or not merged.get('merged_at'):
        fail('completion-only reconciliation requires a confirmed merged PR')
    from .workdirs import github_api_cwd
    context = get_pr_review_context(runner, config=read_config, pr_number=pr, cwd=github_api_cwd())
    if registry is None:
        registry, _ = restore(runner, read_config, pr, context.comments, admitted_before=merged['merged_at'])
    registry = list(validate_registry(list(registry), repo=config.repo, pr=pr))
    if not set(required_identities) <= {a['identity'] for a in registry}:
        fail('durable admission history cannot reconstruct an already admitted obligation')
    actor = resolve_authenticated_github_actor(runner, config=read_config)
    pending = []
    for admission in active_registry(registry):
        request = parse_request(admission['request'])
        try:
            number, state = validate_tracker(request, repo=config.repo, closing_issues=(),
                read_issue=lambda n: read_api(runner, read_config, f'issues/{n}'), allow_closed=True)
            expected = tracker_record(admission, merged)
            receipt = find_publication(runner, read_config, number, expected)
            if state == 'closed':
                print(f"{admission['identity']}: tracker #{number} is closed; obligation retained.")
            if config.dry_run:
                print(f"{admission['identity']}: {'tracked' if receipt else 'tracking-pending'}; dry-run writes nothing.")
                continue
            if receipt is None:
                try:
                    written = post_verified_trusted_issue_round_comment(runner, config=config,
                        issue_number=number, body=record_body(expected),
                        expected_author_login=actor[0], expected_author_id=actor[1])
                    receipt = written.comment_id
                except AgentLoopError:
                    # A timeout can follow a successful POST. Read before any
                    # later retry; never issue a second blind write in this pass.
                    receipt = find_publication(runner, read_config, number, expected)
                    if receipt is None:
                        raise
            admission['state'], admission['receipt'] = 'tracked', receipt
        except AgentLoopError as exc:
            admission['state'], admission['receipt'] = 'tracking-pending', None
            pending.append(retained_diagnostic(admission['identity'], str(exc)))
    if not config.dry_run and registry:
        try:
            persist(runner, config, pr, merged['head']['sha'], registry, pending)
        except AgentLoopError as exc:
            raise AgentLoopError(f'PR #{pr} merged, tracking receipt publication pending: {exc}. Run `{recovery_command(pr)}`.') from exc
    if pending:
        print(f'PR #{pr} merged, tracking pending. Retry `{recovery_command(pr)}`. ' + ' '.join(pending))
        return 1
    return 0


def complete_merge(runner, config, pr):
    """Completion I/O runs after, and outside, the final head-guarded merge."""
    run = getattr(runner, '_post_merge_run', None)
    if run is None or not run.registry:
        return
    try:
        result = reconcile(runner, config, pr, required_identities=tuple(a['identity'] for a in run.registry))
    except AgentLoopError as exc:
        raise AgentLoopError(f'PR #{pr} merged, tracking pending: {exc}. Run `{recovery_command(pr)}`.') from exc
    if result:
        raise AgentLoopError(f'PR #{pr} merged, tracking pending. Run `{recovery_command(pr)}`.')


def closing_set(metadata):
    from .github_issue_references import parse_issue_reference_evidence
    return tuple(e.issue_number for e in parse_issue_reference_evidence(metadata.body, repo=metadata.repo, include_non_closing=False) if e.target_repo.casefold() == metadata.repo.casefold())


def round_start(run, context, items, requirements, number):
    run.restore(context.comments)
    requirements = run.operator_inputs(requirements, items=items,
        closing=closing_set(context.metadata), head=context.metadata.head_sha, round_number=number)
    return run.apply(items), requirements


def reviewer_requests(run, reviews, *, items, metadata, round_number, board, next_number=None):
    # A rejected new declaration remains a required pre-merge evidence demand.
    # Minting it through the ordinary evidence path also allocates its durable
    # source identity, which is shown in the next prompt/freeze notice.
    from .unresolved_items import _upsert_evidence_obligation
    result = run.reviewer_inputs(reviews, items=items, closing=closing_set(metadata),
        head=metadata.head_sha, round_number=round_number, board=board)
    next_number = next_number or max((int(i.item_id[5:]) for i in items
                       if i.item_id.startswith('item-') and i.item_id[5:].isdigit()), default=0) + 1
    for reviewer, review in reviews.items():
        for request in review.post_merge_obligations:
            if request.target_id is None and not any(
                    a['source']['target_kind'] == 'declaration'
                    and a['source'].get('requester') == reviewer
                    and a['request'] == request.payload()
                    for a in active_registry(run.registry)):
                result, consumed = _upsert_evidence_obligation(result, item_number=next_number,
                    reviewer=reviewer, text=request.requirement, source_round=round_number,
                    current_head_sha=metadata.head_sha)
                next_number += int(consumed)
    return run.apply(result), next_number


def scheduling_guidance(items):
    lines = ['For work that can only occur after merge, schedule it with a non-empty owner and an existing open same-repository tracking issue outside this PR closing set. Manually create/select that tracker first. A configured trusted operator must post one fenced JSON block labeled post-merge-schedule in a signed PR comment, then rerun. Required fields per entry in obligations: target_kind, target_id, source_requirement_text, requirement, rationale, owner, tracker_url; converted-row also requires matrix_identity. Both text fields must equal the full target requirement. Scheduling is not evidence fulfillment.']
    for item in items:
        if item.obligation_kind == 'human-exact-head-evidence':
            lines.append('Scheduling target: ' + json.dumps(source_for_item(item), sort_keys=True))
    return '\n'.join(lines)


def manual_pending(run):
    if run.registry:
        print(run.context() + '\nAfter confirmed manual merge, run `' + recovery_command(run.pr) + '`. Tracking publication has not occurred.')


def validate_recovery_options(args, invocation):
    allowed = {'--reconcile-post-merge-obligations', '--repo', '--gh-cmd', '--dry-run',
               '--quiet', '--log-dir', '--human-reviewer-trusted-actor'}
    for token in invocation:
        if token.startswith('--') and token.split('=', 1)[0] not in allowed:
            fail('completion-only reconciliation cannot be combined with execution options: ' + token.split('=', 1)[0])


def resume_scheduled(run, *, config, head, items, round_number, baseline, collect_requirement_ids, clearances, references):
    from .unresolved_items import _pending_evidence_obligations
    from .review_images_coverage_run import boundary
    from .pr_loop_support import _publish_evidence_release
    if not run.registry or _pending_evidence_obligations(items):
        return False
    result = boundary(run.runner, config=config, pr=run.pr, head=head,
        baseline=baseline.signed_requirement_ids_at_freeze, collect_requirement_ids=collect_requirement_ids)
    if result.action != 'coverage_satisfied':
        return result
    _publish_evidence_release(run.runner, config=config, pr_number=run.pr,
        round_number=round_number, head_sha=head, items=items, reason='post-merge-scheduled',
        surfaced_requirement_ids=baseline.signed_requirement_ids_at_freeze,
        allowed_rounds=baseline.allowed_rounds,
        watch_failure_extension_used=baseline.watch_failure_extension_used,
        watch_head_extension_used=baseline.watch_head_extension_used,
        clearances=clearances, signed_evidence_references=references)
    return replace(result, round_number=round_number)
