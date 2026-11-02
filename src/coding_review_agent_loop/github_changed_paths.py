"""Bounded complete changed-file reads pinned to one live PR head.

An unavailable listing is distinct from an empty listing or a path nonmatch.
This reader does not reserve an attempt, fetch Git objects or launch capture.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import re

from .errors import AgentLoopError
from .github_retry import run_gh_read
from .review_images_config import _git_path
from .workdirs import github_api_cwd


@dataclass(frozen=True)
class ChangedPath:
    status: str
    filename: str
    previous_filename: str | None = None

    def as_record(self) -> dict[str, str]:
        result = {'status': self.status, 'filename': self.filename}
        if self.previous_filename is not None:
            result['previous_filename'] = self.previous_filename
        return result


@dataclass(frozen=True)
class PullRequestChangedPaths:
    head_sha: str
    available: bool
    changes: tuple[ChangedPath, ...] = ()
    diagnostic: str = ''


def _metadata(value: object, expected_head: str) -> int:
    if not isinstance(value, dict) or not isinstance(value.get('head'), dict):
        raise ValueError('malformed PR metadata')
    if value['head'].get('sha') != expected_head:
        raise ValueError('PR head changed during changed-file enumeration')
    count = value.get('changed_files')
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError('authoritative changed_files count is unavailable')
    if count >= 3000:
        raise ValueError('changed-file listing reaches the GitHub API ceiling')
    return count


def _change(value: object) -> ChangedPath:
    if not isinstance(value, dict):
        raise ValueError('malformed changed-file record')
    status = value.get('status')
    if not isinstance(status, str):
        raise ValueError('unsupported changed-file status')
    status = {'copied': 'added', 'changed': 'modified', 'deleted': 'removed'}.get(status, status)
    if status not in {'added', 'modified', 'removed', 'renamed'}:
        raise ValueError('unsupported changed-file status')
    name = _git_path(value.get('filename'))
    previous = _git_path(value.get('previous_filename')) if status == 'renamed' else None
    if status != 'renamed' and 'previous_filename' in value:
        raise ValueError('previous filename is only valid for renames')
    return ChangedPath(status, name, previous)


def get_pr_changed_paths(runner, *, config, pr_number: int,
                         head_sha: str) -> PullRequestChangedPaths:
    """Read all REST pages and recheck authoritative head/count before returning.

    Fixed pagination bounds and count/uniqueness checks prevent truncated API
    responses from becoming nonmatches. API failures return no partial changes.
    Ordinary review continuation is the caller's responsibility.
    """
    def read(path):
        result = run_gh_read(runner, [config.gh_cmd, 'api', path],
                             cwd=github_api_cwd(), check=False)
        if result.returncode != 0:
            # Do not reflect untrusted API error text or credentials.
            raise ValueError('GitHub changed-file read is unavailable')
        return json.loads(result.stdout)

    try:
        if not isinstance(head_sha, str) or not re.fullmatch(r'[0-9a-f]{40}', head_sha):
            raise ValueError('changed-file enumeration requires a full head SHA')
        if isinstance(pr_number, bool) or not isinstance(pr_number, int) or pr_number <= 0:
            raise ValueError('invalid PR number')
        root = f'repos/{config.repo}/pulls/{pr_number}'
        count = _metadata(read(root), head_sha)
        changes = []
        for page in range(1, 31):
            batch = read(f'{root}/files?per_page=100&page={page}')
            if not isinstance(batch, list) or len(batch) > 100:
                raise ValueError('malformed changed-file page')
            changes.extend(_change(value) for value in batch)
            if len(changes) > count:
                raise ValueError('changed-file count exceeds authoritative count')
            if len(batch) < 100:
                break
        if len(changes) != count:
            raise ValueError('changed-file listing is incomplete')
        if len({value.filename for value in changes}) != count:
            raise ValueError('duplicate changed-file names')
        if _metadata(read(root), head_sha) != count:
            raise ValueError('changed-file count changed during enumeration')
        return PullRequestChangedPaths(head_sha, True, tuple(changes))
    except (ValueError, TypeError, OSError, AgentLoopError):
        # All failure messages below are service-owned, bounded classifications.
        # Preserve a useful diagnostic without exposing command stderr.
        return PullRequestChangedPaths(head_sha, False, diagnostic='Changed-file eligibility unavailable: incomplete or unstable GitHub listing.')
