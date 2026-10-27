"""Typed historical managed results, separate from dispatch authorization."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import re


@dataclass(frozen=True)
class FailedManagedRun:
    repository: str
    pr_number: int
    head_sha: str
    base_ref: str
    obligation_identity: str
    source_round: int
    audit_comment_id: int
    nonce: str
    run_id: int
    failed_attempt: int
    generation: str
    workflow_revision: str

    def __post_init__(self):
        for name in ('pr_number', 'source_round', 'audit_comment_id', 'run_id', 'failed_attempt'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f'invalid historical result {name}')
        for name in ('repository', 'head_sha', 'base_ref', 'obligation_identity', 'nonce', 'generation', 'workflow_revision'):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or any(c.isspace() for c in value):
                raise ValueError(f'invalid historical result {name}')
        if not re.fullmatch(r'[^/]+/[^/]+', self.repository):
            raise ValueError('invalid repository')
        if not re.fullmatch(r'[A-Za-z0-9_-]{32}', self.nonce):
            raise ValueError('invalid nonce')
        if any(not re.fullmatch(r'[a-f0-9]{40}', getattr(self, name)) for name in ('head_sha', 'workflow_revision')):
            raise ValueError('invalid result revision')

    def as_dict(self):
        return asdict(self)

    @classmethod
    def from_mapping(cls, value):
        if not isinstance(value, dict) or set(value) != {f.name for f in fields(cls)}:
            raise ValueError('partial historical result')
        return cls(**value)


@dataclass(frozen=True)
class VerifiedManagedResult:
    failed_run: FailedManagedRun
    successful_attempt: int

    def __post_init__(self):
        if not isinstance(self.failed_run, FailedManagedRun):
            raise ValueError('result lacks failed-run binding')
        if type(self.successful_attempt) is not int or self.successful_attempt <= self.failed_run.failed_attempt:
            raise ValueError('result must identify a later attempt')

    def as_dict(self):
        return {'failed_run': self.failed_run.as_dict(), 'successful_attempt': self.successful_attempt}

    @classmethod
    def from_mapping(cls, value):
        if not isinstance(value, dict) or set(value) != {'failed_run', 'successful_attempt'}:
            raise ValueError('partial verified result')
        return cls(FailedManagedRun.from_mapping(value['failed_run']), value['successful_attempt'])

    def require_binding(self, *, repository, pr_number, head_sha, base_ref):
        ref = self.failed_run
        if (ref.repository, ref.pr_number, ref.head_sha, ref.base_ref) != (repository, pr_number, head_sha, base_ref):
            raise ValueError('historical result binding drifted')
