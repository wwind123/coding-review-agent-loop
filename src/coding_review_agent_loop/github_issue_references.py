"""Pure issue-reference parsing with compatibility exports from github.py."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from .errors import AgentLoopError

# This intentionally remains looser than the recovery parser below.  It is
# used by PR review context inference, where a same-repository issue URL is
# useful even when it is only quoted as background context.
_GITHUB_ISSUE_URL_RE = re.compile(
    r"https?://github\.com/(?P<repo>[^/\s#]+/[^/\s#]+)/issues/(?P<number>[1-9]\d*)(?![\w/-])",
    re.IGNORECASE,
)

# GitHub's supported auto-close grammar.  Recovery must be anchored to one of
# these keywords; a bare issue number, Refs sentence, URL, title, or branch
# name is not implementation provenance.
_CLOSING_ISSUE_REFERENCE_RE = re.compile(
    r"\b(?P<keyword>close|closes|closed|fix|fixes|fixed|resolve|resolves|resolved)\b"
    r"[ \t]*:?[ \t]*(?:"
    r"(?P<unqualified>#[1-9]\d*)|"
    r"(?P<qualified>[^\s/#]+/[^\s/#]+#[1-9]\d*(?![\w/-]))|"
    r"(?P<url>https?://github\.com/[^/\s#]+/[^/\s#]+/issues/[1-9]\d*(?![\w/-]))"
    r")",
    re.IGNORECASE,
)

# Non-closing references are intentionally parsed by the same target grammar,
# but are never returned as strong recovery evidence.  This narrow form is
# used only when a caller explicitly says it is validating a staged parent.
_NON_CLOSING_ISSUE_REFERENCE_RE = re.compile(
    r"\b(?P<keyword>refs?|references?)\b[ \t]*:?[ \t]*(?:"
    r"(?P<unqualified>#[1-9]\d*)|"
    r"(?P<qualified>[^\s/#]+/[^\s/#]+#[1-9]\d*(?![\w/-]))|"
    r"(?P<url>https?://github\.com/[^/\s#]+/[^/\s#]+/issues/[1-9]\d*(?![\w/-]))"
    r")",
    re.IGNORECASE,
)

def affirmative_markdown_view(body: str | None) -> str:
    """Return active Markdown text that can count as affirmative issue evidence.

    GitHub does not interpret code samples, inline code, or HTML comments as
    closing instructions. List items are classified before indentation so a
    nested list item remains active evidence, and blockquotes remain active
    because GitHub linkifies their references.
    """
    if not body:
        return ""
    text = re.sub(r"<!--.*?-->", "", body, flags=re.DOTALL)
    output: list[str] = []
    fenced = False
    fence_char = ""
    fence_length = 0
    list_line_re = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)")
    for line in text.splitlines():
        stripped = line.lstrip(" \t")
        fence = re.match(r"(`{3,}|~{3,})", stripped)
        if fence:
            token = fence.group(1)
            if not fenced:
                fenced = True
                fence_char = token[0]
                fence_length = len(token)
            elif (
                token[0] == fence_char
                and len(token) >= fence_length
                and not stripped[len(token) :].strip()
            ):
                fenced = False
            continue
        if fenced:
            continue
        is_list_item = bool(list_line_re.match(line))
        if (line.startswith("    ") or line.startswith("\t")) and not is_list_item:
            continue
        # Inline code spans are non-rendered code, even when they are inside a
        # list item or blockquote. Preserve surrounding prose and references.
        line = re.sub(r"(`+)(.+?)\1", "", line)
        output.append(line)
    return "\n".join(output)


@dataclass(frozen=True)
class IssueReferenceEvidence:
    """One parsed issue reference, retaining enough detail for diagnostics."""

    keyword: str
    target_repo: str
    issue_number: int
    reference_form: Literal["unqualified", "qualified", "url"]
    matched_text: str
    closing: bool

    @property
    def is_closing(self) -> bool:
        return self.closing


class OpenPrClosingMatch(int):
    """An open PR number with its strong closing-reference evidence.

    This is an ``int`` subclass for compatibility with callers that compared
    the old discovery result directly to a PR number, while exposing the
    structured evidence required by the safer resolver.
    """

    def __new__(cls, pr_number: int, evidence: tuple[IssueReferenceEvidence, ...]):
        instance = int.__new__(cls, pr_number)
        instance.evidence = evidence
        return instance

    @property
    def pr_number(self) -> int:
        return int(self)


def _issue_reference_evidence_from_match(
    match: re.Match[str], *, closing: bool, default_repo: str
) -> IssueReferenceEvidence:
    reference = match.group("unqualified") or match.group("qualified") or match.group("url")
    if reference is None:  # pragma: no cover - every grammar branch has a target
        raise AgentLoopError("Issue reference parser produced an empty target.")
    if match.group("unqualified"):
        target_repo = default_repo
        reference_form: Literal["unqualified", "qualified", "url"] = "unqualified"
        issue_number = int(reference[1:])
    elif match.group("qualified"):
        target_repo, raw_number = reference.rsplit("#", 1)
        reference_form = "qualified"
        issue_number = int(raw_number)
    else:
        url_match = _GITHUB_ISSUE_URL_RE.fullmatch(reference)
        if url_match is None:  # pragma: no cover - guarded by the shared grammar
            raise AgentLoopError("Issue reference parser produced an invalid URL target.")
        target_repo = url_match.group("repo")
        reference_form = "url"
        issue_number = int(url_match.group("number"))
    return IssueReferenceEvidence(
        keyword=match.group("keyword").casefold(),
        target_repo=target_repo.casefold(),
        issue_number=issue_number,
        reference_form=reference_form,
        matched_text=match.group(0),
        closing=closing,
    )


def parse_issue_reference_evidence(
    body: str | None,
    *,
    repo: str,
    include_non_closing: bool = True,
    affirmative: bool = True,
) -> tuple[IssueReferenceEvidence, ...]:
    """Parse closing and, optionally, explicit ``Refs`` issue evidence.

    The parser deliberately returns cross-repository matches too, with their
    normalized target repository, so callers can explain why a candidate was
    rejected.  Use :func:`parse_strong_issue_reference_evidence` for recovery.
    """
    if not body:
        return ()
    searchable_body = affirmative_markdown_view(body) if affirmative else body
    matches: list[tuple[int, IssueReferenceEvidence]] = [
        (
            match.start(),
            _issue_reference_evidence_from_match(match, closing=True, default_repo=repo),
        )
        for match in _CLOSING_ISSUE_REFERENCE_RE.finditer(searchable_body)
    ]
    if include_non_closing:
        matches.extend(
            (
                match.start(),
                _issue_reference_evidence_from_match(match, closing=False, default_repo=repo),
            )
            for match in _NON_CLOSING_ISSUE_REFERENCE_RE.finditer(searchable_body)
        )
    return tuple(evidence for _position, evidence in sorted(matches, key=lambda item: item[0]))


def parse_strong_issue_reference_evidence(
    body: str | None, *, repo: str, issue_number: int
) -> tuple[IssueReferenceEvidence, ...]:
    """Return only same-repository closing evidence for ``issue_number``."""
    normalized_repo = repo.casefold()
    return tuple(
        evidence
        for evidence in parse_issue_reference_evidence(body, repo=repo, include_non_closing=False)
        if evidence.closing
        and evidence.target_repo.casefold() == normalized_repo
        and evidence.issue_number == issue_number
    )


def parse_raw_strong_issue_reference_evidence(
    body: str | None, *, repo: str, issue_number: int
) -> tuple[IssueReferenceEvidence, ...]:
    """Return raw-body closing evidence for fail-closed safety prohibitions."""
    normalized_repo = repo.casefold()
    return tuple(
        evidence
        for evidence in parse_issue_reference_evidence(
            body, repo=repo, include_non_closing=False, affirmative=False
        )
        if evidence.closing
        and evidence.target_repo.casefold() == normalized_repo
        and evidence.issue_number == issue_number
    )


def parse_non_closing_issue_reference_evidence(
    body: str | None, *, repo: str, issue_number: int
) -> tuple[IssueReferenceEvidence, ...]:
    """Return explicit same-repository ``Refs`` evidence for a staged role."""
    normalized_repo = repo.casefold()
    return tuple(
        evidence
        for evidence in parse_issue_reference_evidence(body, repo=repo)
        if not evidence.closing
        and evidence.target_repo.casefold() == normalized_repo
        and evidence.issue_number == issue_number
    )


def parse_linked_issue_numbers(pr_body: str | None, *, repo: str) -> tuple[int, ...]:
    """Return same-repository issues linked by a PR body, in first-seen order.

    GitHub closing phrases accept unqualified and ``owner/repo#N`` forms.  A
    same-repository issue URL is also a link even without a closing phrase.
    Repository comparison is case-insensitive because GitHub repository names
    are case-insensitive.
    """
    if not pr_body:
        return ()

    normalized_repo = repo.casefold()
    matches: list[tuple[int, int]] = []
    for match in _CLOSING_ISSUE_REFERENCE_RE.finditer(pr_body):
        reference = match.group("unqualified") or match.group("qualified") or match.group("url")
        if reference is None:
            continue
        if reference.startswith("#"):
            matches.append((match.start(), int(reference[1:])))
            continue
        if reference.lower().startswith(("http://", "https://")):
            url_match = _GITHUB_ISSUE_URL_RE.fullmatch(reference)
            if url_match and url_match.group("repo").casefold() == normalized_repo:
                matches.append((match.start(), int(url_match.group("number"))))
            continue
        qualified_repo, number = reference.rsplit("#", 1)
        if qualified_repo.casefold() == normalized_repo:
            matches.append((match.start(), int(number)))

    for match in _GITHUB_ISSUE_URL_RE.finditer(pr_body):
        if match.group("repo").casefold() == normalized_repo:
            matches.append((match.start(), int(match.group("number"))))

    seen: set[int] = set()
    numbers: list[int] = []
    for _, number in sorted(matches, key=lambda item: item[0]):
        if number not in seen:
            seen.add(number)
            numbers.append(number)
    return tuple(numbers)
