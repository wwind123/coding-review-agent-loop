"""Advisory terminal-response context; never an authenticated PR handoff."""

from __future__ import annotations

import json
import re


def terminal_response_diagnostic(
    text: str, *, repo: str, parsed: object = None, terminal: bool = False,
) -> str:
    """Describe only the final response, without discovering or adopting a PR.

    Successful implementation envelopes also use a blocking state.  Acquisition
    failures therefore need an explicit blocking declaration in the summary or
    a null PR, rather than trusting that state alone.
    """
    payload = None
    try:
        payload, _ = json.JSONDecoder().raw_decode(text.lstrip())
    except (ValueError, TypeError):
        pass
    if parsed is not None:
        summary = getattr(parsed, "summary", "")
        number = getattr(parsed, "pr_number", None)
    elif isinstance(payload, dict) and payload.get("kind") == "issue_implementation":
        summary = payload.get("summary", "")
        number = payload.get("pr_number")
        terminal = terminal or number is None
    else:
        if payload is not None:
            # Other structured roles are never legacy coder prose.
            return ""
        # Only a terminal footer in this response establishes legacy blocking.
        footer = re.search(
            r"(?m)^<!--\s*AGENT_(?:STATE:\s*blocking|CLARIFY)\s*-->\s*(?:\n--[^\n]*)?\s*\Z",
            text,
        )
        if not footer:
            return ""
        prose = text[:footer.start()].strip()
        # Quoted history and code blocks are not the agent's current report.
        prose = re.split(r"(?m)^\s*(?:>|```|#{1,6} )", prose, maxsplit=1)[0]
        summary = re.split(r"\n\s*\n", prose, maxsplit=1)[0]
        summary = re.sub(r"<!--.*?-->", "", summary, flags=re.S).strip()
        number = None
        terminal = True
    if not isinstance(summary, str) or not summary.strip():
        return ""
    if not terminal and not re.search(r"\b(?:blocking|blocked|cannot proceed)\b", summary, re.I):
        return ""
    reference_text = summary
    summary = " ".join(summary.split())[:700]
    refs: set[int] = set()
    invalid = False
    if number is not None:
        if type(number) is int and number > 0:
            refs.add(number)
        else:
            invalid = True
    urls = re.findall(r"https?://github\.com/[^\s<>]+", reference_text)
    for url in urls:
        match = re.fullmatch(r"https://github\.com/([^/]+/[^/]+)/pull/([1-9][0-9]*)", url.rstrip(".,);"))
        if match and match[1].casefold() == repo.casefold():
            refs.add(int(match[2]))
        else:
            invalid = True
    for reference in re.findall(r"\bPR\s+#([0-9]+)\b", reference_text, re.I):
        if int(reference) > 0:
            refs.add(int(reference))
        else:
            invalid = True
    diagnostic = "\nCoder terminal context (advisory): " + summary
    if len(refs) == 1 and not invalid:
        diagnostic += (
            f"\nReported PR (unverified): https://github.com/{repo}/pull/{next(iter(refs))}."
        )
    return diagnostic
