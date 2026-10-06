"""Trusted integration-base matcher for managed CI (#1285).

The block between the BEGIN/END markers is copied verbatim into
``.github/workflows/managed-ci.yml``; the contract tests compare the two so
the hosted validator and the local tool always agree on which bases are
trusted.  Grammar of ``AGENT_LOOP_TRUSTED_BASES``: whitespace- or
comma-separated entries, each an exact git branch name or ``<prefix>/*`` with
a non-empty, wildcard-free prefix.  Any other entry (``*``, ``**``, embedded
wildcards, invalid ref names) makes the whole variable invalid, and an unset
or invalid variable trusts the default branch only.
"""

import re


# BEGIN MANAGED_CI_TRUSTED_BASES
# Source repository: wwind123/coding-review-agent-loop
# Source path: src/coding_review_agent_loop/managed_ci_bases.py
# Extraction boundary: trusted_base_patterns() through base_is_trusted().
def trusted_base_patterns(text):
    def valid_branch(name):
        return (
            type(name) is str and name not in {'', '@'}
            and not re.search(r'[\x00-\x20\x7f~^:?*\[\\]', name)
            and not re.search(r'\.\.|@\{|//|\.lock(/|$)|(^|/)\.|[./]$|^[/-]', name)
        )

    if type(text) is not str:
        return ()
    patterns = []
    for entry in re.split(r'[\s,]+', text.strip()):
        if not entry:
            continue
        if entry.endswith('/*'):
            if not valid_branch(entry[:-2]):
                return ()
            patterns.append((entry[:-1], True))
        elif valid_branch(entry):
            patterns.append((entry, False))
        else:
            return ()
    return tuple(patterns)


def base_is_trusted(base, default_branch, text):
    if type(base) is not str or type(default_branch) is not str or not base:
        return False
    if base == default_branch:
        return True
    for pattern, is_prefix in trusted_base_patterns(text):
        if is_prefix:
            if base.startswith(pattern) and len(base) > len(pattern):
                return True
        elif base == pattern:
            return True
    return False
# END MANAGED_CI_TRUSTED_BASES
