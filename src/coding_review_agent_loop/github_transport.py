"""Choose between GitHub GraphQL and repository-scoped REST (#1029).

Some hosts (Claude Code cloud sessions) refuse every ``api.github.com/graphql``
request at an egress proxy while allowing repository-scoped REST.  The
``agent-loop-gh`` shim and the commit-provenance reader share this module so
both detect that environment the same way.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Literal

# The proxy's refusal names this exact phrase; anything else (network, auth,
# rate limit) is an ordinary failure and must not switch transports.
GRAPHQL_REFUSAL_TEXT = "GitHub GraphQL is not available"

TRANSPORT_ENV = "AGENT_LOOP_GH_TRANSPORT"
TRANSPORT_MODES = ("auto", "rest", "graphql")
PROBE_CACHE_TTL_SECONDS = 15 * 60
_PROBE_QUERY = "query={viewer{login}}"
_CACHE_KEY_ENV = ("GH_HOST", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy")

TransportMode = Literal["auto", "rest", "graphql"]
Transport = Literal["rest", "graphql"]


class TransportConfigError(ValueError):
    """Raised for an unrecognized ``AGENT_LOOP_GH_TRANSPORT`` value."""


def is_graphql_refusal(text: str | None) -> bool:
    return bool(text) and GRAPHQL_REFUSAL_TEXT in text


def transport_mode(env: Mapping[str, str] | None = None) -> TransportMode:
    values = os.environ if env is None else env
    raw = (values.get(TRANSPORT_ENV) or "auto").strip().lower()
    if raw not in TRANSPORT_MODES:
        raise TransportConfigError(
            f"{TRANSPORT_ENV}={raw!r} is not one of {', '.join(TRANSPORT_MODES)}."
        )
    return raw  # type: ignore[return-value]


def _cache_path(env: Mapping[str, str]) -> Path:
    base = env.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "agent-loop-gh" / "transport.json"


def _cache_key(real_gh: str, env: Mapping[str, str]) -> str:
    material = {name: env.get(name, "") for name in _CACHE_KEY_ENV}
    material["real_gh"] = real_gh
    return hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()


def _read_cached(real_gh: str, env: Mapping[str, str], now: float) -> Transport | None:
    try:
        payload = json.loads(_cache_path(env).read_text(encoding="utf-8"))
        entry = payload[_cache_key(real_gh, env)]
        decision = entry["transport"]
        stamp = float(entry["at"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if decision not in ("rest", "graphql") or not 0 <= now - stamp < PROBE_CACHE_TTL_SECONDS:
        return None
    return decision


def _write_cached(real_gh: str, env: Mapping[str, str], decision: Transport, now: float) -> None:
    path = _cache_path(env)
    try:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                payload = {}
        except (OSError, ValueError):
            payload = {}
        payload[_cache_key(real_gh, env)] = {"transport": decision, "at": now}
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, delete=False, prefix=".transport-"
        ) as handle:
            json.dump(payload, handle)
            temp_name = handle.name
        os.replace(temp_name, path)
    except OSError:
        # The cache only saves a probe; an unwritable cache is not an error.
        return


ProbeRunner = Callable[[list[str]], "subprocess.CompletedProcess[str]"]


def _default_probe(argv: list[str]) -> "subprocess.CompletedProcess[str]":
    return subprocess.run(argv, capture_output=True, text=True, check=False)


def select_transport(
    real_gh: str,
    *,
    env: Mapping[str, str] | None = None,
    probe: ProbeRunner | None = None,
    now: float | None = None,
) -> Transport:
    """Decide the transport before any GitHub command runs.

    ``auto`` probes GraphQL once and chooses REST only for the proxy's exact
    refusal.  A transient probe failure keeps GraphQL (so ordinary ``gh``
    errors surface unchanged) and is never cached.
    """
    values = os.environ if env is None else env
    mode = transport_mode(values)
    if mode == "rest":
        return "rest"
    if mode == "graphql":
        return "graphql"
    stamp = time.time() if now is None else now
    cached = _read_cached(real_gh, values, stamp)
    if cached is not None:
        return cached
    result = (probe or _default_probe)([real_gh, "api", "graphql", "-f", _PROBE_QUERY])
    if result.returncode == 0:
        _write_cached(real_gh, values, "graphql", stamp)
        return "graphql"
    if is_graphql_refusal(f"{result.stderr or ''}\n{result.stdout or ''}"):
        _write_cached(real_gh, values, "rest", stamp)
        return "rest"
    return "graphql"
