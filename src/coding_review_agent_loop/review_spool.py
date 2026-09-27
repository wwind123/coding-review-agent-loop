"""Private holding area for a parallel review round's validated responses (#1025).

Parallel reviewers publish only after every same-round worker has returned, so
no reviewer can read a peer's findings mid-turn.  Publication itself is one
GitHub comment per reviewer, though, so an interruption between two posts
leaves part of the round public.  Without this spool, a rerun would re-invoke
the unpublished reviewer while its peer's body is visible on the PR/issue.

Every settled outcome -- a validated response, or a failure that settles the
reviewer as unavailable for the round -- is written here after the workers
return and before the first publication.  When a reviewer must instead be
re-invoked (a fatal failure), the healthy responses stay here unpublished until
that reviewer has completed an independent turn.  A rerun replays a spooled
outcome instead of re-invoking that reviewer, and the round's spool is
discarded once every publication succeeded.  The spool is local,
operator-private state; it is never posted and never shown in a prompt.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

SPOOL_SCHEMA_VERSION = 1

# ValidatedAgentResponse fields that are persisted verbatim.  Usage metadata is
# excluded on purpose: the original invocation already accounted for it.
SPOOLED_RESPONSE_FIELDS: tuple[str, ...] = (
    "text",
    "session_id",
    "model_used",
    "provider",
    "role",
    "configured_model",
    "configured_effort",
    "effort_source",
    "observed_model",
    "observed_effort",
    "observation_provenance",
    "acquisition_outcome",
    "acquisition_returncode",
)

_UNSAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class ReviewRoundSpool:
    """The spool for one parallel review round on one plan/PR subject."""

    root: Path
    repo: str
    surface: str
    number: int
    round_number: int
    subject: str

    def _identity(self) -> dict[str, object]:
        return {
            "repo": self.repo,
            "surface": self.surface,
            "number": self.number,
            "round_number": self.round_number,
            "subject": self.subject,
        }

    @property
    def directory(self) -> Path:
        digest = hashlib.sha256(
            json.dumps(self._identity(), sort_keys=True).encode("utf-8")
        ).hexdigest()[:32]
        return self.root / digest

    def _path(self, reviewer_name: str) -> Path:
        slug = _UNSAFE_NAME_RE.sub("-", reviewer_name).strip("-") or "reviewer"
        return self.directory / f"{slug}.json"

    def store(self, reviewer_name: str, fields: dict[str, object]) -> None:
        """Atomically persist one reviewer's validated response fields."""
        self._write(
            reviewer_name,
            {"response": {name: fields.get(name) for name in SPOOLED_RESPONSE_FIELDS}},
        )

    def store_failure(self, reviewer_name: str, *, message: str, failure_category: str | None) -> None:
        """Atomically persist a failure that settled the reviewer as unavailable."""
        self._write(
            reviewer_name,
            {"failure": {"message": message, "failure_category": failure_category}},
        )

    def _write(self, reviewer_name: str, outcome: dict[str, object]) -> None:
        payload = {
            "schema_version": SPOOL_SCHEMA_VERSION,
            **self._identity(),
            "reviewer": reviewer_name,
            **outcome,
        }
        directory = self.directory
        directory.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(directory, 0o700)
        except OSError:
            pass
        handle, temp_name = tempfile.mkstemp(dir=directory, prefix=".spool-", suffix=".tmp")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, sort_keys=True)
            os.replace(temp_name, self._path(reviewer_name))
        except BaseException:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
            raise

    def load(self, reviewer_name: str) -> dict[str, object] | None:
        """Return the spooled outcome, or ``None`` when absent, malformed, or foreign.

        A response outcome is its field mapping; a failure outcome is
        ``{"failure": {"message": ..., "failure_category": ...}}``.
        """
        try:
            payload = json.loads(self._path(reviewer_name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(payload, dict) or payload.get("schema_version") != SPOOL_SCHEMA_VERSION:
            return None
        if any(payload.get(key) != value for key, value in self._identity().items()):
            return None
        if payload.get("reviewer") != reviewer_name:
            return None
        failure = payload.get("failure")
        if failure is not None:
            if not isinstance(failure, dict) or not isinstance(failure.get("message"), str):
                return None
            category = failure.get("failure_category")
            if category is not None and not isinstance(category, str):
                return None
            return {"failure": {"message": failure["message"], "failure_category": category}}
        response = payload.get("response")
        if not isinstance(response, dict) or not isinstance(response.get("text"), str):
            return None
        return {name: response.get(name) for name in SPOOLED_RESPONSE_FIELDS}

    def remove(self, reviewer_name: str) -> None:
        try:
            self._path(reviewer_name).unlink()
        except FileNotFoundError:
            pass

    def discard(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)


def review_spool_root(agent_memory_dir: Path) -> Path:
    """Repo-scoped spool root beside, not inside, the agent memory directory."""
    return agent_memory_dir.parent / "review-round-spool"
