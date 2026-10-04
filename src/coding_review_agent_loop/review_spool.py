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

# Written into every record stored from #1258 on.  A record carrying it and no
# publication carrier provably never began publishing, because the carrier is
# always persisted before the first post; a record without it is legacy, with
# unknown publication progress.
CARRIER_PROTOCOL = 1

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
            {
                "carrier_protocol": CARRIER_PROTOCOL,
                "response": {name: fields.get(name) for name in SPOOLED_RESPONSE_FIELDS},
            },
        )

    def store_publication(self, reviewer_name: str, carrier: dict[str, object]) -> None:
        """Atomically add a frozen publication carrier to a reviewer's response record.

        Everything else in the record is preserved.  Raises ``ValueError`` when
        there is no readable response record to attach the carrier to.
        """
        payload = self._read_payload(reviewer_name)
        if payload is None or not isinstance(payload.get("response"), dict):
            raise ValueError(f"No spooled response record for {reviewer_name} to freeze.")
        payload = {**payload, "carrier_protocol": CARRIER_PROTOCOL, "publication": carrier}
        self._replace(reviewer_name, payload)

    def publication_state(self, reviewer_name: str) -> tuple[str, dict[str, object] | None]:
        """Classify a record's publication progress.

        ``absent``: no record that belongs to this reviewer and round.
        ``legacy``: a readable response record written before the carrier
        protocol.  ``fresh``: a protocol record whose carrier was never frozen.
        ``carrier``: a frozen carrier (returned).  ``malformed``: a record that
        has a ``publication`` field of any kind (including ``null``) but is not
        a readable protocol response record with a mapping carrier.  Carrier
        presence is detected independently of response readability, so an
        unreadable response can never hide earlier publication.
        """
        if not self._path(reviewer_name).exists():
            return "absent", None
        payload = self._read_payload(reviewer_name)
        if payload is None:
            # The file exists but cannot be decoded into a mapping: it may hide a
            # frozen publication carrier, so it is never treated as absent.
            return "malformed", None
        if (
            payload.get("schema_version") != SPOOL_SCHEMA_VERSION
            or any(payload.get(key) != value for key, value in self._identity().items())
            or payload.get("reviewer") != reviewer_name
        ):
            # The path is derived from this round's identity, so a record here
            # with a damaged schema or identity is not "foreign": when it still
            # carries publication state it may hide a published prefix.
            if "publication" in payload or "carrier_protocol" in payload:
                return "malformed", None
            return "absent", None
        if "publication" in payload:
            publication = payload.get("publication")
            if (
                not isinstance(publication, dict)
                or payload.get("carrier_protocol") != CARRIER_PROTOCOL
                or self.load(reviewer_name) is None
                or "response" not in payload
            ):
                return "malformed", None
            return "carrier", publication
        if self.load(reviewer_name) is None or "response" not in payload:
            return "absent", None
        if payload.get("carrier_protocol") == CARRIER_PROTOCOL:
            return "fresh", None
        return "legacy", None

    def record_mtime(self, reviewer_name: str) -> float | None:
        try:
            return self._path(reviewer_name).stat().st_mtime
        except OSError:
            return None

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
        self._replace(reviewer_name, payload)

    def _read_payload(self, reviewer_name: str) -> dict[str, object] | None:
        try:
            payload = json.loads(self._path(reviewer_name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return payload if isinstance(payload, dict) else None

    def _replace(self, reviewer_name: str, payload: dict[str, object]) -> None:
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

    def record_exists(self, reviewer_name: str) -> bool:
        return self._path(reviewer_name).exists()

    def has_records(self) -> bool:
        try:
            return any(self.directory.glob("*.json"))
        except OSError:
            return False

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
