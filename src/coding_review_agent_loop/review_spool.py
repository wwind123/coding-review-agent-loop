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

The spool root lives under a non-listable ``private/`` directory beside the
agent memory directory, never directly beside it, so a reviewer that
explores the prompt-advertised memory directory's parent cannot enumerate
round records (#1560).  A round directory left at the legacy location by an
older version is migrated by an atomic rename before first use; when both
locations hold the round, the spool fails closed.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .errors import AgentLoopError
from .scratch import make_private_dirs, mkdir_private

SPOOL_SCHEMA_VERSION = 1

# Written into every record stored from #1258 on.  A record carrying it and no
# publication mark provably never began publishing, because the mark is always
# persisted before the first post; a record without it is legacy, with unknown
# publication progress.  A publication that began is never resumed: the
# reviewer is unreplayable and its round closes as incomplete (#2064).
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
    "seat_binding",
)

# Optional response fields persisted only when present, so records without
# them stay byte-identical: the #2100 ignored non-carried disposition IDs.
OPTIONAL_SPOOLED_RESPONSE_FIELDS: tuple[str, ...] = ("ignored_prior_disposition_ids",)


def _spooled_response(fields: Mapping[str, object]) -> dict[str, object]:
    return {
        **{name: fields.get(name) for name in SPOOLED_RESPONSE_FIELDS},
        **{
            name: fields[name]
            for name in OPTIONAL_SPOOLED_RESPONSE_FIELDS
            if fields.get(name) is not None
        },
    }


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
    # The pre-#1560 root.  A round directory found there is migrated into
    # ``root`` before any use.
    legacy_root: Path | None = None

    def _identity(self) -> dict[str, object]:
        return {
            "repo": self.repo,
            "surface": self.surface,
            "number": self.number,
            "round_number": self.round_number,
            "subject": self.subject,
        }

    @property
    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(self._identity(), sort_keys=True).encode("utf-8")
        ).hexdigest()[:32]

    @property
    def directory(self) -> Path:
        directory = self.root / self.digest
        self._migrate_legacy(directory)
        return directory

    def _migrate_legacy(self, directory: Path) -> None:
        """Move this round's legacy directory into ``root``; fail closed on conflict.

        A legacy record may mark a begun publication, so it is never
        ignored: both locations present, or a failed rename, raises instead
        of falling back to a fresh reviewer turn (#1258 semantics).
        """
        if self.legacy_root is None:
            return
        legacy = self.legacy_root / self.digest
        if not os.path.lexists(legacy):
            return
        if os.path.lexists(directory):
            raise AgentLoopError(
                f"Review-round spool for this round exists at both {legacy} (legacy) and "
                f"{directory}. Inspect both, keep the one holding the round's records, and "
                "remove the other before rerunning."
            )
        try:
            make_private_dirs(self.root)
            os.rename(legacy, directory)
        except OSError as exc:
            raise AgentLoopError(
                f"Could not migrate the legacy review-round spool {legacy} to {directory}: {exc}"
            ) from exc

    def _path(self, reviewer_name: str) -> Path:
        slug = _UNSAFE_NAME_RE.sub("-", reviewer_name).strip("-") or "reviewer"
        return self.directory / f"{slug}.json"

    def store(self, reviewer_name: str, fields: dict[str, object]) -> None:
        """Atomically persist one reviewer's validated response fields."""
        self._write(
            reviewer_name,
            {
                "carrier_protocol": CARRIER_PROTOCOL,
                "response": _spooled_response(fields),
            },
        )

    def mark_publication_began(self, reviewer_name: str) -> None:
        """Atomically record that this reviewer's publication is about to begin.

        Everything else in the record is preserved.  A record with the mark may
        be partly public, so it is never replayed again: the round closes as
        incomplete instead (#2064).  Raises ``ValueError`` when there is no
        readable response record to mark.
        """
        payload = self._read_payload(reviewer_name)
        if payload is None or not isinstance(payload.get("response"), dict):
            raise ValueError(f"No spooled response record for {reviewer_name} to mark.")
        began_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        payload = {**payload, "carrier_protocol": CARRIER_PROTOCOL, "publication": {"began_at": began_at}}
        self._replace(reviewer_name, payload)

    def publication_state(self, reviewer_name: str) -> str:
        """Classify a record's publication progress.

        ``absent``: no record that belongs to this reviewer and round.
        ``legacy``: a readable response record written before the publication
        protocol, with unknown progress.  ``fresh``: a protocol record whose
        publication never began.  ``began``: a record whose publication began
        (including a pre-#2064 frozen carrier), so it may be partly public.
        ``malformed``: a record that has a ``publication`` field of any kind
        (including ``null``) but is not a readable protocol response record
        with a mapping mark.  Publication state is detected independently of
        response readability, so an unreadable response can never hide
        earlier publication.
        """
        if not self._path(reviewer_name).exists():
            return "absent"
        payload = self._read_payload(reviewer_name)
        if payload is None:
            # The file exists but cannot be decoded into a mapping: it may hide a
            # begun publication, so it is never treated as absent.
            return "malformed"
        if (
            payload.get("schema_version") != SPOOL_SCHEMA_VERSION
            or any(payload.get(key) != value for key, value in self._identity().items())
            or payload.get("reviewer") != reviewer_name
        ):
            # The path is derived from this round's identity, so a record here
            # with a damaged schema or identity is not "foreign": when it still
            # carries publication state it may hide a published prefix.
            if "publication" in payload or "carrier_protocol" in payload:
                return "malformed"
            return "absent"
        if "publication" in payload:
            if (
                not isinstance(payload.get("publication"), dict)
                or payload.get("carrier_protocol") != CARRIER_PROTOCOL
                or self.load(reviewer_name) is None
                or "response" not in payload
            ):
                return "malformed"
            return "began"
        if self.load(reviewer_name) is None or "response" not in payload:
            return "absent"
        if payload.get("carrier_protocol") == CARRIER_PROTOCOL:
            return "fresh"
        return "legacy"

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
        make_private_dirs(self.root)
        mkdir_private(directory)
        handle, temp_name = tempfile.mkstemp(dir=directory, prefix=".spool-", suffix=".tmp")
        try:
            if hasattr(os, "fchmod"):
                # Exactly 0600 whatever the umask: a mode-000 record would
                # read back as malformed.
                os.fchmod(handle, 0o600)
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
        return _spooled_response(response)

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


SPOOL_DIR_NAME = "review-round-spool"
PRIVATE_DIR_NAME = "private"


def review_spool_root(agent_memory_dir: Path) -> Path:
    """Repo-scoped spool root under a non-listable ``private/`` directory.

    It sits beside, not inside, the agent memory directory, and is no longer a
    direct sibling of it (#1560).
    """
    return agent_memory_dir.parent / PRIVATE_DIR_NAME / SPOOL_DIR_NAME


def legacy_review_spool_root(agent_memory_dir: Path) -> Path:
    """The pre-#1560 spool root, a direct sibling of the memory directory."""
    return agent_memory_dir.parent / SPOOL_DIR_NAME
