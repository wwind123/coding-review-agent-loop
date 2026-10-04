"""Frozen publication carriers and cross-invocation resume of spooled rounds (#1258).

A parallel review round persists each reviewer's validated response in a local
spool before the first post (#1025).  Publication is still several GitHub
comments (sidecars, then an anchor), so an outage can leave a reviewer's
bodies partly public.  A rerun must neither re-invoke the reviewer nor post a
duplicate or recomposed body.

Just before the first body is posted, the exact prepared bodies are *frozen*
into that reviewer's spool record together with the authenticated actor, the
response digest and a digest of the validation context.  A rerun then
classifies the *whole* frozen sequence against a complete actor-bound listing
and posts only a strictly missing suffix.  Any other pattern -- a gap, an
ambiguous or out-of-order match, a changed actor, a changed validation
context, an incomplete read -- stops the run with the record kept, nothing
posted, and no reviewer launched; the operator repairs it with the existing
partial-round recovery list (#1142).

Records written before this protocol are *legacy*: their original actor and
publication progress are unknown, so a complete cross-author listing must
show no possible prior publication before they are composed and published.
"""

from __future__ import annotations

import datetime
import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .errors import AgentLoopError
from .logging import log
from .review_spool import ReviewRoundSpool
from .round_transport import ROUND_RESUME_MARKER_RE, decode_mapping, is_round_transport_sidecar

# The record is written before publication, and GitHub and local clocks differ.
CLOCK_SKEW = datetime.timedelta(seconds=120)

_RECOVERY_HINT = (
    "Repair the round with the partial-round recovery list (#1142): delete the listed "
    "comments so the whole round runs again, then rerun."
)


class PublicationResumeStop(AgentLoopError):
    """A spooled publication must not be resumed automatically.

    Nothing was posted, the spool record was kept, and no reviewer was launched.
    """


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def response_digest(fields: dict[str, object]) -> str:
    """Digest of the spooled response fields a carrier is bound to."""
    return _sha256(json.dumps(fields, sort_keys=True, default=str))


def context_digest(**inputs: object) -> str:
    """Digest of validation inputs the spool identity does not already cover."""
    return _sha256(json.dumps(inputs, sort_keys=True, default=str))


@dataclass(frozen=True)
class PublicationCarrier:
    bodies: tuple[str, ...]
    body_sha256: tuple[str, ...]
    prepared_at: str
    actor_id: int | None
    actor_login: str | None
    baseline_ids: tuple[int, ...] | None
    response_sha256: str
    validation_context_digest: str

    def to_dict(self) -> dict[str, object]:
        return {
            "bodies": list(self.bodies),
            "body_sha256": list(self.body_sha256),
            "prepared_at": self.prepared_at,
            "actor_id": self.actor_id,
            "actor_login": self.actor_login,
            "baseline_ids": list(self.baseline_ids) if self.baseline_ids is not None else None,
            "response_sha256": self.response_sha256,
            "validation_context_digest": self.validation_context_digest,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> "PublicationCarrier":
        """Parse a stored carrier; any defect raises ``ValueError`` (fail closed)."""
        bodies = raw.get("bodies")
        digests = raw.get("body_sha256")
        if (
            not isinstance(bodies, list)
            or not bodies
            or not all(isinstance(item, str) and item for item in bodies)
            or not isinstance(digests, list)
            or len(digests) != len(bodies)
            or any(_sha256(body) != digest for body, digest in zip(bodies, digests))
        ):
            raise ValueError("frozen bodies are missing, empty, or fail their digests")
        prepared_at = raw.get("prepared_at")
        if not isinstance(prepared_at, str):
            raise ValueError("prepared_at is missing")
        datetime.datetime.fromisoformat(prepared_at)
        actor_id = raw.get("actor_id")
        if actor_id is not None and (
            isinstance(actor_id, bool) or not isinstance(actor_id, int) or actor_id < 1
        ):
            raise ValueError("actor_id is invalid")
        actor_login = raw.get("actor_login")
        if actor_login is not None and not isinstance(actor_login, str):
            raise ValueError("actor_login is invalid")
        baseline = raw.get("baseline_ids")
        if baseline is not None and (
            not isinstance(baseline, list)
            or not all(isinstance(item, int) and not isinstance(item, bool) for item in baseline)
        ):
            raise ValueError("baseline_ids is invalid")
        response_sha = raw.get("response_sha256")
        context = raw.get("validation_context_digest")
        if not isinstance(response_sha, str) or not isinstance(context, str):
            raise ValueError("binding digests are missing")
        return cls(
            bodies=tuple(bodies),
            body_sha256=tuple(digests),
            prepared_at=prepared_at,
            actor_id=actor_id,
            actor_login=actor_login,
            baseline_ids=tuple(baseline) if baseline is not None else None,
            response_sha256=response_sha,
            validation_context_digest=context,
        )


@dataclass(frozen=True)
class FrozenClassification:
    """Whole-sequence view of a frozen carrier against the public surface.

    ``kind``: ``complete`` (every body public), ``prefix`` (exactly bodies
    ``1..matched_count`` public, the rest absent), ``gap`` (an absent body is
    followed by a public later one), ``ambiguous`` (a body has surplus
    identical candidates) or ``disorder`` (comment ids do not follow body order).
    """

    kind: str
    matched: tuple[int | None, ...]
    matched_count: int
    detail: str = ""


def classify_frozen_sequence(
    carrier: PublicationCarrier, comments: Sequence[object]
) -> FrozenClassification:
    """Classify *every* frozen body; never stop at the first absent one.

    ``comments`` must be a complete actor-bound listing.  A body matches a
    comment with the exact stored body (host-footer tolerant), created at or
    after ``prepared_at`` minus the clock skew, and not in the carrier's
    baseline.  Identical frozen bodies take distinct comments in id order.
    """
    prepared_at = datetime.datetime.fromisoformat(carrier.prepared_at)
    floor = prepared_at - CLOCK_SKEW
    baseline = set(carrier.baseline_ids or ())
    ordered = sorted(comments, key=lambda item: item.comment_id)  # type: ignore[attr-defined]
    assigned: list[int | None] = [None] * len(carrier.bodies)
    for text in dict.fromkeys(carrier.bodies):
        indexes = [i for i, body in enumerate(carrier.bodies) if body == text]
        candidates = [
            c.comment_id  # type: ignore[attr-defined]
            for c in ordered
            if c.body == text  # type: ignore[attr-defined]
            and c.created >= floor  # type: ignore[attr-defined]
            and c.comment_id not in baseline  # type: ignore[attr-defined]
        ]
        if len(candidates) > len(indexes):
            return FrozenClassification(
                "ambiguous",
                tuple(assigned),
                0,
                f"body {indexes[0] + 1} has {len(candidates)} identical candidates "
                f"({', '.join(str(c) for c in candidates)}) for {len(indexes)} frozen occurrence(s)",
            )
        for index, comment_id in zip(indexes, candidates):
            assigned[index] = comment_id
    present = [(i, c) for i, c in enumerate(assigned) if c is not None]
    ids_in_order = [c for _i, c in present]
    if ids_in_order != sorted(ids_in_order):
        return FrozenClassification(
            "disorder", tuple(assigned), len(present),
            "matched comment ids do not increase with the frozen body order",
        )
    count = len(present)
    if count == len(carrier.bodies):
        return FrozenClassification("complete", tuple(assigned), count)
    if [i for i, _c in present] == list(range(count)):
        return FrozenClassification("prefix", tuple(assigned), count)
    missing = [i + 1 for i, c in enumerate(assigned) if c is None]
    return FrozenClassification(
        "gap", tuple(assigned), count,
        f"frozen body(ies) {', '.join(str(i) for i in missing)} are absent while a later "
        "body of the same carrier is public",
    )


def _describe_matches(classification: FrozenClassification) -> str:
    ids = [str(c) for c in classification.matched if c is not None]
    return ", ".join(ids) if ids else "none"


def _metadata(body: str) -> dict[str, object] | None:
    match = None
    for match in ROUND_RESUME_MARKER_RE.finditer(body):
        pass
    if match is None:
        return None
    try:
        return decode_mapping(match.group("payload"))
    except AgentLoopError:
        return None


@dataclass(frozen=True)
class PreparedPublication:
    """What the post loop publishes: the exact bodies and those already public."""

    bodies: tuple[str, ...]
    already_public: frozenset[int] = frozenset()


@dataclass
class RoundPublication:
    """Publication hook for one reviewer's round comment.

    The post loops call :meth:`prepare` once with the freshly composed bodies,
    before posting anything.  It freezes them on first publication, or returns
    the frozen bodies plus the already-public prefix on a resume.
    """

    spool: ReviewRoundSpool
    reviewer_name: str
    flow: str
    round_number: int
    subject: str
    validation_context: str
    surface: str
    resolve_actor: Callable[[], tuple[str, int]]
    list_comments: Callable[[bool], object]
    config: object | None = None

    # -- helpers -----------------------------------------------------------
    def _stop(self, message: str) -> PublicationResumeStop:
        return PublicationResumeStop(
            f"{self.reviewer_name}'s spooled round-{self.round_number} publication on "
            f"{self.surface} cannot be resumed automatically: {message} The spool record "
            "was kept, nothing was posted, and no reviewer was launched."
        )

    def _complete(self, actor: bool):
        listing = self.list_comments(actor)
        comments = getattr(listing, "comments", None)
        if comments is None:
            raise self._stop(
                "the conversation could not be read completely "
                f"({getattr(listing, 'reason', 'unknown reason')})."
            )
        return comments

    def _names_this_reviewer(self, body: str) -> bool:
        meta = _metadata(body)
        return (
            meta is not None
            and meta.get("flow") == self.flow
            and meta.get("agent") == self.reviewer_name
            and meta.get("round_number") == self.round_number
            and meta.get("subject") == self.subject
        )

    # -- entry -------------------------------------------------------------
    def gate(self, *, published: bool = False) -> PreparedPublication | None:
        """Run every stop condition for an existing record without posting.

        Returns the resume plan of a frozen carrier, or ``None`` for a record
        with nothing to resume.  A legacy record whose reviewer is already
        public is not re-checked (its own anchor would look like prior
        publication); its carrier-less record is simply discarded later.
        """
        state, raw = self.spool.publication_state(self.reviewer_name)
        if state == "malformed":
            raise self._stop("its frozen publication carrier is malformed.")
        if state == "legacy":
            if not published:
                self._check_legacy()
            return None
        if state == "carrier":
            assert raw is not None
            return self._resume(raw)
        return None

    def prepare(self, prepared: Sequence[object]) -> PreparedPublication:
        state, raw = self.spool.publication_state(self.reviewer_name)
        bodies = tuple(str(body) for body in prepared)
        if state == "absent":
            return PreparedPublication(bodies)
        if state == "malformed":
            raise self._stop("its frozen publication carrier is malformed.")
        if state == "legacy":
            self._check_legacy()
        if state in {"legacy", "fresh"}:
            self._freeze(bodies)
            return PreparedPublication(bodies)
        assert raw is not None
        return self._resume(raw)

    # -- freeze ------------------------------------------------------------
    def _freeze(self, bodies: tuple[str, ...]) -> None:
        """Persist the carrier before the first post, or stop without posting.

        A carrier that is not actor-bound with a complete baseline could never
        be resumed safely, so publication does not begin without one; the
        validated response stays spooled and a rerun retries the freeze.
        """
        fields = self.spool.load(self.reviewer_name)
        assert fields is not None
        prepared_at = datetime.datetime.now(datetime.timezone.utc)
        try:
            actor_login, actor_id = self.resolve_actor()
        except AgentLoopError as exc:
            raise self._stop(
                f"a publication carrier could not be frozen because the authenticated actor "
                f"is unavailable ({exc})."
            ) from exc
        listing = self.list_comments(True)
        comments = getattr(listing, "comments", None)
        if comments is None:
            raise self._stop(
                "a publication carrier could not be frozen because the pre-publication "
                f"baseline listing is incomplete ({getattr(listing, 'reason', 'unknown reason')})."
            )
        baseline = tuple(sorted(c.comment_id for c in comments if c.body in bodies))
        carrier = PublicationCarrier(
            bodies=bodies,
            body_sha256=tuple(_sha256(body) for body in bodies),
            prepared_at=prepared_at.isoformat(),
            actor_id=actor_id,
            actor_login=actor_login,
            baseline_ids=baseline,
            response_sha256=response_digest(fields),
            validation_context_digest=self.validation_context,
        )
        # Persisted before the first post: a record with the protocol marker and
        # no carrier therefore provably never began publishing.
        self.spool.store_publication(self.reviewer_name, carrier.to_dict())

    # -- resume ------------------------------------------------------------
    def _resume(self, raw: dict[str, object]) -> PreparedPublication:
        try:
            carrier = PublicationCarrier.from_dict(raw)
        except (ValueError, TypeError) as exc:
            raise self._stop(f"its frozen publication carrier is malformed ({exc}).") from exc
        fields = self.spool.load(self.reviewer_name)
        if fields is None or response_digest(fields) != carrier.response_sha256:
            raise self._stop("the spooled response no longer matches the frozen carrier.")
        anchor = carrier.bodies[-1]
        if not self._names_this_reviewer(anchor):
            raise self._stop(
                "the frozen anchor's round metadata does not name this reviewer, flow, "
                "subject and round."
            )
        if carrier.validation_context_digest != self.validation_context:
            raise self._stop(
                "the validation context (head, candidate or surfaced requirements) changed "
                "since the carrier was frozen."
            )
        if carrier.actor_id is None or carrier.baseline_ids is None:
            raise self._stop(
                "the carrier was frozen without a provable actor or pre-write baseline."
            )
        try:
            _login, current_id = self.resolve_actor()
        except AgentLoopError as exc:
            raise self._stop(f"the authenticated actor could not be resolved ({exc}).") from exc
        if current_id != carrier.actor_id:
            raise self._stop(
                f"it was frozen by actor {carrier.actor_login} (id {carrier.actor_id}) but the "
                f"current credentials authenticate actor id {current_id}."
            )
        comments = self._complete(True)
        result = classify_frozen_sequence(carrier, comments)
        if result.kind in {"ambiguous", "disorder", "gap"}:
            raise self._stop(
                f"{result.detail}; matched comment ids: {_describe_matches(result)}. "
                + _RECOVERY_HINT
            )
        if self.config is not None:
            log(
                self.config,  # type: ignore[arg-type]
                f"{self.reviewer_name}: resuming the frozen round publication "
                f"({result.matched_count} of {len(carrier.bodies)} bodies already public)",
            )
        return PreparedPublication(
            carrier.bodies, frozenset(range(1, result.matched_count + 1))
        )

    # -- legacy ------------------------------------------------------------
    def _check_legacy(self) -> None:
        mtime = self.spool.record_mtime(self.reviewer_name)
        if mtime is None:
            raise self._stop("the legacy spool record's age cannot be read.")
        floor = datetime.datetime.fromtimestamp(mtime, datetime.timezone.utc) - CLOCK_SKEW
        comments = self._complete(False)
        suspects = [
            c for c in comments
            if c.created >= floor  # type: ignore[attr-defined]
            and (
                is_round_transport_sidecar(c.body)  # type: ignore[attr-defined]
                or self._names_this_reviewer(c.body)  # type: ignore[attr-defined]
            )
        ]
        if suspects:
            listed = "; ".join(
                f"comment {c.comment_id} by {c.author_login}" for c in suspects  # type: ignore[attr-defined]
            )
            raise self._stop(
                "this record predates publication carriers and possible earlier publication is "
                f"visible ({listed}). " + _RECOVERY_HINT
            )
