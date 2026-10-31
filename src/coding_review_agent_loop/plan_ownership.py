"""Deterministic ownership diagnostics for published, unapproved plans."""

from .decomposition import RiskMatrixOwnershipError, validate_risk_matrix_ownership
from .comment_rendering import RISK_TEST_MATRIX_MARKER_RE, decode_risk_test_matrix_marker
from .round_state import sanitize_plan_validation_diagnostic


def plan_ownership_diagnostic(plan, recommendation) -> str | None:
    if recommendation is None:
        return None
    match = RISK_TEST_MATRIX_MARKER_RE.search(plan)
    matrix = decode_risk_test_matrix_marker(match.group("payload"))["matrix"] if match else None
    try:
        validate_risk_matrix_ownership(matrix, recommendation)
    except RiskMatrixOwnershipError as exc:
        return sanitize_plan_validation_diagnostic(str(exc))
    return None


def validate_candidate_ownership(candidate) -> None:
    """Permitted historical candidates without recommendations remain unchanged."""
    if candidate.execution_recommendation is not None:
        validate_risk_matrix_ownership(candidate.risk_test_matrix, candidate.execution_recommendation)
