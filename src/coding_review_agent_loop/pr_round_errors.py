"""PR history failures must not degrade into advisory or stale-history fallbacks."""
from functools import wraps

from .errors import AgentLoopError


class PrRoundHistoryError(AgentLoopError):
    """A live PR observation cannot safely supply round history."""


def pr_history_boundary(function):
    """Keep the failure distinguishable at legacy advisory reader boundaries."""
    @wraps(function)
    def checked(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except PrRoundHistoryError:
            raise
        except (AgentLoopError, OSError) as exc:
            raise PrRoundHistoryError(str(exc)) from exc
    return checked
