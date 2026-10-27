"""Conservative reuse checks for historically contradictory approval carriers."""


def _contradictory_evidence_approval(metadata) -> bool:
    """Keep historical ledger evidence without strictly rereading an invalid approval."""
    return metadata.state == "approved" and any(
        d.disposition in {"blocking", "same-pr"} for d in metadata.dispositions
    )
