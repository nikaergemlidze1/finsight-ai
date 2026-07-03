"""Lead-tier classification — single source of truth.

Used by both the /predict response and the /analytics aggregation so the
UI and the API can never drift apart on what "high priority" means.
"""
from __future__ import annotations

# Probability cut-offs (inclusive lower bounds)
TIER_HIGH = 0.60
TIER_MEDIUM = 0.30


def classify(probability: float) -> str:
    """Map a subscription probability in [0, 1] to a lead tier."""
    if probability >= TIER_HIGH:
        return "high"
    if probability >= TIER_MEDIUM:
        return "medium"
    return "low"
