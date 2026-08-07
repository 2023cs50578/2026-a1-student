"""
harness/leaderboard.py — combine raw metrics into the weighted leaderboard
score defined in the assignment (Section 7):

    70% * nDCG@10  +  20% * MAP  +  efficiency bonus/penalty (up to ±10%)

The efficiency term is defined *relative to the class* (index build time
and mean query latency vs. the class median across all submissions), so a
single team running this harness locally cannot compute it in isolation.
`compute_provisional_score()` below reports the 90%-weight nDCG@10/MAP
component plus your raw timings, clearly labelled "provisional" — course
staff apply the efficiency modifier when aggregating the full class
leaderboard (see instructor/aggregate_leaderboard.py).
"""
from typing import Dict, Optional

NDCG_WEIGHT = 0.70
MAP_WEIGHT = 0.20
MAX_EFFICIENCY_WEIGHT = 0.10


def compute_provisional_score(aggregate_metrics: Dict[str, float]) -> float:
    """aggregate_metrics: the `aggregate` dict from harness.metrics.evaluate_run
    (must contain "ndcg@10" and "map"). Returns the 90%-weight portion of
    the leaderboard score; the final ±10% efficiency modifier is applied
    at class-wide aggregation time."""
    return NDCG_WEIGHT * aggregate_metrics["ndcg@10"] + MAP_WEIGHT * aggregate_metrics["map"]


def efficiency_modifier(
    your_index_build_s: float,
    your_mean_query_latency_s: float,
    class_median_index_build_s: float,
    class_median_query_latency_s: float,
) -> float:
    """A simple, transparent efficiency modifier in [-MAX_EFFICIENCY_WEIGHT,
    +MAX_EFFICIENCY_WEIGHT]: reward being faster than the class median on
    both index build time and mean query latency, penalise being slower,
    capped at the stated maximum swing. Course staff may recalibrate the
    exact curve when aggregating the real class leaderboard; this
    implementation is the reference default.
    """
    def relative_speed(yours: float, median: float) -> float:
        if median <= 0:
            return 0.0
        # +1 if you're arbitrarily fast, -1 if you're >=3x slower than median,
        # linear in between.
        ratio = yours / median
        return max(-1.0, min(1.0, (1.0 - ratio) / 2.0 + 0.5)) * 2 - 1

    build_component = relative_speed(your_index_build_s, class_median_index_build_s)
    query_component = relative_speed(your_mean_query_latency_s, class_median_query_latency_s)
    combined = (build_component + query_component) / 2.0
    return combined * MAX_EFFICIENCY_WEIGHT


def summarize(
    aggregate_metrics: Dict[str, float],
    baseline_aggregate_metrics: Optional[Dict[str, float]] = None,
) -> Dict:
    provisional = compute_provisional_score(aggregate_metrics)
    result = {
        "ndcg@10": aggregate_metrics["ndcg@10"],
        "map": aggregate_metrics["map"],
        "provisional_score_90pct": provisional,
        "note": (
            "This is nDCG@10 and MAP combined at their 70%/20% weights "
            "(90% of the total). The remaining ±10% efficiency modifier "
            "is computed relative to the whole class's timings and is "
            "applied when course staff aggregate the real leaderboard."
        ),
    }
    if baseline_aggregate_metrics is not None:
        result["baseline_ndcg@10"] = baseline_aggregate_metrics["ndcg@10"]
        result["baseline_map"] = baseline_aggregate_metrics["map"]
        result["beats_baseline_ndcg@10"] = aggregate_metrics["ndcg@10"] > baseline_aggregate_metrics["ndcg@10"]
        result["beats_baseline_map"] = aggregate_metrics["map"] > baseline_aggregate_metrics["map"]
    return result
