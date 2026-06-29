# SAGE v1.6.3 - Sparse Evidence Router Plugin
#
# Goal:
# - Previous EvidenceAntiLeakRouter selected all organs, then aggregated outputs.
# - SparseEvidenceAntiLeakRouter first reads support evidence cheaply,
#   then activates only top-k organs.
#
# This tests compute-aware routing:
# - Can SAGE preserve most accuracy while reducing organ calls?

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from sage_core import BaseOrgan, BaseRouter, OrganResult, SAGEState

from benchmarks.tasks.anti_leak_routing_task import (
    ACTION_DIM,
    FAMILIES,
    ORGANS,
    AntiLeakRoutingEnvironment,
    AntiLeakRoutingMetric,
    EvidenceAntiLeakRouter,
    FamilyOracleAntiLeakRouter,
    KeyTypeAntiLeakRouter,
    LinearOrgan,
    MemoryOrgan,
    PlannerOrgan,
    RandomAntiLeakRouter,
    ThresholdOrgan,
    majority_action,
    parse_item,
)


def split_support(support: Sequence[Tuple[Any, int]]) -> Tuple[List[Tuple[Any, int]], List[Tuple[Any, int]]]:
    """Deterministic support split for lightweight evidence scoring."""
    support = list(support)
    if len(support) <= 2:
        return support, support

    # No random shuffle here: deterministic runner behavior.
    cut = max(1, int(len(support) * 0.67))
    fit_part = support[:cut]
    val_part = support[cut:] or support
    return fit_part, val_part


def predict_memory(fit: Sequence[Tuple[Any, int]], query: Any) -> int:
    table = {key: int(action) for key, action in fit}
    return table.get(query, majority_action(fit))


def fit_linear(fit: Sequence[Tuple[Any, int]]) -> Tuple[int, int]:
    pairs = [(parse_item(key), int(action)) for key, action in fit]
    default = majority_action(fit)
    if len(pairs) < 2:
        return 0, default

    best_a, best_b, best_score = 0, default, -1
    for a in range(ACTION_DIM):
        for b in range(ACTION_DIM):
            score = sum(1 for x, y in pairs if (a * x + b) % ACTION_DIM == y)
            if score > best_score:
                best_a, best_b, best_score = a, b, score
    return best_a, best_b


def predict_linear(fit: Sequence[Tuple[Any, int]], query: Any) -> int:
    a, b = fit_linear(fit)
    return (a * parse_item(query) + b) % ACTION_DIM


def fit_threshold(fit: Sequence[Tuple[Any, int]]) -> Tuple[int, int, int]:
    pairs = sorted((parse_item(key), int(action)) for key, action in fit)
    default = majority_action(fit)
    if len(pairs) < 3:
        return 0, default, default

    xs = [x for x, _ in pairs]
    best_threshold, best_low, best_high, best_score = 0, default, default, -1

    for threshold in range(min(xs), max(xs) + 2):
        lows = [y for x, y in pairs if x < threshold]
        highs = [y for x, y in pairs if x >= threshold]
        if not lows or not highs:
            continue

        low_action = Counter(lows).most_common(1)[0][0]
        high_action = Counter(highs).most_common(1)[0][0]
        score = sum(1 for x, y in pairs if (low_action if x < threshold else high_action) == y)

        if score > best_score:
            best_threshold, best_low, best_high, best_score = threshold, low_action, high_action, score

    return best_threshold, best_low, best_high


def predict_threshold(fit: Sequence[Tuple[Any, int]], query: Any) -> int:
    threshold, low_action, high_action = fit_threshold(fit)
    x = parse_item(query)
    return low_action if x < threshold else high_action


def predict_planner(query: Any) -> int:
    encoded = parse_item(query)
    pos = encoded // 10
    goal = encoded % 10
    return 1 if goal > pos else 0 if goal < pos else 2


def candidate_predict(organ_name: str, fit: Sequence[Tuple[Any, int]], query: Any) -> int:
    if organ_name == "memory_organ":
        return predict_memory(fit, query)
    if organ_name == "linear_organ":
        return predict_linear(fit, query)
    if organ_name == "threshold_organ":
        return predict_threshold(fit, query)
    if organ_name == "planner_organ":
        return predict_planner(query)
    return majority_action(fit)


def score_candidate(organ_name: str, fit: Sequence[Tuple[Any, int]], val: Sequence[Tuple[Any, int]]) -> float:
    if not val:
        return 0.0

    correct = 0
    for key, target in val:
        pred = candidate_predict(organ_name, fit, key)
        correct += int(pred == int(target))
    return correct / len(val)


class SparseEvidenceAntiLeakRouter(BaseRouter):
    """Compute-aware router.

    It does not run all organs.
    It first performs a cheap support-validation estimate, then activates top-k organs.
    """

    name = "sparse_evidence_anti_leak_router"

    def __init__(self, top_k: int = 1, min_confidence: float = 0.0) -> None:
        self.top_k = max(1, int(top_k))
        self.min_confidence = float(min_confidence)
        self.last_scores: Dict[str, float] = {}

    def route(self, state: SAGEState, signal: Dict[str, Any], organs: Mapping[str, BaseOrgan]) -> List[str]:
        support = signal.get("support", [])
        fit, val = split_support(support)

        scores = {
            organ_name: score_candidate(organ_name, fit, val)
            for organ_name in ORGANS
            if organ_name in organs
        }

        if not scores:
            fallback = next(iter(organs.keys()))
            self.last_scores = {fallback: 0.0}
            return [fallback]

        # Stable order: score first, then ORGANS order.
        ranked = sorted(
            scores,
            key=lambda name: (scores[name], -ORGANS.index(name)),
            reverse=True,
        )

        selected = ranked[: min(self.top_k, len(ranked))]

        # If all confidence is poor, still select the best one.
        selected = [
            name for name in selected
            if scores.get(name, 0.0) >= self.min_confidence
        ] or [ranked[0]]

        self.last_scores = scores
        return selected

    def aggregate(
        self,
        state: SAGEState,
        signal: Dict[str, Any],
        outputs: Mapping[str, OrganResult],
    ) -> Dict[str, Any]:
        if not outputs:
            return {
                "prediction": 0,
                "chosen_organ": None,
                "confidence": 0.0,
                "router_scores": self.last_scores,
            }

        # Choose among the executed organs using organ output confidence.
        best_name = max(
            outputs,
            key=lambda name: (
                outputs[name].confidence,
                self.last_scores.get(name, 0.0),
                -ORGANS.index(name) if name in ORGANS else -999,
            ),
        )
        best = outputs[best_name]

        return {
            "prediction": int(best.action),
            "chosen_organ": best.organ_name,
            "confidence": float(best.confidence),
            "router_scores": dict(self.last_scores),
        }


class SparseAntiLeakRoutingMetric(AntiLeakRoutingMetric):
    """Adds compute-efficiency metrics to the migrated anti-leak task."""

    name = "sparse_anti_leak_routing_metric"

    def __init__(self, full_organ_count: int = 4) -> None:
        self.full_organ_count = max(1, int(full_organ_count))

    def evaluate(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        result = super().evaluate(history)

        steps = max(1, len(history))
        organ_calls = sum(len(item.get("selected_organs", [])) for item in history)
        avg_organs_per_step = organ_calls / steps
        compute_saving = 1.0 - (avg_organs_per_step / self.full_organ_count)

        result["organ_calls"] = organ_calls
        result["avg_organs_per_step"] = avg_organs_per_step
        result["full_organ_count"] = self.full_organ_count
        result["compute_saving_vs_full"] = compute_saving

        # Simple combined score for this stage:
        # preserve accuracy while rewarding sparse execution.
        result["sparse_efficiency_score"] = (
            0.70 * float(result.get("accuracy", 0.0))
            + 0.30 * float(compute_saving)
        )
        return result
