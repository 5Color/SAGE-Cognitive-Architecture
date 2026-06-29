from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Mapping

from sage_core import BaseOrgan, BaseRouter, OrganResult, SAGEState
from benchmarks.tasks.sparse_anti_leak_routing_task import (
    ORGANS,
    AntiLeakRoutingEnvironment,
    EvidenceAntiLeakRouter,
    FamilyOracleAntiLeakRouter,
    LinearOrgan,
    MemoryOrgan,
    PlannerOrgan,
    SparseAntiLeakRoutingMetric,
    SparseEvidenceAntiLeakRouter,
    ThresholdOrgan,
    score_candidate,
    split_support,
)


class AdaptiveComputeModeRouter(BaseRouter):
    """v1.7 router: choose Top1 / Top2 / Full dynamically."""

    name = "adaptive_compute_mode_router"

    def __init__(
        self,
        top1_score_threshold: float = 0.95,
        top1_gap_threshold: float = 0.35,
        full_score_threshold: float = 0.45,
        full_gap_threshold: float = 0.10,
        low_energy_threshold: float = 0.35,
    ) -> None:
        self.top1_score_threshold = float(top1_score_threshold)
        self.top1_gap_threshold = float(top1_gap_threshold)
        self.full_score_threshold = float(full_score_threshold)
        self.full_gap_threshold = float(full_gap_threshold)
        self.low_energy_threshold = float(low_energy_threshold)
        self.last_scores: Dict[str, float] = {}
        self.last_mode = "unknown"
        self.last_gap = 0.0
        self.last_best_score = 0.0
        self.last_second_score = 0.0

    def _rank(self, signal: Dict[str, Any], organs: Mapping[str, BaseOrgan]):
        support = signal.get("support", [])
        fit, val = split_support(support)
        scores = {
            name: score_candidate(name, fit, val)
            for name in ORGANS
            if name in organs
        }
        ranked = sorted(scores, key=lambda n: (scores[n], -ORGANS.index(n)), reverse=True)
        return ranked, scores

    def _mode(self, state: SAGEState, ranked: List[str], scores: Dict[str, float]) -> str:
        if not ranked:
            return "top1"

        best = ranked[0]
        second = ranked[1] if len(ranked) > 1 else ranked[0]
        best_score = float(scores.get(best, 0.0))
        second_score = float(scores.get(second, 0.0)) if len(ranked) > 1 else 0.0
        gap = best_score - second_score

        self.last_best_score = best_score
        self.last_second_score = second_score
        self.last_gap = gap

        if state.energy < self.low_energy_threshold:
            return "top1" if (best_score >= self.top1_score_threshold or gap >= self.top1_gap_threshold) else "top2"

        if best_score >= self.top1_score_threshold and gap >= self.top1_gap_threshold:
            return "top1"

        if best_score < self.full_score_threshold or gap <= self.full_gap_threshold:
            return "full"

        return "top2"

    def route(self, state: SAGEState, signal: Dict[str, Any], organs: Mapping[str, BaseOrgan]) -> List[str]:
        ranked, scores = self._rank(signal, organs)
        self.last_scores = scores

        if not ranked:
            self.last_mode = "top1"
            return [next(iter(organs.keys()))]

        self.last_mode = self._mode(state, ranked, scores)

        if self.last_mode == "top1":
            return ranked[:1]
        if self.last_mode == "top2":
            return ranked[: min(2, len(ranked))]
        return ranked

    def aggregate(self, state: SAGEState, signal: Dict[str, Any], outputs: Mapping[str, OrganResult]) -> Dict[str, Any]:
        if not outputs:
            return {
                "prediction": 0,
                "chosen_organ": None,
                "confidence": 0.0,
                "compute_mode": self.last_mode,
                "router_scores": dict(self.last_scores),
                "confidence_gap": self.last_gap,
                "best_score": self.last_best_score,
                "second_score": self.last_second_score,
            }

        best_name = max(
            outputs,
            key=lambda n: (
                outputs[n].confidence,
                self.last_scores.get(n, 0.0),
                -ORGANS.index(n) if n in ORGANS else -999,
            ),
        )
        best = outputs[best_name]
        return {
            "prediction": int(best.action),
            "chosen_organ": best.organ_name,
            "confidence": float(best.confidence),
            "compute_mode": self.last_mode,
            "router_scores": dict(self.last_scores),
            "confidence_gap": float(self.last_gap),
            "best_score": float(self.last_best_score),
            "second_score": float(self.last_second_score),
        }


class AdaptiveComputeMetric(SparseAntiLeakRoutingMetric):
    name = "adaptive_compute_metric"

    def evaluate(self, history):
        result = super().evaluate(history)

        mode_counts = Counter()
        gaps = []
        best_scores = []

        for item in history:
            selected = item.get("selected_organs", [])
            action = item.get("action", {})

            mode = action.get("compute_mode")
            if not mode:
                n = len(selected)
                mode = "top1" if n == 1 else "top2" if n == 2 else "full" if n >= self.full_organ_count else f"top{n}"

            mode_counts[str(mode)] += 1

            if "confidence_gap" in action:
                gaps.append(float(action.get("confidence_gap", 0.0)))
            if "best_score" in action:
                best_scores.append(float(action.get("best_score", 0.0)))

        steps = max(1, len(history))
        result["mode_counts"] = dict(mode_counts)
        result["mode_usage"] = {k: v / steps for k, v in sorted(mode_counts.items())}
        result["avg_confidence_gap"] = sum(gaps) / len(gaps) if gaps else 0.0
        result["avg_best_router_score"] = sum(best_scores) / len(best_scores) if best_scores else 0.0

        nonzero_modes = sum(1 for v in mode_counts.values() if v > 0)
        mode_diversity_bonus = min(1.0, nonzero_modes / 3.0)

        result["adaptive_efficiency_score"] = (
            0.65 * float(result.get("accuracy", 0.0))
            + 0.25 * float(result.get("compute_saving_vs_full", 0.0))
            + 0.10 * mode_diversity_bonus
        )
        return result
