from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List, Mapping

from sage_core import BaseOrgan, OrganResult, SAGEState

# Re-export environment/organs so config can point to this module only.
from benchmarks.tasks.adaptive_compute_routing_task import (
    AntiLeakRoutingEnvironment,
    LinearOrgan,
    MemoryOrgan,
    PlannerOrgan,
    ThresholdOrgan,
    AdaptiveComputeMetric,
    AdaptiveComputeModeRouter,
)
from benchmarks.tasks.sparse_anti_leak_routing_task import ORGANS, score_candidate, split_support


class CalibratedAdaptiveComputeModeRouter(AdaptiveComputeModeRouter):
    """v1.7.1 calibrated router.

    v1.7 succeeded, but Top2 usage was low.
    This router makes Top2 the default when evidence is useful but not uniquely decisive.

    Rule idea:
    - Top1: one organ is clearly dominant.
    - Top2: evidence is useful but second candidate is still plausible.
    - Full: evidence is too weak.
    """

    name = "calibrated_adaptive_compute_mode_router"

    def __init__(
        self,
        top1_score_threshold: float = 0.95,
        top1_second_score_max: float = 0.25,
        full_best_score_below: float = 0.45,
        low_energy_threshold: float = 0.35,
        low_energy_allow_full: bool = False,
    ) -> None:
        super().__init__(
            top1_score_threshold=top1_score_threshold,
            top1_gap_threshold=0.0,
            full_score_threshold=full_best_score_below,
            full_gap_threshold=-1.0,
            low_energy_threshold=low_energy_threshold,
        )
        self.top1_second_score_max = float(top1_second_score_max)
        self.full_best_score_below = float(full_best_score_below)
        self.low_energy_allow_full = bool(low_energy_allow_full)

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

        # Low energy: avoid full unless explicitly allowed.
        if state.energy < self.low_energy_threshold and not self.low_energy_allow_full:
            if best_score >= self.top1_score_threshold and second_score <= self.top1_second_score_max:
                return "top1"
            return "top2"

        # Full only when the best evidence is weak.
        if best_score < self.full_best_score_below:
            return "full"

        # Top1 only when the best organ is clearly unique.
        if best_score >= self.top1_score_threshold and second_score <= self.top1_second_score_max:
            return "top1"

        # Otherwise prefer Top2 instead of jumping directly to Full.
        return "top2"


class OrganLifecycleMetric(AdaptiveComputeMetric):
    """Adaptive compute metric + organ lifecycle diagnostics.

    This version does not delete or disable organs.
    It only recommends lifecycle status from evidence.

    Lifecycle recommendation labels:
    - keep_active
    - monitor
    - review_or_refactor_candidate
    - dormant_candidate
    - insufficient_data
    """

    name = "organ_lifecycle_metric"

    def evaluate(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        result = super().evaluate(history)

        selected_counts: Counter[str] = Counter()
        chosen_counts: Counter[str] = Counter()
        chosen_success: Counter[str] = Counter()
        chosen_failure: Counter[str] = Counter()
        family_chosen: Dict[str, Counter[str]] = defaultdict(Counter)
        family_success: Dict[str, Counter[str]] = defaultdict(Counter)

        for item in history:
            selected = item.get("selected_organs", [])
            info = item.get("info", {}) or {}
            action = item.get("action", {}) or {}

            for organ in selected:
                selected_counts[str(organ)] += 1

            chosen = (
                info.get("chosen_organ")
                or action.get("chosen_organ")
                or None
            )
            family = str(info.get("family", "unknown"))
            correct = bool(info.get("correct", False))

            if chosen:
                chosen = str(chosen)
                chosen_counts[chosen] += 1
                family_chosen[family][chosen] += 1

                if correct:
                    chosen_success[chosen] += 1
                    family_success[family][chosen] += 1
                else:
                    chosen_failure[chosen] += 1

        total_selected = max(1, sum(selected_counts.values()))
        total_steps = max(1, len(history))

        lifecycle: Dict[str, Dict[str, Any]] = {}

        for organ in ORGANS:
            usage = selected_counts.get(organ, 0)
            chosen = chosen_counts.get(organ, 0)
            success = chosen_success.get(organ, 0)
            failure = chosen_failure.get(organ, 0)

            usage_ratio = usage / total_selected
            step_usage_ratio = usage / total_steps
            chosen_ratio = chosen / total_steps
            success_rate = success / chosen if chosen > 0 else None

            recommendation = "insufficient_data"
            status = "active"

            if chosen == 0 and usage < max(3, int(0.03 * total_steps)):
                recommendation = "dormant_candidate"
                status = "dormant_candidate"
            elif chosen < 5:
                recommendation = "insufficient_data"
                status = "active"
            elif success_rate is not None and success_rate >= 0.90:
                recommendation = "keep_active"
                status = "active"
            elif success_rate is not None and success_rate >= 0.75:
                recommendation = "monitor"
                status = "active_monitor"
            else:
                recommendation = "review_or_refactor_candidate"
                status = "review"

            lifecycle[organ] = {
                "status": status,
                "selected_count": usage,
                "chosen_count": chosen,
                "success_count": success,
                "failure_count": failure,
                "usage_ratio_over_selected_calls": usage_ratio,
                "usage_ratio_over_steps": step_usage_ratio,
                "chosen_ratio_over_steps": chosen_ratio,
                "chosen_success_rate": success_rate,
                "recommendation": recommendation,
                "can_delete": False,
                "delete_policy": "never_auto_delete; archive_or_disable_only_after_human_approval",
            }

        family_specialization: Dict[str, Dict[str, Any]] = {}
        for family, counter in family_chosen.items():
            total = max(1, sum(counter.values()))
            top_organ, top_count = counter.most_common(1)[0]
            top_success = family_success[family].get(top_organ, 0)
            family_specialization[family] = {
                "top_chosen_organ": top_organ,
                "top_chosen_ratio": top_count / total,
                "top_success_when_chosen": top_success / top_count if top_count else None,
                "chosen_distribution": dict(counter),
            }

        result["organ_lifecycle"] = lifecycle
        result["family_specialization_v1_7_1"] = family_specialization
        result["lifecycle_policy"] = {
            "auto_delete": False,
            "auto_disable": False,
            "recommend_only": True,
            "allowed_transitions_later": ["active", "dormant", "candidate", "archived", "quarantined"],
        }
        return result
