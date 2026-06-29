from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Mapping


@dataclass
class OrganLifecycleRecord:
    """Recommendation-only lifecycle state for one organ.

    v1.8 policy:
    - no automatic deletion
    - no automatic disabling
    - archive/dormant transitions require human approval
    """

    name: str
    status: str = "active"
    recommendation: str = "insufficient_data"
    health_score: float = 0.0
    usage_ratio_over_steps: float = 0.0
    chosen_ratio_over_steps: float = 0.0
    chosen_success_rate: float | None = None
    evidence: Dict[str, Any] = field(default_factory=dict)
    allowed_actions: list[str] = field(default_factory=lambda: ["observe", "recommend"])
    can_auto_delete: bool = False
    can_auto_disable: bool = False
    requires_human_approval: bool = True
    notes: list[str] = field(default_factory=list)


@dataclass
class OrganRegistry:
    version: str = "v1.8"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source_result: str | None = None
    source_variant: str | None = None
    policy: Dict[str, Any] = field(default_factory=lambda: {
        "auto_delete": False,
        "auto_disable": False,
        "recommend_only": True,
        "human_approval_required": True,
        "default_delete_policy": "never_auto_delete; archive_or_disable_only_after_human_approval",
    })
    organs: Dict[str, OrganLifecycleRecord] = field(default_factory=dict)

    def to_jsonable(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "source_result": self.source_result,
            "source_variant": self.source_variant,
            "policy": self.policy,
            "organs": {name: asdict(record) for name, record in sorted(self.organs.items())},
        }


class OrganLifecycleManager:
    """Build a registry from v1.7.1 lifecycle summary.

    This manager only creates recommendations.
    It does not mutate the active organ set.
    """

    def __init__(self, core_organs: Iterable[str] | None = None) -> None:
        self.core_organs = set(core_organs or ["memory_organ"])

    def health_score(
        self,
        usage_ratio: float,
        chosen_ratio: float,
        success_rate: float | None,
        recommendation_counts: Mapping[str, int],
    ) -> float:
        success = 0.50 if success_rate is None else max(0.0, min(1.0, float(success_rate)))
        usage = max(0.0, min(1.0, float(usage_ratio)))
        chosen = max(0.0, min(1.0, float(chosen_ratio)))

        rec_bonus = 0.0
        total_recs = max(1, sum(int(v) for v in recommendation_counts.values()))
        rec_bonus += 0.15 * (int(recommendation_counts.get("keep_active", 0)) / total_recs)
        rec_bonus -= 0.10 * (int(recommendation_counts.get("review_or_refactor_candidate", 0)) / total_recs)
        rec_bonus -= 0.10 * (int(recommendation_counts.get("dormant_candidate", 0)) / total_recs)

        score = 0.65 * success + 0.20 * usage + 0.10 * chosen + rec_bonus
        return round(max(0.0, min(1.0, score)), 6)

    def status_from_evidence(
        self,
        organ_name: str,
        health_score: float,
        success_rate: float | None,
        recommendation_counts: Mapping[str, int],
    ) -> tuple[str, str, list[str]]:
        notes: list[str] = []

        if organ_name in self.core_organs:
            notes.append("core organ: never auto-disable")
            if success_rate is not None and success_rate >= 0.90:
                return "core_active", "keep_active", notes
            return "core_monitor", "monitor", notes

        keep = int(recommendation_counts.get("keep_active", 0))
        monitor = int(recommendation_counts.get("monitor", 0))
        review = int(recommendation_counts.get("review_or_refactor_candidate", 0))
        dormant = int(recommendation_counts.get("dormant_candidate", 0))

        if dormant > 0 and keep == 0 and monitor == 0:
            notes.append("low usage evidence; do not delete, only mark dormant candidate")
            return "dormant_candidate", "consider_dormant_after_human_review", notes

        if review > 0 and keep == 0:
            notes.append("review signal detected; consider refactor, not deletion")
            return "review", "review_or_refactor_candidate", notes

        if health_score >= 0.88 and (success_rate is None or success_rate >= 0.88):
            return "active", "keep_active", notes

        if health_score >= 0.72:
            return "active_monitor", "monitor", notes

        notes.append("weak lifecycle evidence; collect more data before action")
        return "active_monitor", "monitor", notes

    def build_registry(
        self,
        result: Mapping[str, Any],
        variant: str,
        source_result: str | None = None,
    ) -> OrganRegistry:
        summary = result.get("summary", {})
        if variant not in summary:
            available = ", ".join(sorted(summary.keys()))
            raise KeyError(f"Variant not found: {variant}. Available: {available}")

        variant_summary = summary[variant]
        lifecycle_summary = variant_summary.get("organ_lifecycle_summary", {})
        if not lifecycle_summary:
            raise ValueError(f"No organ_lifecycle_summary found for variant: {variant}")

        registry = OrganRegistry(
            source_result=source_result,
            source_variant=variant,
        )

        for organ_name, item in sorted(lifecycle_summary.items()):
            usage = float(item.get("usage_ratio_over_steps_mean", 0.0))
            chosen = float(item.get("chosen_ratio_over_steps_mean", 0.0))
            success = item.get("chosen_success_rate_mean", None)
            success_rate = None if success is None else float(success)
            rec_counts = item.get("recommendation_counts", {})

            health = self.health_score(usage, chosen, success_rate, rec_counts)
            status, recommendation, notes = self.status_from_evidence(
                organ_name=organ_name,
                health_score=health,
                success_rate=success_rate,
                recommendation_counts=rec_counts,
            )

            record = OrganLifecycleRecord(
                name=organ_name,
                status=status,
                recommendation=recommendation,
                health_score=health,
                usage_ratio_over_steps=usage,
                chosen_ratio_over_steps=chosen,
                chosen_success_rate=success_rate,
                evidence={
                    "recommendation_counts": dict(rec_counts),
                    "usage_ratio_over_steps_std": item.get("usage_ratio_over_steps_std", 0.0),
                    "chosen_success_rate_std": item.get("chosen_success_rate_std", 0.0),
                },
                notes=notes,
            )

            registry.organs[organ_name] = record

        return registry
