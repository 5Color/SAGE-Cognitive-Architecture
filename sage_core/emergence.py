from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List
import json
import math


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ReflectionCandidate:
    organ: str
    claim: str
    proposal: str
    confidence: float = 0.5
    novelty: float = 0.5
    reuse_value: float = 0.5
    risk: float = 0.1
    cost: float = 0.1
    tags: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    contradiction_flags: List[str] = field(default_factory=list)

    def to_jsonable(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AggregatedReflection:
    version: str = "v2.0"
    created_at: str = field(default_factory=utc_now)
    selected: ReflectionCandidate | None = None
    ranked_candidates: List[Dict[str, Any]] = field(default_factory=list)
    emergence_metrics: Dict[str, float] = field(default_factory=dict)
    final_note: str = ""
    safety_policy: Dict[str, bool] = field(default_factory=lambda: {
        "network_actions": False,
        "shell_actions": False,
        "auto_delete_organs": False,
        "auto_disable_organs": False,
        "memory_approval_required": True,
    })

    def to_jsonable(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "selected": None if self.selected is None else self.selected.to_jsonable(),
            "ranked_candidates": self.ranked_candidates,
            "emergence_metrics": self.emergence_metrics,
            "final_note": self.final_note,
            "safety_policy": self.safety_policy,
        }


class EmergentContext:
    def __init__(
        self,
        state: Dict[str, Any],
        registry: Dict[str, Any],
        memory_inbox_count: int,
        recent_reflection_text: str = "",
    ) -> None:
        self.state = state
        self.registry = registry
        self.memory_inbox_count = memory_inbox_count
        self.recent_reflection_text = recent_reflection_text

    @property
    def registry_exists(self) -> bool:
        return bool(self.registry.get("exists", False))

    @property
    def num_organs(self) -> int:
        return int(self.registry.get("num_organs", 0))

    @property
    def state_step(self) -> int:
        return int(self.state.get("step", 0))

    @property
    def mode(self) -> str:
        return str(self.state.get("mode", "unknown"))


class BaseReflectionOrgan:
    name = "base_reflection_organ"

    def propose(self, ctx: EmergentContext) -> ReflectionCandidate:
        raise NotImplementedError


class ObserverOrgan(BaseReflectionOrgan):
    name = "observer_organ"

    def propose(self, ctx: EmergentContext) -> ReflectionCandidate:
        status_counts = ctx.registry.get("status_counts", {})
        claim = (
            f"Runtime is in {ctx.mode} mode at step {ctx.state_step}; "
            f"registry has {ctx.num_organs} organs with statuses {status_counts}."
        )
        return ReflectionCandidate(
            organ=self.name,
            claim=claim,
            proposal="Keep observing runtime state and registry before changing architecture.",
            confidence=0.82 if ctx.registry_exists else 0.55,
            novelty=0.25,
            reuse_value=0.65,
            risk=0.05,
            cost=0.05,
            tags=["observation", "registry", "state"],
            evidence={
                "mode": ctx.mode,
                "step": ctx.state_step,
                "registry_exists": ctx.registry_exists,
                "num_organs": ctx.num_organs,
                "status_counts": status_counts,
            },
        )


class PlannerOrgan(BaseReflectionOrgan):
    name = "planner_organ"

    def propose(self, ctx: EmergentContext) -> ReflectionCandidate:
        if ctx.memory_inbox_count > 0:
            proposal = "Review memory inbox and convert useful proposals into approved memory before adding larger features."
            reuse_value = 0.82
        else:
            proposal = "Create at least one memory proposal from runtime observations."
            reuse_value = 0.60

        return ReflectionCandidate(
            organ=self.name,
            claim="The next useful action should improve the feedback loop rather than add heavy neural components.",
            proposal=proposal,
            confidence=0.74,
            novelty=0.45,
            reuse_value=reuse_value,
            risk=0.10,
            cost=0.18,
            tags=["planning", "next_step", "memory"],
            evidence={
                "memory_inbox_count": ctx.memory_inbox_count,
                "reason": "SAGE needs accumulated evidence before organ genesis or neural scaling.",
            },
        )


class CriticOrgan(BaseReflectionOrgan):
    name = "critic_organ"

    def propose(self, ctx: EmergentContext) -> ReflectionCandidate:
        flags = []
        if ctx.num_organs < 4:
            flags.append("registry_has_too_few_organs")
        if ctx.memory_inbox_count > 20:
            flags.append("memory_inbox_growth_without_approval")

        return ReflectionCandidate(
            organ=self.name,
            claim="SAGE should not confuse logs or random variation with true emergence.",
            proposal="Track whether a reflection improves later decisions before calling it emergent.",
            confidence=0.78,
            novelty=0.55,
            reuse_value=0.88,
            risk=0.08,
            cost=0.12,
            tags=["critic", "guardrail", "emergence"],
            evidence={
                "contradiction_checks": flags,
                "definition": "Emergence requires reusable patterns that improve later behavior.",
            },
            contradiction_flags=flags,
        )


class MemoryReflectionOrgan(BaseReflectionOrgan):
    name = "memory_reflection_organ"

    def propose(self, ctx: EmergentContext) -> ReflectionCandidate:
        if ctx.memory_inbox_count >= 1:
            claim = f"There are {ctx.memory_inbox_count} memory proposals waiting for review."
            proposal = "Summarize repeated runtime observations and mark only reusable insights for approval."
            novelty = 0.42
        else:
            claim = "No memory proposals are available yet."
            proposal = "Create memory proposals only from observations that can affect future routing."
            novelty = 0.35

        return ReflectionCandidate(
            organ=self.name,
            claim=claim,
            proposal=proposal,
            confidence=0.76,
            novelty=novelty,
            reuse_value=0.80,
            risk=0.06,
            cost=0.09,
            tags=["memory", "consolidation", "reuse"],
            evidence={
                "memory_inbox_count": ctx.memory_inbox_count,
            },
        )


class CuriosityOrgan(BaseReflectionOrgan):
    name = "curiosity_organ"

    def propose(self, ctx: EmergentContext) -> ReflectionCandidate:
        return ReflectionCandidate(
            organ=self.name,
            claim="A small internal disagreement loop may reveal reusable patterns not directly coded into one organ.",
            proposal="Run multiple reflection organs, allow disagreement, then store only high-value selected reflections.",
            confidence=0.66,
            novelty=0.90,
            reuse_value=0.70,
            risk=0.22,
            cost=0.22,
            tags=["curiosity", "emergence", "multi_organ"],
            evidence={
                "hypothesis": "Diverse organs plus selection pressure can produce emergent-like behavior.",
                "warning": "Random novelty alone is not emergence.",
            },
        )


class EmergentAggregator:
    """Selects a reflection from multiple organ candidates.

    This is not a neural model.
    It is a transparent scoring function for v2.0.
    """

    def score(self, candidate: ReflectionCandidate, agreement_bonus: float) -> float:
        contradiction_penalty = 0.12 * len(candidate.contradiction_flags)
        raw = (
            0.25 * candidate.confidence
            + 0.29 * candidate.novelty
            + 0.20 * candidate.reuse_value
            + 0.14 * agreement_bonus
            - 0.08 * candidate.cost
            - 0.08 * candidate.risk
            - contradiction_penalty
        )
        return round(max(0.0, min(1.0, raw)), 6)

    def agreement_bonus(self, candidate: ReflectionCandidate, all_candidates: Iterable[ReflectionCandidate]) -> float:
        tags = set(candidate.tags)
        if not tags:
            return 0.0
        others = [c for c in all_candidates if c is not candidate]
        if not others:
            return 0.0
        overlaps = []
        for other in others:
            other_tags = set(other.tags)
            if not other_tags:
                overlaps.append(0.0)
            else:
                overlaps.append(len(tags & other_tags) / max(1, len(tags | other_tags)))
        return sum(overlaps) / len(overlaps)

    def diversity(self, candidates: List[ReflectionCandidate]) -> float:
        if not candidates:
            return 0.0
        organs = {c.organ for c in candidates}
        tags = set()
        for c in candidates:
            tags.update(c.tags)
        return round(min(1.0, 0.5 * len(organs) / 5.0 + 0.5 * len(tags) / 12.0), 6)

    def aggregate(self, candidates: List[ReflectionCandidate]) -> AggregatedReflection:
        ranked = []
        for c in candidates:
            agreement = self.agreement_bonus(c, candidates)
            score = self.score(c, agreement)
            ranked.append({
                "score": score,
                "agreement_bonus": round(agreement, 6),
                "candidate": c.to_jsonable(),
            })

        ranked.sort(key=lambda item: item["score"], reverse=True)
        selected = None if not ranked else ReflectionCandidate(**ranked[0]["candidate"])

        mean_score = sum(item["score"] for item in ranked) / max(1, len(ranked))
        top_score = 0.0 if not ranked else ranked[0]["score"]
        second_score = 0.0 if len(ranked) < 2 else ranked[1]["score"]

        metrics = {
            "candidate_count": float(len(candidates)),
            "organ_diversity": self.diversity(candidates),
            "mean_candidate_score": round(mean_score, 6),
            "top_score": round(top_score, 6),
            "top_second_gap": round(top_score - second_score, 6),
            "mean_novelty": round(sum(c.novelty for c in candidates) / max(1, len(candidates)), 6),
            "mean_reuse_value": round(sum(c.reuse_value for c in candidates) / max(1, len(candidates)), 6),
        }

        if selected is None:
            note = "No reflection candidate was selected."
        else:
            note = (
                f"Selected {selected.organ}: {selected.proposal} "
                f"This is an emergent reflection candidate, not proof of AGI."
            )

        return AggregatedReflection(
            selected=selected,
            ranked_candidates=ranked,
            emergence_metrics=metrics,
            final_note=note,
        )


def build_default_reflection_organs() -> List[BaseReflectionOrgan]:
    return [
        ObserverOrgan(),
        PlannerOrgan(),
        CriticOrgan(),
        MemoryReflectionOrgan(),
        CuriosityOrgan(),
    ]
