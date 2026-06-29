from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import json

from sage_core.emergence import EmergentAggregator, EmergentContext, build_default_reflection_organs
from sage_core.memory_store import MemoryStore
from sage_core.runtime_state import SAGERuntimeState


@dataclass
class EmergentReflectionConfig:
    state_path: str = "runtime_state/state.json"
    registry_path: str = "registry/organ_registry.json"
    memory_root: str = "memory"
    reflection_log_path: str = "logs/emergent_reflection.md"
    result_path: str = "results/v2_0_emergent_reflection.json"
    create_memory_proposal: bool = True

    @classmethod
    def load(cls, path: str | Path) -> "EmergentReflectionConfig":
        p = Path(path)
        if not p.exists():
            return cls()
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls(**{**cls().__dict__, **data})


class EmergentReflectionLoop:
    def __init__(self, config: EmergentReflectionConfig) -> None:
        self.config = config
        self.memory = MemoryStore(config.memory_root)

    def load_state(self) -> Dict[str, Any]:
        path = Path(self.config.state_path)
        if not path.exists():
            return SAGERuntimeState().to_jsonable()
        return json.loads(path.read_text(encoding="utf-8"))

    def load_registry_summary(self) -> Dict[str, Any]:
        path = Path(self.config.registry_path)
        if not path.exists():
            return {
                "exists": False,
                "num_organs": 0,
                "status_counts": {},
                "recommendation_counts": {},
            }

        data = json.loads(path.read_text(encoding="utf-8"))
        organs = data.get("organs", {})
        status_counts: Dict[str, int] = {}
        recommendation_counts: Dict[str, int] = {}

        for item in organs.values():
            status = item.get("status", "unknown")
            rec = item.get("recommendation", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
            recommendation_counts[rec] = recommendation_counts.get(rec, 0) + 1

        return {
            "exists": True,
            "num_organs": len(organs),
            "status_counts": status_counts,
            "recommendation_counts": recommendation_counts,
            "source_variant": data.get("source_variant"),
            "policy": data.get("policy", {}),
        }

    def load_recent_log_text(self, max_chars: int = 3000) -> str:
        paths = [
            Path(self.config.reflection_log_path),
            Path("logs/daily_reflection.md"),
            Path("logs/v1_9_runtime_smoke_reflection.md"),
        ]
        chunks = []
        for path in paths:
            if path.exists():
                text = path.read_text(encoding="utf-8", errors="replace")
                chunks.append(text[-max_chars:])
        return "\n\n".join(chunks)[-max_chars:]

    def append_log(self, result: Dict[str, Any]) -> None:
        log_path = Path(self.config.reflection_log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        selected = result.get("selected") or {}
        metrics = result.get("emergence_metrics", {})
        block = f"""
## SAGE v2.0 Emergent Reflection

Selected organ: {selected.get("organ")}
Claim: {selected.get("claim")}
Proposal: {selected.get("proposal")}

Metrics:
- candidate_count: {metrics.get("candidate_count")}
- organ_diversity: {metrics.get("organ_diversity")}
- mean_candidate_score: {metrics.get("mean_candidate_score")}
- top_score: {metrics.get("top_score")}
- top_second_gap: {metrics.get("top_second_gap")}
- mean_novelty: {metrics.get("mean_novelty")}
- mean_reuse_value: {metrics.get("mean_reuse_value")}

Note:
{result.get("final_note")}

Safety:
No network action, shell action, organ deletion, or organ disabling was performed.
""".strip() + "\n\n"

        with log_path.open("a", encoding="utf-8") as f:
            f.write(block)

    def run_once(self) -> Dict[str, Any]:
        state = self.load_state()
        registry = self.load_registry_summary()
        recent = self.load_recent_log_text()
        inbox_count_before = self.memory.count_inbox()

        ctx = EmergentContext(
            state=state,
            registry=registry,
            memory_inbox_count=inbox_count_before,
            recent_reflection_text=recent,
        )

        organs = build_default_reflection_organs()
        candidates = [organ.propose(ctx) for organ in organs]
        aggregated = EmergentAggregator().aggregate(candidates)
        result = aggregated.to_jsonable()

        if self.config.create_memory_proposal and result.get("selected"):
            selected = result["selected"]
            proposal = self.memory.propose(
                source="emergent_reflection_v2_0",
                importance=0.62,
                content=f"{selected.get('claim')} Proposal: {selected.get('proposal')}",
                evidence={
                    "emergence_metrics": result.get("emergence_metrics", {}),
                    "selected_organ": selected.get("organ"),
                    "human_review_required": True,
                },
            )
            result["memory_proposal_created"] = proposal.to_jsonable()

        result["memory_inbox_count_before"] = inbox_count_before
        result["memory_inbox_count_after"] = self.memory.count_inbox()
        result["registry_summary"] = registry
        result["state_summary"] = {
            "version": state.get("version"),
            "mode": state.get("mode"),
            "step": state.get("step"),
            "energy": state.get("energy"),
        }

        out_path = Path(self.config.result_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

        self.append_log(result)
        return result
