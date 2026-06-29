from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
from datetime import datetime
import json
import time

from sage_core.memory_store import MemoryStore
from sage_core.runtime_state import SAGERuntimeState

@dataclass
class RuntimeConfig:
    node_name: str = "sage-runtime-node"
    state_path: str = "runtime_state/state.json"
    registry_path: str = "registry/organ_registry.json"
    memory_root: str = "memory"
    log_path: str = "logs/daily_reflection.md"
    tick_seconds: float = 1.0
    max_ticks: int = 5
    mode: str = "safe_idle"

    @classmethod
    def load(cls, path: str | Path) -> "RuntimeConfig":
        p = Path(path)
        if not p.exists():
            return cls()
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls(**{**cls().__dict__, **data})

class SAGERuntimeNode:
    """SAGE v1.9 Runtime Node.

    Scope:
    - persist state
    - read organ registry
    - create memory inbox proposals
    - write reflection logs
    - run safe idle loop

    Non-goals:
    - no autonomous shell execution
    - no autonomous internet access
    - no automatic deletion/disable of organs
    """
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.state = SAGERuntimeState.load(config.state_path)
        self.state.node_name = config.node_name
        self.state.mode = config.mode
        self.memory = MemoryStore(config.memory_root)

    def read_registry_summary(self) -> Dict[str, Any]:
        path = Path(self.config.registry_path)
        if not path.exists():
            return {"exists": False, "num_organs": 0, "status_counts": {}, "recommendation_counts": {}}

        data = json.loads(path.read_text(encoding="utf-8"))
        organs = data.get("organs", {})
        status_counts: Dict[str, int] = {}
        recommendation_counts: Dict[str, int] = {}

        for item in organs.values():
            status = item.get("status", "unknown")
            rec = item.get("recommendation", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
            recommendation_counts[rec] = recommendation_counts.get(rec, 0) + 1

        self.state.counters["registry_reads"] = self.state.counters.get("registry_reads", 0) + 1

        return {
            "exists": True,
            "num_organs": len(organs),
            "status_counts": status_counts,
            "recommendation_counts": recommendation_counts,
            "source_variant": data.get("source_variant"),
            "policy": data.get("policy", {}),
        }

    def write_reflection(self, registry_summary: Dict[str, Any]) -> None:
        log_path = Path(self.config.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        now = datetime.now().isoformat(timespec="seconds")
        block = f"""
## Runtime Tick {self.state.step}

- time: {now}
- mode: {self.state.mode}
- focus: {self.state.focus}
- energy: {self.state.energy:.3f}
- registry_exists: {registry_summary.get("exists")}
- num_organs: {registry_summary.get("num_organs")}
- status_counts: {registry_summary.get("status_counts")}
- recommendation_counts: {registry_summary.get("recommendation_counts")}

Reflection:
SAGE is in safe idle mode. It observed the registry and recorded state.
No autonomous organ deletion, disabling, shell execution, or network action was performed.
""".strip() + "\n\n"
        with log_path.open("a", encoding="utf-8") as f:
            f.write(block)
        self.state.counters["reflections"] = self.state.counters.get("reflections", 0) + 1

    def maybe_propose_memory(self, registry_summary: Dict[str, Any]) -> None:
        if self.state.step == 1:
            proposal = self.memory.propose(
                source="runtime_v1_9",
                importance=0.55,
                content=(
                    "SAGE v1.9 runtime node started in safe idle mode. "
                    "It can persist state, read organ registry, propose memory, and write reflection logs."
                ),
                evidence={
                    "registry_summary": registry_summary,
                    "policy": "human review required before long-term memory approval",
                },
            )
            self.state.counters["memory_proposals"] = self.state.counters.get("memory_proposals", 0) + 1
            self.state.notes.append(f"memory_proposal_created:{proposal.id}")

    def tick_once(self) -> Dict[str, Any]:
        self.state.tick(focus="runtime_observation")
        registry_summary = self.read_registry_summary()
        self.maybe_propose_memory(registry_summary)
        self.write_reflection(registry_summary)
        self.state.save(self.config.state_path)
        return {
            "step": self.state.step,
            "mode": self.state.mode,
            "registry_summary": registry_summary,
            "memory_inbox_count": self.memory.count_inbox(),
            "state_path": self.config.state_path,
            "log_path": self.config.log_path,
        }

    def run(self) -> list[Dict[str, Any]]:
        outputs = []
        for _ in range(max(0, int(self.config.max_ticks))):
            outputs.append(self.tick_once())
            if self.config.tick_seconds > 0:
                time.sleep(float(self.config.tick_seconds))
        return outputs
