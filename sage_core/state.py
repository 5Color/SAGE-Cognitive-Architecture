from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class OrganResult:
    """Output produced by one organ.

    organ_name: organ identifier
    action: organ's proposed action/prediction
    confidence: how strongly the organ supports the action
    evidence: optional diagnostic data for metrics/debugging
    """

    organ_name: str
    action: Any
    confidence: float = 1.0
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SAGEState:
    """Persistent state of the SAGE system.

    step: current cycle number
    energy: global system energy
    reward: last reward
    organ_energy: per-organ energy/homeostasis values
    memory: compact event traces
    router_feedback: diagnostics from the router
    context: optional environment/context vector or metadata
    """

    step: int = 0
    energy: float = 1.0
    reward: float = 0.0
    organ_energy: Dict[str, float] = field(default_factory=dict)
    memory: List[Dict[str, Any]] = field(default_factory=list)
    router_feedback: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)

    def is_alive(self, min_energy: float = 0.05) -> bool:
        return self.energy > min_energy

    def remember(self, event: Dict[str, Any], max_memory: int = 128) -> None:
        self.memory.append(event)
        if len(self.memory) > max_memory:
            self.memory = self.memory[-max_memory:]

    def copy_public(self) -> Dict[str, Any]:
        """JSON-friendly snapshot for metrics and result files."""
        return {
            "step": self.step,
            "energy": self.energy,
            "reward": self.reward,
            "organ_energy": dict(self.organ_energy),
            "memory_size": len(self.memory),
            "router_feedback": dict(self.router_feedback),
            "context": dict(self.context),
        }
