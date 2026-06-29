# SAGE v1.6.1 task plugin
#
# This file contains only task-specific components.
# The repeated control loop is handled by run_experiment.py + sage_core.engine.

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple

from sage_core import BaseEnvironment, BaseOrgan, BaseRouter, OrganResult, SAGEState
from sage_core.metrics import BasicRunMetric


class ParityOrgan(BaseOrgan):
    name = "parity_organ"

    def process(self, state: SAGEState, signal: Dict[str, Any]) -> OrganResult:
        x = int(signal["x"])
        return OrganResult(
            organ_name=self.name,
            action=x % 2,
            confidence=0.90,
            evidence={"rule": "x % 2"},
        )


class BiasOrgan(BaseOrgan):
    name = "bias_organ"

    def __init__(self, bias: int = 0) -> None:
        self.bias = int(bias)

    def process(self, state: SAGEState, signal: Dict[str, Any]) -> OrganResult:
        return OrganResult(
            organ_name=self.name,
            action=self.bias,
            confidence=0.40,
            evidence={"rule": f"always_{self.bias}"},
        )


class EnergyAwareRouter(BaseRouter):
    name = "energy_aware_router"

    def __init__(self, min_energy_for_specialist: float = 0.20) -> None:
        self.min_energy_for_specialist = float(min_energy_for_specialist)

    def route(
        self,
        state: SAGEState,
        signal: Dict[str, Any],
        organs: Mapping[str, BaseOrgan],
    ) -> List[str]:
        if state.energy > self.min_energy_for_specialist:
            return ["parity_organ", "bias_organ"]
        return ["bias_organ"]

    def aggregate(
        self,
        state: SAGEState,
        signal: Dict[str, Any],
        outputs: Mapping[str, OrganResult],
    ) -> Dict[str, Any]:
        if not outputs:
            return {"prediction": 0, "chosen_organ": None}

        best = max(outputs.values(), key=lambda result: result.confidence)
        return {
            "prediction": int(best.action),
            "chosen_organ": best.organ_name,
            "confidence": float(best.confidence),
        }


class ParityEnvironment(BaseEnvironment):
    name = "parity_environment"

    def __init__(self, length: int = 32, start: int = 0) -> None:
        self.length = int(length)
        self.start = int(start)
        self.index = int(start)
        self.count = 0

    def reset(self) -> Dict[str, Any]:
        self.index = self.start
        self.count = 0
        return {"x": self.index}

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        target = self.index % 2
        pred = int(action.get("prediction", 0))
        correct = pred == target
        reward = 1.0 if correct else -1.0

        self.index += 1
        self.count += 1
        done = self.count >= self.length

        return {"x": self.index}, reward, done, {
            "target": target,
            "prediction": pred,
            "correct": correct,
            "chosen_organ": action.get("chosen_organ"),
            "confidence": action.get("confidence", 0.0),
        }


# Alias class so the config can load the metric from this task module if desired.
class ParityMetric(BasicRunMetric):
    name = "parity_metric"
