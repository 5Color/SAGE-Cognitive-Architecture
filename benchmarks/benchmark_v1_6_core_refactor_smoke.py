# SAGE v1.6 - Core Refactor Smoke Benchmark
#
# Goal:
# - Test that sage_core can run a full state/router/organ/environment/metric loop.
# - This is not a performance benchmark.
# - This checks that future benchmarks can reuse a shared engine instead of copy-pasting everything.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

from sage_core import BaseEnvironment, BaseOrgan, BaseRouter, OrganResult, SAGEEngine, SAGEState
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

    def process(self, state: SAGEState, signal: Dict[str, Any]) -> OrganResult:
        return OrganResult(
            organ_name=self.name,
            action=0,
            confidence=0.40,
            evidence={"rule": "always_zero"},
        )


class EnergyAwareRouter(BaseRouter):
    name = "energy_aware_router"

    def route(
        self,
        state: SAGEState,
        signal: Dict[str, Any],
        organs: Mapping[str, BaseOrgan],
    ) -> List[str]:
        # Simple smoke logic:
        # use the specialized organ while energy is healthy,
        # include the cheap bias organ as fallback/diagnostic.
        if state.energy > 0.20:
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
        return {"prediction": int(best.action), "chosen_organ": best.organ_name}


class ParityEnvironment(BaseEnvironment):
    name = "parity_environment"

    def __init__(self, length: int = 32) -> None:
        self.length = length
        self.index = 0

    def reset(self) -> Dict[str, Any]:
        self.index = 0
        return {"x": self.index}

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        target = self.index % 2
        pred = int(action.get("prediction", 0))
        correct = pred == target
        reward = 1.0 if correct else -1.0

        self.index += 1
        done = self.index >= self.length
        next_signal = {"x": self.index}

        return next_signal, reward, done, {
            "target": target,
            "prediction": pred,
            "correct": correct,
            "chosen_organ": action.get("chosen_organ"),
        }


def main() -> None:
    organs = {
        "parity_organ": ParityOrgan(),
        "bias_organ": BiasOrgan(),
    }
    router = EnergyAwareRouter()
    env = ParityEnvironment(length=32)
    metric = BasicRunMetric()

    engine = SAGEEngine(
        organs=organs,
        router=router,
        env=env,
        metric=metric,
        state=SAGEState(energy=1.0),
    )

    result = engine.run(max_steps=64)
    output = {
        "benchmark": "SAGE-v1.6-core-refactor-smoke",
        "goal": "Verify shared sage_core engine/interface structure.",
        "interpretation_guardrail": "This is a refactor smoke test, not an intelligence or AGI benchmark.",
        "result": result,
    }

    out_path = Path("results/v1_6_core_refactor_smoke.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== SAGE v1.6 Core Refactor Smoke ===")
    print(f"accuracy={result['accuracy']:.4f}")
    print(f"avg_reward={result['avg_reward']:.4f}")
    print(f"steps={result['steps']}")
    print(f"final_energy={result['final_state']['energy']:.4f}")
    print(f"organ_usage={result['organ_usage']}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
