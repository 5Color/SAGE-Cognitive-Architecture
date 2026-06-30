from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import json
from sage_core.reflection_stability import ReflectionStabilityProbe

@dataclass
class ReflectionStabilityConfig:
    base_policy_path: str = "configs/reflection_policy_exploratory.json"
    output_path: str = "results/v2_0_3_reflection_stability_probe.json"
    output_dir: str = "results/v2_0_3_stability_probe"
    variant_policy_dir: str = "configs/generated/stability_probe"
    target_organ: str = "curiosity_organ"
    novelty_deltas: List[float] = field(default_factory=lambda: [-0.02, 0.0, 0.02])
    risk_deltas: List[float] = field(default_factory=lambda: [0.0, 0.02])
    min_target_rate: float = 0.80
    @classmethod
    def load(cls, path: str | Path) -> "ReflectionStabilityConfig":
        p = Path(path)
        if not p.exists():
            return cls()
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls(**{**cls().__dict__, **data})

class ReflectionStabilityRuntime:
    def __init__(self, config: ReflectionStabilityConfig) -> None:
        self.config = config
    def run_once(self) -> dict:
        probe = ReflectionStabilityProbe(
            self.config.base_policy_path,
            self.config.output_dir,
            self.config.variant_policy_dir,
            self.config.target_organ,
            self.config.novelty_deltas,
            self.config.risk_deltas,
            self.config.min_target_rate,
        )
        result = probe.run().to_jsonable()
        out = Path(self.config.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        result["output_path"] = str(out)
        return result
