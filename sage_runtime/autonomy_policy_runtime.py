from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import json

from sage_core.autonomy_policy import AutonomyLevelPolicy


@dataclass
class AutonomyPolicyRuntimeConfig:
    policy_path: str = "configs/autonomy_level_policy.json"
    output_path: str = "results/v2_3_autonomy_level_policy_report.json"

    @classmethod
    def load(cls, path: str | Path) -> "AutonomyPolicyRuntimeConfig":
        p = Path(path)
        if not p.exists():
            return cls()
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls(**{**cls().__dict__, **data})


class AutonomyPolicyRuntime:
    def __init__(self, config: AutonomyPolicyRuntimeConfig) -> None:
        self.config = config

    def run_once(self, actions: Optional[List[str]] = None) -> Dict[str, Any]:
        policy = AutonomyLevelPolicy.from_config(self.config.policy_path)
        report = policy.report(actions_to_evaluate=actions).to_jsonable()

        out = Path(self.config.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        report["output_path"] = str(out)
        return report

    def decide(self, action: str) -> Dict[str, Any]:
        policy = AutonomyLevelPolicy.from_config(self.config.policy_path)
        decision = policy.decide(action).to_jsonable()

        out = Path(self.config.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": "v2.3",
            "mode": "single_action_decision",
            "decision": decision,
        }
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        decision["output_path"] = str(out)
        return decision
