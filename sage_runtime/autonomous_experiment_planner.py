from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import json

from sage_core.experiment_planner import AutonomousExperimentPlanner, ResultReader


@dataclass
class AutonomousExperimentPlannerConfig:
    result_root: str = "."
    output_path: str = "results/v2_0_2_autonomous_experiment_plan.json"
    inbox_dir: str = "experiments/inbox"
    selected_proposal_path: str = "experiments/inbox/selected_next_experiment.json"
    min_gap_for_stability: float = 0.03
    max_memory_inbox_before_review: int = 12

    @classmethod
    def load(cls, path: str | Path) -> "AutonomousExperimentPlannerConfig":
        p = Path(path)
        if not p.exists():
            return cls()
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls(**{**cls().__dict__, **data})


class AutonomousExperimentPlannerRuntime:
    def __init__(self, config: AutonomousExperimentPlannerConfig) -> None:
        self.config = config

    def run_once(self) -> Dict[str, Any]:
        reader = ResultReader(self.config.result_root)
        observations = reader.collect_default_observations()
        planner = AutonomousExperimentPlanner(
            min_gap_for_stability=self.config.min_gap_for_stability,
            max_memory_inbox_before_review=self.config.max_memory_inbox_before_review,
        )
        plan = planner.propose(observations)
        data = plan.to_jsonable()

        output_path = Path(self.config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

        inbox_dir = Path(self.config.inbox_dir)
        inbox_dir.mkdir(parents=True, exist_ok=True)

        for proposal in data.get("proposals", []):
            safe_title = proposal["title"].lower().replace(" ", "_").replace("/", "_")
            p = inbox_dir / f"{proposal['id']}_{safe_title}.json"
            p.write_text(json.dumps(proposal, indent=2, ensure_ascii=False), encoding="utf-8")

        selected = None
        for proposal in data.get("proposals", []):
            if proposal.get("id") == data.get("selected_proposal_id"):
                selected = proposal
                break

        if selected:
            selected_path = Path(self.config.selected_proposal_path)
            selected_path.parent.mkdir(parents=True, exist_ok=True)
            selected_path.write_text(json.dumps(selected, indent=2, ensure_ascii=False), encoding="utf-8")

        data["output_path"] = str(output_path)
        data["inbox_dir"] = str(inbox_dir)
        data["selected_proposal_path"] = self.config.selected_proposal_path if selected else None
        return data
