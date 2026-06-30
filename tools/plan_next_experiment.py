from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sage_runtime.autonomous_experiment_planner import AutonomousExperimentPlannerConfig, AutonomousExperimentPlannerRuntime


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SAGE v2.0.2 Autonomous Experiment Planner.")
    parser.add_argument("--config", default="configs/autonomous_experiment_planner.json")
    args = parser.parse_args()

    config = AutonomousExperimentPlannerConfig.load(args.config)
    runtime = AutonomousExperimentPlannerRuntime(config)
    result = runtime.run_once()

    proposals = result.get("proposals", [])
    selected_id = result.get("selected_proposal_id")
    selected = next((p for p in proposals if p.get("id") == selected_id), None)

    print("=== SAGE v2.0.2 Autonomous Experiment Planner ===")
    print(f"observations: {len(result.get('observations', []))}")
    print(f"proposals: {len(proposals)}")
    print(f"output: {result.get('output_path')}")
    print(f"inbox: {result.get('inbox_dir')}")
    if selected:
        print(f"selected: {selected.get('title')}")
        print(f"risk: {selected.get('risk_level')}")
        print(f"approval_required: {selected.get('requires_human_approval')}")
    print()
    print(json.dumps({
        "selected_proposal_id": selected_id,
        "safety_policy": result.get("safety_policy"),
        "final_note": result.get("final_note"),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
