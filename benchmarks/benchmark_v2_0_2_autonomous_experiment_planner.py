from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sage_runtime.autonomous_experiment_planner import AutonomousExperimentPlannerConfig, AutonomousExperimentPlannerRuntime


def main() -> None:
    config = AutonomousExperimentPlannerConfig(
        result_root=".",
        output_path="results/v2_0_2_autonomous_experiment_plan.json",
        inbox_dir="experiments/inbox",
        selected_proposal_path="experiments/inbox/selected_next_experiment.json",
        min_gap_for_stability=0.03,
        max_memory_inbox_before_review=12,
    )
    runtime = AutonomousExperimentPlannerRuntime(config)
    plan = runtime.run_once()
    proposals = plan.get("proposals", [])
    selected = next((p for p in proposals if p.get("id") == plan.get("selected_proposal_id")), None)
    safety = plan.get("safety_policy", {})
    proposal_files = list(Path("experiments/inbox").glob("*.json"))

    result = {
        "benchmark": "SAGE-v2.0.2-autonomous-experiment-planner",
        "version": "v2.0.2",
        "observation_count": len(plan.get("observations", [])),
        "proposal_count": len(proposals),
        "selected_proposal_title": None if selected is None else selected.get("title"),
        "selected_requires_human_approval": None if selected is None else selected.get("requires_human_approval"),
        "proposal_files_count": len(proposal_files),
        "safety_policy": safety,
        "passed": (
            len(proposals) >= 2
            and selected is not None
            and selected.get("requires_human_approval") is True
            and safety.get("network_actions") is False
            and safety.get("shell_actions") is False
            and safety.get("auto_delete_files") is False
            and safety.get("auto_disable_organs") is False
            and safety.get("auto_approve_memory") is False
            and safety.get("git_actions") is False
            and len(proposal_files) >= len(proposals)
        ),
    }

    out = Path("results/v2_0_2_autonomous_experiment_planner_smoke.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== SAGE v2.0.2 Autonomous Experiment Planner Smoke ===")
    print(f"observations: {result['observation_count']}")
    print(f"proposals: {result['proposal_count']}")
    print(f"selected: {result['selected_proposal_title']}")
    print(f"proposal_files_count: {result['proposal_files_count']}")
    print(f"passed: {result['passed']}")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
