from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sage_runtime.guarded_continuous_runtime import (
    GuardedContinuousRuntime,
    GuardedContinuousRuntimeConfig,
)


def main() -> None:
    config = GuardedContinuousRuntimeConfig(
        max_cycles=3,
        interval_seconds=0.0,
        max_failures=1,
        max_memory_inbox_growth=4,
        max_result_files_growth=120,
        max_experiment_inbox_growth=60,
        stop_file_path="runtime_control/STOP",
        output_dir="results/v2_0_5_guarded_runtime_smoke",
        summary_path="results/v2_0_5_guarded_runtime_smoke/summary.json",
        run_reflection=True,
        run_experiment_planner=True,
        run_stability_probe=True,
        create_memory_proposal_each_cycle=False,
    )

    runtime = GuardedContinuousRuntime(config)
    report = runtime.run()

    cycles = report.get("cycles", [])
    action_names = []
    for cycle in cycles:
        action_names.extend(cycle.get("actions", []))

    safety = report.get("safety_policy", {})
    latest = report.get("latest", {})
    baseline = report.get("baseline", {})

    memory_growth = latest.get("memory_inbox_count", 0) - baseline.get("memory_inbox_count", 0)

    result = {
        "benchmark": "SAGE-v2.0.5-runtime-guard-long-run-monitor-smoke",
        "version": "v2.0.5",
        "cycles_completed": report.get("cycles_completed"),
        "failure_count": report.get("failure_count"),
        "stop_reason": report.get("stop_reason"),
        "memory_inbox_growth": memory_growth,
        "action_names": action_names,
        "summary_path": report.get("summary_path"),
        "safety_policy": safety,
        "passed": (
            report.get("passed") is True
            and report.get("cycles_completed") >= 1
            and report.get("failure_count") == 0
            and memory_growth <= config.max_memory_inbox_growth
            and "run_reflection_loop" in action_names
            and "run_autonomous_experiment_planner" in action_names
            and "run_reflection_stability_probe" in action_names
            and safety.get("network_actions") is False
            and safety.get("arbitrary_shell_actions") is False
            and safety.get("file_delete") is False
            and safety.get("core_code_modification") is False
            and safety.get("auto_approve_memory") is False
            and safety.get("git_actions") is False
        ),
    }

    out = Path("results/v2_0_5_runtime_guard_long_run_monitor_smoke.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== SAGE v2.0.5 Runtime Guard Long-Run Monitor Smoke ===")
    print(f"cycles_completed: {result['cycles_completed']}")
    print(f"failure_count: {result['failure_count']}")
    print(f"stop_reason: {result['stop_reason']}")
    print(f"memory_inbox_growth: {result['memory_inbox_growth']}")
    print(f"passed: {result['passed']}")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
