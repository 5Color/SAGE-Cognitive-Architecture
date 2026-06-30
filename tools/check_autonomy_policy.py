from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sage_runtime.autonomy_policy_runtime import (
    AutonomyPolicyRuntime,
    AutonomyPolicyRuntimeConfig,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="SAGE v2.3 autonomy level policy checker")
    parser.add_argument("command", choices=["report", "decide"])
    parser.add_argument("--config", default="configs/autonomy_policy_runtime.json")
    parser.add_argument("--action", default=None)
    args = parser.parse_args()

    runtime = AutonomyPolicyRuntime(AutonomyPolicyRuntimeConfig.load(args.config))

    if args.command == "report":
        report = runtime.run_once()
        print("=== SAGE v2.3 Autonomy Level Policy Report ===")
        print(f"active_level: {report.get('active_level')} ({report.get('level_name')})")
        print(f"allowed_actions: {len(report.get('allowed_actions', []))}")
        print(f"approval_required_actions: {len(report.get('approval_required_actions', []))}")
        print(f"forbidden_actions: {len(report.get('forbidden_actions', []))}")
        print(f"passed: {report.get('passed')}")
        print(f"output: {report.get('output_path')}")
        print()
        print(json.dumps({
            "level_description": report.get("level_description"),
            "allowed_actions": report.get("allowed_actions"),
            "approval_required_actions": report.get("approval_required_actions"),
            "forbidden_actions": report.get("forbidden_actions"),
            "safety_policy": report.get("safety_policy"),
        }, indent=2, ensure_ascii=False))

    elif args.command == "decide":
        if not args.action:
            raise SystemExit("--action is required for decide")
        decision = runtime.decide(args.action)
        print("=== SAGE v2.3 Autonomy Decision ===")
        print(json.dumps(decision, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
