from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sage_runtime.cleanup_retention_advisor import (
    CleanupRetentionAdvisorConfig,
    CleanupRetentionAdvisorRuntime,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SAGE v2.0.6 cleanup retention advisor.")
    parser.add_argument("--config", default="configs/cleanup_retention_policy.json")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = CleanupRetentionAdvisorConfig.load(args.config)
    if args.output:
        config.output_path = args.output

    runtime = CleanupRetentionAdvisorRuntime(config)
    report = runtime.run_once()

    print("=== SAGE v2.0.6 Cleanup & Retention Policy Advisor ===")
    print(f"proposals: {len(report.get('proposals', []))}")
    print(f"passed: {report.get('passed')}")
    print(f"output: {report.get('output_path')}")
    print(f"inbox: {report.get('inbox_path')}")
    print()
    print(json.dumps({
        "selected_summary": report.get("selected_summary"),
        "safety_policy": report.get("safety_policy"),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
