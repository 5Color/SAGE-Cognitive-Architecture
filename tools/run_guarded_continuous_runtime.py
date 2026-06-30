from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description="Run SAGE v2.0.5 guarded continuous runtime.")
    parser.add_argument("--config", default="configs/guarded_continuous_runtime.json")
    parser.add_argument("--max-cycles", type=int, default=None)
    parser.add_argument("--interval", type=float, default=None)
    args = parser.parse_args()

    config = GuardedContinuousRuntimeConfig.load(args.config)
    if args.max_cycles is not None:
        config.max_cycles = args.max_cycles
    if args.interval is not None:
        config.interval_seconds = args.interval

    runtime = GuardedContinuousRuntime(config)
    report = runtime.run()

    print("=== SAGE v2.0.5 Guarded Continuous Runtime ===")
    print(f"cycles_completed: {report.get('cycles_completed')}")
    print(f"failure_count: {report.get('failure_count')}")
    print(f"stopped: {report.get('stopped')}")
    print(f"stop_reason: {report.get('stop_reason')}")
    print(f"passed: {report.get('passed')}")
    print(f"summary: {report.get('summary_path')}")
    print()
    print(json.dumps({
        "baseline": report.get("baseline"),
        "latest": report.get("latest"),
        "safety_policy": report.get("safety_policy"),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
