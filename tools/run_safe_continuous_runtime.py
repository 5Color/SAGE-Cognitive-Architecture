from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sage_runtime.safe_continuous_runtime import SafeContinuousRuntimeConfig, SafeContinuousRuntime


def main() -> None:
    parser = argparse.ArgumentParser(description='Run SAGE v2.0.4 Safe Continuous Runtime.')
    parser.add_argument('--config', default='configs/safe_continuous_runtime.json')
    parser.add_argument('--max-cycles', type=int, default=None)
    parser.add_argument('--interval', type=float, default=None)
    args = parser.parse_args()

    config = SafeContinuousRuntimeConfig.load(args.config)
    if args.max_cycles is not None:
        config.max_cycles = args.max_cycles
    if args.interval is not None:
        config.cycle_interval_seconds = args.interval

    runtime = SafeContinuousRuntime(config)
    summary = runtime.run()

    print('=== SAGE v2.0.4 Safe Continuous Runtime ===')
    print(f'cycles_requested: {summary.get("cycles_requested")}')
    print(f'cycles_completed: {summary.get("cycles_completed")}')
    print(f'stopped_by_file: {summary.get("stopped_by_file")}')
    print(f'passed: {summary.get("passed")}')
    print(f'summary: {config.output_dir}/summary.json')
    print()
    print(json.dumps({
        'allowed_actions': summary.get('allowed_actions'),
        'forbidden_actions': summary.get('forbidden_actions'),
        'safety_policy': summary.get('safety_policy'),
    }, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
