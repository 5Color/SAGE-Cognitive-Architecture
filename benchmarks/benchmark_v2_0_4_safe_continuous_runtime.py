from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sage_runtime.safe_continuous_runtime import SafeContinuousRuntimeConfig, SafeContinuousRuntime


def main() -> None:
    stop_file = Path('runtime_control/STOP')
    if stop_file.exists():
        stop_file.unlink()

    config = SafeContinuousRuntimeConfig(
        max_cycles=2,
        cycle_interval_seconds=0.0,
        stop_file='runtime_control/STOP',
        output_dir='results/v2_0_4_continuous_runtime_smoke',
        log_path='logs/v2_0_4_safe_continuous_runtime_smoke.md',
        memory_root='memory',
        max_memory_inbox=999,
        create_memory_proposal=False,
        run_reflection=True,
        run_experiment_planner=True,
        run_stability_probe=True,
        stability_probe_every_n_cycles=1,
    )

    summary = SafeContinuousRuntime(config).run()
    cycle_reports = summary.get('cycle_reports', [])
    action_names = []
    for report in cycle_reports:
        for action in report.get('actions', []):
            action_names.append(action.get('action'))

    result = {
        'benchmark': 'SAGE-v2.0.4-safe-continuous-runtime-smoke',
        'version': 'v2.0.4',
        'cycles_requested': summary.get('cycles_requested'),
        'cycles_completed': summary.get('cycles_completed'),
        'stopped_by_file': summary.get('stopped_by_file'),
        'action_names': action_names,
        'safety_policy': summary.get('safety_policy'),
        'summary_path': 'results/v2_0_4_continuous_runtime_smoke/summary.json',
        'passed': (
            summary.get('passed') is True
            and summary.get('cycles_completed') == 2
            and 'run_reflection_loop' in action_names
            and 'run_autonomous_experiment_planner' in action_names
            and 'run_reflection_stability_probe' in action_names
            and summary.get('safety_policy', {}).get('network_actions') is False
            and summary.get('safety_policy', {}).get('arbitrary_shell_actions') is False
            and summary.get('safety_policy', {}).get('auto_approve_memory') is False
            and summary.get('safety_policy', {}).get('git_actions') is False
        ),
    }

    out = Path('results/v2_0_4_safe_continuous_runtime_smoke.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')

    print('=== SAGE v2.0.4 Safe Continuous Runtime Smoke ===')
    print(f'cycles_completed: {result["cycles_completed"]}')
    print(f'action_names: {result["action_names"]}')
    print(f'passed: {result["passed"]}')
    print(f'saved: {out}')


if __name__ == '__main__':
    main()
