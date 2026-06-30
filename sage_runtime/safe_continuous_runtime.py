from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
import json
import time

from sage_core.memory_store import MemoryStore
from sage_runtime.emergent_reflection_loop import EmergentReflectionConfig, EmergentReflectionLoop
from sage_runtime.autonomous_experiment_planner import AutonomousExperimentPlannerConfig, AutonomousExperimentPlannerRuntime
from sage_runtime.reflection_stability_runtime import ReflectionStabilityConfig, ReflectionStabilityRuntime


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class SafeContinuousRuntimeConfig:
    version: str = 'v2.0.4'
    max_cycles: int = 3
    cycle_interval_seconds: float = 2.0
    stop_file: str = 'runtime_control/STOP'
    output_dir: str = 'results/v2_0_4_continuous_runtime'
    log_path: str = 'logs/v2_0_4_safe_continuous_runtime.md'
    memory_root: str = 'memory'
    max_memory_inbox: int = 20
    create_memory_proposal: bool = False
    run_reflection: bool = True
    run_experiment_planner: bool = True
    run_stability_probe: bool = True
    stability_probe_every_n_cycles: int = 2
    allowed_actions: List[str] = field(default_factory=lambda: [
        'run_reflection_loop',
        'run_autonomous_experiment_planner',
        'run_reflection_stability_probe',
        'write_results_json',
        'write_logs',
        'write_experiments_inbox',
        'write_generated_configs',
    ])
    forbidden_actions: List[str] = field(default_factory=lambda: [
        'network_access',
        'arbitrary_shell_execution',
        'file_delete',
        'core_code_modification',
        'organ_auto_disable',
        'organ_auto_delete',
        'memory_auto_approve',
        'git_commit',
        'git_push',
    ])

    @classmethod
    def load(cls, path: str | Path) -> 'SafeContinuousRuntimeConfig':
        p = Path(path)
        if not p.exists():
            return cls()
        data = json.loads(p.read_text(encoding='utf-8'))
        return cls(**{**cls().__dict__, **data})

    def to_jsonable(self) -> Dict[str, Any]:
        return self.__dict__.copy()


class SafeContinuousRuntime:
    def __init__(self, config: SafeContinuousRuntimeConfig) -> None:
        self.config = config
        self.memory = MemoryStore(config.memory_root)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        Path(config.log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(config.stop_file).parent.mkdir(parents=True, exist_ok=True)

    def safety_policy(self) -> Dict[str, bool]:
        return {
            'network_actions': False,
            'arbitrary_shell_actions': False,
            'file_delete': False,
            'core_code_modification': False,
            'auto_disable_organs': False,
            'auto_delete_organs': False,
            'auto_approve_memory': False,
            'git_actions': False,
            'human_approval_required': True,
            'limited_whitelist_execution': True,
        }

    def should_stop(self) -> bool:
        return Path(self.config.stop_file).exists()

    def append_log(self, text: str) -> None:
        with Path(self.config.log_path).open('a', encoding='utf-8') as f:
            f.write(text.rstrip() + '\n\n')

    def run_reflection(self, cycle: int) -> Dict[str, Any]:
        result_path = self.output_dir / f'cycle_{cycle:03d}_reflection.json'
        reflection_config = EmergentReflectionConfig(
            state_path='runtime_state/state.json',
            registry_path='registry/organ_registry.json',
            memory_root=self.config.memory_root,
            reflection_log_path=f'logs/v2_0_4_cycle_{cycle:03d}_reflection.md',
            result_path=str(result_path),
            create_memory_proposal=self.config.create_memory_proposal,
            reflection_policy_path='configs/reflection_policy_exploratory.json',
        )
        result = EmergentReflectionLoop(reflection_config).run_once()
        return {
            'action': 'run_reflection_loop',
            'result_path': str(result_path),
            'selected_organ': (result.get('selected') or {}).get('organ'),
            'top_second_gap': result.get('emergence_metrics', {}).get('top_second_gap'),
            'memory_inbox_after': result.get('memory_inbox_count_after'),
        }

    def run_experiment_planner(self, cycle: int) -> Dict[str, Any]:
        output_path = self.output_dir / f'cycle_{cycle:03d}_experiment_plan.json'
        planner_config = AutonomousExperimentPlannerConfig(
            result_root='.',
            output_path=str(output_path),
            inbox_dir=f'experiments/inbox/v2_0_4_cycle_{cycle:03d}',
            selected_proposal_path=f'experiments/inbox/v2_0_4_cycle_{cycle:03d}/selected_next_experiment.json',
            min_gap_for_stability=0.03,
            max_memory_inbox_before_review=12,
        )
        result = AutonomousExperimentPlannerRuntime(planner_config).run_once()
        proposals = result.get('proposals', [])
        selected_id = result.get('selected_proposal_id')
        selected = next((p for p in proposals if p.get('id') == selected_id), None)
        return {
            'action': 'run_autonomous_experiment_planner',
            'result_path': str(output_path),
            'proposal_count': len(proposals),
            'selected_proposal_title': None if selected is None else selected.get('title'),
            'selected_requires_human_approval': None if selected is None else selected.get('requires_human_approval'),
        }

    def run_stability_probe(self, cycle: int) -> Dict[str, Any]:
        output_path = self.output_dir / f'cycle_{cycle:03d}_stability_probe.json'
        stability_config = ReflectionStabilityConfig(
            base_policy_path='configs/reflection_policy_exploratory.json',
            output_path=str(output_path),
            output_dir=str(self.output_dir / f'cycle_{cycle:03d}_stability_variants'),
            variant_policy_dir=f'configs/generated/v2_0_4_cycle_{cycle:03d}_stability_probe',
            target_organ='curiosity_organ',
            novelty_deltas=[-0.02, 0.0, 0.02],
            risk_deltas=[0.0, 0.02],
            min_target_rate=0.80,
        )
        result = ReflectionStabilityRuntime(stability_config).run_once()
        return {
            'action': 'run_reflection_stability_probe',
            'result_path': str(output_path),
            'target_selected_rate': result.get('target_selected_rate'),
            'selected_counts': result.get('selected_counts'),
            'passed': result.get('passed'),
        }

    def run_cycle(self, cycle: int) -> Dict[str, Any]:
        started_at = utc_now()
        actions: List[Dict[str, Any]] = []
        errors: List[str] = []
        inbox_count = self.memory.count_inbox()

        if inbox_count >= self.config.max_memory_inbox:
            return {
                'cycle': cycle,
                'started_at': started_at,
                'ended_at': utc_now(),
                'mode': 'safe_pause_memory_inbox_limit',
                'memory_inbox_count': inbox_count,
                'actions': [],
                'errors': [f'memory inbox limit reached: {inbox_count} >= {self.config.max_memory_inbox}'],
                'safety_policy': self.safety_policy(),
            }

        try:
            if self.config.run_reflection:
                actions.append(self.run_reflection(cycle))
        except Exception as exc:
            errors.append(f'reflection_error: {type(exc).__name__}: {exc}')

        try:
            if self.config.run_experiment_planner:
                actions.append(self.run_experiment_planner(cycle))
        except Exception as exc:
            errors.append(f'experiment_planner_error: {type(exc).__name__}: {exc}')

        try:
            if self.config.run_stability_probe and self.config.stability_probe_every_n_cycles > 0:
                if cycle % self.config.stability_probe_every_n_cycles == 0:
                    actions.append(self.run_stability_probe(cycle))
        except Exception as exc:
            errors.append(f'stability_probe_error: {type(exc).__name__}: {exc}')

        return {
            'cycle': cycle,
            'started_at': started_at,
            'ended_at': utc_now(),
            'mode': 'safe_continuous_runtime',
            'memory_inbox_count': self.memory.count_inbox(),
            'actions': actions,
            'errors': errors,
            'safety_policy': self.safety_policy(),
        }

    def run(self) -> Dict[str, Any]:
        cycle_reports: List[Dict[str, Any]] = []
        stopped_by_file = False
        started_at = utc_now()

        self.append_log(f'# SAGE v2.0.4 Safe Continuous Runtime\n\nStarted: {started_at}')

        for cycle in range(1, self.config.max_cycles + 1):
            if self.should_stop():
                stopped_by_file = True
                self.append_log(f'STOP file detected before cycle {cycle}.')
                break

            report = self.run_cycle(cycle)
            cycle_reports.append(report)
            cycle_path = self.output_dir / f'cycle_{cycle:03d}_report.json'
            cycle_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')

            action_names = [a.get('action') for a in report.get('actions', [])]
            self.append_log(
                f'## Cycle {cycle}\n'
                f'- mode: {report.get("mode")}\n'
                f'- actions: {action_names}\n'
                f'- errors: {report.get("errors")}\n'
                f'- memory_inbox_count: {report.get("memory_inbox_count")}\n'
            )

            if report.get('mode') == 'safe_pause_memory_inbox_limit':
                break

            if self.config.cycle_interval_seconds > 0 and cycle < self.config.max_cycles:
                time.sleep(self.config.cycle_interval_seconds)

        summary = {
            'version': self.config.version,
            'runtime': 'safe_continuous_runtime',
            'started_at': started_at,
            'ended_at': utc_now(),
            'cycles_requested': self.config.max_cycles,
            'cycles_completed': len(cycle_reports),
            'stopped_by_file': stopped_by_file,
            'stop_file': self.config.stop_file,
            'allowed_actions': self.config.allowed_actions,
            'forbidden_actions': self.config.forbidden_actions,
            'safety_policy': self.safety_policy(),
            'cycle_reports': cycle_reports,
            'passed': (
                len(cycle_reports) >= 1
                and all(not r.get('errors') for r in cycle_reports)
                and self.safety_policy().get('network_actions') is False
                and self.safety_policy().get('arbitrary_shell_actions') is False
                and self.safety_policy().get('auto_approve_memory') is False
                and self.safety_policy().get('git_actions') is False
            ),
        }

        summary_path = self.output_dir / 'summary.json'
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
        self.append_log(f'Finished: {summary["ended_at"]}\nPassed: {summary["passed"]}')
        return summary
