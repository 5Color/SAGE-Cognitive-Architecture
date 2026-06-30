from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import json
import time
import traceback

from sage_core.runtime_guard import GuardLimits, RuntimeGuard
from sage_runtime.emergent_reflection_loop import EmergentReflectionConfig, EmergentReflectionLoop
from sage_runtime.autonomous_experiment_planner import (
    AutonomousExperimentPlannerConfig,
    AutonomousExperimentPlannerRuntime,
)
from sage_runtime.reflection_stability_runtime import (
    ReflectionStabilityConfig,
    ReflectionStabilityRuntime,
)


@dataclass
class GuardedContinuousRuntimeConfig:
    max_cycles: int = 10
    interval_seconds: float = 1.0
    max_failures: int = 2
    max_memory_inbox_growth: int = 8
    max_result_files_growth: int = 80
    max_experiment_inbox_growth: int = 40
    stop_file_path: str = "runtime_control/STOP"
    output_dir: str = "results/v2_0_5_guarded_runtime"
    summary_path: str = "results/v2_0_5_guarded_runtime/summary.json"
    run_reflection: bool = True
    run_experiment_planner: bool = True
    run_stability_probe: bool = True
    create_memory_proposal_each_cycle: bool = False

    @classmethod
    def load(cls, path: str | Path) -> "GuardedContinuousRuntimeConfig":
        p = Path(path)
        if not p.exists():
            return cls()
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls(**{**cls().__dict__, **data})


class GuardedContinuousRuntime:
    """Guarded long-run loop for SAGE.

    This is a safer version of continuous runtime.
    It allows only whitelisted internal actions and monitors growth/failures.
    """

    allowed_actions = [
        "run_reflection_loop",
        "run_autonomous_experiment_planner",
        "run_reflection_stability_probe",
        "write_results_json",
        "write_logs",
        "write_experiments_inbox",
        "write_generated_configs",
        "monitor_runtime_growth",
        "stop_file_check",
    ]

    forbidden_actions = [
        "network_access",
        "arbitrary_shell_execution",
        "file_delete",
        "core_code_modification",
        "organ_auto_disable",
        "organ_auto_delete",
        "memory_auto_approve",
        "git_commit",
        "git_push",
    ]

    def __init__(self, config: GuardedContinuousRuntimeConfig) -> None:
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        limits = GuardLimits(
            max_cycles=config.max_cycles,
            max_failures=config.max_failures,
            max_memory_inbox_growth=config.max_memory_inbox_growth,
            max_result_files_growth=config.max_result_files_growth,
            max_experiment_inbox_growth=config.max_experiment_inbox_growth,
            stop_file_path=config.stop_file_path,
        )
        self.guard = RuntimeGuard(limits=limits)

    def run_reflection_once(self, cycle_index: int) -> Dict[str, Any]:
        cfg = EmergentReflectionConfig(
            state_path="runtime_state/state.json",
            registry_path="registry/organ_registry.json",
            memory_root="memory",
            reflection_log_path=f"logs/v2_0_5_guarded_cycle_{cycle_index}_reflection.md",
            result_path=str(self.output_dir / f"cycle_{cycle_index}_reflection.json"),
            create_memory_proposal=self.config.create_memory_proposal_each_cycle,
            reflection_policy_path="configs/reflection_policy_exploratory.json",
        )
        return EmergentReflectionLoop(cfg).run_once()

    def run_planner_once(self, cycle_index: int) -> Dict[str, Any]:
        cfg = AutonomousExperimentPlannerConfig(
            result_root=".",
            output_path=str(self.output_dir / f"cycle_{cycle_index}_experiment_plan.json"),
            inbox_dir="experiments/inbox",
            selected_proposal_path=f"experiments/inbox/v2_0_5_cycle_{cycle_index}_selected_next_experiment.json",
            min_gap_for_stability=0.03,
            max_memory_inbox_before_review=12,
        )
        return AutonomousExperimentPlannerRuntime(cfg).run_once()

    def run_stability_once(self, cycle_index: int) -> Dict[str, Any]:
        cfg = ReflectionStabilityConfig(
            base_policy_path="configs/reflection_policy_exploratory.json",
            output_path=str(self.output_dir / f"cycle_{cycle_index}_stability_probe.json"),
            output_dir=str(self.output_dir / f"cycle_{cycle_index}_stability_variants"),
            variant_policy_dir=f"configs/generated/v2_0_5_cycle_{cycle_index}_stability_probe",
            target_organ="curiosity_organ",
            novelty_deltas=[-0.02, 0.0, 0.02],
            risk_deltas=[0.0, 0.02],
            min_target_rate=0.80,
        )
        return ReflectionStabilityRuntime(cfg).run_once()

    def run_cycle(self, cycle_index: int) -> tuple[bool, List[str], List[str]]:
        actions: List[str] = []
        errors: List[str] = []

        if self.config.run_reflection:
            actions.append("run_reflection_loop")
            self.run_reflection_once(cycle_index)

        if self.config.run_experiment_planner:
            actions.append("run_autonomous_experiment_planner")
            self.run_planner_once(cycle_index)

        if self.config.run_stability_probe:
            actions.append("run_reflection_stability_probe")
            self.run_stability_once(cycle_index)

        actions.extend([
            "write_results_json",
            "write_logs",
            "write_experiments_inbox",
            "write_generated_configs",
            "monitor_runtime_growth",
            "stop_file_check",
        ])

        return True, actions, errors

    def run(self) -> Dict[str, Any]:
        failure_count = 0
        stop_reason = None
        stopped = False
        latest = self.guard.snapshot(cycle_index=0, failure_count=0)

        for cycle_index in range(1, self.config.max_cycles + 1):
            decision = self.guard.evaluate(latest)
            if not decision.should_continue:
                stop_reason = decision.reason
                stopped = True
                break

            record = self.guard.begin_cycle(cycle_index)
            ok = False
            actions: List[str] = []
            errors: List[str] = []

            try:
                ok, actions, errors = self.run_cycle(cycle_index)
            except Exception as exc:
                failure_count += 1
                ok = False
                errors.append(f"{type(exc).__name__}: {exc}")
                errors.append(traceback.format_exc())

            latest = self.guard.end_cycle(
                record=record,
                ok=ok,
                actions=actions,
                errors=errors,
                failure_count=failure_count,
            )

            decision = self.guard.evaluate(latest)
            if not decision.should_continue:
                stop_reason = decision.reason
                stopped = True
                break

            if cycle_index < self.config.max_cycles and self.config.interval_seconds > 0:
                time.sleep(self.config.interval_seconds)

        if stop_reason is None:
            final_decision = self.guard.evaluate(latest)
            if not final_decision.should_continue:
                stop_reason = final_decision.reason
                stopped = True
            else:
                stop_reason = "completed_requested_cycles"

        report = self.guard.report(
            latest=latest,
            failure_count=failure_count,
            stopped=stopped,
            stop_reason=stop_reason,
        ).to_jsonable()

        report["allowed_actions"] = self.allowed_actions
        report["forbidden_actions"] = self.forbidden_actions

        summary_path = Path(self.config.summary_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        report["summary_path"] = str(summary_path)

        return report
