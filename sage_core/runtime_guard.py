from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
import json


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class GuardLimits:
    max_cycles: int = 10
    max_failures: int = 2
    max_memory_inbox_growth: int = 8
    max_result_files_growth: int = 80
    max_experiment_inbox_growth: int = 40
    stop_file_path: str = "runtime_control/STOP"

    def to_jsonable(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GuardSnapshot:
    created_at: str = field(default_factory=utc_now)
    memory_inbox_count: int = 0
    result_files_count: int = 0
    experiment_inbox_count: int = 0
    cycle_index: int = 0
    failure_count: int = 0

    def to_jsonable(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CycleGuardRecord:
    cycle_index: int
    started_at: str
    ended_at: str | None = None
    ok: bool = False
    actions: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    memory_inbox_count: int = 0
    result_files_count: int = 0
    experiment_inbox_count: int = 0
    stop_reason: str | None = None

    def to_jsonable(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GuardDecision:
    should_continue: bool
    reason: str = "continue"

    def to_jsonable(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GuardReport:
    version: str = "v2.0.5"
    created_at: str = field(default_factory=utc_now)
    guard_name: str = "runtime_guard_long_run_monitor"
    limits: Dict[str, Any] = field(default_factory=dict)
    baseline: Dict[str, Any] = field(default_factory=dict)
    latest: Dict[str, Any] = field(default_factory=dict)
    cycles: List[Dict[str, Any]] = field(default_factory=list)
    cycles_completed: int = 0
    failure_count: int = 0
    stopped: bool = False
    stop_reason: str | None = None
    safety_policy: Dict[str, bool] = field(default_factory=lambda: {
        "network_actions": False,
        "arbitrary_shell_actions": False,
        "file_delete": False,
        "core_code_modification": False,
        "auto_disable_organs": False,
        "auto_delete_organs": False,
        "auto_approve_memory": False,
        "git_actions": False,
        "human_approval_required": True,
        "limited_whitelist_execution": True,
        "guarded_long_run": True,
    })
    passed: bool = False

    def to_jsonable(self) -> Dict[str, Any]:
        return asdict(self)


class RuntimeGuard:
    """Safety guard for SAGE continuous runtime.

    It does not run experiments by itself.
    It only observes counts, failures, STOP file, and growth limits.
    """

    def __init__(
        self,
        limits: GuardLimits | None = None,
        root: str | Path = ".",
    ) -> None:
        self.root = Path(root)
        self.limits = limits or GuardLimits()
        self.baseline = self.snapshot(cycle_index=0, failure_count=0)
        self.records: List[CycleGuardRecord] = []

    def count_files(self, rel_dir: str, pattern: str = "*.json") -> int:
        path = self.root / rel_dir
        if not path.exists():
            return 0
        return len(list(path.rglob(pattern)))

    def snapshot(self, cycle_index: int, failure_count: int) -> GuardSnapshot:
        return GuardSnapshot(
            memory_inbox_count=self.count_files("memory/inbox"),
            result_files_count=self.count_files("results"),
            experiment_inbox_count=self.count_files("experiments/inbox"),
            cycle_index=cycle_index,
            failure_count=failure_count,
        )

    def stop_file_exists(self) -> bool:
        return (self.root / self.limits.stop_file_path).exists()

    def evaluate(self, latest: GuardSnapshot) -> GuardDecision:
        if self.stop_file_exists():
            return GuardDecision(False, "stop_file_detected")

        if latest.failure_count > self.limits.max_failures:
            return GuardDecision(False, "max_failures_exceeded")

        memory_growth = latest.memory_inbox_count - self.baseline.memory_inbox_count
        if memory_growth > self.limits.max_memory_inbox_growth:
            return GuardDecision(False, "memory_inbox_growth_exceeded")

        result_growth = latest.result_files_count - self.baseline.result_files_count
        if result_growth > self.limits.max_result_files_growth:
            return GuardDecision(False, "result_files_growth_exceeded")

        exp_growth = latest.experiment_inbox_count - self.baseline.experiment_inbox_count
        if exp_growth > self.limits.max_experiment_inbox_growth:
            return GuardDecision(False, "experiment_inbox_growth_exceeded")

        if latest.cycle_index >= self.limits.max_cycles:
            return GuardDecision(False, "max_cycles_reached")

        return GuardDecision(True, "continue")

    def begin_cycle(self, cycle_index: int) -> CycleGuardRecord:
        record = CycleGuardRecord(
            cycle_index=cycle_index,
            started_at=utc_now(),
        )
        self.records.append(record)
        return record

    def end_cycle(
        self,
        record: CycleGuardRecord,
        ok: bool,
        actions: List[str],
        errors: List[str],
        failure_count: int,
        stop_reason: str | None = None,
    ) -> GuardSnapshot:
        latest = self.snapshot(record.cycle_index, failure_count)
        record.ended_at = utc_now()
        record.ok = ok
        record.actions = list(actions)
        record.errors = list(errors)
        record.memory_inbox_count = latest.memory_inbox_count
        record.result_files_count = latest.result_files_count
        record.experiment_inbox_count = latest.experiment_inbox_count
        record.stop_reason = stop_reason
        return latest

    def report(
        self,
        latest: GuardSnapshot,
        failure_count: int,
        stopped: bool,
        stop_reason: str | None,
    ) -> GuardReport:
        cycles_completed = len([r for r in self.records if r.ok])
        passed = (
            failure_count <= self.limits.max_failures
            and cycles_completed >= 1
            and stop_reason not in {
                "max_failures_exceeded",
                "memory_inbox_growth_exceeded",
                "result_files_growth_exceeded",
                "experiment_inbox_growth_exceeded",
            }
        )

        return GuardReport(
            limits=self.limits.to_jsonable(),
            baseline=self.baseline.to_jsonable(),
            latest=latest.to_jsonable(),
            cycles=[r.to_jsonable() for r in self.records],
            cycles_completed=cycles_completed,
            failure_count=failure_count,
            stopped=stopped,
            stop_reason=stop_reason,
            passed=passed,
        )
