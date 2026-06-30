from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import json


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class AutonomyAction:
    name: str
    description: str
    category: str
    min_level: int
    requires_human_approval: bool = False
    forbidden: bool = False


@dataclass
class AutonomyDecision:
    action: str
    requested_level: int
    allowed: bool
    requires_human_approval: bool
    reason: str
    category: str = "unknown"
    forbidden: bool = False

    def to_jsonable(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AutonomyPolicyReport:
    version: str = "v2.3"
    created_at: str = field(default_factory=utc_now)
    policy_name: str = "autonomy_level_policy"
    active_level: int = 2
    level_name: str = "safe_auto"
    level_description: str = ""
    allowed_actions: List[str] = field(default_factory=list)
    approval_required_actions: List[str] = field(default_factory=list)
    forbidden_actions: List[str] = field(default_factory=list)
    decisions: List[Dict[str, Any]] = field(default_factory=list)
    safety_policy: Dict[str, bool] = field(default_factory=lambda: {
        "auto_approve_memory": False,
        "auto_delete_files": False,
        "auto_archive_files": False,
        "auto_modify_core_code": False,
        "auto_git_commit": False,
        "auto_git_push": False,
        "network_actions": False,
        "arbitrary_shell_actions": False,
        "human_approval_required_for_irreversible_actions": True,
        "bounded_autonomy_only": True,
    })
    passed: bool = False

    def to_jsonable(self) -> Dict[str, Any]:
        return asdict(self)


class AutonomyLevelPolicy:
    """Explicit autonomy level policy for SAGE.

    This policy is a gatekeeper. It does not perform actions.
    It only answers whether an action is allowed, approval-required, or forbidden.

    Main rule:
    SAGE may observe, run whitelisted internal loops, propose, and write reports.
    SAGE may not approve memory, delete files, modify core code, or run git/network actions automatically.
    """

    LEVELS: Dict[int, Dict[str, str]] = {
        0: {
            "name": "manual",
            "description": "Human manually runs every action. SAGE only reports status.",
        },
        1: {
            "name": "assisted",
            "description": "SAGE can inspect and suggest actions, but human executes them.",
        },
        2: {
            "name": "safe_auto",
            "description": "SAGE can automatically run whitelisted local loops and write proposals/results.",
        },
        3: {
            "name": "approval_required",
            "description": "SAGE can prepare reviewable actions, but approval is required for memory/archive decisions.",
        },
        4: {
            "name": "restricted_self_management",
            "description": "SAGE can rank, summarize, deduplicate, and prioritize; irreversible actions still need approval.",
        },
        5: {
            "name": "full_autonomy_disabled",
            "description": "Reserved. Full autonomy is intentionally disabled in this project.",
        },
    }

    ACTIONS: Dict[str, AutonomyAction] = {
        # Level 0/1 observation and reporting
        "read_state": AutonomyAction(
            "read_state", "Read local runtime state and summary files.", "observe", 0
        ),
        "view_results": AutonomyAction(
            "view_results", "View local result JSON files.", "observe", 0
        ),
        "view_docs": AutonomyAction(
            "view_docs", "View local project docs.", "observe", 0
        ),
        "list_memory_candidates": AutonomyAction(
            "list_memory_candidates", "List memory/inbox candidates.", "observe", 1
        ),
        "show_memory_candidate": AutonomyAction(
            "show_memory_candidate", "Show a memory candidate for human review.", "observe", 1
        ),
        "propose_cleanup": AutonomyAction(
            "propose_cleanup", "Generate cleanup/retention proposals only.", "proposal", 1
        ),
        "propose_experiment": AutonomyAction(
            "propose_experiment", "Generate next-experiment proposals only.", "proposal", 1
        ),

        # Level 2 safe auto
        "run_reflection_loop": AutonomyAction(
            "run_reflection_loop", "Run whitelisted reflection loop.", "safe_auto", 2
        ),
        "run_experiment_planner": AutonomyAction(
            "run_experiment_planner", "Run whitelisted autonomous experiment planner.", "safe_auto", 2
        ),
        "run_stability_probe": AutonomyAction(
            "run_stability_probe", "Run whitelisted reflection stability probe.", "safe_auto", 2
        ),
        "run_cleanup_advisor": AutonomyAction(
            "run_cleanup_advisor", "Run cleanup advisor in proposal-only mode.", "safe_auto", 2
        ),
        "write_results_json": AutonomyAction(
            "write_results_json", "Write local result JSON files.", "write", 2
        ),
        "write_logs": AutonomyAction(
            "write_logs", "Write local logs.", "write", 2
        ),
        "write_memory_proposal": AutonomyAction(
            "write_memory_proposal", "Write memory proposal to memory/inbox only.", "write", 2
        ),
        "write_experiment_proposal": AutonomyAction(
            "write_experiment_proposal", "Write experiment proposal to experiments/inbox only.", "write", 2
        ),
        "create_stop_file": AutonomyAction(
            "create_stop_file", "Create runtime_control/STOP safety file.", "control", 2
        ),
        "remove_stop_file": AutonomyAction(
            "remove_stop_file", "Remove runtime_control/STOP to allow another run.", "control", 2, requires_human_approval=True
        ),

        # Level 3 approval-gated
        "approve_memory": AutonomyAction(
            "approve_memory", "Move a memory proposal into memory/approved.", "approval", 3, requires_human_approval=True
        ),
        "reject_memory": AutonomyAction(
            "reject_memory", "Move a memory proposal into memory/rejected.", "approval", 3, requires_human_approval=True
        ),
        "archive_results": AutonomyAction(
            "archive_results", "Move old result files into archive folders.", "approval", 3, requires_human_approval=True
        ),
        "archive_generated_configs": AutonomyAction(
            "archive_generated_configs", "Move generated config folders into archive folders.", "approval", 3, requires_human_approval=True
        ),

        # Level 4 reversible/self-management proposals
        "rank_memory_candidates": AutonomyAction(
            "rank_memory_candidates", "Rank memory candidates by usefulness.", "self_management", 4
        ),
        "deduplicate_memory_candidates": AutonomyAction(
            "deduplicate_memory_candidates", "Detect duplicate memory candidates and suggest decisions.", "self_management", 4
        ),
        "summarize_results": AutonomyAction(
            "summarize_results", "Summarize result files into a reviewable report.", "self_management", 4
        ),

        # Forbidden regardless of level in current project
        "auto_approve_memory": AutonomyAction(
            "auto_approve_memory", "Approve memory without human confirmation.", "forbidden", 99, forbidden=True
        ),
        "file_delete": AutonomyAction(
            "file_delete", "Delete local project files.", "forbidden", 99, forbidden=True
        ),
        "core_code_modification": AutonomyAction(
            "core_code_modification", "Modify sage_core or sage_runtime automatically.", "forbidden", 99, forbidden=True
        ),
        "git_commit": AutonomyAction(
            "git_commit", "Commit to git automatically.", "forbidden", 99, forbidden=True
        ),
        "git_push": AutonomyAction(
            "git_push", "Push to remote automatically.", "forbidden", 99, forbidden=True
        ),
        "network_access": AutonomyAction(
            "network_access", "Perform external network actions.", "forbidden", 99, forbidden=True
        ),
        "arbitrary_shell_execution": AutonomyAction(
            "arbitrary_shell_execution", "Run arbitrary shell commands.", "forbidden", 99, forbidden=True
        ),
    }

    def __init__(self, active_level: int = 2) -> None:
        if active_level not in self.LEVELS:
            raise ValueError(f"Unknown autonomy level: {active_level}")
        if active_level >= 5:
            # Level 5 exists only as a documented boundary.
            active_level = 4
        self.active_level = active_level

    @classmethod
    def from_config(cls, path: str | Path) -> "AutonomyLevelPolicy":
        p = Path(path)
        if not p.exists():
            return cls(active_level=2)
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls(active_level=int(data.get("active_level", 2)))

    def decide(self, action_name: str) -> AutonomyDecision:
        action = self.ACTIONS.get(action_name)
        if not action:
            return AutonomyDecision(
                action=action_name,
                requested_level=self.active_level,
                allowed=False,
                requires_human_approval=True,
                reason="unknown_action_denied_by_default",
                category="unknown",
                forbidden=True,
            )

        if action.forbidden:
            return AutonomyDecision(
                action=action.name,
                requested_level=self.active_level,
                allowed=False,
                requires_human_approval=True,
                reason="forbidden_action_denied",
                category=action.category,
                forbidden=True,
            )

        if self.active_level < action.min_level:
            return AutonomyDecision(
                action=action.name,
                requested_level=self.active_level,
                allowed=False,
                requires_human_approval=action.requires_human_approval,
                reason=f"requires_level_{action.min_level}",
                category=action.category,
                forbidden=False,
            )

        if action.requires_human_approval:
            return AutonomyDecision(
                action=action.name,
                requested_level=self.active_level,
                allowed=False,
                requires_human_approval=True,
                reason="human_approval_required",
                category=action.category,
                forbidden=False,
            )

        return AutonomyDecision(
            action=action.name,
            requested_level=self.active_level,
            allowed=True,
            requires_human_approval=False,
            reason="allowed_by_active_level",
            category=action.category,
            forbidden=False,
        )

    def report(self, actions_to_evaluate: Optional[List[str]] = None) -> AutonomyPolicyReport:
        if actions_to_evaluate is None:
            actions_to_evaluate = list(self.ACTIONS.keys())

        decisions = [self.decide(action) for action in actions_to_evaluate]
        allowed = [d.action for d in decisions if d.allowed]
        approval = [d.action for d in decisions if d.requires_human_approval and not d.forbidden]
        forbidden = [d.action for d in decisions if d.forbidden]

        level = self.LEVELS[self.active_level]
        passed = (
            "auto_approve_memory" in forbidden
            and "file_delete" in forbidden
            and "core_code_modification" in forbidden
            and "git_commit" in forbidden
            and "git_push" in forbidden
            and "network_access" in forbidden
            and "arbitrary_shell_execution" in forbidden
            and "run_reflection_loop" in allowed
            and "run_cleanup_advisor" in allowed
            and "approve_memory" in approval
        )

        return AutonomyPolicyReport(
            active_level=self.active_level,
            level_name=level["name"],
            level_description=level["description"],
            allowed_actions=allowed,
            approval_required_actions=approval,
            forbidden_actions=forbidden,
            decisions=[d.to_jsonable() for d in decisions],
            passed=passed,
        )
