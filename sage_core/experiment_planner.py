from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
import hashlib
import json


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_id(*parts: object, length: int = 12) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(str(part).encode("utf-8", errors="replace"))
        h.update(b"|")
    return h.hexdigest()[:length]


@dataclass
class ExperimentObservation:
    source_path: str
    kind: str
    summary: str
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_jsonable(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentProposal:
    title: str
    goal: str
    rationale: str
    expected_effect: str
    proposed_actions: List[str]
    success_criteria: List[str]
    risk_level: str = "low"
    requires_human_approval: bool = True
    forbidden_actions: List[str] = field(default_factory=lambda: [
        "network_access",
        "shell_command_execution_without_human",
        "auto_delete_files",
        "auto_disable_organs",
        "auto_approve_memory",
        "git_commit_or_push_without_human",
    ])
    suggested_config_changes: Dict[str, Any] = field(default_factory=dict)
    source_observations: List[str] = field(default_factory=list)
    id: str = ""

    def finalize_id(self) -> None:
        if not self.id:
            self.id = stable_id(self.title, self.goal, self.rationale)

    def to_jsonable(self) -> Dict[str, Any]:
        self.finalize_id()
        return asdict(self)


@dataclass
class ExperimentPlan:
    version: str = "v2.0.2"
    created_at: str = field(default_factory=utc_now)
    planner_name: str = "autonomous_experiment_planner"
    observations: List[ExperimentObservation] = field(default_factory=list)
    proposals: List[ExperimentProposal] = field(default_factory=list)
    selected_proposal_id: str | None = None
    final_note: str = ""
    safety_policy: Dict[str, bool] = field(default_factory=lambda: {
        "network_actions": False,
        "shell_actions": False,
        "auto_delete_files": False,
        "auto_disable_organs": False,
        "auto_approve_memory": False,
        "git_actions": False,
        "human_approval_required": True,
    })

    def to_jsonable(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "planner_name": self.planner_name,
            "observations": [o.to_jsonable() for o in self.observations],
            "proposals": [p.to_jsonable() for p in self.proposals],
            "selected_proposal_id": self.selected_proposal_id,
            "final_note": self.final_note,
            "safety_policy": self.safety_policy,
        }


class ResultReader:
    def __init__(self, root: str | Path = ".") -> None:
        self.root = Path(root)

    def read_json(self, rel_path: str) -> Dict[str, Any] | None:
        path = self.root / rel_path
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def observation_from_reflection_result(self, rel_path: str) -> ExperimentObservation | None:
        data = self.read_json(rel_path)
        if not data:
            return None
        selected = data.get("selected") or {}
        metrics = data.get("emergence_metrics", {})
        policy = data.get("reflection_policy", {})
        summary = (
            f"Reflection result selected {selected.get('organ')} "
            f"with policy {policy.get('name') or metrics.get('policy_name')} "
            f"and top_second_gap {metrics.get('top_second_gap')}."
        )
        return ExperimentObservation(
            source_path=rel_path,
            kind="reflection_result",
            summary=summary,
            metrics={
                "selected_organ": selected.get("organ"),
                "policy_name": policy.get("name") or metrics.get("policy_name"),
                "top_score": metrics.get("top_score"),
                "top_second_gap": metrics.get("top_second_gap"),
                "candidate_count": metrics.get("candidate_count"),
                "organ_diversity": metrics.get("organ_diversity"),
                "mean_novelty": metrics.get("mean_novelty"),
                "mean_reuse_value": metrics.get("mean_reuse_value"),
                "memory_inbox_count_before": data.get("memory_inbox_count_before"),
                "memory_inbox_count_after": data.get("memory_inbox_count_after"),
            },
        )

    def observation_from_policy_benchmark(self, rel_path: str) -> ExperimentObservation | None:
        data = self.read_json(rel_path)
        if not data:
            return None
        rows = data.get("rows", [])
        selected_by_policy = {row.get("policy"): row.get("selected_organ") for row in rows}
        unique_selected = data.get("unique_selected_organs", [])
        passed = data.get("passed")
        summary = (
            f"Policy benchmark passed={passed}; unique selected organs={unique_selected}; "
            f"selected_by_policy={selected_by_policy}."
        )
        return ExperimentObservation(
            source_path=rel_path,
            kind="policy_benchmark",
            summary=summary,
            metrics={
                "passed": passed,
                "unique_selected_organs": unique_selected,
                "selected_by_policy": selected_by_policy,
                "row_count": len(rows),
            },
        )

    def collect_default_observations(self) -> List[ExperimentObservation]:
        candidates = [
            self.observation_from_reflection_result("results/v2_0_emergent_reflection.json"),
            self.observation_from_reflection_result("results/v2_0_emergent_reflection_smoke_detail.json"),
            self.observation_from_policy_benchmark("results/v2_0_1_reflection_policy_config.json"),
            self.observation_from_policy_benchmark("results/v2_0_1_policy_config.json"),
        ]
        return [obs for obs in candidates if obs is not None]


class AutonomousExperimentPlanner:
    """Proposal-only autonomous planner.

    It reads allowed result JSON files and proposes next experiments.
    It does not execute experiments, edit files, approve memory, run shell commands,
    delete files, or perform git actions.
    """

    def __init__(self, min_gap_for_stability: float = 0.03, max_memory_inbox_before_review: int = 12) -> None:
        self.min_gap_for_stability = min_gap_for_stability
        self.max_memory_inbox_before_review = max_memory_inbox_before_review

    def _latest_metric(self, observations: List[ExperimentObservation], key: str) -> Any:
        for obs in observations:
            if key in obs.metrics and obs.metrics[key] is not None:
                return obs.metrics[key]
        return None

    def propose(self, observations: List[ExperimentObservation]) -> ExperimentPlan:
        proposals: List[ExperimentProposal] = []
        top_gap = self._latest_metric(observations, "top_second_gap")
        memory_after = self._latest_metric(observations, "memory_inbox_count_after")
        policy_obs = next((o for o in observations if o.kind == "policy_benchmark"), None)
        selected_by_policy = {}
        unique_selected = []
        policy_passed = None
        if policy_obs:
            selected_by_policy = dict(policy_obs.metrics.get("selected_by_policy", {}))
            unique_selected = list(policy_obs.metrics.get("unique_selected_organs", []))
            policy_passed = policy_obs.metrics.get("passed")
        source_paths = [o.source_path for o in observations]

        if top_gap is None or float(top_gap) < self.min_gap_for_stability:
            proposals.append(ExperimentProposal(
                title="Stability Probe for Reflection Selection",
                goal="Check whether the selected reflection organ remains stable under small policy perturbations.",
                rationale=(
                    f"The current top_second_gap is {top_gap}, below the stability threshold "
                    f"{self.min_gap_for_stability}. A small gap means the selected organ may be sensitive to scoring weights."
                ),
                expected_effect="Identify whether curiosity-biased selection is robust or only barely preferred.",
                proposed_actions=[
                    "Generate nearby exploratory policies by changing novelty by -0.02, 0.00, and +0.02.",
                    "Run the reflection loop once per policy without creating memory proposals.",
                    "Compare selected_organ, top_score, and top_second_gap.",
                    "Save a JSON stability report for human review.",
                ],
                success_criteria=[
                    "At least three policy variants are evaluated.",
                    "No memory is auto-approved.",
                    "No files outside results/ and logs/ are modified by the benchmark.",
                    "A report shows whether selected_organ stays the same or changes.",
                ],
                risk_level="low",
                suggested_config_changes={
                    "base_policy": "configs/reflection_policy_exploratory.json",
                    "sweep": {"novelty_delta": [-0.02, 0.0, 0.02], "risk_delta": [0.0, 0.02]},
                    "create_memory_proposal": False,
                },
                source_observations=source_paths,
            ))

        if policy_obs is None or len(unique_selected) < 2 or policy_passed is False:
            proposals.append(ExperimentProposal(
                title="Policy Diversity Repair",
                goal="Ensure reflection policies can produce meaningfully different selected organs.",
                rationale=(
                    "A configurable policy system should be able to switch behavior. "
                    f"Current unique selected organs: {unique_selected}. selected_by_policy={selected_by_policy}."
                ),
                expected_effect="Make conservative, exploratory, balanced, and memory-focused modes easier to distinguish.",
                proposed_actions=[
                    "Review reflection_policy_*.json files.",
                    "Increase separation between novelty-focused and reuse-focused policies.",
                    "Run benchmark_v2_0_1_reflection_policy_config again.",
                    "Check that at least two different selected organs appear.",
                ],
                success_criteria=[
                    "Policy benchmark passes.",
                    "At least two unique selected organs appear.",
                    "Exploratory policy selects curiosity_organ.",
                    "At least one conservative or balanced policy selects critic_organ.",
                ],
                risk_level="low",
                suggested_config_changes={
                    "exploratory": {"increase": "novelty", "decrease": "reuse_value"},
                    "conservative": {"increase": ["confidence", "reuse_value"], "decrease": "novelty"},
                    "memory_focused": {"increase": "reuse_value", "increase_penalty": ["cost", "risk"]},
                },
                source_observations=source_paths,
            ))

        if memory_after is not None and int(memory_after) >= self.max_memory_inbox_before_review:
            proposals.append(ExperimentProposal(
                title="Memory Inbox Review Tool",
                goal="Prevent memory/inbox from growing without human review.",
                rationale=(
                    f"memory_inbox_count_after is {memory_after}. If inbox keeps growing, "
                    "SAGE accumulates proposals without consolidation."
                ),
                expected_effect="Create a safe review workflow that moves memory proposals to approved or rejected with human choice.",
                proposed_actions=[
                    "Create a read-only listing tool for memory/inbox.",
                    "Add approve/reject commands that require explicit human input.",
                    "Keep auto-approval disabled.",
                    "Write a review summary to results/.",
                ],
                success_criteria=[
                    "Inbox entries can be listed.",
                    "A selected entry can be moved only after human approval.",
                    "No automatic approval happens.",
                    "Approved and rejected counts are reported.",
                ],
                risk_level="medium",
                suggested_config_changes={"memory_review": {"auto_approve": False, "human_approval_required": True}},
                source_observations=source_paths,
            ))

        proposals.append(ExperimentProposal(
            title="Experiment Planner Self-Check",
            goal="Validate that SAGE can propose the next experiment without executing it.",
            rationale="v2.0.2 should introduce controlled autonomy: proposal generation without unsafe action execution.",
            expected_effect="Establish the first autonomous planning loop while preserving human approval.",
            proposed_actions=[
                "Read existing result JSON files.",
                "Generate ranked next-experiment proposals.",
                "Save proposals to experiments/inbox/.",
                "Do not run the proposed experiments automatically.",
            ],
            success_criteria=[
                "At least two proposals are generated.",
                "Every proposal has requires_human_approval=true.",
                "Safety policy disables network, shell, deletion, memory approval, and git actions.",
                "One selected proposal is recommended for human review.",
            ],
            risk_level="low",
            suggested_config_changes={"autonomy_level": "proposal_only", "human_approval_required": True},
            source_observations=source_paths,
        ))

        priority = {
            "Stability Probe for Reflection Selection": 0,
            "Policy Diversity Repair": 1,
            "Memory Inbox Review Tool": 2,
            "Experiment Planner Self-Check": 3,
        }
        proposals.sort(key=lambda p: priority.get(p.title, 99))
        for proposal in proposals:
            proposal.finalize_id()
        selected_id = proposals[0].id if proposals else None
        final_note = (
            "SAGE generated next-experiment proposals but did not execute them. "
            "Human approval is required before any experiment is run."
            if selected_id else "No experiment proposals were generated."
        )
        return ExperimentPlan(observations=observations, proposals=proposals, selected_proposal_id=selected_id, final_note=final_note)
