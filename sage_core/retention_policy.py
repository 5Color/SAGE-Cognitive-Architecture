from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import hashlib
import json


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_id(text: str, length: int = 12) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


@dataclass
class RetentionLimits:
    max_result_json_files: int = 120
    max_generated_config_dirs: int = 20
    max_experiment_inbox_items: int = 30
    max_memory_inbox_items: int = 20
    max_log_markdown_files: int = 50

    def to_jsonable(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RepoAreaStats:
    area: str
    path: str
    exists: bool
    file_count: int = 0
    dir_count: int = 0
    json_count: int = 0
    markdown_count: int = 0
    total_bytes: int = 0
    examples: List[str] = field(default_factory=list)

    def to_jsonable(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RetentionProposal:
    proposal_id: str
    area: str
    priority: str
    action_type: str
    title: str
    rationale: str
    suggested_target: str
    affected_paths: List[str] = field(default_factory=list)
    destructive: bool = False
    requires_human_approval: bool = True
    execute_now: bool = False

    def to_jsonable(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RetentionReport:
    version: str = "v2.0.6"
    created_at: str = field(default_factory=utc_now)
    advisor_name: str = "cleanup_retention_policy_advisor"
    limits: Dict[str, Any] = field(default_factory=dict)
    area_stats: List[Dict[str, Any]] = field(default_factory=list)
    proposals: List[Dict[str, Any]] = field(default_factory=list)
    selected_summary: List[str] = field(default_factory=list)
    safety_policy: Dict[str, bool] = field(default_factory=lambda: {
        "file_delete": False,
        "file_move": False,
        "file_rename": False,
        "git_actions": False,
        "auto_archive": False,
        "auto_cleanup": False,
        "auto_approve_memory": False,
        "human_approval_required": True,
        "proposal_only": True,
    })
    passed: bool = False

    def to_jsonable(self) -> Dict[str, Any]:
        return asdict(self)


class RetentionPolicyAdvisor:
    """Proposal-only cleanup/retention advisor.

    This class never deletes, moves, renames, archives, approves memory,
    commits, or pushes. It only scans and writes a proposal report.
    """

    def __init__(self, root: str | Path = ".", limits: Optional[RetentionLimits] = None) -> None:
        self.root = Path(root)
        self.limits = limits or RetentionLimits()

    def _safe_rel(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.root)).replace("\\", "/")
        except ValueError:
            return str(path).replace("\\", "/")

    def scan_area(self, area: str, rel_path: str) -> RepoAreaStats:
        base = self.root / rel_path
        if not base.exists():
            return RepoAreaStats(area=area, path=rel_path, exists=False)

        files = [p for p in base.rglob("*") if p.is_file()]
        dirs = [p for p in base.rglob("*") if p.is_dir()]
        json_files = [p for p in files if p.suffix.lower() == ".json"]
        md_files = [p for p in files if p.suffix.lower() in {".md", ".markdown"}]
        total_bytes = 0
        for p in files:
            try:
                total_bytes += p.stat().st_size
            except OSError:
                pass

        examples = [self._safe_rel(p) for p in files[:10]]

        return RepoAreaStats(
            area=area,
            path=rel_path,
            exists=True,
            file_count=len(files),
            dir_count=len(dirs),
            json_count=len(json_files),
            markdown_count=len(md_files),
            total_bytes=total_bytes,
            examples=examples,
        )

    def _proposal(
        self,
        *,
        area: str,
        priority: str,
        action_type: str,
        title: str,
        rationale: str,
        suggested_target: str,
        affected_paths: List[str],
    ) -> RetentionProposal:
        pid = stable_id("|".join([area, priority, action_type, title, suggested_target]))
        return RetentionProposal(
            proposal_id=pid,
            area=area,
            priority=priority,
            action_type=action_type,
            title=title,
            rationale=rationale,
            suggested_target=suggested_target,
            affected_paths=affected_paths[:50],
            destructive=False,
            requires_human_approval=True,
            execute_now=False,
        )

    def generate_proposals(self, stats: Dict[str, RepoAreaStats]) -> List[RetentionProposal]:
        proposals: List[RetentionProposal] = []

        results = stats.get("results")
        if results and results.exists and results.json_count > self.limits.max_result_json_files:
            proposals.append(self._proposal(
                area="results",
                priority="high",
                action_type="archive_candidate",
                title="Archive older result JSON files by version",
                rationale=(
                    f"results/ contains {results.json_count} JSON files, above the configured "
                    f"limit of {self.limits.max_result_json_files}. This can make git status and "
                    "manual review noisy. The advisor recommends archiving by version only after review."
                ),
                suggested_target="results/archive/<version>/",
                affected_paths=results.examples,
            ))

        generated = stats.get("configs_generated")
        if generated and generated.exists and generated.dir_count > self.limits.max_generated_config_dirs:
            proposals.append(self._proposal(
                area="configs/generated",
                priority="medium",
                action_type="archive_candidate",
                title="Archive generated config folders by run/version",
                rationale=(
                    f"configs/generated/ contains {generated.dir_count} directories, above the configured "
                    f"limit of {self.limits.max_generated_config_dirs}. Generated stability probe configs "
                    "are useful history, but they should not be mixed with active configs forever."
                ),
                suggested_target="configs/generated/archive/<version_or_run>/",
                affected_paths=generated.examples,
            ))

        exp = stats.get("experiments_inbox")
        if exp and exp.exists and exp.file_count > self.limits.max_experiment_inbox_items:
            proposals.append(self._proposal(
                area="experiments/inbox",
                priority="medium",
                action_type="review_candidate",
                title="Review old experiment proposals before archiving",
                rationale=(
                    f"experiments/inbox/ contains {exp.file_count} files, above the configured limit of "
                    f"{self.limits.max_experiment_inbox_items}. Inbox should represent pending proposals; "
                    "old cycle proposals should be reviewed before being marked archived."
                ),
                suggested_target="experiments/archive/<version>/",
                affected_paths=exp.examples,
            ))

        memory = stats.get("memory_inbox")
        if memory and memory.exists and memory.file_count > self.limits.max_memory_inbox_items:
            proposals.append(self._proposal(
                area="memory/inbox",
                priority="high",
                action_type="human_review_candidate",
                title="Review memory inbox proposals",
                rationale=(
                    f"memory/inbox/ contains {memory.file_count} files, above the configured limit of "
                    f"{self.limits.max_memory_inbox_items}. Memory proposals must not be auto-approved. "
                    "A human should review, approve, reject, or summarize them."
                ),
                suggested_target="memory/approved or memory/rejected after human review",
                affected_paths=memory.examples,
            ))

        logs = stats.get("logs")
        if logs and logs.exists and logs.markdown_count > self.limits.max_log_markdown_files:
            proposals.append(self._proposal(
                area="logs",
                priority="low",
                action_type="curation_candidate",
                title="Curate important runtime logs into docs/logs",
                rationale=(
                    f"logs/ contains {logs.markdown_count} markdown files, above the configured limit of "
                    f"{self.limits.max_log_markdown_files}. logs/ is generated runtime output; only important "
                    "human-reviewed logs should be copied into docs/logs/."
                ),
                suggested_target="docs/logs/<curated_log_name>.md",
                affected_paths=logs.examples,
            ))

        # Always add a conservative policy proposal even when thresholds are not exceeded.
        proposals.append(self._proposal(
            area="repository",
            priority="baseline",
            action_type="policy_candidate",
            title="Keep cleanup proposal-only until approved",
            rationale=(
                "SAGE should not delete, move, rename, auto-archive, auto-approve memory, or perform git actions "
                "without human approval. Cleanup should first create a report and exact commands marked review-only."
            ),
            suggested_target="docs/versions or docs/logs for reviewed docs; archive folders only after approval",
            affected_paths=[],
        ))

        return proposals

    def run(self) -> RetentionReport:
        area_defs = {
            "results": "results",
            "configs_generated": "configs/generated",
            "experiments_inbox": "experiments/inbox",
            "memory_inbox": "memory/inbox",
            "logs": "logs",
            "docs": "docs",
        }
        stats_map = {
            name: self.scan_area(name, path)
            for name, path in area_defs.items()
        }
        proposals = self.generate_proposals(stats_map)

        selected_summary = []
        for p in proposals:
            selected_summary.append(f"[{p.priority}] {p.area}: {p.title}")

        passed = (
            len(proposals) >= 1
            and all(p.destructive is False for p in proposals)
            and all(p.execute_now is False for p in proposals)
            and all(p.requires_human_approval is True for p in proposals)
        )

        return RetentionReport(
            limits=self.limits.to_jsonable(),
            area_stats=[s.to_jsonable() for s in stats_map.values()],
            proposals=[p.to_jsonable() for p in proposals],
            selected_summary=selected_summary,
            passed=passed,
        )
