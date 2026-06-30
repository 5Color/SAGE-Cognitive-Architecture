from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import json

from sage_core.retention_policy import RetentionLimits, RetentionPolicyAdvisor


@dataclass
class CleanupRetentionAdvisorConfig:
    output_path: str = "results/v2_0_6_cleanup_retention_policy.json"
    inbox_path: str = "experiments/inbox/v2_0_6_cleanup_retention_policy_proposal.json"
    max_result_json_files: int = 120
    max_generated_config_dirs: int = 20
    max_experiment_inbox_items: int = 30
    max_memory_inbox_items: int = 20
    max_log_markdown_files: int = 50
    write_experiment_inbox_proposal: bool = True

    @classmethod
    def load(cls, path: str | Path) -> "CleanupRetentionAdvisorConfig":
        p = Path(path)
        if not p.exists():
            return cls()
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls(**{**cls().__dict__, **data})


class CleanupRetentionAdvisorRuntime:
    def __init__(self, config: CleanupRetentionAdvisorConfig) -> None:
        self.config = config

    def run_once(self) -> Dict[str, Any]:
        limits = RetentionLimits(
            max_result_json_files=self.config.max_result_json_files,
            max_generated_config_dirs=self.config.max_generated_config_dirs,
            max_experiment_inbox_items=self.config.max_experiment_inbox_items,
            max_memory_inbox_items=self.config.max_memory_inbox_items,
            max_log_markdown_files=self.config.max_log_markdown_files,
        )
        advisor = RetentionPolicyAdvisor(limits=limits)
        report = advisor.run().to_jsonable()

        output_path = Path(self.config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

        if self.config.write_experiment_inbox_proposal:
            inbox_path = Path(self.config.inbox_path)
            inbox_path.parent.mkdir(parents=True, exist_ok=True)
            inbox_payload = {
                "version": report["version"],
                "proposal_type": "cleanup_retention_policy",
                "created_at": report["created_at"],
                "requires_human_approval": True,
                "execute_now": False,
                "summary": report["selected_summary"],
                "report_path": str(output_path),
                "safety_policy": report["safety_policy"],
            }
            inbox_path.write_text(json.dumps(inbox_payload, indent=2, ensure_ascii=False), encoding="utf-8")
            report["inbox_path"] = str(inbox_path)

        report["output_path"] = str(output_path)
        return report
