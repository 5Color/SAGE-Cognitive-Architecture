from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sage_runtime.cleanup_retention_advisor import (
    CleanupRetentionAdvisorConfig,
    CleanupRetentionAdvisorRuntime,
)


def main() -> None:
    config = CleanupRetentionAdvisorConfig(
        output_path="results/v2_0_6_cleanup_retention_policy_smoke.json",
        inbox_path="experiments/inbox/v2_0_6_cleanup_retention_policy_smoke_proposal.json",
        max_result_json_files=1,
        max_generated_config_dirs=1,
        max_experiment_inbox_items=1,
        max_memory_inbox_items=1,
        max_log_markdown_files=1,
        write_experiment_inbox_proposal=True,
    )
    report = CleanupRetentionAdvisorRuntime(config).run_once()

    proposals = report.get("proposals", [])
    safety = report.get("safety_policy", {})
    proposal_action_types = [p.get("action_type") for p in proposals]
    destructive_flags = [p.get("destructive") for p in proposals]
    execute_flags = [p.get("execute_now") for p in proposals]

    result = {
        "benchmark": "SAGE-v2.0.6-cleanup-retention-policy-smoke",
        "version": "v2.0.6",
        "proposal_count": len(proposals),
        "proposal_action_types": proposal_action_types,
        "destructive_flags": destructive_flags,
        "execute_flags": execute_flags,
        "output_path": report.get("output_path"),
        "inbox_path": report.get("inbox_path"),
        "safety_policy": safety,
        "passed": (
            report.get("passed") is True
            and len(proposals) >= 1
            and all(flag is False for flag in destructive_flags)
            and all(flag is False for flag in execute_flags)
            and safety.get("file_delete") is False
            and safety.get("file_move") is False
            and safety.get("file_rename") is False
            and safety.get("auto_archive") is False
            and safety.get("auto_cleanup") is False
            and safety.get("auto_approve_memory") is False
            and safety.get("human_approval_required") is True
            and safety.get("proposal_only") is True
        ),
    }

    out = Path("results/v2_0_6_cleanup_retention_policy_smoke_result.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== SAGE v2.0.6 Cleanup Retention Policy Smoke ===")
    print(f"proposal_count: {result['proposal_count']}")
    print(f"action_types: {result['proposal_action_types']}")
    print(f"passed: {result['passed']}")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
