from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sage_core.memory_consolidation import ConsolidationConfig, MemoryConsolidationOrgan


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    workspace = Path("results/v2_5_memory_consolidation_smoke_workspace")
    if workspace.exists():
        shutil.rmtree(workspace)

    memory_root = workspace / "memory"
    inbox = memory_root / "inbox"
    approved = memory_root / "approved"
    inbox.mkdir(parents=True, exist_ok=True)
    approved.mkdir(parents=True, exist_ok=True)

    write_json(
        approved / "approved_emergence_caution.json",
        {
            "content": "SAGE should not confuse random variation with true emergence. Safety policy and validation are required."
        },
    )

    write_json(
        inbox / "good_result_memory.json",
        {
            "content": "SAGE v2.4 CPU Language Core benchmark passed true. It validated CPU-only state extraction, Korean chunking, and approved memory retrieval."
        },
    )

    write_json(
        inbox / "policy_memory.json",
        {
            "content": "SAGE v2.3 autonomy policy result passed true. run_reflection_loop is allowed, file_delete and git_push are forbidden, and human approval is required for risky memory approval."
        },
    )

    write_json(
        inbox / "duplicate_memory.json",
        {
            "content": "SAGE should not confuse random variation with true emergence. Safety policy and validation are required."
        },
    )

    write_json(
        inbox / "risky_memory.json",
        {
            "content": "Maybe SAGE should enable network_access, git_push, file_delete, and unbounded full autonomy automatically."
        },
    )

    config = ConsolidationConfig(
        memory_root=str(memory_root),
        output_path=str(workspace / "report.json"),
        audit_log_path=str(workspace / "memory" / "consolidation_log.jsonl"),
        auto_move_to_provisional=True,
        auto_promote_to_validated=True,
        auto_approve_strict=True,
        auto_reject_duplicates=True,
    )

    organ = MemoryConsolidationOrgan(config)
    report = organ.run().to_jsonable()

    actions = report["actions"]
    approved_actions = [a for a in actions if a["action"] == "move_to_approved" and a["executed"]]
    duplicate_actions = [a for a in actions if a["action"] == "move_to_rejected_duplicate" and a["executed"]]
    risky_assessments = [a for a in report["assessments"] if "risk_flags_present" in a["reasons"]]

    passed = (
        report["passed"] is True
        and report["counts_before"]["inbox"] == 4
        and report["counts_after"]["inbox"] == 0
        and len(approved_actions) >= 1
        and len(duplicate_actions) == 1
        and len(risky_assessments) == 1
        and all("file_delete" not in a["action"] for a in actions)
        and report["safety_policy"]["file_delete"] is False
        and report["safety_policy"]["bounded_auto_approve"] is True
    )

    result = {
        "benchmark": "SAGE-v2.5-memory-consolidation-organ-smoke",
        "version": "v2.5",
        "report": report,
        "passed": passed,
    }

    out = Path("results/v2_5_memory_consolidation_smoke_result.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== SAGE v2.5 Memory Consolidation Organ Smoke ===")
    print(f"counts_before: {report['counts_before']}")
    print(f"counts_after : {report['counts_after']}")
    print(f"approved_actions: {len(approved_actions)}")
    print(f"duplicate_actions: {len(duplicate_actions)}")
    print(f"risky_assessments: {len(risky_assessments)}")
    print(f"passed: {passed}")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
