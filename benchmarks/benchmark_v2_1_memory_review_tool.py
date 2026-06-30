from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sage_runtime.memory_review_runtime import MemoryReviewRuntime, MemoryReviewRuntimeConfig


def main() -> None:
    workspace = Path("results/v2_1_memory_review_smoke_workspace")
    if workspace.exists():
        shutil.rmtree(workspace)

    memory_root = workspace / "memory"
    inbox = memory_root / "inbox"
    approved = memory_root / "approved"
    rejected = memory_root / "rejected"
    inbox.mkdir(parents=True, exist_ok=True)
    approved.mkdir(parents=True, exist_ok=True)
    rejected.mkdir(parents=True, exist_ok=True)

    sample = {
        "source": "smoke_test",
        "claim": "SAGE should review memory proposals before using them as approved memory.",
        "proposal": "Add a human-in-the-loop memory approval and rejection tool.",
        "confidence": 0.95,
        "risk": 0.05,
    }
    sample_path = inbox / "sample_memory_proposal.json"
    sample_path.write_text(json.dumps(sample, indent=2, ensure_ascii=False), encoding="utf-8")

    config = MemoryReviewRuntimeConfig(
        memory_root=str(memory_root),
        output_path="results/v2_1_memory_review_tool_smoke.json",
    )
    runtime = MemoryReviewRuntime(config)

    initial = runtime.list()
    candidates = initial.get("candidates", [])
    candidate_id = candidates[0]["candidate_id"] if candidates else ""

    shown = runtime.show(candidate_id)
    approved_report = runtime.decide(
        candidate_id=candidate_id,
        action="approve",
        reason="Smoke test approves a synthetic memory proposal.",
        confirm=True,
    )

    result = {
        "benchmark": "SAGE-v2.1-memory-review-tool-smoke",
        "version": "v2.1",
        "initial_inbox_count": initial.get("inbox_count"),
        "candidate_id": candidate_id,
        "show_has_raw": "raw" in shown,
        "final_inbox_count": approved_report.get("inbox_count"),
        "final_approved_count": approved_report.get("approved_count"),
        "final_rejected_count": approved_report.get("rejected_count"),
        "last_decision": approved_report.get("last_decision"),
        "safety_policy": approved_report.get("safety_policy"),
        "passed": (
            initial.get("inbox_count") == 1
            and bool(candidate_id)
            and "raw" in shown
            and approved_report.get("inbox_count") == 0
            and approved_report.get("approved_count") == 1
            and approved_report.get("rejected_count") == 0
            and approved_report.get("last_decision", {}).get("action") == "approve"
            and approved_report.get("last_decision", {}).get("auto_approved") is False
            and approved_report.get("safety_policy", {}).get("auto_approve_memory") is False
            and approved_report.get("safety_policy", {}).get("auto_delete_memory") is False
        ),
    }

    out = Path("results/v2_1_memory_review_tool_smoke_result.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== SAGE v2.1 Memory Review Tool Smoke ===")
    print(f"initial_inbox_count: {result['initial_inbox_count']}")
    print(f"final_inbox_count: {result['final_inbox_count']}")
    print(f"final_approved_count: {result['final_approved_count']}")
    print(f"passed: {result['passed']}")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
