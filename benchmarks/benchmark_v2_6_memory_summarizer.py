from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sage_core.memory_summarizer import MemorySummarizer, MemorySummarizerConfig


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    workspace = Path("results/v2_6_memory_summarizer_smoke_workspace")
    if workspace.exists():
        shutil.rmtree(workspace)

    memory_root = workspace / "memory"

    write_json(memory_root / "approved" / "v2_4_cpu_language_core.json", {
        "content": "SAGE v2.4 CPU Language Core benchmark passed true. It validated CPU-only state extraction, Korean chunking, and approved memory retrieval."
    })
    write_json(memory_root / "approved" / "v2_3_policy.json", {
        "content": "SAGE v2.3 autonomy policy passed true. file_delete, git_push, and network_access are forbidden. run_reflection_loop is allowed."
    })
    write_json(memory_root / "validated" / "v2_5_consolidation.json", {
        "content": "SAGE v2.5 memory consolidation moves memory from inbox to provisional, validated, approved, or rejected with audit logs."
    })
    write_json(memory_root / "provisional" / "reflection_note.json", {
        "content": "Reflection, critic, curiosity, and planner organs create candidate memories and experiments."
    })

    config = MemorySummarizerConfig(
        memory_root=str(memory_root),
        summary_dir=str(workspace / "memory" / "summaries"),
        output_path=str(workspace / "report.json"),
        max_representative_items=4,
    )

    summarizer = MemorySummarizer(config)
    report = summarizer.run().to_jsonable()

    md_path = Path(report["summary_markdown_path"])
    json_path = Path(report["summary_json_path"])

    passed = (
        report["passed"] is True
        and report["item_count"] == 4
        and report["stage_counts"].get("approved") == 2
        and md_path.exists()
        and json_path.exists()
        and "safety" in report["topic_counts"]
        and "language_core" in report["topic_counts"]
        and report["safety_policy"]["read_only_source_memory"] is True
        and report["safety_policy"]["file_delete"] is False
    )

    result = {
        "benchmark": "SAGE-v2.6-memory-summarizer-smoke",
        "version": "v2.6",
        "report": report,
        "passed": passed,
    }

    out = Path("results/v2_6_memory_summarizer_smoke_result.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== SAGE v2.6 Memory Summarizer Smoke ===")
    print(f"item_count: {report['item_count']}")
    print(f"stage_counts: {report['stage_counts']}")
    print(f"summary_markdown_path: {report['summary_markdown_path']}")
    print(f"passed: {passed}")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
