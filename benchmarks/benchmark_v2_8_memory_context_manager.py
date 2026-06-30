from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sage_core.memory_context_manager import MemoryContextConfig, MemoryContextManager


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    workspace = Path("results/v2_8_memory_context_manager_smoke_workspace")
    if workspace.exists():
        shutil.rmtree(workspace)

    memory_root = workspace / "memory"

    write_json(memory_root / "approved" / "v2_4_cpu_language_core.json", {
        "content": "SAGE v2.4 CPU Language Core benchmark passed true. It validates CPU-only Korean chunking, state extraction, and approved memory retrieval."
    })
    write_json(memory_root / "approved" / "v2_3_safety_policy.json", {
        "content": "SAGE v2.3 autonomy policy passed true. file_delete, git_push, and network_access are forbidden. Level 2 Safe Auto allows reflection and cleanup advisor only."
    })
    write_json(memory_root / "validated" / "v2_7_chat_loop.json", {
        "content": "SAGE v2.7 Local Chat Loop connects user input, CPU Language Core, memory summary, approved memory retrieval, and memory candidate proposal."
    })
    write_json(memory_root / "provisional" / "random_note.json", {
        "content": "A low priority note about unrelated strawberry phenotype experiments."
    })

    summary_dir = memory_root / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / "latest_memory_summary.md"
    summary_path.write_text(
        "# SAGE Memory Summary\n\n"
        "## language_core\n"
        "- CPU Language Core passed true and handles Korean chunking.\n\n"
        "## safety\n"
        "- file_delete, git_push, and network_access are forbidden under Level 2 Safe Auto.\n\n"
        "## chat_loop\n"
        "- Local Chat Loop links user input to memory candidate proposals.\n",
        encoding="utf-8",
    )

    config = MemoryContextConfig(
        memory_root=str(memory_root),
        memory_summary_path=str(summary_path),
        output_path=str(workspace / "v2_8_memory_context_bundle.json"),
        max_selected_items=4,
        max_summary_snippets=4,
    )

    manager = MemoryContextManager(config)
    bundle = manager.build_context("CPU Language Core와 safety policy 기준으로 다음 단계 알려줘").to_jsonable()

    selected_paths = " ".join(item["path"] for item in bundle["selected_memory_items"])
    selected_topics = set(bundle["inferred_topics"])

    passed = (
        bundle["passed"] is True
        and "language_core" in selected_topics
        and "safety" in selected_topics
        and "v2_4_cpu_language_core" in selected_paths
        and "v2_3_safety_policy" in selected_paths
        and bundle["safety_policy"]["read_only_memory"] is True
        and bundle["safety_policy"]["source_memory_delete"] is False
        and Path(config.output_path).exists()
    )

    result = {
        "benchmark": "SAGE-v2.8-memory-context-manager-smoke",
        "version": "v2.8",
        "bundle": bundle,
        "selected_paths": selected_paths,
        "passed": passed,
    }

    out = Path("results/v2_8_memory_context_manager_smoke_result.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== SAGE v2.8 Memory Context Manager Smoke ===")
    print(f"inferred_topics: {bundle['inferred_topics']}")
    print(f"summary_snippets: {len(bundle['selected_summary_snippets'])}")
    print(f"memory_items: {len(bundle['selected_memory_items'])}")
    print(f"context_chars: {bundle['stats']['context_char_count']}")
    print(f"passed: {passed}")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
