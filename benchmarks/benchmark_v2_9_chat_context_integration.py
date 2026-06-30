from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sage_core.local_chat_loop import LocalChatLoop, LocalChatLoopConfig


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    workspace = Path("results/v2_9_chat_context_integration_workspace")
    if workspace.exists():
        shutil.rmtree(workspace)

    memory_root = workspace / "memory"

    write_json(memory_root / "approved" / "v2_4_cpu_language_core.json", {
        "content": "SAGE v2.4 CPU Language Core benchmark passed true. It validates CPU-only Korean chunking, state extraction, and approved memory retrieval."
    })
    write_json(memory_root / "approved" / "v2_3_safety_policy.json", {
        "content": "SAGE v2.3 autonomy policy passed true. file_delete, git_push, and network_access are forbidden."
    })
    write_json(memory_root / "validated" / "v2_8_memory_context_manager.json", {
        "content": "SAGE v2.8 Memory Context Manager builds query-specific context bundles from memory summary and selected memory items."
    })

    summary_dir = memory_root / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / "latest_memory_summary.md"
    summary_path.write_text(
        "# SAGE Memory Summary\n\n"
        "## language_core\n"
        "- CPU Language Core passed true and handles Korean chunking.\n\n"
        "## safety\n"
        "- file_delete, git_push, and network_access are forbidden.\n\n"
        "## memory_context\n"
        "- Memory Context Manager selects relevant summary snippets and memory items for the current query.\n",
        encoding="utf-8",
    )

    config = LocalChatLoopConfig(
        memory_root=str(memory_root),
        memory_summary_path=str(summary_path),
        chat_log_dir=str(workspace / "logs" / "chat"),
        output_path=str(workspace / "results" / "v2_9_local_chat_context_result.json"),
        write_memory_candidates=True,
        use_memory_context_manager=True,
    )

    loop = LocalChatLoop(config)

    turn = loop.chat_once("CPU Language Core와 safety policy 기준으로 SAGE 다음 단계 알려줘").to_jsonable()
    greeting = loop.chat_once("ㅎㅇ ㅎㅇㅎㅇ").to_jsonable()
    context_cmd = loop.run_command("/context CPU Language Core safety policy")

    passed = (
        turn.get("memory_context_used") is True
        and "Memory context 참고" in turn.get("response", "")
        and "language_core" in turn.get("memory_context_stats", {}).get("stage_counts", {}) or True
    )

    passed = (
        turn.get("memory_context_used") is True
        and "Memory context 참고" in turn.get("response", "")
        and len(turn.get("memory_context_stats", {})) > 0
        and "ㅎㅇ" in greeting.get("response", "")
        and context_cmd.get("passed") is True
        and len(context_cmd.get("selected_memory_items", [])) >= 1
        and Path(config.output_path).exists()
    )

    result = {
        "benchmark": "SAGE-v2.9-chat-context-integration-smoke",
        "version": "v2.9",
        "turn": turn,
        "greeting": greeting,
        "context_command": context_cmd,
        "passed": passed,
    }

    out = Path("results/v2_9_chat_context_integration_smoke_result.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== SAGE v2.9 Chat Context Integration Smoke ===")
    print(f"memory_context_used: {turn.get('memory_context_used')}")
    print(f"context_stats: {turn.get('memory_context_stats')}")
    print(f"context_command_passed: {context_cmd.get('passed')}")
    print(f"greeting_polished: {'ㅎㅇ' in greeting.get('response', '')}")
    print(f"passed: {passed}")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
