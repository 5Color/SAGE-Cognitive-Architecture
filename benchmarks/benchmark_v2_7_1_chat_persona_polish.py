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
    workspace = Path("results/v2_7_1_chat_persona_polish_workspace")
    if workspace.exists():
        shutil.rmtree(workspace)

    memory_root = workspace / "memory"
    write_json(memory_root / "approved" / "identity_memory.json", {
        "content": "SAGE is a controlled cognitive architecture prototype with CPU Language Core, memory system, and local chat loop."
    })

    summary_dir = memory_root / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / "latest_memory_summary.md"
    summary_path.write_text(
        "# SAGE Memory Summary\n\n- language_core: CPU Language Core passed true.\n- safety: file_delete and git_push are forbidden.\n",
        encoding="utf-8",
    )

    config = LocalChatLoopConfig(
        memory_root=str(memory_root),
        memory_summary_path=str(summary_path),
        chat_log_dir=str(workspace / "logs" / "chat"),
        output_path=str(workspace / "results" / "v2_7_1_local_chat_loop_result.json"),
        write_memory_candidates=True,
    )

    loop = LocalChatLoop(config)

    greeting = loop.chat_once("안녕?").to_jsonable()
    identity = loop.chat_once("넌 누구야").to_jsonable()
    capability = loop.chat_once("뭐 할 수 있어?").to_jsonable()

    passed = (
        "로컬 대화 루프" in greeting["response"]
        and "Self-organizing Adaptive Generative Ecosystem" in identity["response"]
        and "/status" in capability["response"]
        and greeting["memory_summary_used"] is True
        and identity["memory_summary_used"] is True
    )

    result = {
        "benchmark": "SAGE-v2.7.1-chat-persona-polish-smoke",
        "version": "v2.7.1",
        "greeting_response": greeting["response"],
        "identity_response": identity["response"],
        "capability_response": capability["response"],
        "passed": passed,
    }

    out = Path("results/v2_7_1_chat_persona_polish_smoke_result.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== SAGE v2.7.1 Chat Persona Polish Smoke ===")
    print(f"greeting_ok: {'로컬 대화 루프' in greeting['response']}")
    print(f"identity_ok: {'Self-organizing Adaptive Generative Ecosystem' in identity['response']}")
    print(f"capability_ok: {'/status' in capability['response']}")
    print(f"passed: {passed}")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
