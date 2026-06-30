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
    workspace = Path("results/v2_10_response_composer_organ_workspace")
    if workspace.exists():
        shutil.rmtree(workspace)

    memory_root = workspace / "memory"
    write_json(memory_root / "approved" / "sage_identity.json", {
        "content": "SAGE is not AGI. It is a controlled cognitive architecture prototype using memory, reflection, safety policy, CPU language core, memory context manager, and response composer."
    })
    write_json(memory_root / "validated" / "composer_note.json", {
        "content": "Response Composer Organ converts state and context bundle into Korean sentence plans."
    })

    summary_dir = memory_root / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / "latest_memory_summary.md"
    summary_path.write_text(
        "# SAGE Memory Summary\n\n"
        "## agi\n"
        "- SAGE does not claim to be AGI. It tests AGI-oriented cognitive architecture components.\n\n"
        "## response_composer\n"
        "- Response Composer turns state and selected context into generated Korean sentences.\n",
        encoding="utf-8",
    )

    config = LocalChatLoopConfig(
        memory_root=str(memory_root),
        memory_summary_path=str(summary_path),
        chat_log_dir=str(workspace / "logs" / "chat"),
        output_path=str(workspace / "results" / "v2_10_response_composer_chat_result.json"),
        write_memory_candidates=True,
        use_memory_context_manager=True,
        use_response_composer=True,
    )

    loop = LocalChatLoop(config)
    agi = loop.chat_once("AGI가 뭔지 설명해줘").to_jsonable()
    next_step = loop.chat_once("SAGE 다음단계 뭐해야해?").to_jsonable()
    greeting = loop.chat_once("ㅎㅇ ㅎㅇㅎㅇ").to_jsonable()

    agi_response = agi.get("response", "")
    next_response = next_step.get("response", "")
    greeting_response = greeting.get("response", "")

    bad_debug_phrase = "추천 출력 전략: 먼저 state JSON"

    passed = (
        agi.get("composer_used") is True
        and "인공일반지능" in agi_response
        and "현재 SAGE는 AGI라고 주장하는 단계가 아니라" in agi_response
        and bad_debug_phrase not in agi_response
        and "Response Composer" in next_response
        and "ㅎㅇ" in greeting_response
        and agi.get("memory_context_used") is True
        and Path(config.output_path).exists()
    )

    result = {
        "benchmark": "SAGE-v2.10-response-composer-organ-smoke",
        "version": "v2.10",
        "agi_response": agi_response,
        "next_step_response": next_response,
        "greeting_response": greeting_response,
        "agi_turn": agi,
        "next_turn": next_step,
        "passed": passed,
    }

    out = Path("results/v2_10_response_composer_organ_smoke_result.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== SAGE v2.10 Response Composer Organ Smoke ===")
    print(f"composer_used: {agi.get('composer_used')}")
    print(f"agi_definition_ok: {'인공일반지능' in agi_response}")
    print(f"not_debug_dump: {bad_debug_phrase not in agi_response}")
    print(f"next_step_mentions_composer: {'Response Composer' in next_response}")
    print(f"greeting_ok: {'ㅎㅇ' in greeting_response}")
    print(f"memory_context_used: {agi.get('memory_context_used')}")
    print(f"passed: {passed}")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
