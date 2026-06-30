from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sage_core.cpu_language_core import CPULanguageCore


def main() -> None:
    core = CPULanguageCore(memory_root="memory")

    cases = [
        "다음 단계 ㄱㄱ. SAGE 현재 진행상황을 요약하고 CPU 언어모델 구조를 검증해줘.",
        "SAGE는 AGI가 아니라 통제 가능한 cognitive architecture prototype이라고 정리하자.",
        "CPU에서 한국어 chunk tokenizer와 memory retriever를 연결할 수 있을까?",
        "file_delete 같은 위험 행동은 금지하고 reflection loop는 허용해야 한다.",
    ]

    outputs = []
    started = time.perf_counter()

    for text in cases:
        t0 = time.perf_counter()
        result = core.run(text).to_jsonable()
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)
        outputs.append({
            "input": text,
            "elapsed_ms": elapsed_ms,
            "passed": result.get("passed"),
            "intent": result.get("state", {}).get("intent"),
            "topics": result.get("state", {}).get("topics"),
            "chunk_count": len(result.get("analysis", {}).get("chunk_tokens", [])),
            "word_count": len(result.get("analysis", {}).get("word_tokens", [])),
            "memory_hit_count": len(result.get("memory_hits", [])),
            "response_preview": result.get("response", "")[:300],
        })

    total_ms = round((time.perf_counter() - started) * 1000, 3)

    passed = (
        len(outputs) == len(cases)
        and all(o["passed"] for o in outputs)
        and any(o["intent"] in {"request_next_step", "summary_request"} for o in outputs)
        and any("cpu_language_model" in o["topics"] for o in outputs)
        and total_ms < 5000
    )

    report = {
        "benchmark": "SAGE-v2.4-cpu-language-core-architecture-probe",
        "version": "v2.4",
        "case_count": len(cases),
        "total_elapsed_ms": total_ms,
        "outputs": outputs,
        "safety_policy": {
            "cpu_only": True,
            "network_actions": False,
            "external_model_required": False,
            "memory_read_only": True,
            "file_delete": False,
            "core_code_auto_modify": False,
        },
        "passed": passed,
    }

    out = Path("results/v2_4_cpu_language_core_benchmark.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== SAGE v2.4 CPU Language Core Benchmark ===")
    print(f"case_count: {report['case_count']}")
    print(f"total_elapsed_ms: {report['total_elapsed_ms']}")
    print(f"passed: {report['passed']}")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
