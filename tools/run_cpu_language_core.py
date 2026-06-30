from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sage_runtime.cpu_language_core_runtime import (
    CPULanguageCoreRuntime,
    CPULanguageCoreRuntimeConfig,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="SAGE v2.4 CPU Language Core")
    parser.add_argument("--config", default="configs/cpu_language_core.json")
    parser.add_argument("--text", default=None)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    if args.demo:
        text = "다음 단계 ㄱㄱ. SAGE 현재 진행상황을 요약하고 CPU 언어모델 구조를 검증해줘."
    elif args.text:
        text = args.text
    else:
        text = input("Input text: ").strip()

    runtime = CPULanguageCoreRuntime(CPULanguageCoreRuntimeConfig.load(args.config))
    result = runtime.run_once(text)

    print("=== SAGE v2.4 CPU Language Core ===")
    print(f"passed: {result.get('passed')}")
    print(f"output: {result.get('output_path')}")
    print()
    print("--- State ---")
    print(json.dumps(result.get("state", {}), indent=2, ensure_ascii=False))
    print()
    print("--- Analysis Summary ---")
    analysis = result.get("analysis", {})
    print(json.dumps({
        "char_count": analysis.get("char_count"),
        "syllable_count": len(analysis.get("syllable_tokens", [])),
        "word_count": len(analysis.get("word_tokens", [])),
        "chunk_count": len(analysis.get("chunk_tokens", [])),
        "chunk_tokens": analysis.get("chunk_tokens", []),
        "compression": analysis.get("compression", {}),
    }, indent=2, ensure_ascii=False))
    print()
    print("--- Response ---")
    print(result.get("response", ""))


if __name__ == "__main__":
    main()
