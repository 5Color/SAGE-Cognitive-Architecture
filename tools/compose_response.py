from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sage_core.cpu_language_core import CPULanguageCore
from sage_core.response_composer import ResponseComposer


def main() -> None:
    parser = argparse.ArgumentParser(description="SAGE v2.10 response composer")
    parser.add_argument("--text", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    core = CPULanguageCore(memory_root=Path("memory"))
    core_result = core.run(args.text, retrieve_memory=True).to_jsonable()

    context_bundle = {}
    try:
        from sage_core.memory_context_manager import MemoryContextConfig, MemoryContextManager
        context_bundle = MemoryContextManager(MemoryContextConfig()).build_context(args.text).to_jsonable()
    except Exception:
        context_bundle = {}

    composer = ResponseComposer()
    result = composer.compose(args.text, core_result, context_bundle).to_jsonable()

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print("=== SAGE v2.10 Response Composer ===")
        print(result["response"])
        print()
        print(f"intent: {result['intent']}")
        print(f"concept: {result['concept_key']}")
        print(f"plan: {result['sentence_plan']}")
        print(f"passed: {result['passed']}")


if __name__ == "__main__":
    main()
