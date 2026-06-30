from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sage_runtime.memory_context_runtime import MemoryContextRuntime, MemoryContextRuntimeConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="SAGE v2.8 memory context manager")
    parser.add_argument("--config", default="configs/memory_context_runtime.json")
    parser.add_argument("--query", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    runtime = MemoryContextRuntime(MemoryContextRuntimeConfig.load(args.config))
    bundle = runtime.build(args.query)

    if args.json:
        print(json.dumps(bundle, indent=2, ensure_ascii=False))
        return

    print("=== SAGE v2.8 Memory Context Manager ===")
    print(f"query: {bundle.get('query')}")
    print(f"inferred_topics: {bundle.get('inferred_topics')}")
    print(f"summary_snippets: {len(bundle.get('selected_summary_snippets', []))}")
    print(f"memory_items: {len(bundle.get('selected_memory_items', []))}")
    print(f"context_chars: {bundle.get('stats', {}).get('context_char_count')}")
    print(f"passed: {bundle.get('passed')}")
    print(f"output: {bundle.get('output_path')}")
    print()
    print("Selected memory:")
    for item in bundle.get("selected_memory_items", [])[:5]:
        print(f"- [{item.get('stage')}] {item.get('path')} · score {item.get('score')}")


if __name__ == "__main__":
    main()
