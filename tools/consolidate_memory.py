from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sage_runtime.memory_consolidation_runtime import (
    MemoryConsolidationRuntime,
    MemoryConsolidationRuntimeConfig,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="SAGE v2.5 memory consolidation organ")
    parser.add_argument("--config", default="configs/memory_consolidation_runtime.json")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    runtime = MemoryConsolidationRuntime(MemoryConsolidationRuntimeConfig.load(args.config))
    report = runtime.run_once()

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    print("=== SAGE v2.5 Memory Consolidation Organ ===")
    print(f"mode: {report.get('mode')}")
    print(f"passed: {report.get('passed')}")
    print(f"output: {report.get('output_path')}")
    print()
    print("counts_before:", report.get("counts_before"))
    print("counts_after :", report.get("counts_after"))
    print()
    for line in report.get("selected_summary", []):
        print("- " + line)

    print()
    print("Note: This organ may move low-risk memory candidates automatically.")
    print("It never deletes files, never runs git/network/shell actions, and writes an audit log.")


if __name__ == "__main__":
    main()
