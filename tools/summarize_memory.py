from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sage_runtime.memory_summarizer_runtime import (
    MemorySummarizerRuntime,
    MemorySummarizerRuntimeConfig,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="SAGE v2.6 memory summarizer")
    parser.add_argument("--config", default="configs/memory_summarizer_runtime.json")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    runtime = MemorySummarizerRuntime(MemorySummarizerRuntimeConfig.load(args.config))
    report = runtime.run_once()

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    print("=== SAGE v2.6 Memory Summarizer ===")
    print(f"mode: {report.get('mode')}")
    print(f"item_count: {report.get('item_count')}")
    print(f"stage_counts: {report.get('stage_counts')}")
    print(f"summary_markdown_path: {report.get('summary_markdown_path')}")
    print(f"summary_json_path: {report.get('summary_json_path')}")
    print(f"passed: {report.get('passed')}")
    print()
    print("Top topics:")
    for topic, count in report.get("topic_counts", {}).items():
        print(f"- {topic}: {count}")
    print()
    print("This tool is read-only for source memory and writes derived summaries only.")


if __name__ == "__main__":
    main()
