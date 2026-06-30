from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sage_runtime.memory_review_runtime import MemoryReviewRuntime, MemoryReviewRuntimeConfig


def print_candidate_table(report: dict) -> None:
    candidates = report.get("candidates", [])
    print(f"inbox_count: {report.get('inbox_count')}")
    print(f"approved_count: {report.get('approved_count')}")
    print(f"rejected_count: {report.get('rejected_count')}")
    print()

    if not candidates:
        print("No memory candidates in memory/inbox.")
        return

    for idx, c in enumerate(candidates, start=1):
        preview = c.get("preview") or []
        short = " | ".join(preview[:2])
        if len(short) > 240:
            short = short[:237] + "..."
        print(f"[{idx}] id={c.get('candidate_id')} file={c.get('filename')}")
        print(f"    size={c.get('size_bytes')} bytes")
        print(f"    preview={short}")
        if c.get("load_error"):
            print(f"    load_error={c.get('load_error')}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="SAGE v2.1 memory review tool")
    parser.add_argument("command", choices=["list", "show", "approve", "reject", "report"])
    parser.add_argument("--config", default="configs/memory_review_tool.json")
    parser.add_argument("--id", default=None, help="Candidate id or unique prefix from list command.")
    parser.add_argument("--reason", default="", help="Human review reason.")
    parser.add_argument("--confirm", action="store_true", help="Required for approve/reject moves.")
    args = parser.parse_args()

    config = MemoryReviewRuntimeConfig.load(args.config)
    runtime = MemoryReviewRuntime(config)

    if args.command in {"show", "approve", "reject"} and not args.id:
        raise SystemExit("--id is required for show/approve/reject")

    if args.command == "list":
        report = runtime.list()
        print("=== SAGE v2.1 Memory Review - List ===")
        print_candidate_table(report)
        print(f"report: {report.get('output_path')}")

    elif args.command == "report":
        report = runtime.list()
        print("=== SAGE v2.1 Memory Review - Report ===")
        print(json.dumps({
            "inbox_count": report.get("inbox_count"),
            "approved_count": report.get("approved_count"),
            "rejected_count": report.get("rejected_count"),
            "safety_policy": report.get("safety_policy"),
            "output_path": report.get("output_path"),
        }, indent=2, ensure_ascii=False))

    elif args.command == "show":
        payload = runtime.show(args.id)
        print("=== SAGE v2.1 Memory Review - Show ===")
        print(json.dumps(payload, indent=2, ensure_ascii=False))

    elif args.command in {"approve", "reject"}:
        report = runtime.decide(
            candidate_id=args.id,
            action=args.command,
            reason=args.reason,
            confirm=args.confirm,
        )
        print(f"=== SAGE v2.1 Memory Review - {args.command.upper()} ===")
        print(json.dumps({
            "last_decision": report.get("last_decision"),
            "inbox_count": report.get("inbox_count"),
            "approved_count": report.get("approved_count"),
            "rejected_count": report.get("rejected_count"),
            "output_path": report.get("output_path"),
        }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
