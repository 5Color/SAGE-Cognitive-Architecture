from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED = [
    "input", "intent", "concept", "state", "selected_context",
    "sentence_plan", "target_response", "memory_update",
    "safety_label", "difficulty", "tags", "quality_notes"
]


def validate_obj(obj: dict) -> list[str]:
    errors = []
    for key in REQUIRED:
        if key not in obj:
            errors.append(f"missing {key}")
    if "target_response" in obj and len(str(obj["target_response"]).strip()) < 10:
        errors.append("target_response too short")
    if "sentence_plan" in obj and not obj["sentence_plan"]:
        errors.append("empty sentence_plan")
    if "memory_update" in obj:
        mu = obj["memory_update"]
        for k in ["importance", "confidence", "reuse_value", "novelty", "contradiction_risk"]:
            v = mu.get(k)
            if not isinstance(v, (int, float)) or not (0 <= v <= 1):
                errors.append(f"bad memory_update.{k}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="data/teacher/sage_teacher_examples.jsonl")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        raise SystemExit(f"missing file: {path}")

    total = 0
    ok = 0
    bad = 0
    first_errors = []

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, start=1):
            if not line.strip():
                continue
            total += 1
            try:
                obj = json.loads(line)
            except Exception as exc:
                bad += 1
                first_errors.append((i, [f"json parse error: {exc}"]))
                continue

            errors = validate_obj(obj)
            if errors:
                bad += 1
                if len(first_errors) < 5:
                    first_errors.append((i, errors))
            else:
                ok += 1

    print("=== SAGE Teacher Dataset Validation ===")
    print(f"file: {path}")
    print(f"total: {total}")
    print(f"ok: {ok}")
    print(f"bad: {bad}")
    if first_errors:
        print("first_errors:")
        for line_no, errors in first_errors:
            print(f"- line {line_no}: {errors}")


if __name__ == "__main__":
    main()
