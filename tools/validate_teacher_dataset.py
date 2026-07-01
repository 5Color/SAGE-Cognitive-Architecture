#!/usr/bin/env python3
"""
SAGE Teacher Dataset Validator

Purpose
- Clean raw Teacher JSONL before training a small SAGE language organ / cpuLM.
- Reject samples that teach bad habits, especially context-dependent requests with no actual context.

Expected input schema, flexible:
- input + target_response  (SAGE teacher examples)
- question + answer
- prompt + response
- messages-like objects are not rewritten here; this validator is for flattened JSONL.

Outputs
- clean JSONL
- rejected JSONL with _validation.reasons
- validation report JSON
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

INPUT_KEYS = ("input", "question", "prompt", "instruction")
RESPONSE_KEYS = ("target_response", "response", "answer", "output", "completion")

# Context-dependent references that are unsafe unless actual target/context is included.
CONTEXT_DEPENDENT_PATTERNS = [
    r"방금\s*(말한|쓴|답변|내용)",
    r"이\s*내용",
    r"이\s*문장",
    r"위\s*(내용|문장|답변)",
    r"앞의\s*(내용|문장|답변)",
    r"아까\s*(말한|쓴|답변|내용)",
]

# Requests that require an explicit target text.
TARGET_REQUIRED_PATTERNS = [
    r"요약해\s*줘",
    r"정리해\s*줘",
    r"바꿔\s*줘",
    r"고쳐\s*줘",
    r"수정해\s*줘",
    r"다듬어\s*줘",
    r"풀어\s*써\s*줘",
    r"한\s*줄로",
]

# If any of these is present, the input probably includes an actual target.
CONTEXT_MARKERS = [":", "：", "\n", "\"", "'", "「", "」", "『", "』", "다음 문장", "다음 글", "다음 답변"]

BAD_META_RESPONSE_PATTERNS = [
    r"요약\s*요청입니다",
    r"요약\s*요청\s*입니다",
    r"요약해\s*달라는\s*요청",
    r"정리해\s*달라는\s*요청",
    r"바꿔\s*달라는\s*요청",
    r"사용자가\s*(요약|정리|수정|변경)을\s*요청했다",
    r"질문은\s*.*요청입니다",
    r"입력은\s*.*요청입니다",
]

IDENTITY_CONTAMINATION_PATTERNS = [
    r"저는\s*(ChatGPT|챗GPT|GPT|OpenAI)",
    r"나는\s*(ChatGPT|챗GPT|GPT|OpenAI)",
    r"OpenAI에서\s*개발",
    r"ChatGPT로서",
    r"GPT로서",
]

SAGE_OVERCLAIM_PATTERNS = [
    r"SAGE는\s*AGI입니다",
    r"SAGE는\s*완성된\s*AGI",
    r"SAGE는\s*인간처럼\s*(의식|감정|느낌)을",
    r"SAGE는\s*모든\s*문제를\s*해결",
    r"AGI를\s*완성",
]

ODD_KOREAN_PATTERNS = [
    r"[가-힣]+을\s*설명해줘\s*짧게\s*말해줘",
    r"설명\s*설명해줘",
    r"되묻기을",
    r"말투을",
    r"형식이\s*있으면\s*그\s*형식에\s*맞추어\s*답한다$",  # teacher-like meta sentence as direct response
]

MIN_RESPONSE_CHARS_DEFAULT = 8
MAX_RESPONSE_CHARS_DEFAULT = 1200


def read_jsonl(path: Path) -> Iterable[Tuple[int, Optional[Dict[str, Any]], Optional[str]]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            raw = line.rstrip("\n")
            if not raw.strip():
                yield line_no, None, "empty_line"
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as e:
                yield line_no, None, f"json_parse_error:{e.msg}"
                continue
            if not isinstance(obj, dict):
                yield line_no, None, "not_json_object"
                continue
            yield line_no, obj, None


def first_str(obj: Dict[str, Any], keys: Tuple[str, ...]) -> Tuple[Optional[str], Optional[str]]:
    for key in keys:
        value = obj.get(key)
        if isinstance(value, str):
            return key, value.strip()
    return None, None


def contains_re(patterns: List[str], text: str) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def has_context_marker(text: str) -> bool:
    if any(m in text for m in CONTEXT_MARKERS):
        # "다음 문장" alone is not enough unless there is a separator or quoted/text-like payload.
        if "다음 문장" in text or "다음 글" in text or "다음 답변" in text:
            return (":" in text or "：" in text or "\n" in text or "\"" in text or "'" in text or "「" in text or "『" in text)
        return True
    return False


def selected_context_nonempty(obj: Dict[str, Any]) -> bool:
    ctx = obj.get("selected_context")
    if isinstance(ctx, list) and len(ctx) > 0:
        return True
    if isinstance(ctx, str) and ctx.strip():
        return True
    state = obj.get("state")
    if isinstance(state, dict):
        for k in ("context", "source", "given_text"):
            v = state.get(k)
            if isinstance(v, str) and v.strip():
                return True
            if isinstance(v, list) and v:
                return True
    return False


def fingerprint(text: str) -> str:
    compact = re.sub(r"\s+", " ", text.strip())
    return hashlib.sha256(compact.encode("utf-8")).hexdigest()[:16]


def validate_one(
    obj: Dict[str, Any],
    line_no: int,
    seen_inputs: set[str],
    min_response_chars: int,
    max_response_chars: int,
    strict_context: bool,
) -> Tuple[bool, List[str], Dict[str, Any]]:
    reasons: List[str] = []
    input_key, inp = first_str(obj, INPUT_KEYS)
    response_key, resp = first_str(obj, RESPONSE_KEYS)

    if input_key is None or inp is None or not inp:
        reasons.append("missing_or_empty_input")
        inp = ""
    if response_key is None or resp is None or not resp:
        reasons.append("missing_or_empty_target_response")
        resp = ""

    if inp:
        fp = fingerprint(inp)
        if fp in seen_inputs:
            reasons.append("duplicate_input")
        else:
            seen_inputs.add(fp)

    if resp and len(resp) < min_response_chars:
        reasons.append("target_response_too_short")
    if resp and len(resp) > max_response_chars:
        reasons.append("target_response_too_long")

    if strict_context and inp:
        context_dependent = contains_re(CONTEXT_DEPENDENT_PATTERNS, inp)
        target_required = contains_re(TARGET_REQUIRED_PATTERNS, inp)
        has_explicit_context = has_context_marker(inp) or selected_context_nonempty(obj)

        # Strong reject: context-dependent edit/summary with no actual target.
        if context_dependent and target_required and not has_explicit_context:
            reasons.append("context_dependent_request_without_context")

        # "다음 문장/글/답변" requires colon/newline/quote payload.
        if re.search(r"다음\s*(문장|글|답변)", inp) and not has_context_marker(inp):
            reasons.append("target_required_but_missing_text")

    if inp and contains_re(ODD_KOREAN_PATTERNS, inp):
        reasons.append("odd_korean_pattern_in_input")
    if resp and contains_re(ODD_KOREAN_PATTERNS, resp):
        reasons.append("odd_korean_pattern_in_response")

    if resp and contains_re(BAD_META_RESPONSE_PATTERNS, resp):
        reasons.append("meta_response_instead_of_task_answer")
    if resp and contains_re(IDENTITY_CONTAMINATION_PATTERNS, resp):
        reasons.append("teacher_identity_contamination")
    if resp and contains_re(SAGE_OVERCLAIM_PATTERNS, resp):
        reasons.append("sage_agi_overclaim")

    # Detect suspicious English-only target responses for Korean-first seed data.
    if resp:
        hangul_count = len(re.findall(r"[가-힣]", resp))
        alpha_count = len(re.findall(r"[A-Za-z]", resp))
        if alpha_count > 40 and hangul_count < 10:
            reasons.append("mostly_english_response")

    annotated = dict(obj)
    annotated["_validation"] = {
        "line_no": line_no,
        "accepted": not reasons,
        "reasons": reasons,
        "input_key": input_key,
        "response_key": response_key,
        "input_fingerprint": fingerprint(inp) if inp else None,
    }
    return not reasons, reasons, annotated


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate and clean SAGE teacher dataset JSONL.")
    ap.add_argument("--input", required=True, help="Input teacher JSONL path")
    ap.add_argument("--out", required=True, help="Clean output JSONL path")
    ap.add_argument("--rejected", required=True, help="Rejected output JSONL path")
    ap.add_argument("--report", required=True, help="Validation report JSON path")
    ap.add_argument("--min-response-chars", type=int, default=MIN_RESPONSE_CHARS_DEFAULT)
    ap.add_argument("--max-response-chars", type=int, default=MAX_RESPONSE_CHARS_DEFAULT)
    ap.add_argument("--no-strict-context", action="store_true", help="Do not reject context-dependent requests without explicit context")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.out)
    rejected_path = Path(args.rejected)
    report_path = Path(args.report)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    rejected_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    seen_inputs: set[str] = set()
    total = clean = rejected = parse_errors = 0
    reason_counts: collections.Counter[str] = collections.Counter()
    intent_counts: collections.Counter[str] = collections.Counter()
    clean_intent_counts: collections.Counter[str] = collections.Counter()
    rejected_intent_counts: collections.Counter[str] = collections.Counter()

    rejected_examples: List[Dict[str, Any]] = []

    with out_path.open("w", encoding="utf-8") as fout, rejected_path.open("w", encoding="utf-8") as frej:
        for line_no, obj, load_error in read_jsonl(in_path):
            total += 1
            if load_error is not None:
                parse_errors += 1
                reason_counts[load_error] += 1
                bad = {"_validation": {"line_no": line_no, "accepted": False, "reasons": [load_error]}}
                frej.write(json.dumps(bad, ensure_ascii=False) + "\n")
                rejected += 1
                continue

            assert obj is not None
            intent = str(obj.get("intent") or obj.get("concept") or obj.get("_meta", {}).get("seed", {}).get("category") or "unknown")
            intent_counts[intent] += 1

            ok, reasons, annotated = validate_one(
                obj=obj,
                line_no=line_no,
                seen_inputs=seen_inputs,
                min_response_chars=args.min_response_chars,
                max_response_chars=args.max_response_chars,
                strict_context=not args.no_strict_context,
            )

            if ok:
                # Keep clean dataset unchanged, except no validation metadata by default.
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                clean += 1
                clean_intent_counts[intent] += 1
            else:
                for r in reasons:
                    reason_counts[r] += 1
                frej.write(json.dumps(annotated, ensure_ascii=False) + "\n")
                rejected += 1
                rejected_intent_counts[intent] += 1
                if len(rejected_examples) < 30:
                    rejected_examples.append({
                        "line_no": line_no,
                        "input": first_str(obj, INPUT_KEYS)[1],
                        "target_response": first_str(obj, RESPONSE_KEYS)[1],
                        "reasons": reasons,
                    })

    report = {
        "input": str(in_path),
        "clean_output": str(out_path),
        "rejected_output": str(rejected_path),
        "total_lines": total,
        "clean": clean,
        "rejected": rejected,
        "parse_errors": parse_errors,
        "clean_ratio": round(clean / total, 6) if total else 0.0,
        "reason_counts": dict(reason_counts.most_common()),
        "intent_counts": dict(intent_counts.most_common()),
        "clean_intent_counts": dict(clean_intent_counts.most_common()),
        "rejected_intent_counts": dict(rejected_intent_counts.most_common()),
        "rejected_examples_first_30": rejected_examples,
        "settings": {
            "min_response_chars": args.min_response_chars,
            "max_response_chars": args.max_response_chars,
            "strict_context": not args.no_strict_context,
        },
    }

    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== SAGE Teacher Dataset Validation ===")
    print(f"input:    {in_path}")
    print(f"clean:    {clean}")
    print(f"rejected: {rejected}")
    print(f"total:    {total}")
    print(f"ratio:    {report['clean_ratio']}")
    print(f"out:      {out_path}")
    print(f"rejected: {rejected_path}")
    print(f"report:   {report_path}")
    if reason_counts:
        print("\nTop rejection reasons:")
        for reason, count in reason_counts.most_common(10):
            print(f"- {reason}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
