from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI


SYSTEM_PROMPT = """\
You are the Teacher model for the SAGE project.

SAGE is not AGI.
SAGE is an AGI-oriented controlled cognitive architecture prototype.
Its current architecture includes:
- CPU Language Core
- Memory Consolidation
- Memory Summarizer
- Memory Context Manager
- Response Composer
- Weighted Learning Memory
- Raw Archive
- Training Dataset Builder

Generate one supervised training example for SAGE.
Do not include hidden chain-of-thought.
Do not output analysis prose outside the schema.
The target_response must be natural Korean.
The sentence_plan must be a visible high-level plan, not private reasoning.
Respect safety: no network actions, no shell execution, no git, no file deletion.
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_schema(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def make_user_prompt(seed: dict[str, Any]) -> str:
    return f"""\
Create a SAGE training example.

Seed category: {seed.get("category")}
Seed target_concept: {seed.get("target_concept")}
User input: {seed.get("question")}

Requirements:
- Use Korean target_response.
- Fit the JSON schema exactly.
- The response should train SAGE to produce state-aware, memory-aware answers.
- If the input is explicit feedback, set intent to learning_feedback and memory_type to weighted_learning_candidate.
- If the input asks what something is, set intent to definition.
- If the input is short/noisy, set intent to smalltalk and ask for clarification.
"""


def extract_content(completion: Any) -> str:
    if hasattr(completion, "choices"):
        return completion.choices[0].message.content
    raw = completion.model_dump() if hasattr(completion, "model_dump") else completion
    return raw["choices"][0]["message"]["content"]


async def generate_one(
    client: AsyncOpenAI,
    schema: dict[str, Any],
    seed: dict[str, Any],
    model: str,
    temperature: float,
    max_retries: int,
    require_parameters: bool,
) -> dict[str, Any]:
    user_prompt = make_user_prompt(seed)
    last_error = None

    extra_body: dict[str, Any] = {}
    if require_parameters:
        # Ask OpenRouter to only route to providers that support required parameters.
        extra_body["provider"] = {
            "require_parameters": True
        }

    for attempt in range(max_retries + 1):
        try:
            completion = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema["name"],
                        "strict": True,
                        "schema": schema["schema"],
                    },
                },
                extra_body=extra_body or None,
            )
            content = extract_content(completion)
            obj = json.loads(content)
            obj["_meta"] = {
                "created_at": utc_now(),
                "teacher_model": model,
                "seed": seed,
                "generator": "sage_v2.13.1_openrouter_teacher_generator",
                "api": "openrouter_chat_completions",
            }
            return obj
        except Exception as exc:
            last_error = exc
            await asyncio.sleep(0.8 * (attempt + 1))

    raise RuntimeError(f"failed after retries: {last_error}")


async def run(args: argparse.Namespace) -> None:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set. In PowerShell: $env:OPENROUTER_API_KEY='sk-or-...'")

    random.seed(args.seed_random)

    seed_rows = load_jsonl(Path(args.seeds))
    if args.shuffle:
        random.shuffle(seed_rows)
    seed_rows = seed_rows[: args.limit]

    schema = load_schema(Path(args.schema))

    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_headers={
            "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost"),
            "X-OpenRouter-Title": os.getenv("OPENROUTER_APP_TITLE", "SAGE Teacher Dataset Generator"),
        },
    )

    out_path = Path(args.out)
    err_path = Path(args.errors)
    raw_path = Path(args.raw)

    sem = asyncio.Semaphore(args.concurrency)
    written = 0
    failed = 0

    async def worker(seed: dict[str, Any], idx: int) -> None:
        nonlocal written, failed
        async with sem:
            try:
                obj = await generate_one(
                    client=client,
                    schema=schema,
                    seed=seed,
                    model=args.model,
                    temperature=args.temperature,
                    max_retries=args.max_retries,
                    require_parameters=args.require_parameters,
                )
                append_jsonl(out_path, obj)
                append_jsonl(raw_path, {"created_at": utc_now(), "seed": seed, "object": obj})
                written += 1
                print(f"[ok {idx}/{len(seed_rows)}] {seed.get('question')[:50]}")
            except Exception as exc:
                failed += 1
                append_jsonl(err_path, {
                    "created_at": utc_now(),
                    "seed": seed,
                    "error": repr(exc),
                })
                print(f"[fail {idx}/{len(seed_rows)}] {repr(exc)}")

    if args.dry_run:
        print("=== DRY RUN ===")
        print(f"api: OpenRouter Chat Completions")
        print(f"model: {args.model}")
        print(f"seeds: {len(seed_rows)}")
        print(f"out: {out_path}")
        print(f"require_parameters: {args.require_parameters}")
        print("first prompt:")
        print(make_user_prompt(seed_rows[0]))
        return

    start = time.time()
    tasks = [worker(seed, i + 1) for i, seed in enumerate(seed_rows)]
    await asyncio.gather(*tasks)

    print("=== SAGE v2.13.1 OpenRouter Teacher Dataset Generation ===")
    print(f"written: {written}")
    print(f"failed: {failed}")
    print(f"out: {out_path}")
    print(f"errors: {err_path}")
    print(f"elapsed_sec: {round(time.time() - start, 2)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SAGE teacher dataset examples with OpenRouter API.")
    parser.add_argument("--model", default=os.getenv("SAGE_TEACHER_MODEL", "openai/gpt-4o-mini"))
    parser.add_argument("--seeds", default="data/seeds/seed_questions.jsonl")
    parser.add_argument("--schema", default="schemas/sage_teacher_example.schema.json")
    parser.add_argument("--out", default="data/teacher/sage_teacher_examples.jsonl")
    parser.add_argument("--raw", default="data/teacher/raw_teacher_outputs.jsonl")
    parser.add_argument("--errors", default="data/teacher/errors.jsonl")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--seed-random", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--require-parameters", action="store_true", default=True)
    args = parser.parse_args()

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
