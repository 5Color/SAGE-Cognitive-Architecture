from __future__ import annotations

import argparse
import json
import os
import urllib.request


def main() -> None:
    parser = argparse.ArgumentParser(description="List OpenRouter models.")
    parser.add_argument("--contains", default="")
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()

    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/models",
        headers={
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY', '')}",
            "Accept": "application/json",
        },
    )

    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    rows = data.get("data", [])
    if args.contains:
        needle = args.contains.lower()
        rows = [r for r in rows if needle in r.get("id", "").lower() or needle in r.get("name", "").lower()]

    for r in rows[: args.limit]:
        print(r.get("id"), "-", r.get("name", ""))


if __name__ == "__main__":
    main()
