from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sage_core.local_chat_loop import LocalChatLoop, LocalChatLoopConfig


def load_config(path: Path) -> LocalChatLoopConfig:
    if not path.exists():
        return LocalChatLoopConfig()

    data: Dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    allowed = set(LocalChatLoopConfig.__dataclass_fields__.keys())
    filtered = {k: v for k, v in data.items() if k in allowed}
    return LocalChatLoopConfig(**filtered)


def print_command_result(result: Dict[str, Any]) -> None:
    print(json.dumps(result, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="SAGE local chat loop")
    parser.add_argument("--config", default="configs/local_chat_loop.json")
    parser.add_argument("--text", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    loop = LocalChatLoop(config)

    if args.text:
        turn = loop.chat_once(args.text)
        if args.json:
            print(json.dumps(turn.to_jsonable(), indent=2, ensure_ascii=False))
        else:
            print("=== SAGE v2.10.1 Local Chat Loop ===")
            print(turn.response)
            print()
            print(f"turn_id: {turn.turn_id}")
            print(f"memory_candidate_path: {turn.memory_candidate_path}")
        return

    print("=== SAGE v2.10.1 Local Chat Loop ===")
    print("Commands: /status, /summary, /context <query>, /consolidate, /summarize, /help, /exit")
    print()

    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue

        if user_input == "/exit":
            break

        if user_input.startswith("/"):
            result = loop.run_command(user_input)
            print()
            print("sage>")
            print_command_result(result)
            print()
            continue

        turn = loop.chat_once(user_input)
        print()
        print("sage>")
        print(turn.response)
        print()


if __name__ == "__main__":
    main()

