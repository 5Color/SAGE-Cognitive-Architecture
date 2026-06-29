from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sage_runtime.sage_node import RuntimeConfig, SAGERuntimeNode

def main() -> None:
    parser = argparse.ArgumentParser(description="Run SAGE v1.9 Runtime Node.")
    parser.add_argument("--config", default="configs/runtime_node.json")
    parser.add_argument("--max-ticks", type=int, default=None)
    parser.add_argument("--tick-seconds", type=float, default=None)
    args = parser.parse_args()

    config = RuntimeConfig.load(args.config)
    if args.max_ticks is not None:
        config.max_ticks = args.max_ticks
    if args.tick_seconds is not None:
        config.tick_seconds = args.tick_seconds

    node = SAGERuntimeNode(config)
    outputs = node.run()

    print("=== SAGE v1.9 Runtime Node ===")
    print(f"node: {config.node_name}")
    print(f"mode: {config.mode}")
    print(f"ticks: {len(outputs)}")
    print(f"state: {config.state_path}")
    print(f"log: {config.log_path}")
    print(f"memory_root: {config.memory_root}")
    print()
    if outputs:
        print(json.dumps(outputs[-1], indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
