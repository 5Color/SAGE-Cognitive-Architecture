from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sage_runtime.sage_node import RuntimeConfig, SAGERuntimeNode

def main() -> None:
    config = RuntimeConfig(
        node_name="sage-runtime-node-smoke",
        state_path="runtime_state/smoke_state.json",
        registry_path="registry/organ_registry.json",
        memory_root="memory",
        log_path="logs/v1_9_runtime_smoke_reflection.md",
        tick_seconds=0.0,
        max_ticks=3,
        mode="safe_idle",
    )
    node = SAGERuntimeNode(config)
    outputs = node.run()

    state_path = Path(config.state_path)
    log_path = Path(config.log_path)
    inbox_count = node.memory.count_inbox()

    result = {
        "benchmark": "SAGE-v1.9-runtime-node-smoke",
        "version": "v1.9",
        "ticks_requested": config.max_ticks,
        "ticks_completed": len(outputs),
        "state_exists": state_path.exists(),
        "log_exists": log_path.exists(),
        "memory_inbox_count": inbox_count,
        "last_output": outputs[-1] if outputs else None,
        "safety_policy": {
            "network_actions": False,
            "shell_actions": False,
            "auto_delete_organs": False,
            "auto_disable_organs": False,
            "memory_approval_required": True,
        },
        "passed": (
            len(outputs) == config.max_ticks
            and state_path.exists()
            and log_path.exists()
            and inbox_count >= 1
        ),
    }

    out = Path("results/v1_9_runtime_node_smoke.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== SAGE v1.9 Runtime Node Smoke ===")
    print(f"ticks_completed: {result['ticks_completed']}")
    print(f"state_exists: {result['state_exists']}")
    print(f"log_exists: {result['log_exists']}")
    print(f"memory_inbox_count: {result['memory_inbox_count']}")
    print(f"passed: {result['passed']}")
    print(f"saved: {out}")

if __name__ == "__main__":
    main()
