from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sage_runtime.emergent_reflection_loop import EmergentReflectionConfig, EmergentReflectionLoop


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SAGE v2.0 Emergent Reflection Loop.")
    parser.add_argument("--config", default="configs/emergent_reflection.json")
    args = parser.parse_args()

    config = EmergentReflectionConfig.load(args.config)
    loop = EmergentReflectionLoop(config)
    result = loop.run_once()

    print("=== SAGE v2.0 Emergent Reflection Loop ===")
    print(f"result: {config.result_path}")
    print(f"log: {config.reflection_log_path}")
    print(f"selected_organ: {result.get('selected', {}).get('organ')}")
    print(f"proposal: {result.get('selected', {}).get('proposal')}")
    print()
    print(json.dumps(result.get("emergence_metrics", {}), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
