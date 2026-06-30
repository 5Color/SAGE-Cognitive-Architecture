from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from sage_runtime.reflection_stability_runtime import ReflectionStabilityConfig, ReflectionStabilityRuntime

def main() -> None:
    parser = argparse.ArgumentParser(description="Run SAGE v2.0.3 Reflection Stability Probe.")
    parser.add_argument("--config", default="configs/reflection_stability_probe.json")
    args = parser.parse_args()
    result = ReflectionStabilityRuntime(ReflectionStabilityConfig.load(args.config)).run_once()
    print("=== SAGE v2.0.3 Reflection Stability Probe ===")
    print(f"target_organ: {result.get('target_organ')}")
    print(f"variant_count: {result.get('variant_count')}")
    print(f"selected_counts: {result.get('selected_counts')}")
    print(f"target_selected_rate: {result.get('target_selected_rate')}")
    print(f"mean_top_second_gap: {result.get('mean_top_second_gap')}")
    print(f"passed: {result.get('passed')}")
    print(f"saved: {result.get('output_path')}")
    print(json.dumps({"interpretation": result.get("interpretation"), "safety_policy": result.get("safety_policy")}, indent=2, ensure_ascii=False))
if __name__ == "__main__":
    main()
