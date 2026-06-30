from __future__ import annotations
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from sage_runtime.reflection_stability_runtime import ReflectionStabilityConfig, ReflectionStabilityRuntime

def main() -> None:
    cfg = ReflectionStabilityConfig(
        base_policy_path="configs/reflection_policy_exploratory.json",
        output_path="results/v2_0_3_reflection_stability_probe.json",
        output_dir="results/v2_0_3_stability_probe",
        variant_policy_dir="configs/generated/stability_probe",
        target_organ="curiosity_organ",
        novelty_deltas=[-0.02, 0.0, 0.02],
        risk_deltas=[0.0, 0.02],
        min_target_rate=0.80,
    )
    detail = ReflectionStabilityRuntime(cfg).run_once()
    result = {
        "benchmark": "SAGE-v2.0.3-reflection-stability-probe-smoke",
        "version": "v2.0.3",
        "target_organ": detail.get("target_organ"),
        "variant_count": detail.get("variant_count"),
        "selected_counts": detail.get("selected_counts"),
        "target_selected_rate": detail.get("target_selected_rate"),
        "min_top_second_gap": detail.get("min_top_second_gap"),
        "max_top_second_gap": detail.get("max_top_second_gap"),
        "mean_top_second_gap": detail.get("mean_top_second_gap"),
        "passed_detail": detail.get("passed"),
        "passed": (
            detail.get("passed") is True
            and detail.get("variant_count", 0) >= 6
            and detail.get("target_selected_rate", 0.0) >= 0.80
            and detail.get("safety_policy", {}).get("create_memory_proposal") is False
        ),
    }
    out = Path("results/v2_0_3_reflection_stability_probe_smoke.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("=== SAGE v2.0.3 Reflection Stability Probe Smoke ===")
    print(f"target_organ: {result['target_organ']}")
    print(f"variant_count: {result['variant_count']}")
    print(f"selected_counts: {result['selected_counts']}")
    print(f"target_selected_rate: {result['target_selected_rate']}")
    print(f"mean_top_second_gap: {result['mean_top_second_gap']}")
    print(f"passed: {result['passed']}")
    print(f"saved: {out}")
if __name__ == "__main__":
    main()
