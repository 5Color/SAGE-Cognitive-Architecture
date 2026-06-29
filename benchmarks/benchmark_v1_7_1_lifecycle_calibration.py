from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_experiment import build_component, set_seed
from sage_core import SAGEEngine, SAGEState

DEFAULT_CONFIGS = {
    "full_evidence": "benchmarks/configs/anti_leak_sparse_full_evidence.json",
    "sparse_top1": "benchmarks/configs/anti_leak_sparse_top1.json",
    "sparse_top2": "benchmarks/configs/anti_leak_sparse_top2.json",
    "adaptive_v1_7": "benchmarks/configs/anti_leak_adaptive_compute.json",
    "calibrated_balanced": "benchmarks/configs/anti_leak_adaptive_calibrated_balanced.json",
    "calibrated_top2_friendly": "benchmarks/configs/anti_leak_adaptive_calibrated_top2_friendly.json",
    "calibrated_safe": "benchmarks/configs/anti_leak_adaptive_calibrated_safe.json",
    "oracle": "benchmarks/configs/anti_leak_sparse_oracle.json",
}

TRACKED = [
    "accuracy",
    "avg_reward",
    "task_diversity",
    "avg_organs_per_step",
    "compute_saving_vs_full",
    "sparse_efficiency_score",
    "adaptive_efficiency_score",
    "avg_confidence_gap",
    "avg_best_router_score",
]

def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0

def std(xs):
    xs = list(xs)
    if len(xs) <= 1:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))

def update_config(config: Dict[str, Any], seed: int):
    cfg = copy.deepcopy(config)
    cfg["seed"] = int(seed)
    cfg["environment"]["params"]["seed"] = int(seed)
    cfg["output_path"] = f"results/tmp_v1_7_1_{cfg.get('benchmark','run')}_seed{seed}.json"
    return cfg

def run_config(config: Mapping[str, Any], seed: int):
    cfg = update_config(dict(config), seed)
    set_seed(seed)

    organs = {}
    for spec in cfg["organs"]:
        organ = build_component(spec)
        organs[organ.name] = organ

    router = build_component(cfg["router"])
    env = build_component(cfg["environment"])
    metric = build_component(cfg["metric"])
    state = SAGEState(**dict(cfg.get("state", {})))

    engine = SAGEEngine(
        organs=organs,
        router=router,
        env=env,
        metric=metric,
        state=state,
        **dict(cfg.get("engine", {})),
    )
    result = engine.run(max_steps=int(cfg.get("max_steps", 64)))
    result["seed"] = int(seed)
    result["router"] = router.name
    return result

def summarize(runs: List[Dict[str, Any]]):
    s = {}
    for metric in TRACKED:
        vals = [float(r.get(metric, 0.0)) for r in runs]
        s[f"{metric}_mean"] = mean(vals)
        s[f"{metric}_std"] = std(vals)

    modes = sorted({m for r in runs for m in r.get("mode_usage", {}).keys()})
    s["mode_usage"] = {}
    for m in modes:
        vals = [float(r.get("mode_usage", {}).get(m, 0.0)) for r in runs]
        s["mode_usage"][m] = {"mean": mean(vals), "std": std(vals)}

    organs = sorted({o for r in runs for o in r.get("organ_lifecycle", {}).keys()})
    lifecycle_summary = {}
    for organ in organs:
        success_vals = []
        usage_vals = []
        chosen_vals = []
        recs = []
        for r in runs:
            item = r.get("organ_lifecycle", {}).get(organ, {})
            if item.get("chosen_success_rate") is not None:
                success_vals.append(float(item["chosen_success_rate"]))
            usage_vals.append(float(item.get("usage_ratio_over_steps", 0.0)))
            chosen_vals.append(float(item.get("chosen_ratio_over_steps", 0.0)))
            if item.get("recommendation"):
                recs.append(str(item["recommendation"]))

        lifecycle_summary[organ] = {
            "usage_ratio_over_steps_mean": mean(usage_vals),
            "usage_ratio_over_steps_std": std(usage_vals),
            "chosen_ratio_over_steps_mean": mean(chosen_vals),
            "chosen_success_rate_mean": mean(success_vals),
            "chosen_success_rate_std": std(success_vals),
            "recommendation_counts": dict(sorted({x: recs.count(x) for x in set(recs)}.items())),
        }

    s["organ_lifecycle_summary"] = lifecycle_summary
    return s

def line(name, s):
    eff = s.get("adaptive_efficiency_score_mean", 0.0) or s.get("sparse_efficiency_score_mean", 0.0)
    return (
        f"{name:26s} | "
        f"acc {s.get('accuracy_mean',0):.4f}±{s.get('accuracy_std',0):.4f} | "
        f"org/step {s.get('avg_organs_per_step_mean',0):.2f}±{s.get('avg_organs_per_step_std',0):.2f} | "
        f"saving {s.get('compute_saving_vs_full_mean',0):.2f}±{s.get('compute_saving_vs_full_std',0):.2f} | "
        f"eff {eff:.4f}"
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--episodes-per-family", type=int, default=80)
    parser.add_argument("--support-size", type=int, default=None)
    parser.add_argument("--out", type=str, default="results/v1_7_1_lifecycle_calibration_multiseed.json")
    args = parser.parse_args()

    all_runs = {}
    summary = {}

    for name, path in DEFAULT_CONFIGS.items():
        cfg = json.loads(Path(path).read_text(encoding="utf-8"))
        cfg["environment"]["params"]["episodes_per_family"] = int(args.episodes_per_family)
        if args.support_size is not None:
            cfg["environment"]["params"]["support_size"] = int(args.support_size)

        runs = [run_config(cfg, seed) for seed in args.seeds]
        all_runs[name] = runs
        summary[name] = summarize(runs)

    output = {
        "benchmark": "SAGE-v1.7.1-lifecycle-calibration-multiseed",
        "version": "v1.7.1",
        "goal": "Calibrate adaptive compute mode and add organ lifecycle diagnostics.",
        "seeds": args.seeds,
        "episodes_per_family": args.episodes_per_family,
        "interpretation_guardrail": "Synthetic benchmark evidence only. Lifecycle output is recommendation-only.",
        "summary": summary,
        "runs": all_runs,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== SAGE v1.7.1 Lifecycle Calibration Multi-seed ===")
    print(f"seeds={args.seeds}")
    print(f"episodes_per_family={args.episodes_per_family}\n")

    for name in DEFAULT_CONFIGS:
        print(line(name, summary[name]))

    print("\nMode usage:")
    for name in ["adaptive_v1_7", "calibrated_balanced", "calibrated_top2_friendly", "calibrated_safe"]:
        print(f"- {name}: {summary[name].get('mode_usage', {})}")

    print("\nOrgan lifecycle summary for calibrated_balanced:")
    for organ, item in summary["calibrated_balanced"].get("organ_lifecycle_summary", {}).items():
        print(f"- {organ}: {item}")

    print(f"\nSaved: {out}")

if __name__ == "__main__":
    main()
