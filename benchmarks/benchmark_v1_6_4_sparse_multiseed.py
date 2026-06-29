# SAGE v1.6.4 - Sparse Routing Multi-seed Validation
#
# Goal:
# - v1.6.3 showed a strong single-seed result.
# - v1.6.4 checks whether the sparse routing result is stable across multiple seeds.
#
# Run:
# python benchmarks/benchmark_v1_6_4_sparse_multiseed.py
#
# Optional:
# python benchmarks/benchmark_v1_6_4_sparse_multiseed.py --seeds 0 1 2 3 4 5 6 7 8 9
# python benchmarks/benchmark_v1_6_4_sparse_multiseed.py --episodes-per-family 80
#
# Output:
# results/v1_6_4_sparse_multiseed_validation.json

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from run_experiment import build_component, set_seed
from sage_core import SAGEEngine, SAGEState


DEFAULT_CONFIGS = {
    "sparse_top1": "benchmarks/configs/anti_leak_sparse_top1.json",
    "sparse_top2": "benchmarks/configs/anti_leak_sparse_top2.json",
    "full_evidence": "benchmarks/configs/anti_leak_sparse_full_evidence.json",
    "oracle": "benchmarks/configs/anti_leak_sparse_oracle.json",
}


TRACKED_METRICS = [
    "accuracy",
    "avg_reward",
    "task_diversity",
    "organ_calls",
    "avg_organs_per_step",
    "compute_saving_vs_full",
    "sparse_efficiency_score",
]


def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def std(values: Iterable[float]) -> float:
    values = list(values)
    if len(values) <= 1:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1))


def update_nested_seed(config: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """Return a config copy with seed propagated to relevant places."""
    cfg = copy.deepcopy(config)
    cfg["seed"] = int(seed)

    env_params = cfg.get("environment", {}).setdefault("params", {})
    env_params["seed"] = int(seed)

    router_params = cfg.get("router", {}).setdefault("params", {})
    if "seed" in router_params:
        router_params["seed"] = int(seed)

    # Prevent each seed from overwriting single-run v1.6.3 result files.
    benchmark = str(cfg.get("benchmark", "run")).replace(" ", "_")
    cfg["output_path"] = f"results/tmp_v1_6_4_{benchmark}_seed{seed}.json"

    return cfg


def run_config(config: Mapping[str, Any], seed: int) -> Dict[str, Any]:
    cfg = update_nested_seed(dict(config), seed)
    set_seed(seed)

    organs = {}
    for organ_spec in cfg["organs"]:
        organ = build_component(organ_spec)
        organs[organ.name] = organ

    router = build_component(cfg["router"])
    environment = build_component(cfg["environment"])
    metric = build_component(cfg["metric"])

    state = SAGEState(**dict(cfg.get("state", {})))
    engine = SAGEEngine(
        organs=organs,
        router=router,
        env=environment,
        metric=metric,
        state=state,
        **dict(cfg.get("engine", {})),
    )

    result = engine.run(max_steps=int(cfg.get("max_steps", 64)))
    result["seed"] = int(seed)
    result["router"] = router.name
    return result


def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}

    for metric in TRACKED_METRICS:
        vals = [float(r.get(metric, 0.0)) for r in results]
        summary[f"{metric}_mean"] = mean(vals)
        summary[f"{metric}_std"] = std(vals)
        summary[f"{metric}_min"] = min(vals) if vals else 0.0
        summary[f"{metric}_max"] = max(vals) if vals else 0.0

    # Family accuracy summary
    all_families = sorted({
        family
        for r in results
        for family in r.get("family_accuracy", {}).keys()
    })

    family_summary = {}
    for family in all_families:
        vals = [
            float(r.get("family_accuracy", {}).get(family, 0.0))
            for r in results
        ]
        family_summary[family] = {
            "mean": mean(vals),
            "std": std(vals),
            "min": min(vals) if vals else 0.0,
            "max": max(vals) if vals else 0.0,
        }

    summary["family_accuracy"] = family_summary
    return summary


def compact_line(name: str, summary: Mapping[str, Any]) -> str:
    return (
        f"{name:14s} | "
        f"acc {summary.get('accuracy_mean', 0):.4f}±{summary.get('accuracy_std', 0):.4f} | "
        f"org/step {summary.get('avg_organs_per_step_mean', 0):.2f}±{summary.get('avg_organs_per_step_std', 0):.2f} | "
        f"saving {summary.get('compute_saving_vs_full_mean', 0):.2f}±{summary.get('compute_saving_vs_full_std', 0):.2f} | "
        f"eff {summary.get('sparse_efficiency_score_mean', 0):.4f}±{summary.get('sparse_efficiency_score_std', 0):.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="SAGE v1.6.4 sparse routing multi-seed validation")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--episodes-per-family", type=int, default=None)
    parser.add_argument("--support-size", type=int, default=None)
    parser.add_argument("--out", type=str, default="results/v1_6_4_sparse_multiseed_validation.json")
    args = parser.parse_args()

    all_runs: Dict[str, List[Dict[str, Any]]] = {}
    summaries: Dict[str, Dict[str, Any]] = {}

    for name, path in DEFAULT_CONFIGS.items():
        config_path = Path(path)
        config = json.loads(config_path.read_text(encoding="utf-8"))

        if args.episodes_per_family is not None:
            config["environment"]["params"]["episodes_per_family"] = int(args.episodes_per_family)
        if args.support_size is not None:
            config["environment"]["params"]["support_size"] = int(args.support_size)

        runs = []
        for seed in args.seeds:
            result = run_config(config, int(seed))
            runs.append(result)

        all_runs[name] = runs
        summaries[name] = summarize_results(runs)

    output = {
        "benchmark": "SAGE-v1.6.4-sparse-routing-multiseed-validation",
        "version": "v1.6.4",
        "goal": (
            "Validate whether sparse evidence routing remains stable across multiple seeds."
        ),
        "seeds": args.seeds,
        "configs": DEFAULT_CONFIGS,
        "episodes_per_family_override": args.episodes_per_family,
        "support_size_override": args.support_size,
        "tracked_metrics": TRACKED_METRICS,
        "interpretation_guardrail": (
            "This is a multi-seed validation on a small synthetic benchmark, not an AGI proof."
        ),
        "summary": summaries,
        "runs": all_runs,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== SAGE v1.6.4 Sparse Routing Multi-seed Validation ===")
    print(f"seeds={args.seeds}")
    if args.episodes_per_family is not None:
        print(f"episodes_per_family={args.episodes_per_family}")
    if args.support_size is not None:
        print(f"support_size={args.support_size}")
    print()

    for name in ["full_evidence", "sparse_top1", "sparse_top2", "oracle"]:
        print(compact_line(name, summaries[name]))

    print()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
