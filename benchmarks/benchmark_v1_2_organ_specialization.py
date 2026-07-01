from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.nn.functional as F

import benchmark_v0_7_state_replay as v07
import benchmark_v0_9_context_gated_router as v09
import benchmark_v1_1_adaptive_context_gate as v11


SEEDS = [0, 1, 2, 3, 4]

MODEL_SPECS = [
    {
        "name": "SAGE-v0.7-StateReplay",
        "version": "v0.7",
        "mode": "baseline",
        "gate_scale": None,
        "base_gate": None,
        "use_energy": True,
        "use_state": True,
        "use_replay": True,
        "use_context_gate": False,
    },
    {
        "name": "SAGE-v1.1-AdaptiveGate-base0.10",
        "version": "v1.1",
        "mode": "adaptive",
        "gate_scale": None,
        "base_gate": 0.10,
        "use_energy": True,
        "use_state": True,
        "use_replay": True,
        "use_context_gate": True,
    },
]


def round_float(value: Optional[float], digits: int = 4) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


def normalized_entropy(vector: Optional[Iterable[float]]) -> Optional[float]:
    if vector is None:
        return None

    vals = [max(0.0, float(v)) for v in vector]
    total = sum(vals)
    if total <= 0.0 or len(vals) <= 1:
        return None

    probs = [v / total for v in vals if v > 0.0]
    entropy = -sum(p * math.log(p) for p in probs)
    return entropy / math.log(len(vals))


def vector_std(vector: Optional[Iterable[float]]) -> Optional[float]:
    if vector is None:
        return None

    vals = [float(v) for v in vector]
    if not vals:
        return None
    return stdev(vals) if len(vals) >= 2 else 0.0


def safe_ratio_vector(counts: torch.Tensor) -> Optional[List[float]]:
    total = counts.sum().item()
    if total <= 0:
        return None
    return [round_float(x, 4) for x in (counts / total).tolist()]


def get_num_organs(model: Any) -> int:
    for attr in ["num_organs", "num_experts"]:
        if hasattr(model, attr):
            return int(getattr(model, attr))
    if hasattr(model, "organ_energy"):
        return int(model.organ_energy.numel())
    return 0


def final_energy_vector(model: Any) -> Optional[List[float]]:
    if not hasattr(model, "organ_energy"):
        return None
    values = model.organ_energy.detach().cpu().tolist()
    return [round_float(v, 4) for v in values]


def compute_specialization_score(
    usage_vector: Optional[List[float]],
    phase_usage_vector: Dict[str, Optional[List[float]]],
    phase_usage_entropy: Dict[str, Optional[float]],
) -> Optional[float]:
    if not usage_vector or not phase_usage_vector:
        return None

    valid_phase_vectors = {
        phase: vec for phase, vec in phase_usage_vector.items() if vec is not None
    }
    valid_phase_entropy = [
        entropy for entropy in phase_usage_entropy.values() if entropy is not None
    ]

    if not valid_phase_vectors or not valid_phase_entropy:
        return None

    n_organs = len(usage_vector)
    n_phases = len(valid_phase_vectors)
    top_organs = [max(range(n_organs), key=lambda idx: vec[idx]) for vec in valid_phase_vectors.values()]

    entropy_component = 1.0 - mean(valid_phase_entropy)
    diversity_component = len(set(top_organs)) / max(1, min(n_organs, n_phases))

    # Low phase entropy can mean specialization, but low global entropy is collapse.
    # Keep the score near zero when all phases route into one dominant organ.
    global_entropy = normalized_entropy(usage_vector)
    if global_entropy is None:
        return None
    collapse_floor = 0.35
    collapse_penalty = max(0.0, min(1.0, (global_entropy - collapse_floor) / (1.0 - collapse_floor)))

    score = (0.55 * entropy_component + 0.45 * diversity_component) * collapse_penalty
    return round_float(max(0.0, min(1.0, score)))


def adapt_metrics(phase_acc_history: Any, cfg: Any) -> Dict[str, Any]:
    first_window: Dict[str, Optional[float]] = {}
    last_window: Dict[str, Optional[float]] = {}
    adapt_gain: Dict[str, Optional[float]] = {}

    for phase in range(int(cfg.num_phases)):
        values = list(phase_acc_history[phase]) if phase in phase_acc_history else []
        key = f"phase_{phase}"
        if not values:
            first_window[key] = None
            last_window[key] = None
            adapt_gain[key] = None
            continue

        n = min(int(cfg.adaptation_window), len(values))
        first = sum(values[:n]) / n
        last = sum(values[-n:]) / n
        first_window[key] = round_float(first)
        last_window[key] = round_float(last)
        adapt_gain[key] = round_float(last - first)

    valid_gains = [v for v in adapt_gain.values() if v is not None]
    return {
        "first_window_acc": first_window,
        "last_window_acc": last_window,
        "adapt_gain": adapt_gain,
        "adapt": round_float(mean(valid_gains)) if valid_gains else None,
    }


@torch.no_grad()
def evaluate_v07_specialization(
    *,
    name: str,
    model: Any,
    cfg: Any,
    phase_acc_history: Any,
    controller: Any,
) -> Dict[str, Any]:
    env = v07.ContinualSocialEnv()
    model.eval()
    if controller is not None:
        controller.reset()

    n_organs = get_num_organs(model)
    usage_counts = torch.zeros(n_organs)
    phase_usage_counts = {f"phase_{phase}": torch.zeros(n_organs) for phase in range(cfg.num_phases)}

    total_correct = 0
    total_samples = 0
    total_reward = 0.0
    total_time = 0.0
    phase_accuracy: Dict[str, float] = {}

    for phase in range(cfg.num_phases):
        phase_correct = 0
        phase_samples = 0
        phase_key = f"phase_{phase}"

        for _ in range(cfg.eval_batches):
            obs, target_actions, _ = v07.make_current_batch(env, phase, cfg.batch_size, cfg.device)

            start = time.perf_counter()
            out = v07.forward_model(model, obs, controller)
            total_time += time.perf_counter() - start

            pred_actions = out["action_logits"].argmax(dim=-1)
            correct_count = (pred_actions == target_actions).sum().item()
            phase_correct += correct_count
            phase_samples += cfg.batch_size
            total_correct += correct_count
            total_samples += cfg.batch_size

            for i in range(cfg.batch_size):
                _, reward, _ = env.step(phase, obs[i].cpu(), int(pred_actions[i].item()))
                total_reward += reward

            selected = out.get("selected_organs")
            if selected is not None and n_organs > 0:
                counts = torch.bincount(selected.flatten().cpu(), minlength=n_organs).float()
                usage_counts += counts
                phase_usage_counts[phase_key] += counts

        phase_accuracy[phase_key] = round_float(phase_correct / phase_samples)

    return build_specialization_metrics(
        name=name,
        cfg=cfg,
        model=model,
        total_correct=total_correct,
        total_samples=total_samples,
        total_reward=total_reward,
        total_time=total_time,
        phase_accuracy=phase_accuracy,
        phase_acc_history=phase_acc_history,
        usage_counts=usage_counts,
        phase_usage_counts=phase_usage_counts,
    )


@torch.no_grad()
def evaluate_v09_specialization(
    *,
    name: str,
    model: Any,
    cfg: Any,
    phase_acc_history: Any,
    controller: Any,
    use_context_gate: bool,
) -> Dict[str, Any]:
    env = v09.ContinualSocialEnv()
    model.eval()
    if controller is not None:
        controller.reset()

    eval_context = v09.ContextTracker(cfg.context_window)
    n_organs = get_num_organs(model)
    usage_counts = torch.zeros(n_organs)
    phase_usage_counts = {f"phase_{phase}": torch.zeros(n_organs) for phase in range(cfg.num_phases)}

    total_correct = 0
    total_samples = 0
    total_reward = 0.0
    total_time = 0.0
    phase_accuracy: Dict[str, float] = {}

    for phase in range(cfg.num_phases):
        phase_correct = 0
        phase_samples = 0
        phase_key = f"phase_{phase}"

        for _ in range(cfg.eval_batches):
            obs8, target_actions, _ = v09.make_current_batch(env, phase, cfg.batch_size, cfg.device)
            context = eval_context.get_context(obs8.shape[0], cfg.device)

            start = time.perf_counter()
            out = v09.forward_model(
                model=model,
                obs8=obs8,
                context=context if use_context_gate else None,
                controller=controller,
            )
            total_time += time.perf_counter() - start

            pred_actions = out["action_logits"].argmax(dim=-1)
            correct = (pred_actions == target_actions).float()
            eval_loss = F.cross_entropy(out["action_logits"], target_actions).item()
            confidence = F.softmax(out["action_logits"], dim=-1).max(dim=-1).values.mean().item()
            eval_context.update(
                acc=correct.mean().item(),
                loss=eval_loss,
                uncertainty=1.0 - confidence,
            )

            correct_count = correct.sum().item()
            phase_correct += correct_count
            phase_samples += cfg.batch_size
            total_correct += correct_count
            total_samples += cfg.batch_size

            for i in range(cfg.batch_size):
                _, reward, _ = env.step(phase, obs8[i].cpu(), int(pred_actions[i].item()))
                total_reward += reward

            selected = out.get("selected_organs")
            if selected is not None and n_organs > 0:
                counts = torch.bincount(selected.flatten().cpu(), minlength=n_organs).float()
                usage_counts += counts
                phase_usage_counts[phase_key] += counts

        phase_accuracy[phase_key] = round_float(phase_correct / phase_samples)

    return build_specialization_metrics(
        name=name,
        cfg=cfg,
        model=model,
        total_correct=total_correct,
        total_samples=total_samples,
        total_reward=total_reward,
        total_time=total_time,
        phase_accuracy=phase_accuracy,
        phase_acc_history=phase_acc_history,
        usage_counts=usage_counts,
        phase_usage_counts=phase_usage_counts,
    )


def build_specialization_metrics(
    *,
    name: str,
    cfg: Any,
    model: Any,
    total_correct: int,
    total_samples: int,
    total_reward: float,
    total_time: float,
    phase_accuracy: Dict[str, float],
    phase_acc_history: Any,
    usage_counts: torch.Tensor,
    phase_usage_counts: Dict[str, torch.Tensor],
) -> Dict[str, Any]:
    usage_vector = safe_ratio_vector(usage_counts)
    phase_usage_vector = {
        phase: safe_ratio_vector(counts) for phase, counts in phase_usage_counts.items()
    }
    phase_usage_entropy = {
        phase: round_float(normalized_entropy(vec)) for phase, vec in phase_usage_vector.items()
    }
    phase_entropy_values = [v for v in phase_usage_entropy.values() if v is not None]

    energy_vector = final_energy_vector(model)
    adapt = adapt_metrics(phase_acc_history, cfg)

    metrics = {
        "model": name,
        "avg_acc": round_float(total_correct / total_samples),
        "reward": round_float(total_reward / total_samples),
        "adapt": adapt["adapt"],
        "phase_accuracy": round_float(mean(phase_accuracy.values())),
        "phase_accuracy_by_phase": phase_accuracy,
        "first_window_acc": adapt["first_window_acc"],
        "last_window_acc": adapt["last_window_acc"],
        "adapt_gain": adapt["adapt_gain"],
        "ms_per_batch": round_float((total_time / (cfg.eval_batches * cfg.num_phases)) * 1000.0),
        "usage_vector": usage_vector,
        "usage_mean_aux": round_float(mean(usage_vector)) if usage_vector else None,
        "usage_std": round_float(vector_std(usage_vector)),
        "usage_entropy": round_float(normalized_entropy(usage_vector)),
        "usage_max": round_float(max(usage_vector)) if usage_vector else None,
        "usage_min": round_float(min(usage_vector)) if usage_vector else None,
        "phase_usage_vector": phase_usage_vector,
        "phase_usage_entropy": phase_usage_entropy,
        "phase_usage_entropy_mean": round_float(mean(phase_entropy_values)) if phase_entropy_values else None,
        "final_energy_vector": energy_vector,
        "energy_std": round_float(vector_std(energy_vector)),
        "energy_max": round_float(max(energy_vector)) if energy_vector else None,
        "energy_min": round_float(min(energy_vector)) if energy_vector else None,
        "energy_range": round_float(max(energy_vector) - min(energy_vector)) if energy_vector else None,
    }
    metrics["specialization_score"] = compute_specialization_score(
        usage_vector,
        phase_usage_vector,
        phase_usage_entropy,
    )
    return metrics


def train_and_evaluate_one(
    *,
    spec: Dict[str, Any],
    seed: int,
    train_steps: Optional[int],
    eval_batches: Optional[int],
    device: str,
    adaptive_sharpness: float,
    adaptive_threshold: float,
    adaptive_min_gate_ratio: float,
) -> Dict[str, Any]:
    v11.set_global_seed(seed)

    if spec["version"] == "v0.7":
        cfg = v11.make_v07_config(
            seed=seed,
            model_name=spec["name"],
            train_steps=train_steps,
            eval_batches=eval_batches,
            device=device,
        )
        model = v11.make_v07_model()
        train_out = v07.train_model(
            name=spec["name"],
            model=model,
            cfg=cfg,
            use_energy=spec["use_energy"],
            use_state=spec["use_state"],
            use_replay=spec["use_replay"],
        )
        trained_model, history, controller, train_metrics = v11.parse_train_output(train_out, model)
        metrics = evaluate_v07_specialization(
            name=spec["name"],
            model=trained_model,
            cfg=cfg,
            phase_acc_history=history,
            controller=controller,
        )
    else:
        base_gate = float(spec["base_gate"])
        cfg = v11.make_v09_config(
            seed=seed,
            model_name=spec["name"],
            gate_scale=base_gate,
            train_steps=train_steps,
            eval_batches=eval_batches,
            device=device,
        )
        model = v11.instantiate_v09_model(cfg, base_gate)
        model = v11.patch_adaptive_gate(
            model,
            base_gate=base_gate,
            sharpness=adaptive_sharpness,
            threshold=adaptive_threshold,
            min_gate_ratio=adaptive_min_gate_ratio,
        )
        train_out = v09.train_model(
            spec["name"],
            model,
            cfg,
            spec["use_energy"],
            spec["use_state"],
            spec["use_replay"],
            spec["use_context_gate"],
        )
        trained_model, history, controller, train_metrics = v11.parse_train_output(train_out, model)
        metrics = evaluate_v09_specialization(
            name=spec["name"],
            model=trained_model,
            cfg=cfg,
            phase_acc_history=history,
            controller=controller,
            use_context_gate=spec["use_context_gate"],
        )
        gate_summary = v11.summarize_adaptive_gate(trained_model)
        if gate_summary is not None:
            metrics["adaptive_gate"] = v11.safe_jsonable(gate_summary)

    return {
        "seed": seed,
        "model": spec["name"],
        "version": spec["version"],
        "mode": spec["mode"],
        "gate_scale": spec["gate_scale"],
        "base_gate": spec["base_gate"],
        "use_energy": spec["use_energy"],
        "use_state": spec["use_state"],
        "use_replay": spec["use_replay"],
        "use_context_gate": spec["use_context_gate"],
        "model_class": trained_model.__class__.__name__,
        "metrics": v11.safe_jsonable(metrics),
        "train_metrics": v11.safe_jsonable(train_metrics),
        "config_snapshot": v11.safe_jsonable(v11.config_to_dict(cfg)),
    }


def summarize_runs(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    scalar_keys = [
        "avg_acc",
        "reward",
        "adapt",
        "phase_accuracy",
        "usage_std",
        "usage_entropy",
        "usage_max",
        "usage_min",
        "phase_usage_entropy_mean",
        "energy_std",
        "energy_max",
        "energy_min",
        "energy_range",
        "specialization_score",
    ]

    for key in scalar_keys:
        values = [
            run["metrics"].get(key)
            for run in runs
            if run.get("metrics", {}).get(key) is not None
        ]
        summary[key] = {
            "mean": mean(values) if values else None,
            "std": stdev(values) if len(values) >= 2 else 0.0 if values else None,
            "n": len(values),
        }

    return summary


def fmt(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def print_table(results: Dict[str, Any]) -> None:
    metric_keys = [
        "avg_acc",
        "reward",
        "adapt",
        "usage_entropy",
        "energy_std",
        "specialization_score",
    ]
    widths = [35, 18, 18, 18, 22, 18, 28]
    header = ["model"] + [f"{key} mean+/-std" for key in metric_keys]

    print("\n=== SAGE v1.2 Organ Specialization Metrics ===")
    print(f"seeds: {results['seeds']}")
    print()
    print(" | ".join(value.ljust(width) for value, width in zip(header, widths)))
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))

    for spec in results["models"]:
        model_summary = results["summary"][spec["name"]]
        row = [spec["name"]]
        for key in metric_keys:
            item = model_summary[key]
            if item["mean"] is None:
                row.append("N/A")
            else:
                row.append(f"{item['mean']:.4f}+/-{item['std']:.4f}")
        print(" | ".join(value.ljust(width) for value, width in zip(row, widths)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SAGE v1.2 organ specialization metric benchmark"
    )
    parser.add_argument("--train-steps", type=int, default=None)
    parser.add_argument("--eval-batches", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--adaptive-sharpness", type=float, default=2.0)
    parser.add_argument("--adaptive-threshold", type=float, default=1.0)
    parser.add_argument("--adaptive-min-gate-ratio", type=float, default=0.05)
    parser.add_argument(
        "--out",
        type=str,
        default="results/v1_2_organ_specialization_benchmark.json",
    )
    args = parser.parse_args()

    all_runs: List[Dict[str, Any]] = []
    for seed in SEEDS:
        print(f"\n[Seed {seed}]")
        for spec in MODEL_SPECS:
            print(f"  running {spec['name']} ...", flush=True)
            run = train_and_evaluate_one(
                spec=spec,
                seed=seed,
                train_steps=args.train_steps,
                eval_batches=args.eval_batches,
                device=args.device,
                adaptive_sharpness=args.adaptive_sharpness,
                adaptive_threshold=args.adaptive_threshold,
                adaptive_min_gate_ratio=args.adaptive_min_gate_ratio,
            )
            metrics = run["metrics"]
            print(
                "    "
                f"avg_acc={fmt(metrics.get('avg_acc'))}, "
                f"reward={fmt(metrics.get('reward'))}, "
                f"adapt={fmt(metrics.get('adapt'))}, "
                f"usage_entropy={fmt(metrics.get('usage_entropy'))}, "
                f"energy_std={fmt(metrics.get('energy_std'))}, "
                f"specialization={fmt(metrics.get('specialization_score'))}"
            )
            all_runs.append(run)

    summary = {
        spec["name"]: summarize_runs([run for run in all_runs if run["model"] == spec["name"]])
        for spec in MODEL_SPECS
    }
    output = {
        "benchmark": "SAGE-v1.2-organ-specialization-metrics",
        "goal": "Measure organ role differentiation, not maximize task performance.",
        "seeds": SEEDS,
        "models": MODEL_SPECS,
        "specialization_score_formula": {
            "entropy_component": "1 - mean(normalized phase_usage_entropy)",
            "diversity_component": "unique phase top organs / min(num_organs, num_phases)",
            "collapse_penalty": "clamp((usage_entropy - 0.35) / 0.65, 0, 1)",
            "score": "clamp((0.55 * entropy_component + 0.45 * diversity_component) * collapse_penalty, 0, 1)",
        },
        "cli": {
            "train_steps": args.train_steps,
            "eval_batches": args.eval_batches,
            "device": args.device,
            "adaptive_sharpness": args.adaptive_sharpness,
            "adaptive_threshold": args.adaptive_threshold,
            "adaptive_min_gate_ratio": args.adaptive_min_gate_ratio,
        },
        "runs": all_runs,
        "summary": summary,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(v11.safe_jsonable(output), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print_table(output)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
