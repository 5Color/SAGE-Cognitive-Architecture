# benchmark_v1_0_multiseed.py
# SAGE v1.0 Multi-seed Benchmark
#
# 핵심 수정:
# - v0.7 baseline은 benchmark_v0_7_state_replay.py의 SAGEv0/train_model/evaluate_model을 직접 사용
# - v0.9 gate 모델들은 benchmark_v0_9_context_gated_router.py의 SAGEContextGated/train_model/evaluate_model 사용
#
# 이유:
# - SAGEContextGated는 context input이 필요한 모델이므로,
#   gate OFF만으로 v0.7-StateReplay를 안전하게 흉내내면 안 됨.
# - v0.7은 SAGEv0 + Persistent State + Replay + Energy 조합으로 별도 실행해야 함.

from __future__ import annotations

import argparse
import copy
import inspect
import json
import math
import os
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Optional, Tuple


# v0.7 baseline
import benchmark_v0_7_state_replay as v07

# v0.9 context-gated router
import benchmark_v0_9_context_gated_router as v09


SEEDS = [0, 1, 2, 3, 4]

MODEL_SPECS = [
    {
        "name": "SAGE-v0.7-StateReplay",
        "version": "v0.7",
        "gate_scale": None,
        "use_energy": True,
        "use_state": True,
        "use_replay": True,
        "use_context_gate": False,
    },
    {
        "name": "SAGE-v0.9-Gate-0.03",
        "version": "v0.9",
        "gate_scale": 0.03,
        "use_energy": True,
        "use_state": True,
        "use_replay": True,
        "use_context_gate": True,
    },
    {
        "name": "SAGE-v0.9-Gate-0.10",
        "version": "v0.9",
        "gate_scale": 0.10,
        "use_energy": True,
        "use_state": True,
        "use_replay": True,
        "use_context_gate": True,
    },
    {
        "name": "SAGE-v0.9-Gate-0.20",
        "version": "v0.9",
        "gate_scale": 0.20,
        "use_energy": True,
        "use_state": True,
        "use_replay": True,
        "use_context_gate": True,
    },
    {
        "name": "SAGE-v0.9-Gate-0.35",
        "version": "v0.9",
        "gate_scale": 0.35,
        "use_energy": True,
        "use_state": True,
        "use_replay": True,
        "use_context_gate": True,
    },
]


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    except Exception:
        pass

    # 각 모듈의 자체 set_seed도 호출
    if hasattr(v07, "set_seed"):
        try:
            v07.set_seed(seed)
        except Exception:
            pass

    if hasattr(v09, "set_seed"):
        try:
            v09.set_seed(seed)
        except Exception:
            pass


def set_cfg(cfg: Any, key: str, value: Any) -> None:
    try:
        setattr(cfg, key, value)
    except Exception:
        pass


def has_attr(obj: Any, key: str) -> bool:
    try:
        return hasattr(obj, key)
    except Exception:
        return False


def config_to_dict(cfg: Any) -> Dict[str, Any]:
    if is_dataclass(cfg):
        return asdict(cfg)
    if hasattr(cfg, "__dict__"):
        return dict(cfg.__dict__)
    return {}


def apply_common_config(
    cfg: Any,
    *,
    seed: int,
    model_name: str,
    gate_scale: Optional[float],
    train_steps: Optional[int],
    eval_batches: Optional[int],
    device: str,
) -> Any:
    # seed
    for key in ["seed", "random_seed", "run_seed"]:
        set_cfg(cfg, key, seed)

    # 기록용 이름
    for key in ["model_name", "experiment_name", "run_name"]:
        set_cfg(cfg, key, model_name)

    # gate scale
    if gate_scale is not None:
        for key in ["gate_scale", "context_gate_scale", "router_gate_scale"]:
            set_cfg(cfg, key, gate_scale)

    # v0.8 raw context input은 v1.0 비교에서 꺼둠
    for key in ["use_raw_context", "raw_context_input", "concat_context_to_obs"]:
        set_cfg(cfg, key, False)

    # device
    for key in ["device", "runtime_device"]:
        set_cfg(cfg, key, device)

    # CPU 시간 조절:
    # 이 프로젝트 Config는 보통 steps_per_phase * num_phases 구조라서
    # train_steps는 전체 step 목표치로 받고 steps_per_phase로 환산.
    if train_steps is not None:
        if has_attr(cfg, "steps_per_phase") and has_attr(cfg, "num_phases"):
            num_phases = max(1, int(getattr(cfg, "num_phases")))
            set_cfg(cfg, "steps_per_phase", max(1, int(math.ceil(train_steps / num_phases))))

        for key in [
            "train_steps",
            "num_train_steps",
            "steps",
            "total_steps",
            "n_steps",
        ]:
            set_cfg(cfg, key, train_steps)

    # eval은 v0.7 기준 eval_batches를 직접 쓰므로 이것만 명확히 override.
    if eval_batches is not None:
        for key in [
            "eval_batches",
            "num_eval_batches",
            "evaluation_batches",
        ]:
            set_cfg(cfg, key, eval_batches)

    return cfg


def make_v07_config(
    *,
    seed: int,
    model_name: str,
    train_steps: Optional[int],
    eval_batches: Optional[int],
    device: str,
) -> Any:
    cfg = copy.deepcopy(v07.Config())
    return apply_common_config(
        cfg,
        seed=seed,
        model_name=model_name,
        gate_scale=None,
        train_steps=train_steps,
        eval_batches=eval_batches,
        device=device,
    )


def make_v09_config(
    *,
    seed: int,
    model_name: str,
    gate_scale: float,
    train_steps: Optional[int],
    eval_batches: Optional[int],
    device: str,
) -> Any:
    cfg = copy.deepcopy(v09.Config())
    return apply_common_config(
        cfg,
        seed=seed,
        model_name=model_name,
        gate_scale=gate_scale,
        train_steps=train_steps,
        eval_batches=eval_batches,
        device=device,
    )


def make_v07_model(cfg: Any) -> Any:
    env = v07.ContinualSocialEnv()
    return v07.SAGEv0(
        env.obs_dim,
        64,
        128,
        4,
        env.action_dim,
        2,
    )


def instantiate_v09_model(cfg: Any, gate_scale: float) -> Any:
    cls = v09.SAGEContextGated

    candidates = [
        ("SAGEContextGated(cfg)", lambda: cls(cfg)),
        ("SAGEContextGated(config=cfg)", lambda: cls(config=cfg)),
        ("SAGEContextGated(gate_scale=gate_scale)", lambda: cls(gate_scale=gate_scale)),
        ("SAGEContextGated(context_gate_scale=gate_scale)", lambda: cls(context_gate_scale=gate_scale)),
        ("SAGEContextGated()", lambda: cls()),
    ]

    errors = []
    for label, call in candidates:
        try:
            model = call()

            # 생성자에서 gate scale이 안 들어간 경우를 대비해 속성도 직접 덮어쓰기
            for key in ["gate_scale", "context_gate_scale", "router_gate_scale"]:
                if hasattr(model, key):
                    try:
                        setattr(model, key, gate_scale)
                    except Exception:
                        pass

            return model
        except Exception as exc:
            errors.append(f"{label}: {type(exc).__name__}: {exc}")

    raise RuntimeError(
        "SAGEContextGated 생성 실패\n" + "\n".join(f"  - {e}" for e in errors)
    )


def parse_train_output(out: Any, fallback_model: Any) -> Tuple[Any, Any, Any, Dict[str, Any]]:
    """
    train_model 출력 파싱.

    v0.7 파일 기준:
    return model, phase_acc_history, sage_controller

    v0.9도 같은 계열일 가능성이 높으므로 우선 이 형태를 표준으로 처리.
    """
    trained_model = fallback_model
    phase_acc_history = []
    controller = None
    train_metrics: Dict[str, Any] = {}

    if isinstance(out, dict):
        train_metrics.update(out)
        trained_model = (
            out.get("model")
            or out.get("trained_model")
            or out.get("agent")
            or fallback_model
        )
        phase_acc_history = (
            out.get("phase_acc_history")
            or out.get("phase_accuracy_history")
            or out.get("phase_history")
            or []
        )
        controller = (
            out.get("controller")
            or out.get("sage_controller")
        )
        return trained_model, phase_acc_history, controller, train_metrics

    if isinstance(out, tuple):
        if len(out) >= 3:
            trained_model = out[0]
            phase_acc_history = out[1]
            controller = out[2]
        elif len(out) == 2:
            phase_acc_history = out[0]
            controller = out[1]
        elif len(out) == 1:
            trained_model = out[0]

        for item in out:
            if isinstance(item, dict):
                train_metrics.update(item)

        return trained_model, phase_acc_history, controller, train_metrics

    if out is not None:
        trained_model = out

    return trained_model, phase_acc_history, controller, train_metrics


def parse_eval_output(out: Any) -> Dict[str, Any]:
    if isinstance(out, dict):
        return dict(out)

    if isinstance(out, tuple):
        dicts = [item for item in out if isinstance(item, dict)]
        if dicts:
            merged: Dict[str, Any] = {}
            for d in dicts:
                merged.update(d)
            return merged

        keys = ["final_avg_acc", "avg_reward", "avg_adapt_gain", "phase_accuracy", "usage"]
        parsed: Dict[str, Any] = {}
        for key, value in zip(keys, out):
            parsed[key] = value
        return parsed

    return {"raw_eval_output": repr(out)}


def call_v07_train_eval(spec: Dict[str, Any], cfg: Any, model: Any) -> Dict[str, Any]:
    train_out = v07.train_model(
        name=spec["name"],
        model=model,
        cfg=cfg,
        use_energy=spec["use_energy"],
        use_state=spec["use_state"],
        use_replay=spec["use_replay"],
    )

    trained_model, history, controller, train_metrics = parse_train_output(
        train_out,
        fallback_model=model,
    )

    eval_out = v07.evaluate_model(
        name=spec["name"],
        model=trained_model,
        cfg=cfg,
        phase_acc_history=history,
        sage_controller=controller,
        use_state=spec["use_state"],
        use_replay=spec["use_replay"],
        use_energy=spec["use_energy"],
    )

    eval_metrics = parse_eval_output(eval_out)

    raw = {}
    raw.update(train_metrics)
    raw.update(eval_metrics)
    return raw


def call_v09_train_eval(spec: Dict[str, Any], cfg: Any, model: Any) -> Dict[str, Any]:
    train_out = v09.train_model(
        spec["name"],
        model,
        cfg,
        spec["use_energy"],
        spec["use_state"],
        spec["use_replay"],
        spec["use_context_gate"],
    )

    trained_model, history, controller, train_metrics = parse_train_output(
        train_out,
        fallback_model=model,
    )

    eval_out = v09.evaluate_model(
        spec["name"],
        trained_model,
        cfg,
        history,
        controller,
        spec["use_state"],
        spec["use_replay"],
        spec["use_energy"],
        spec["use_context_gate"],
    )

    eval_metrics = parse_eval_output(eval_out)

    raw = {}
    raw.update(train_metrics)
    raw.update(eval_metrics)
    return raw


def to_float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None

    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        if hasattr(value, "item"):
            value = value.item()
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    except Exception:
        return None


def mean_from_list_or_dict(value: Any) -> Optional[float]:
    if value is None:
        return None

    if hasattr(value, "detach"):
        try:
            value = value.detach().cpu().tolist()
        except Exception:
            pass

    if isinstance(value, dict):
        vals = [to_float_or_none(v) for v in value.values()]
        vals = [v for v in vals if v is not None]
        return mean(vals) if vals else None

    if isinstance(value, (list, tuple)):
        vals = [to_float_or_none(v) for v in value]
        vals = [v for v in vals if v is not None]
        return mean(vals) if vals else None

    return to_float_or_none(value)


def pick_metric(metrics: Dict[str, Any], aliases: Iterable[str]) -> Optional[float]:
    for key in aliases:
        if key in metrics:
            return to_float_or_none(metrics[key])
    return None


def normalize_metrics(metrics: Dict[str, Any]) -> Dict[str, Optional[float]]:
    avg_acc = pick_metric(
        metrics,
        [
            "avg_acc",
            "final_avg_acc",
            "accuracy",
            "acc",
            "mean_acc",
            "eval_acc",
            "avg_accuracy",
        ],
    )

    reward = pick_metric(
        metrics,
        [
            "reward",
            "avg_reward",
            "mean_reward",
            "eval_reward",
        ],
    )

    adapt = pick_metric(
        metrics,
        [
            "adapt",
            "adapt_gain_mean",
            "avg_adapt_gain",
            "adapt_gain",
            "avg_adapt",
            "mean_adapt",
            "adaptation",
            "adapt_score",
        ],
    )

    # phase_acc는 {"phase_0": ..., ...} 형태이므로 평균 phase accuracy로 정규화
    phase_accuracy = pick_metric(
        metrics,
        [
            "phase_accuracy",
            "phase_acc_mean",
            "avg_phase_acc",
            "mean_phase_acc",
            "phase_correct",
        ],
    )

    if phase_accuracy is None and "phase_acc" in metrics:
        phase_accuracy = mean_from_list_or_dict(metrics["phase_acc"])

    usage_mean = None
    for key in [
        "usage",
        "avg_usage",
        "mean_usage",
        "organ_usage",
        "usage_mean",
        "router_usage",
        "organ_usage_mean",
    ]:
        if key in metrics:
            usage_mean = mean_from_list_or_dict(metrics[key])
            break

    return {
        "avg_acc": avg_acc,
        "reward": reward,
        "adapt": adapt,
        "phase_accuracy": phase_accuracy,
        "usage_mean": usage_mean,
    }


def summarize_runs(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    metric_keys = ["avg_acc", "reward", "adapt", "phase_accuracy", "usage_mean"]

    for key in metric_keys:
        values = [
            run["metrics"].get(key)
            for run in runs
            if run.get("metrics", {}).get(key) is not None
        ]

        if not values:
            summary[key] = {
                "mean": None,
                "std": None,
                "n": 0,
            }
        else:
            summary[key] = {
                "mean": mean(values),
                "std": stdev(values) if len(values) >= 2 else 0.0,
                "n": len(values),
            }

    return summary


def safe_jsonable(obj: Any) -> Any:
    try:
        if hasattr(obj, "detach"):
            obj = obj.detach().cpu()
        if hasattr(obj, "tolist"):
            return obj.tolist()
        if hasattr(obj, "item"):
            return obj.item()
    except Exception:
        pass

    if isinstance(obj, dict):
        return {str(k): safe_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [safe_jsonable(v) for v in obj]

    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    return repr(obj)


def run_one(
    *,
    spec: Dict[str, Any],
    seed: int,
    train_steps: Optional[int],
    eval_batches: Optional[int],
    device: str,
) -> Dict[str, Any]:
    set_global_seed(seed)

    if spec["version"] == "v0.7":
        cfg = make_v07_config(
            seed=seed,
            model_name=spec["name"],
            train_steps=train_steps,
            eval_batches=eval_batches,
            device=device,
        )
        model = make_v07_model(cfg)
        raw_metrics = call_v07_train_eval(spec, cfg, model)
        model_class = model.__class__.__name__
    else:
        assert spec["gate_scale"] is not None
        cfg = make_v09_config(
            seed=seed,
            model_name=spec["name"],
            gate_scale=float(spec["gate_scale"]),
            train_steps=train_steps,
            eval_batches=eval_batches,
            device=device,
        )
        model = instantiate_v09_model(cfg, float(spec["gate_scale"]))
        raw_metrics = call_v09_train_eval(spec, cfg, model)
        model_class = model.__class__.__name__

    normalized = normalize_metrics(raw_metrics)

    return {
        "seed": seed,
        "model": spec["name"],
        "version": spec["version"],
        "gate_scale": spec["gate_scale"],
        "use_energy": spec["use_energy"],
        "use_state": spec["use_state"],
        "use_replay": spec["use_replay"],
        "use_context_gate": spec["use_context_gate"],
        "model_class": model_class,
        "metrics": normalized,
        "raw_metrics": safe_jsonable(raw_metrics),
        "config_snapshot": safe_jsonable(config_to_dict(cfg)),
    }


def fmt(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value:.4f}"


def print_table(results: Dict[str, Any]) -> None:
    metric_keys = ["avg_acc", "reward", "adapt", "phase_accuracy", "usage_mean"]

    print("\n=== SAGE v1.0 Multi-seed Benchmark ===")
    print(f"seeds: {results['seeds']}")
    print()

    header = ["model"] + [f"{m} mean±std" for m in metric_keys]
    widths = [26, 18, 18, 18, 22, 18]

    print(" | ".join(h.ljust(w) for h, w in zip(header, widths)))
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))

    for spec in MODEL_SPECS:
        name = spec["name"]
        summary = results["summary"][name]

        row = [name]
        for key in metric_keys:
            item = summary[key]
            if item["mean"] is None:
                row.append("N/A")
            else:
                row.append(f"{item['mean']:.4f}±{item['std']:.4f}")

        print(" | ".join(v.ljust(w) for v, w in zip(row, widths)))

    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SAGE v1.0 multi-seed benchmark"
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=None,
        help=(
            "전체 train step 목표치. "
            "Config가 steps_per_phase*num_phases 구조이면 steps_per_phase로 환산됩니다."
        ),
    )
    parser.add_argument(
        "--eval-batches",
        type=int,
        default=None,
        help="eval_batches를 직접 덮어씁니다. CPU에서 빠르게 보려면 10~15 추천.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
    )
    parser.add_argument(
        "--out",
        type=str,
        default="results/v1_0_multiseed_benchmark.json",
    )
    args = parser.parse_args()

    print("Imported function signatures:")
    print(f"  v0.7 train_model: {inspect.signature(v07.train_model)}")
    print(f"  v0.7 evaluate_model: {inspect.signature(v07.evaluate_model)}")
    print(f"  v0.9 train_model: {inspect.signature(v09.train_model)}")
    print(f"  v0.9 evaluate_model: {inspect.signature(v09.evaluate_model)}")

    all_runs: List[Dict[str, Any]] = []

    for seed in SEEDS:
        print(f"\n[Seed {seed}]")
        for spec in MODEL_SPECS:
            print(f"  running {spec['name']} ...", flush=True)

            run = run_one(
                spec=spec,
                seed=seed,
                train_steps=args.train_steps,
                eval_batches=args.eval_batches,
                device=args.device,
            )

            m = run["metrics"]
            print(
                "    "
                f"avg_acc={fmt(m.get('avg_acc'))}, "
                f"reward={fmt(m.get('reward'))}, "
                f"adapt={fmt(m.get('adapt'))}, "
                f"phase_accuracy={fmt(m.get('phase_accuracy'))}, "
                f"usage_mean={fmt(m.get('usage_mean'))}"
            )

            all_runs.append(run)

    summary: Dict[str, Any] = {}
    for spec in MODEL_SPECS:
        model_runs = [r for r in all_runs if r["model"] == spec["name"]]
        summary[spec["name"]] = summarize_runs(model_runs)

    output = {
        "benchmark": "SAGE-v1.0-multiseed",
        "seeds": SEEDS,
        "models": MODEL_SPECS,
        "runs": all_runs,
        "summary": summary,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(safe_jsonable(output), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print_table(output)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
