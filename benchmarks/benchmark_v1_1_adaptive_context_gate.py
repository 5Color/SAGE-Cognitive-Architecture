# benchmark_v1_1_adaptive_context_gate.py
# SAGE v1.1 Adaptive Context Gate Benchmark
#
# 목표:
# - v1.0 결론: fixed context gate는 adapt를 조금 올리지만 avg_acc/reward를 깎음.
# - v1.1 방향: context를 항상 강하게 쓰지 않고,
#   uncertainty + phase_shift_score가 높을 때만 gate를 키움.
#
# 비교 모델:
# 1. SAGE-v0.7-StateReplay
# 2. SAGE-v0.9-FixedGate-0.10
# 3. SAGE-v0.9-FixedGate-0.20
# 4. SAGE-v1.1-AdaptiveGate-base0.10
# 5. SAGE-v1.1-AdaptiveGate-base0.20
#
# 저장:
# results/v1_1_adaptive_context_gate_benchmark.json

from __future__ import annotations

import argparse
import copy
import inspect
import json
import math
import os
import random
import types
from dataclasses import asdict, is_dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Optional, Tuple


import benchmark_v0_7_state_replay as v07
import benchmark_v0_9_context_gated_router as v09


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
        "name": "SAGE-v0.9-FixedGate-0.10",
        "version": "v0.9",
        "mode": "fixed",
        "gate_scale": 0.10,
        "base_gate": 0.10,
        "use_energy": True,
        "use_state": True,
        "use_replay": True,
        "use_context_gate": True,
    },
    {
        "name": "SAGE-v0.9-FixedGate-0.20",
        "version": "v0.9",
        "mode": "fixed",
        "gate_scale": 0.20,
        "base_gate": 0.20,
        "use_energy": True,
        "use_state": True,
        "use_replay": True,
        "use_context_gate": True,
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
    {
        "name": "SAGE-v1.1-AdaptiveGate-base0.20",
        "version": "v1.1",
        "mode": "adaptive",
        "gate_scale": None,
        "base_gate": 0.20,
        "use_energy": True,
        "use_state": True,
        "use_replay": True,
        "use_context_gate": True,
    },
]


EXTRA_035_SPECS = [
    {
        "name": "SAGE-v0.9-FixedGate-0.35",
        "version": "v0.9",
        "mode": "fixed",
        "gate_scale": 0.35,
        "base_gate": 0.35,
        "use_energy": True,
        "use_state": True,
        "use_replay": True,
        "use_context_gate": True,
    },
    {
        "name": "SAGE-v1.1-AdaptiveGate-base0.35",
        "version": "v1.1",
        "mode": "adaptive",
        "gate_scale": None,
        "base_gate": 0.35,
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

    for module in [v07, v09]:
        if hasattr(module, "set_seed"):
            try:
                module.set_seed(seed)
            except Exception:
                pass


def set_cfg(cfg: Any, key: str, value: Any) -> None:
    try:
        setattr(cfg, key, value)
    except Exception:
        pass


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
    for key in ["seed", "random_seed", "run_seed"]:
        set_cfg(cfg, key, seed)

    for key in ["model_name", "experiment_name", "run_name"]:
        set_cfg(cfg, key, model_name)

    if gate_scale is not None:
        for key in ["gate_scale", "context_gate_scale", "router_gate_scale"]:
            set_cfg(cfg, key, float(gate_scale))

    for key in ["use_raw_context", "raw_context_input", "concat_context_to_obs"]:
        set_cfg(cfg, key, False)

    for key in ["device", "runtime_device"]:
        set_cfg(cfg, key, device)

    # 기존 Config가 steps_per_phase * num_phases 형태라면 전체 step을 phase당 step으로 환산.
    if train_steps is not None:
        if hasattr(cfg, "steps_per_phase") and hasattr(cfg, "num_phases"):
            num_phases = max(1, int(getattr(cfg, "num_phases")))
            set_cfg(cfg, "steps_per_phase", max(1, int(math.ceil(train_steps / num_phases))))

        for key in ["train_steps", "num_train_steps", "steps", "total_steps", "n_steps"]:
            set_cfg(cfg, key, int(train_steps))

    if eval_batches is not None:
        for key in ["eval_batches", "num_eval_batches", "evaluation_batches"]:
            set_cfg(cfg, key, int(eval_batches))

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


def make_v07_model() -> Any:
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
            force_gate_attrs(model, gate_scale)
            return model
        except Exception as exc:
            errors.append(f"{label}: {type(exc).__name__}: {exc}")

    raise RuntimeError(
        "SAGEContextGated 생성 실패\n" + "\n".join(f"  - {e}" for e in errors)
    )


def force_gate_attrs(model: Any, gate_value: float) -> None:
    """
    v0.9 파일에서 gate scale 필드명이 무엇이든 최대한 반영.
    """
    for key in [
        "gate_scale",
        "context_gate_scale",
        "router_gate_scale",
        "context_scale",
        "gate_strength",
    ]:
        try:
            setattr(model, key, float(gate_value))
        except Exception:
            pass


def infer_context_dim(model: Any, default: int = 4) -> int:
    """
    context_router의 첫 Linear 입력 차원에서 context dim 추정.
    실패하면 v0.8/v0.9 설계값인 4 사용.
    """
    try:
        router = getattr(model, "context_router")
        net = getattr(router, "net", None)
        if net is not None:
            for layer in net:
                if hasattr(layer, "in_features"):
                    return int(layer.in_features)
    except Exception:
        pass

    try:
        router = getattr(model, "context_router")
        if hasattr(router, "in_features"):
            return int(router.in_features)
    except Exception:
        pass

    return default


def patch_adaptive_gate(
    model: Any,
    *,
    base_gate: float,
    sharpness: float,
    threshold: float,
    min_gate_ratio: float,
) -> Any:
    """
    v0.9 SAGEContextGated 모델의 forward를 monkey-patch해서,
    매 forward마다 context 기반 dynamic gate를 계산한다.

    context 벡터 순서 가정:
    [recent_acc, recent_loss, phase_shift_score, uncertainty]

    dynamic_gate:
    base_gate * (min_gate_ratio + (1 - min_gate_ratio) * sigmoid(
        sharpness * (phase_shift_score + uncertainty - threshold)
    ))

    의도:
    - 변화/불확실성 낮음: gate 약하게
    - 변화/불확실성 높음: gate 강하게
    """
    import torch

    original_forward = model.forward
    context_dim = infer_context_dim(model, default=4)

    model.adaptive_gate_base = float(base_gate)
    model.adaptive_gate_sharpness = float(sharpness)
    model.adaptive_gate_threshold = float(threshold)
    model.adaptive_gate_min_ratio = float(min_gate_ratio)
    model.adaptive_gate_history = []

    def adaptive_forward(self, *args, **kwargs):
        # context 추출
        obs = None
        context = None

        if len(args) >= 1:
            obs = args[0]
        else:
            obs = kwargs.get("obs") or kwargs.get("obs8") or kwargs.get("x")

        if len(args) >= 2:
            context = args[1]
        else:
            context = kwargs.get("context")

        # 혹시 context가 None이면 zero context를 넣어 crash 방지
        if context is None:
            if obs is None:
                dynamic_gate = 0.0
            else:
                context = torch.zeros(
                    obs.shape[0],
                    context_dim,
                    dtype=obs.dtype,
                    device=obs.device,
                )

                if len(args) >= 2:
                    args = (args[0], context, *args[2:])
                else:
                    kwargs["context"] = context

        # dynamic gate 계산
        if context is None:
            dynamic_gate = 0.0
        else:
            if context.dim() == 1:
                ctx = context.unsqueeze(0)
            else:
                ctx = context

            if ctx.shape[-1] >= 4:
                phase_shift_score = ctx[:, 2].detach()
                uncertainty = ctx[:, 3].detach()
            elif ctx.shape[-1] >= 2:
                phase_shift_score = ctx[:, -2].detach()
                uncertainty = ctx[:, -1].detach()
            else:
                phase_shift_score = ctx.mean(dim=-1).detach()
                uncertainty = ctx.mean(dim=-1).detach()

            raw = sharpness * (phase_shift_score + uncertainty - threshold)
            gate_ratio = torch.sigmoid(raw)
            gate_ratio = min_gate_ratio + (1.0 - min_gate_ratio) * gate_ratio
            dynamic_gate = float((base_gate * gate_ratio.mean()).item())

        # 기존 gate attr 저장
        old_values = {}
        for key in [
            "gate_scale",
            "context_gate_scale",
            "router_gate_scale",
            "context_scale",
            "gate_strength",
        ]:
            if hasattr(self, key):
                try:
                    old_values[key] = getattr(self, key)
                    setattr(self, key, dynamic_gate)
                except Exception:
                    pass

        self.adaptive_gate_history.append(dynamic_gate)

        try:
            return original_forward(*args, **kwargs)
        finally:
            # 다음 call 전에 복구
            for key, value in old_values.items():
                try:
                    setattr(self, key, value)
                except Exception:
                    pass

    model.forward = types.MethodType(adaptive_forward, model)
    return model


def summarize_adaptive_gate(model: Any) -> Optional[Dict[str, Any]]:
    hist = getattr(model, "adaptive_gate_history", None)
    if not hist:
        return None

    values = [float(x) for x in hist]
    return {
        "mean": mean(values),
        "std": stdev(values) if len(values) >= 2 else 0.0,
        "min": min(values),
        "max": max(values),
        "n": len(values),
        # JSON 크기 폭증 방지: 앞뒤 일부만 저장
        "head": values[:10],
        "tail": values[-10:],
    }


def parse_train_output(out: Any, fallback_model: Any) -> Tuple[Any, Any, Any, Dict[str, Any]]:
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
        controller = out.get("controller") or out.get("sage_controller")
        return trained_model, phase_acc_history, controller, train_metrics

    if isinstance(out, tuple):
        # v0.7 기준: return model, phase_acc_history, sage_controller
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

    elif out is not None:
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

    gate_summary = summarize_adaptive_gate(trained_model)
    if gate_summary is not None:
        raw["adaptive_gate"] = gate_summary

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

    adaptive_gate_mean = None
    if isinstance(metrics.get("adaptive_gate"), dict):
        adaptive_gate_mean = to_float_or_none(metrics["adaptive_gate"].get("mean"))

    return {
        "avg_acc": avg_acc,
        "reward": reward,
        "adapt": adapt,
        "phase_accuracy": phase_accuracy,
        "usage_mean": usage_mean,
        "adaptive_gate_mean": adaptive_gate_mean,
    }


def summarize_runs(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    metric_keys = [
        "avg_acc",
        "reward",
        "adapt",
        "phase_accuracy",
        "usage_mean",
        "adaptive_gate_mean",
    ]

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
    adaptive_sharpness: float,
    adaptive_threshold: float,
    adaptive_min_gate_ratio: float,
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
        model = make_v07_model()
        raw_metrics = call_v07_train_eval(spec, cfg, model)
        model_class = model.__class__.__name__
    else:
        base_gate = float(spec["base_gate"])
        fixed_gate_for_config = (
            float(spec["gate_scale"])
            if spec["gate_scale"] is not None
            else base_gate
        )

        cfg = make_v09_config(
            seed=seed,
            model_name=spec["name"],
            gate_scale=fixed_gate_for_config,
            train_steps=train_steps,
            eval_batches=eval_batches,
            device=device,
        )
        model = instantiate_v09_model(cfg, fixed_gate_for_config)

        if spec["mode"] == "adaptive":
            model = patch_adaptive_gate(
                model,
                base_gate=base_gate,
                sharpness=adaptive_sharpness,
                threshold=adaptive_threshold,
                min_gate_ratio=adaptive_min_gate_ratio,
            )

        raw_metrics = call_v09_train_eval(spec, cfg, model)
        model_class = model.__class__.__name__

    normalized = normalize_metrics(raw_metrics)

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
        "model_class": model_class,
        "metrics": normalized,
        "raw_metrics": safe_jsonable(raw_metrics),
        "config_snapshot": safe_jsonable(config_to_dict(cfg)),
    }


def fmt(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value:.4f}"


def print_table(results: Dict[str, Any], model_specs: List[Dict[str, Any]]) -> None:
    metric_keys = [
        "avg_acc",
        "reward",
        "adapt",
        "phase_accuracy",
        "usage_mean",
        "adaptive_gate_mean",
    ]

    print("\n=== SAGE v1.1 Adaptive Context Gate Benchmark ===")
    print(f"seeds: {results['seeds']}")
    print()

    header = ["model"] + [f"{m} mean±std" for m in metric_keys]
    widths = [35, 18, 18, 18, 22, 18, 24]

    print(" | ".join(h.ljust(w) for h, w in zip(header, widths)))
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))

    for spec in model_specs:
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
        description="SAGE v1.1 adaptive context gate benchmark"
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
        help="eval_batches를 직접 덮어씁니다. CPU 빠른 확인은 10~15 추천.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
    )
    parser.add_argument(
        "--adaptive-sharpness",
        type=float,
        default=2.0,
        help="adaptive gate sigmoid 민감도",
    )
    parser.add_argument(
        "--adaptive-threshold",
        type=float,
        default=1.0,
        help="phase_shift_score + uncertainty 기준점",
    )
    parser.add_argument(
        "--adaptive-min-gate-ratio",
        type=float,
        default=0.05,
        help="평상시에도 유지할 최소 gate 비율. 0이면 완전히 꺼질 수 있음.",
    )
    parser.add_argument(
        "--include-035",
        action="store_true",
        help="fixed/adaptive 0.35 비교군도 추가",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="results/v1_1_adaptive_context_gate_benchmark.json",
    )
    args = parser.parse_args()

    model_specs = list(MODEL_SPECS)
    if args.include_035:
        model_specs.extend(EXTRA_035_SPECS)

    print("Imported function signatures:")
    print(f"  v0.7 train_model: {inspect.signature(v07.train_model)}")
    print(f"  v0.7 evaluate_model: {inspect.signature(v07.evaluate_model)}")
    print(f"  v0.9 train_model: {inspect.signature(v09.train_model)}")
    print(f"  v0.9 evaluate_model: {inspect.signature(v09.evaluate_model)}")

    print("\nAdaptive gate formula:")
    print(
        "  dynamic_gate = base_gate * "
        "(min_ratio + (1-min_ratio) * sigmoid(sharpness * "
        "(phase_shift_score + uncertainty - threshold)))"
    )
    print(
        f"  sharpness={args.adaptive_sharpness}, "
        f"threshold={args.adaptive_threshold}, "
        f"min_ratio={args.adaptive_min_gate_ratio}"
    )

    all_runs: List[Dict[str, Any]] = []

    for seed in SEEDS:
        print(f"\n[Seed {seed}]")
        for spec in model_specs:
            print(f"  running {spec['name']} ...", flush=True)

            run = run_one(
                spec=spec,
                seed=seed,
                train_steps=args.train_steps,
                eval_batches=args.eval_batches,
                device=args.device,
                adaptive_sharpness=args.adaptive_sharpness,
                adaptive_threshold=args.adaptive_threshold,
                adaptive_min_gate_ratio=args.adaptive_min_gate_ratio,
            )

            m = run["metrics"]
            print(
                "    "
                f"avg_acc={fmt(m.get('avg_acc'))}, "
                f"reward={fmt(m.get('reward'))}, "
                f"adapt={fmt(m.get('adapt'))}, "
                f"phase_accuracy={fmt(m.get('phase_accuracy'))}, "
                f"usage_mean={fmt(m.get('usage_mean'))}, "
                f"adaptive_gate_mean={fmt(m.get('adaptive_gate_mean'))}"
            )

            all_runs.append(run)

    summary: Dict[str, Any] = {}
    for spec in model_specs:
        model_runs = [r for r in all_runs if r["model"] == spec["name"]]
        summary[spec["name"]] = summarize_runs(model_runs)

    output = {
        "benchmark": "SAGE-v1.1-adaptive-context-gate",
        "seeds": SEEDS,
        "adaptive_gate_formula": {
            "base": "base_gate * (min_ratio + (1-min_ratio) * sigmoid(sharpness * (phase_shift_score + uncertainty - threshold)))",
            "sharpness": args.adaptive_sharpness,
            "threshold": args.adaptive_threshold,
            "min_gate_ratio": args.adaptive_min_gate_ratio,
            "context_order_assumption": [
                "recent_acc",
                "recent_loss",
                "phase_shift_score",
                "uncertainty",
            ],
        },
        "models": model_specs,
        "runs": all_runs,
        "summary": summary,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(safe_jsonable(output), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print_table(output, model_specs)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
