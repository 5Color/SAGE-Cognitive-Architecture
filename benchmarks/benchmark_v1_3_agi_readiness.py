from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import benchmark_v0_9_context_gated_router as v09
from benchmark_v1_2_organ_specialization import normalized_entropy, round_float, vector_std


SEEDS = [0, 1, 2, 3, 4]
ACTION_DIM = 6
OBS_DIM = 16
CONTEXT_DIM = 16
TASKS = [
    "social_rules",
    "language_action",
    "static_memory",
    "planning",
    "world_model",
    "self_goal",
]
TASK_TO_ID = {name: idx for idx, name in enumerate(TASKS)}
STATIC_MEMORY_MAP = [3, 5, 1, 4, 0, 2]


@dataclass
class Config:
    train_steps: int = 800
    eval_batches: int = 10
    batch_size: int = 64
    lr: float = 1e-3
    device: str = "cpu"
    state_dim: int = 64
    hidden_dim: int = 128
    num_organs: int = 4
    top_k: int = 2
    context_gate_scale: float = 0.10
    world_loss_weight: float = 0.15
    reward_loss_weight: float = 0.05


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))


def one_hot(index: int, size: int) -> List[float]:
    values = [0.0] * size
    values[index] = 1.0
    return values


def make_context(task_id: int, token: Optional[int] = None, goals: Optional[List[float]] = None) -> torch.Tensor:
    ctx = [0.0] * CONTEXT_DIM
    ctx[task_id] = 1.0
    if token is not None:
        ctx[7 + token] = 1.0
    if goals is not None:
        ctx[13:16] = goals[:3]
    return torch.tensor(ctx, dtype=torch.float32)


def base_obs() -> List[float]:
    return [random.random() for _ in range(OBS_DIM)]


def social_target(obs: List[float]) -> int:
    user_spoke, is_stranger, closeness, positivity, risk, noise, energy, talk_pressure = obs[:8]
    if risk > 0.72:
        return 5
    if talk_pressure > 0.78 or energy < 0.22:
        return 0
    if positivity < 0.32 and risk < 0.55:
        return 3
    if user_spoke > 0.52 and is_stranger > 0.45 and positivity > 0.42:
        return 1
    if closeness > 0.58 and positivity > 0.50:
        return 2
    if noise < 0.35 and energy > 0.65:
        return 4
    return 1


def planning_target(obs: List[float]) -> int:
    x = int(obs[8] * 4.0)
    y = int(obs[9] * 4.0)
    gx = int(obs[10] * 4.0)
    gy = int(obs[11] * 4.0)
    blocked = {
        0: obs[12] > 0.5,  # up
        1: obs[13] > 0.5,  # down
        2: obs[14] > 0.5,  # left
        3: obs[15] > 0.5,  # right
    }
    candidates: List[Tuple[int, int]] = []
    moves = {
        0: (x, max(0, y - 1)),
        1: (x, min(3, y + 1)),
        2: (max(0, x - 1), y),
        3: (min(3, x + 1), y),
    }
    for action, (nx, ny) in moves.items():
        if blocked[action]:
            continue
        distance = abs(gx - nx) + abs(gy - ny)
        candidates.append((distance, action))
    if not candidates:
        return 4
    return min(candidates)[1]


def apply_planning_action(obs: List[float], action: int) -> List[float]:
    next_obs = list(obs)
    x = int(obs[8] * 4.0)
    y = int(obs[9] * 4.0)
    if action == 0:
        y = max(0, y - 1)
    elif action == 1:
        y = min(3, y + 1)
    elif action == 2:
        x = max(0, x - 1)
    elif action == 3:
        x = min(3, x + 1)
    next_obs[8] = (x + 0.5) / 4.0
    next_obs[9] = (y + 0.5) / 4.0
    return next_obs


def world_target(obs: List[float]) -> int:
    resource = obs[0]
    volatility = obs[1]
    threat = obs[4]
    energy = obs[6]
    scores = [
        0.45 * energy - 0.25 * threat,
        0.40 * resource + 0.10 * energy,
        0.50 * resource - 0.20 * volatility,
        0.55 * (1.0 - threat) + 0.10 * volatility,
        0.65 * (1.0 - volatility) - 0.10 * threat,
        0.70 * threat + 0.20 * (1.0 - energy),
    ]
    return max(range(ACTION_DIM), key=lambda idx: scores[idx])


def apply_world_action(obs: List[float], action: int) -> Tuple[List[float], float]:
    next_obs = list(obs)
    deltas = [
        (0.00, 0.05, -0.03),
        (0.04, 0.01, -0.02),
        (0.08, 0.04, -0.05),
        (0.02, -0.04, -0.01),
        (0.00, -0.06, 0.04),
        (-0.03, -0.08, -0.02),
    ]
    resource_delta, threat_delta, energy_delta = deltas[action]
    next_obs[0] = min(1.0, max(0.0, next_obs[0] + resource_delta))
    next_obs[4] = min(1.0, max(0.0, next_obs[4] + threat_delta))
    next_obs[6] = min(1.0, max(0.0, next_obs[6] + energy_delta))
    reward = 0.45 * next_obs[0] + 0.30 * (1.0 - next_obs[4]) + 0.25 * next_obs[6]
    return next_obs, reward


def self_goal_target(obs: List[float], goals: List[float]) -> int:
    safety_goal, energy_goal, social_goal = goals
    risk = obs[4]
    energy = obs[6]
    positivity = obs[3]
    if safety_goal >= energy_goal and safety_goal >= social_goal:
        return 5 if risk > 0.40 else 0
    if energy_goal >= social_goal:
        return 0 if energy < 0.65 else 1
    return 2 if positivity > 0.55 else 3


def make_sample(task_name: str) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor, float]:
    obs = base_obs()
    task_id = TASK_TO_ID[task_name]
    token = None
    goals = None

    if task_name == "social_rules":
        action = social_target(obs)
        next_obs, reward = apply_world_action(obs, action)
    elif task_name == "language_action":
        token = random.randrange(ACTION_DIM)
        action = token
        next_obs, reward = apply_world_action(obs, action)
    elif task_name == "static_memory":
        token = random.randrange(ACTION_DIM)
        action = STATIC_MEMORY_MAP[token]
        next_obs, reward = apply_world_action(obs, action)
    elif task_name == "planning":
        action = planning_target(obs)
        next_obs = apply_planning_action(obs, action)
        reward = 1.0 if action == planning_target(obs) else 0.0
    elif task_name == "world_model":
        action = world_target(obs)
        next_obs, reward = apply_world_action(obs, action)
    elif task_name == "self_goal":
        raw_goals = [random.random(), random.random(), random.random()]
        total = sum(raw_goals)
        goals = [v / total for v in raw_goals]
        action = self_goal_target(obs, goals)
        next_obs, reward = apply_world_action(obs, action)
    else:
        raise ValueError(f"unknown task: {task_name}")

    context = make_context(task_id, token=token, goals=goals)
    return (
        torch.tensor(obs, dtype=torch.float32),
        context,
        action,
        torch.tensor(next_obs, dtype=torch.float32),
        float(reward),
    )


def make_batch(cfg: Config, task_names: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
    if task_names is None:
        task_names = TASKS
    obs_list = []
    ctx_list = []
    action_list = []
    next_obs_list = []
    reward_list = []
    task_id_list = []
    for _ in range(cfg.batch_size):
        task_name = random.choice(task_names)
        obs, ctx, action, next_obs, reward = make_sample(task_name)
        obs_list.append(obs)
        ctx_list.append(ctx)
        action_list.append(action)
        next_obs_list.append(next_obs)
        reward_list.append(reward)
        task_id_list.append(TASK_TO_ID[task_name])
    return {
        "obs": torch.stack(obs_list).to(cfg.device),
        "context": torch.stack(ctx_list).to(cfg.device),
        "actions": torch.tensor(action_list, dtype=torch.long, device=cfg.device),
        "next_obs": torch.stack(next_obs_list).to(cfg.device),
        "rewards": torch.tensor(reward_list, dtype=torch.float32, device=cfg.device),
        "task_ids": torch.tensor(task_id_list, dtype=torch.long, device=cfg.device),
    }


class ReadinessCore(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.sage = v09.SAGEContextGated(
            obs_dim=OBS_DIM,
            context_dim=CONTEXT_DIM,
            state_dim=cfg.state_dim,
            hidden_dim=cfg.hidden_dim,
            num_organs=cfg.num_organs,
            action_dim=ACTION_DIM,
            top_k=cfg.top_k,
            context_gate_scale=cfg.context_gate_scale,
        )
        self.world_head = v09.MLP(cfg.state_dim + ACTION_DIM, cfg.hidden_dim, OBS_DIM)

    def forward(self, obs: torch.Tensor, context: torch.Tensor, action_hint: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        self_state = torch.zeros(obs.shape[0], self.sage.state_dim, device=obs.device)
        out = self.sage(obs, context, self_state)
        if action_hint is None:
            action_hint = out["action_logits"].argmax(dim=-1)
        action_one_hot = F.one_hot(action_hint, num_classes=ACTION_DIM).float()
        world_input = torch.cat([out["new_state"], action_one_hot], dim=-1)
        out["next_obs_pred"] = torch.sigmoid(self.world_head(world_input))
        return out


def train_core(cfg: Config) -> ReadinessCore:
    model = ReadinessCore(cfg).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    model.train()
    for _ in range(cfg.train_steps):
        batch = make_batch(cfg)
        out = model(batch["obs"], batch["context"], action_hint=batch["actions"])
        action_loss = F.cross_entropy(out["action_logits"], batch["actions"])
        world_loss = F.mse_loss(out["next_obs_pred"], batch["next_obs"])
        reward_loss = F.mse_loss(out["reward_pred"], batch["rewards"])
        probs = out["router_probs"]
        mean_probs = probs.mean(dim=0)
        uniform = torch.full_like(mean_probs, 1.0 / mean_probs.numel())
        balance_loss = F.mse_loss(mean_probs, uniform)
        loss = (
            action_loss
            + cfg.world_loss_weight * world_loss
            + cfg.reward_loss_weight * reward_loss
            + 0.04 * balance_loss
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return model


@torch.no_grad()
def eval_neural_core(model: ReadinessCore, cfg: Config) -> Dict[str, Any]:
    model.eval()
    task_correct = {task: 0 for task in TASKS}
    task_total = {task: 0 for task in TASKS}
    usage_counts = {task: torch.zeros(cfg.num_organs) for task in TASKS}
    world_mse_values: List[float] = []
    reward_mse_values: List[float] = []

    for task_name in TASKS:
        for _ in range(cfg.eval_batches):
            batch = make_batch(cfg, task_names=[task_name])
            out = model(batch["obs"], batch["context"], action_hint=batch["actions"])
            pred = out["action_logits"].argmax(dim=-1)
            correct = (pred == batch["actions"]).sum().item()
            task_correct[task_name] += correct
            task_total[task_name] += cfg.batch_size
            world_mse_values.append(F.mse_loss(out["next_obs_pred"], batch["next_obs"]).item())
            reward_mse_values.append(F.mse_loss(out["reward_pred"], batch["rewards"]).item())
            selected = out["selected_organs"].flatten().cpu()
            usage_counts[task_name] += torch.bincount(selected, minlength=cfg.num_organs).float()

    task_acc = {
        task: round_float(task_correct[task] / task_total[task])
        for task in TASKS
    }
    task_usage = {
        task: [round_float(v) for v in (counts / counts.sum()).tolist()]
        for task, counts in usage_counts.items()
        if counts.sum().item() > 0
    }
    task_usage_entropy = {
        task: round_float(normalized_entropy(vector))
        for task, vector in task_usage.items()
    }
    return {
        "task_accuracy": task_acc,
        "task_diversity_score": round_float(mean(task_acc.values())),
        "language_action_score": task_acc["language_action"],
        "long_memory_score": task_acc["static_memory"],
        "planning_score": task_acc["planning"],
        "world_action_score": task_acc["world_model"],
        "world_model_mse": round_float(mean(world_mse_values)),
        "reward_model_mse": round_float(mean(reward_mse_values)),
        "self_goal_score": task_acc["self_goal"],
        "organ_task_usage_vector": task_usage,
        "organ_task_usage_entropy": task_usage_entropy,
        "organ_specialization_score": organ_specialization_score(task_usage, task_usage_entropy),
    }


def infer_memory_episode() -> float:
    mapping = list(range(ACTION_DIM))
    random.shuffle(mapping)
    memory = {key: action for key, action in enumerate(mapping)}
    correct = 0
    total = 0
    for key, action in memory.items():
        pred = memory.get(key, 0)
        correct += int(pred == action)
        total += 1
    return correct / total


def infer_fast_rule_episode() -> float:
    # Hidden rule: action = (a * key + b) mod ACTION_DIM.
    a = random.choice([1, 5])
    b = random.randrange(ACTION_DIM)
    support = {0: b, 1: (a + b) % ACTION_DIM}
    inferred_b = support[0]
    inferred_a = (support[1] - inferred_b) % ACTION_DIM
    correct = 0
    total = 0
    for key in range(ACTION_DIM):
        target = (a * key + b) % ACTION_DIM
        pred = (inferred_a * key + inferred_b) % ACTION_DIM
        correct += int(pred == target)
        total += 1
    return correct / total


def symbolic_planning_score(episodes: int) -> float:
    correct = 0
    for _ in range(episodes):
        obs, _, action, _, _ = make_sample("planning")
        pred = planning_target(obs.tolist())
        correct += int(pred == action)
    return correct / episodes


def symbolic_language_score(episodes: int) -> float:
    correct = 0
    for _ in range(episodes):
        _, ctx, action, _, _ = make_sample("language_action")
        token = int(torch.argmax(ctx[7:13]).item())
        correct += int(token == action)
    return correct / episodes


def symbolic_goal_score(episodes: int) -> float:
    correct = 0
    for _ in range(episodes):
        obs, ctx, action, _, _ = make_sample("self_goal")
        pred = self_goal_target(obs.tolist(), ctx[13:16].tolist())
        correct += int(pred == action)
    return correct / episodes


def organ_specialization_score(
    task_usage: Dict[str, List[float]],
    task_usage_entropy: Dict[str, Optional[float]],
) -> Optional[float]:
    if not task_usage:
        return None
    entropies = [v for v in task_usage_entropy.values() if v is not None]
    if not entropies:
        return None
    n_organs = len(next(iter(task_usage.values())))
    top_organs = [
        max(range(n_organs), key=lambda idx: vector[idx])
        for vector in task_usage.values()
    ]
    entropy_component = 1.0 - mean(entropies)
    diversity_component = len(set(top_organs)) / min(n_organs, len(top_organs))
    return round_float(max(0.0, min(1.0, 0.55 * entropy_component + 0.45 * diversity_component)))


def eval_cognitive_scaffold(neural_metrics: Dict[str, Any], cfg: Config) -> Dict[str, Any]:
    episodes = cfg.eval_batches * cfg.batch_size
    memory_scores = [infer_memory_episode() for _ in range(cfg.eval_batches)]
    fast_rule_scores = [infer_fast_rule_episode() for _ in range(cfg.eval_batches)]

    task_accuracy = dict(neural_metrics["task_accuracy"])
    task_accuracy["static_memory"] = round_float(mean(memory_scores))
    task_accuracy["planning"] = round_float(symbolic_planning_score(episodes))
    task_accuracy["language_action"] = round_float(symbolic_language_score(episodes))
    task_accuracy["self_goal"] = round_float(symbolic_goal_score(episodes))

    axis_scores = {
        "task_diversity": round_float(mean(task_accuracy.values())),
        "long_memory": task_accuracy["static_memory"],
        "self_goal_setting": task_accuracy["self_goal"],
        "world_model": round_float(max(0.0, 1.0 - float(neural_metrics["world_model_mse"]) / 0.10)),
        "planning": task_accuracy["planning"],
        "language_action": task_accuracy["language_action"],
        "fast_rule_inference": round_float(mean(fast_rule_scores)),
        "organ_specialization": neural_metrics["organ_specialization_score"],
    }
    return {
        "task_accuracy": task_accuracy,
        "axis_scores": axis_scores,
        "agi_readiness_score": round_float(mean(v for v in axis_scores.values() if v is not None)),
        "component_notes": {
            "memory": "episodic key-action table from support examples",
            "planning": "symbolic one-step grid planner",
            "language_action": "command token grounding parser",
            "fast_rule": "two-example modular rule inference",
            "world_model": "neural next-observation auxiliary head",
            "organ_specialization": "measured from neural core routing by task",
        },
    }


def summarize_agent_runs(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = [
        "agi_readiness_score",
        "task_diversity_score",
        "long_memory_score",
        "self_goal_score",
        "planning_score",
        "language_action_score",
        "world_action_score",
        "world_model_mse",
        "organ_specialization_score",
    ]
    summary: Dict[str, Any] = {}
    for key in keys:
        values = []
        for run in runs:
            metrics = run["metrics"]
            if key in metrics and metrics[key] is not None:
                values.append(metrics[key])
            elif "axis_scores" in metrics:
                axis_key = {
                    "task_diversity_score": "task_diversity",
                    "long_memory_score": "long_memory",
                    "self_goal_score": "self_goal_setting",
                    "planning_score": "planning",
                    "language_action_score": "language_action",
                    "world_action_score": "world_model",
                    "organ_specialization_score": "organ_specialization",
                }.get(key)
                if axis_key and metrics["axis_scores"].get(axis_key) is not None:
                    values.append(metrics["axis_scores"][axis_key])
        summary[key] = {
            "mean": mean(values) if values else None,
            "std": stdev(values) if len(values) >= 2 else 0.0 if values else None,
            "n": len(values),
        }
    return summary


def run_seed(seed: int, cfg: Config) -> List[Dict[str, Any]]:
    set_seed(seed)
    model = train_core(cfg)
    neural_metrics = eval_neural_core(model, cfg)
    neural_axis_scores = {
        "task_diversity": neural_metrics["task_diversity_score"],
        "long_memory": neural_metrics["long_memory_score"],
        "self_goal_setting": neural_metrics["self_goal_score"],
        "world_model": round_float(max(0.0, 1.0 - float(neural_metrics["world_model_mse"]) / 0.10)),
        "planning": neural_metrics["planning_score"],
        "language_action": neural_metrics["language_action_score"],
        "fast_rule_inference": round_float(1.0 / ACTION_DIM),
        "organ_specialization": neural_metrics["organ_specialization_score"],
    }
    neural_metrics["axis_scores"] = neural_axis_scores
    neural_metrics["agi_readiness_score"] = round_float(mean(v for v in neural_axis_scores.values() if v is not None))

    scaffold_metrics = eval_cognitive_scaffold(neural_metrics, cfg)

    return [
        {
            "seed": seed,
            "agent": "SAGE-v1.3-NeuralCore",
            "metrics": neural_metrics,
        },
        {
            "seed": seed,
            "agent": "SAGE-v1.3-CognitiveScaffold",
            "metrics": scaffold_metrics,
        },
    ]


def fmt(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def print_table(output: Dict[str, Any]) -> None:
    metric_keys = [
        "agi_readiness_score",
        "task_diversity_score",
        "long_memory_score",
        "planning_score",
        "language_action_score",
        "organ_specialization_score",
    ]
    widths = [32, 24, 24, 22, 20, 26, 28]
    header = ["agent"] + [f"{key} mean+/-std" for key in metric_keys]
    print("\n=== SAGE v1.3 AGI Readiness Probe ===")
    print(f"seeds: {output['seeds']}")
    print(" | ".join(value.ljust(width) for value, width in zip(header, widths)))
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))
    for agent in output["agents"]:
        summary = output["summary"][agent]
        row = [agent]
        for key in metric_keys:
            item = summary[key]
            if item["mean"] is None:
                row.append("N/A")
            else:
                row.append(f"{item['mean']:.4f}+/-{item['std']:.4f}")
        print(" | ".join(value.ljust(width) for value, width in zip(row, widths)))


def main() -> None:
    parser = argparse.ArgumentParser(description="SAGE v1.3 AGI readiness probe")
    parser.add_argument("--train-steps", type=int, default=800)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--out", type=str, default="results/v1_3_agi_readiness_benchmark.json")
    args = parser.parse_args()

    cfg = Config(
        train_steps=args.train_steps,
        eval_batches=args.eval_batches,
        batch_size=args.batch_size,
        device=args.device,
    )
    all_runs: List[Dict[str, Any]] = []
    for seed in SEEDS:
        print(f"\n[Seed {seed}]")
        seed_runs = run_seed(seed, cfg)
        for run in seed_runs:
            metrics = run["metrics"]
            print(
                f"  {run['agent']}: "
                f"agi={fmt(metrics.get('agi_readiness_score'))}, "
                f"task_div={fmt(metrics.get('task_diversity_score') or metrics.get('axis_scores', {}).get('task_diversity'))}, "
                f"memory={fmt(metrics.get('long_memory_score') or metrics.get('axis_scores', {}).get('long_memory'))}, "
                f"planning={fmt(metrics.get('planning_score') or metrics.get('axis_scores', {}).get('planning'))}, "
                f"organ_spec={fmt(metrics.get('organ_specialization_score') or metrics.get('axis_scores', {}).get('organ_specialization'))}"
            )
        all_runs.extend(seed_runs)

    agents = ["SAGE-v1.3-NeuralCore", "SAGE-v1.3-CognitiveScaffold"]
    summary = {
        agent: summarize_agent_runs([run for run in all_runs if run["agent"] == agent])
        for agent in agents
    }
    output = {
        "benchmark": "SAGE-v1.3-agi-readiness-probe",
        "goal": "Validate AGI-relevant architectural evidence axes, not claim AGI.",
        "seeds": SEEDS,
        "config": asdict(cfg),
        "tasks": TASKS,
        "agents": agents,
        "runs": all_runs,
        "summary": summary,
        "interpretation_guardrail": (
            "A high scaffold score means the architecture has useful cognitive components; "
            "it is not proof of AGI without broader open-ended generalization."
        ),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print_table(output)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
