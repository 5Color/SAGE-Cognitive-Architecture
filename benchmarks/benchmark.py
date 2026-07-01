import json
import os
import random
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from main import ToySocialEnv, SAGEv0, MLP


@dataclass
class Config:
    episodes: int = 2000
    batch_size: int = 64
    eval_batches: int = 50
    lr: float = 1e-3
    device: str = "cpu"
    seed: int = 42


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def make_batch(env: ToySocialEnv, batch_size: int, device: str):
    obs_list = []
    action_list = []
    reward_list = []

    for _ in range(batch_size):
        obs = env.sample_obs()
        target = env.best_action(obs)
        _, reward, _ = env.step(obs, target)

        obs_list.append(obs)
        action_list.append(target)
        reward_list.append(reward)

    obs = torch.stack(obs_list).to(device)
    actions = torch.tensor(action_list, dtype=torch.long, device=device)
    rewards = torch.tensor(reward_list, dtype=torch.float32, device=device)

    return obs, actions, rewards


class SingleMLP(nn.Module):
    def __init__(self, obs_dim=8, state_dim=64, hidden_dim=128, action_dim=6):
        super().__init__()
        self.encoder = MLP(obs_dim, hidden_dim, state_dim)
        self.action_head = MLP(state_dim, hidden_dim, action_dim)

    def forward(self, obs):
        state = self.encoder(obs)
        action_logits = self.action_head(state)

        return {
            "action_logits": action_logits,
            "router_probs": None,
            "selected_organs": None,
        }

    def effective_param_count(self):
        return count_params(self)


class BasicMoE(nn.Module):
    """
    일반 MoE baseline.
    SAGE처럼 energy나 self-state는 없고,
    router가 expert top-k만 선택한다.
    """

    def __init__(
        self,
        obs_dim=8,
        state_dim=64,
        hidden_dim=128,
        num_experts=4,
        action_dim=6,
        top_k=2,
    ):
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k

        self.perception = MLP(obs_dim, hidden_dim, state_dim)
        self.router = MLP(state_dim, hidden_dim, num_experts)

        self.experts = nn.ModuleList([
            MLP(state_dim, hidden_dim, state_dim)
            for _ in range(num_experts)
        ])

        self.integrator = MLP(state_dim * 2, hidden_dim, state_dim)
        self.action_head = MLP(state_dim, hidden_dim, action_dim)

    def forward(self, obs):
        state = self.perception(obs)

        router_logits = self.router(state)
        router_probs = F.softmax(router_logits, dim=-1)
        top_vals, top_idx = torch.topk(router_probs, self.top_k, dim=-1)

        expert_mix = torch.zeros_like(state)

        for k in range(self.top_k):
            idx = top_idx[:, k]
            weight = top_vals[:, k].unsqueeze(-1)

            out = torch.zeros_like(state)

            for expert_id in range(self.num_experts):
                mask = idx == expert_id
                if mask.any():
                    out[mask] = self.experts[expert_id](state[mask])

            expert_mix += out * weight

        integrated = self.integrator(torch.cat([state, expert_mix], dim=-1))
        organism_state = F.layer_norm(state + integrated, [state.shape[-1]])

        action_logits = self.action_head(organism_state)

        return {
            "action_logits": action_logits,
            "router_probs": router_probs,
            "selected_organs": top_idx,
        }

    def effective_param_count(self):
        common = (
            count_params(self.perception)
            + count_params(self.router)
            + count_params(self.integrator)
            + count_params(self.action_head)
        )

        expert_one = count_params(self.experts[0])
        return common + self.top_k * expert_one


def sage_effective_param_count(model: SAGEv0):
    common = (
        count_params(model.perception)
        + count_params(model.self_update)
        + count_params(model.router)
        + count_params(model.integrator)
        + count_params(model.action_head)
        + count_params(model.reward_head)
        + count_params(model.cost_head)
    )

    organ_one = count_params(model.organs[0])
    return common + model.top_k * organ_one


def forward_model(model, obs):
    if isinstance(model, SAGEv0):
        self_state = torch.zeros(obs.shape[0], model.state_dim, device=obs.device)
        return model(obs, self_state)

    return model(obs)


def train_model(name, model, cfg: Config):
    env = ToySocialEnv()
    model = model.to(cfg.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    avg_acc = 0.0
    pbar = tqdm(range(cfg.episodes), desc=f"train {name}")

    for step in pbar:
        obs, target_actions, rewards = make_batch(env, cfg.batch_size, cfg.device)

        out = forward_model(model, obs)
        action_loss = F.cross_entropy(out["action_logits"], target_actions)

        loss = action_loss

        if out["router_probs"] is not None:
            probs = out["router_probs"]
            mean_probs = probs.mean(dim=0)
            uniform = torch.full_like(mean_probs, 1.0 / mean_probs.numel())
            load_balance_loss = F.mse_loss(mean_probs, uniform)
            loss = loss + 0.05 * load_balance_loss

        if "cost_pred" in out:
            cost_loss = out["cost_pred"].mean()
            loss = loss + 0.001 * cost_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        with torch.no_grad():
            pred_actions = out["action_logits"].argmax(dim=-1)
            acc = (pred_actions == target_actions).float().mean().item()

            if isinstance(model, SAGEv0):
                correct = (pred_actions == target_actions).float()
                model.update_energy(out["selected_organs"], correct)

            avg_acc = 0.98 * avg_acc + 0.02 * acc

        if step % 100 == 0:
            if isinstance(model, SAGEv0):
                energy = [round(x, 2) for x in model.organ_energy.cpu().tolist()]
                pbar.set_description(
                    f"train {name} loss={loss.item():.3f} acc={avg_acc:.3f} energy={energy}"
                )
            else:
                pbar.set_description(
                    f"train {name} loss={loss.item():.3f} acc={avg_acc:.3f}"
                )

    return model


@torch.no_grad()
def evaluate_model(name, model, cfg: Config):
    env = ToySocialEnv()
    model.eval()

    total_correct = 0
    total_samples = 0
    total_reward = 0.0
    total_time = 0.0

    usage_counts = None

    for _ in range(cfg.eval_batches):
        obs, target_actions, _ = make_batch(env, cfg.batch_size, cfg.device)

        start = time.perf_counter()
        out = forward_model(model, obs)
        end = time.perf_counter()

        total_time += end - start

        pred_actions = out["action_logits"].argmax(dim=-1)

        total_correct += (pred_actions == target_actions).sum().item()
        total_samples += cfg.batch_size

        for i in range(cfg.batch_size):
            _, reward, _ = env.step(obs[i].cpu(), int(pred_actions[i].item()))
            total_reward += reward

        selected = out["selected_organs"]

        if selected is not None:
            if isinstance(model, SAGEv0):
                n = model.num_organs
            else:
                n = model.num_experts

            if usage_counts is None:
                usage_counts = torch.zeros(n)

            usage_counts += torch.bincount(
                selected.flatten().cpu(),
                minlength=n
            ).float()

    accuracy = total_correct / total_samples
    avg_reward = total_reward / total_samples
    ms_per_batch = (total_time / cfg.eval_batches) * 1000.0

    if isinstance(model, SingleMLP):
        effective_params = model.effective_param_count()
        active_units = 1
    elif isinstance(model, BasicMoE):
        effective_params = model.effective_param_count()
        active_units = model.top_k
    else:
        effective_params = sage_effective_param_count(model)
        active_units = model.top_k

    reward_per_million_params = avg_reward / (effective_params / 1_000_000)

    if usage_counts is not None and usage_counts.sum() > 0:
        usage = (usage_counts / usage_counts.sum()).tolist()
        usage = [round(x, 3) for x in usage]
    else:
        usage = None

    result = {
        "model": name,
        "accuracy": round(accuracy, 4),
        "avg_reward": round(avg_reward, 4),
        "ms_per_batch": round(ms_per_batch, 4),
        "active_units": active_units,
        "effective_params": effective_params,
        "reward_per_million_effective_params": round(reward_per_million_params, 4),
        "usage": usage,
    }

    if isinstance(model, SAGEv0):
        result["final_energy"] = [
            round(x, 4) for x in model.organ_energy.cpu().tolist()
        ]

    return result


def print_results(results):
    print("\n=== SAGE-v0.5 Benchmark Results ===")

    header = (
        f"{'model':<12} "
        f"{'acc':>8} "
        f"{'reward':>8} "
        f"{'ms/batch':>10} "
        f"{'active':>8} "
        f"{'eff_params':>12} "
        f"{'reward/Mparam':>14} "
        f"{'usage':>24}"
    )

    print(header)
    print("-" * len(header))

    for r in results:
        print(
            f"{r['model']:<12} "
            f"{r['accuracy']:>8.4f} "
            f"{r['avg_reward']:>8.4f} "
            f"{r['ms_per_batch']:>10.4f} "
            f"{r['active_units']:>8} "
            f"{r['effective_params']:>12} "
            f"{r['reward_per_million_effective_params']:>14.4f} "
            f"{str(r['usage']):>24}"
        )

    print("\nNote:")
    print("- reward/Mparam은 매우 단순화한 proxy 지표임.")
    print("- 진짜 compute는 FLOPs 측정이 필요하지만, v0.5에서는 구조 비교용으로 충분함.")


def main():
    cfg = Config()
    set_seed(cfg.seed)

    env = ToySocialEnv()

    model_specs = [
        (
            "SingleMLP",
            SingleMLP(
                obs_dim=env.obs_dim,
                state_dim=64,
                hidden_dim=128,
                action_dim=env.action_dim,
            ),
        ),
        (
            "BasicMoE",
            BasicMoE(
                obs_dim=env.obs_dim,
                state_dim=64,
                hidden_dim=128,
                num_experts=4,
                action_dim=env.action_dim,
                top_k=2,
            ),
        ),
        (
            "SAGE-v0.4",
            SAGEv0(
                obs_dim=env.obs_dim,
                state_dim=64,
                hidden_dim=128,
                num_organs=4,
                action_dim=env.action_dim,
                top_k=2,
            ),
        ),
    ]

    results = []

    for name, model in model_specs:
        print(f"\n\n===== {name} =====")
        set_seed(cfg.seed)

        trained_model = train_model(name, model, cfg)
        result = evaluate_model(name, trained_model, cfg)
        results.append(result)

    print_results(results)

    os.makedirs("results", exist_ok=True)

    with open("results/v0_5_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\nSaved: results/v0_5_benchmark.json")


if __name__ == "__main__":
    main()