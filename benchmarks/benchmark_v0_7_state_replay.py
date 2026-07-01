import json
import os
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from main import SAGEv0, MLP


# =========================
# Config
# =========================

@dataclass
class Config:
    steps_per_phase: int = 500
    num_phases: int = 4
    batch_size: int = 64
    eval_batches: int = 30
    adaptation_window: int = 80
    lr: float = 1e-3
    device: str = "cpu"
    seed: int = 42

    replay_capacity: int = 5000
    replay_ratio: float = 0.35
    state_ema: float = 0.05


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


# =========================
# Continual Environment
# =========================

class ContinualSocialEnv:
    ACTIONS = {
        0: "silence",
        1: "short_greeting",
        2: "ask_question",
        3: "empathize",
        4: "explain",
        5: "avoid",
    }

    def __init__(self):
        self.obs_dim = 8
        self.action_dim = 6

    def sample_obs(self):
        return torch.tensor([
            random.random(),
            random.random(),
            random.random(),
            random.random(),
            random.random(),
            random.random(),
            random.random(),
            random.random(),
        ], dtype=torch.float32)

    def best_action(self, phase: int, obs: torch.Tensor) -> int:
        user_spoke, is_stranger, closeness, positivity, risk, noise, energy, talk_pressure = obs.tolist()

        # Phase 0: 기본 사회 규칙
        if phase == 0:
            if risk > 0.75:
                return 5
            if talk_pressure > 0.75 or energy < 0.2:
                return 0
            if user_spoke > 0.5 and is_stranger > 0.5 and positivity > 0.4:
                return 1
            if closeness > 0.6 and positivity > 0.5:
                return 2
            if positivity < 0.35 and risk < 0.5:
                return 3
            if noise < 0.4 and energy > 0.6:
                return 4
            return 1

        # Phase 1: 호기심/대화 확장 규칙
        if phase == 1:
            if risk > 0.8:
                return 5
            if energy < 0.15 or talk_pressure > 0.85:
                return 0
            if positivity > 0.55 and user_spoke > 0.4:
                return 2
            if closeness > 0.5 and positivity > 0.45:
                return 2
            if positivity < 0.3:
                return 3
            if noise < 0.3 and energy > 0.7:
                return 4
            return 1

        # Phase 2: 위험 회피 강화 규칙
        if phase == 2:
            if risk > 0.5:
                return 5
            if talk_pressure > 0.6 or energy < 0.3:
                return 0
            if positivity < 0.45:
                return 3
            if is_stranger > 0.6:
                return 1
            if closeness > 0.65:
                return 2
            return 1

        # Phase 3: 에너지 절약 규칙
        if phase == 3:
            if risk > 0.65:
                return 5
            if energy < 0.5:
                return 0
            if noise > 0.7:
                return 0
            if talk_pressure > 0.55:
                return 0
            if positivity > 0.7 and closeness > 0.5:
                return 2
            if positivity < 0.35:
                return 3
            if noise < 0.25 and energy > 0.75:
                return 4
            return 1

        raise ValueError(f"Unknown phase: {phase}")

    def step(self, phase: int, obs: torch.Tensor, action: int):
        target = self.best_action(phase, obs)
        reward = 1.0 if action == target else -0.3

        risk = obs[4].item()
        talk_pressure = obs[7].item()
        energy = obs[6].item()
        noise = obs[5].item()

        if risk > 0.75 and action != 5:
            reward -= 1.0

        if talk_pressure > 0.7 and action == 4:
            reward -= 0.7

        if energy < 0.25 and action in [2, 4]:
            reward -= 0.5

        if noise > 0.75 and action == 4:
            reward -= 0.5

        action_cost = {
            0: 0.02,
            1: 0.05,
            2: 0.12,
            3: 0.10,
            4: 0.25,
            5: 0.06,
        }[action]

        reward -= action_cost
        return self.sample_obs(), reward, target


# =========================
# Replay Memory
# =========================

class ReplayMemory:
    def __init__(self, capacity: int):
        self.data = deque(maxlen=capacity)

    def __len__(self):
        return len(self.data)

    def add_batch(self, obs, actions, phases):
        obs = obs.detach().cpu()
        actions = actions.detach().cpu()
        phases = phases.detach().cpu()

        for i in range(obs.shape[0]):
            self.data.append((obs[i], actions[i], phases[i]))

    def sample(self, n: int, device: str):
        n = min(n, len(self.data))

        if n <= 0:
            return None

        samples = random.sample(self.data, n)

        obs = torch.stack([x[0] for x in samples]).to(device)
        actions = torch.stack([x[1] for x in samples]).long().to(device)
        phases = torch.stack([x[2] for x in samples]).long().to(device)

        return obs, actions, phases


def make_current_batch(env, phase: int, batch_size: int, device: str):
    obs_list = []
    action_list = []
    phase_list = []

    for _ in range(batch_size):
        obs = env.sample_obs()
        target = env.best_action(phase, obs)

        obs_list.append(obs)
        action_list.append(target)
        phase_list.append(phase)

    obs = torch.stack(obs_list).to(device)
    actions = torch.tensor(action_list, dtype=torch.long, device=device)
    phases = torch.tensor(phase_list, dtype=torch.long, device=device)

    return obs, actions, phases


def make_training_batch(env, memory, phase: int, cfg: Config, use_replay: bool):
    if not use_replay or len(memory) < cfg.batch_size:
        return make_current_batch(env, phase, cfg.batch_size, cfg.device)

    replay_n = int(cfg.batch_size * cfg.replay_ratio)
    current_n = cfg.batch_size - replay_n

    current_obs, current_actions, current_phases = make_current_batch(
        env, phase, current_n, cfg.device
    )

    replay = memory.sample(replay_n, cfg.device)

    if replay is None:
        return make_current_batch(env, phase, cfg.batch_size, cfg.device)

    replay_obs, replay_actions, replay_phases = replay

    obs = torch.cat([current_obs, replay_obs], dim=0)
    actions = torch.cat([current_actions, replay_actions], dim=0)
    phases = torch.cat([current_phases, replay_phases], dim=0)

    return obs, actions, phases


# =========================
# Models
# =========================

class FeedForwardPolicy(nn.Module):
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
        state = F.layer_norm(state + integrated, [state.shape[-1]])

        action_logits = self.action_head(state)

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


class PersistentSAGEController:
    """
    SAGE에 persistent self-state를 붙이는 래퍼.
    모델 파라미터는 바꾸지 않고, 입력 self_state만 누적한다.
    """

    def __init__(self, model: SAGEv0, cfg: Config, enabled: bool):
        self.model = model
        self.cfg = cfg
        self.enabled = enabled
        self.state = torch.zeros(1, model.state_dim, device=cfg.device)

    def reset(self):
        self.state = torch.zeros(1, self.model.state_dim, device=self.cfg.device)

    def forward(self, obs):
        if not self.enabled:
            self_state = torch.zeros(obs.shape[0], self.model.state_dim, device=obs.device)
            return self.model(obs, self_state)

        self_state = self.state.expand(obs.shape[0], -1)
        out = self.model(obs, self_state)

        with torch.no_grad():
            batch_state = out["new_state"].mean(dim=0, keepdim=True)
            self.state = (1.0 - self.cfg.state_ema) * self.state + self.cfg.state_ema * batch_state
            self.state = F.layer_norm(self.state, [self.model.state_dim])

        return out


def forward_model(model, obs, sage_controller=None):
    if isinstance(model, SAGEv0):
        return sage_controller.forward(obs)

    return model(obs)


# =========================
# Train / Eval
# =========================

def train_model(name, model, cfg: Config, use_energy: bool, use_state: bool, use_replay: bool):
    env = ContinualSocialEnv()
    model = model.to(cfg.device)

    memory = ReplayMemory(cfg.replay_capacity)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    sage_controller = None

    if isinstance(model, SAGEv0):
        sage_controller = PersistentSAGEController(model, cfg, enabled=use_state)

    total_steps = cfg.steps_per_phase * cfg.num_phases
    phase_acc_history = defaultdict(list)

    pbar = tqdm(range(total_steps), desc=f"train {name}")

    for step in pbar:
        phase = min(step // cfg.steps_per_phase, cfg.num_phases - 1)

        obs, target_actions, phases = make_training_batch(
            env=env,
            memory=memory,
            phase=phase,
            cfg=cfg,
            use_replay=use_replay,
        )

        out = forward_model(model, obs, sage_controller)

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
            correct = (pred_actions == target_actions).float()
            acc = correct.mean().item()

            # 현재 phase 샘플만 history에 기록
            current_mask = phases == phase
            if current_mask.any():
                current_acc = (
                    pred_actions[current_mask] == target_actions[current_mask]
                ).float().mean().item()
                phase_acc_history[phase].append(current_acc)

            if isinstance(model, SAGEv0) and use_energy:
                model.update_energy(out["selected_organs"], correct)

            # Replay에는 현재 phase에서 새로 만든 샘플만 저장
            current_obs, current_actions, current_phases = make_current_batch(
                env, phase, cfg.batch_size // 2, cfg.device
            )
            memory.add_batch(current_obs, current_actions, current_phases)

        if step % 100 == 0:
            if isinstance(model, SAGEv0):
                energy = [round(x, 2) for x in model.organ_energy.cpu().tolist()]
                pbar.set_description(
                    f"train {name} phase={phase} loss={loss.item():.3f} acc={acc:.3f} energy={energy}"
                )
            else:
                pbar.set_description(
                    f"train {name} phase={phase} loss={loss.item():.3f} acc={acc:.3f}"
                )

    return model, phase_acc_history, sage_controller


@torch.no_grad()
def evaluate_model(name, model, cfg: Config, phase_acc_history, sage_controller, use_state, use_replay, use_energy):
    env = ContinualSocialEnv()
    model.eval()

    if sage_controller is not None:
        sage_controller.reset()

    total_correct = 0
    total_samples = 0
    total_reward = 0.0
    total_time = 0.0

    phase_acc = {}
    usage_counts = None

    for phase in range(cfg.num_phases):
        phase_correct = 0
        phase_samples = 0

        for _ in range(cfg.eval_batches):
            obs, target_actions, phases = make_current_batch(
                env=env,
                phase=phase,
                batch_size=cfg.batch_size,
                device=cfg.device,
            )

            start = time.perf_counter()
            out = forward_model(model, obs, sage_controller)
            end = time.perf_counter()

            total_time += end - start

            pred_actions = out["action_logits"].argmax(dim=-1)

            correct_count = (pred_actions == target_actions).sum().item()

            phase_correct += correct_count
            phase_samples += cfg.batch_size

            total_correct += correct_count
            total_samples += cfg.batch_size

            for i in range(cfg.batch_size):
                _, reward, _ = env.step(
                    phase,
                    obs[i].cpu(),
                    int(pred_actions[i].item())
                )
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
                    minlength=n,
                ).float()

        phase_acc[f"phase_{phase}"] = round(phase_correct / phase_samples, 4)

    final_avg_acc = total_correct / total_samples
    avg_reward = total_reward / total_samples
    ms_per_batch = (total_time / (cfg.eval_batches * cfg.num_phases)) * 1000.0

    first_window = {}
    last_window = {}
    adapt_gain = {}

    for phase in range(cfg.num_phases):
        values = phase_acc_history[phase]

        if len(values) == 0:
            first_window[f"phase_{phase}"] = None
            last_window[f"phase_{phase}"] = None
            adapt_gain[f"phase_{phase}"] = None
            continue

        n = min(cfg.adaptation_window, len(values))
        first = sum(values[:n]) / n
        last = sum(values[-n:]) / n

        first_window[f"phase_{phase}"] = round(first, 4)
        last_window[f"phase_{phase}"] = round(last, 4)
        adapt_gain[f"phase_{phase}"] = round(last - first, 4)

    valid_gains = [v for v in adapt_gain.values() if v is not None]
    avg_adapt_gain = sum(valid_gains) / len(valid_gains)

    if hasattr(model, "effective_param_count"):
        effective_params = model.effective_param_count()
    elif isinstance(model, SAGEv0):
        effective_params = sage_effective_param_count(model)
    else:
        effective_params = count_params(model)

    reward_per_million_params = avg_reward / (effective_params / 1_000_000)

    if usage_counts is not None and usage_counts.sum() > 0:
        usage = (usage_counts / usage_counts.sum()).tolist()
        usage = [round(x, 3) for x in usage]
    else:
        usage = None

    result = {
        "model": name,
        "final_avg_acc": round(final_avg_acc, 4),
        "avg_reward": round(avg_reward, 4),
        "avg_adapt_gain": round(avg_adapt_gain, 4),
        "ms_per_batch": round(ms_per_batch, 4),
        "effective_params": effective_params,
        "reward_per_million_effective_params": round(reward_per_million_params, 4),
        "phase_acc": phase_acc,
        "first_window_acc": first_window,
        "last_window_acc": last_window,
        "adapt_gain": adapt_gain,
        "usage": usage,
        "use_state": use_state,
        "use_replay": use_replay,
        "use_energy": use_energy,
    }

    if isinstance(model, SAGEv0):
        result["final_energy"] = [
            round(x, 4) for x in model.organ_energy.cpu().tolist()
        ]

    return result


def print_results(results):
    print("\n=== SAGE-v0.7 Persistent State + Replay Benchmark ===")

    header = (
        f"{'model':<22} "
        f"{'avg_acc':>8} "
        f"{'reward':>8} "
        f"{'adapt':>8} "
        f"{'ms/batch':>10} "
        f"{'eff_params':>12} "
        f"{'reward/Mparam':>14} "
        f"{'usage':>26}"
    )

    print(header)
    print("-" * len(header))

    for r in results:
        print(
            f"{r['model']:<22} "
            f"{r['final_avg_acc']:>8.4f} "
            f"{r['avg_reward']:>8.4f} "
            f"{r['avg_adapt_gain']:>8.4f} "
            f"{r['ms_per_batch']:>10.4f} "
            f"{r['effective_params']:>12} "
            f"{r['reward_per_million_effective_params']:>14.4f} "
            f"{str(r['usage']):>26}"
        )

    print("\nPhase accuracy:")
    for r in results:
        print(r["model"], r["phase_acc"])

    print("\nAdapt gain:")
    for r in results:
        print(r["model"], r["adapt_gain"])


def main():
    cfg = Config()
    set_seed(cfg.seed)

    env = ContinualSocialEnv()

    model_specs = [
        (
            "SingleMLP",
            FeedForwardPolicy(env.obs_dim, 64, 128, env.action_dim),
            False,
            False,
            False,
        ),
        (
            "SingleMLP-Replay",
            FeedForwardPolicy(env.obs_dim, 64, 128, env.action_dim),
            False,
            False,
            True,
        ),
        (
            "BasicMoE",
            BasicMoE(env.obs_dim, 64, 128, 4, env.action_dim, 2),
            False,
            False,
            False,
        ),
        (
            "BasicMoE-Replay",
            BasicMoE(env.obs_dim, 64, 128, 4, env.action_dim, 2),
            False,
            False,
            True,
        ),
        (
            "SAGE-v0.4",
            SAGEv0(env.obs_dim, 64, 128, 4, env.action_dim, 2),
            True,
            False,
            False,
        ),
        (
            "SAGE-StateOnly",
            SAGEv0(env.obs_dim, 64, 128, 4, env.action_dim, 2),
            True,
            True,
            False,
        ),
        (
            "SAGE-ReplayOnly",
            SAGEv0(env.obs_dim, 64, 128, 4, env.action_dim, 2),
            True,
            False,
            True,
        ),
        (
            "SAGE-v0.7-StateReplay",
            SAGEv0(env.obs_dim, 64, 128, 4, env.action_dim, 2),
            True,
            True,
            True,
        ),
    ]

    results = []

    for name, model, use_energy, use_state, use_replay in model_specs:
        print(f"\n\n===== {name} =====")
        set_seed(cfg.seed)

        trained_model, history, controller = train_model(
            name=name,
            model=model,
            cfg=cfg,
            use_energy=use_energy,
            use_state=use_state,
            use_replay=use_replay,
        )

        result = evaluate_model(
            name=name,
            model=trained_model,
            cfg=cfg,
            phase_acc_history=history,
            sage_controller=controller,
            use_state=use_state,
            use_replay=use_replay,
            use_energy=use_energy,
        )

        results.append(result)

    print_results(results)

    os.makedirs("results", exist_ok=True)
    save_path = "results/v0_7_state_replay_benchmark.json"

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    main()