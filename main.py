import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# =========================
# 1. Toy Social Environment
# =========================

class ToySocialEnv:
    """
    SAGE-v0 검증용 가짜 사회 환경.
    입력 obs:
    [0] user_spoke        상대가 말했는가
    [1] is_stranger       처음 보는 사람인가
    [2] closeness         친밀도
    [3] positivity        분위기 긍정도
    [4] risk              위험도
    [5] noise             월드 소음도
    [6] energy            현재 에너지
    [7] talk_pressure     최근 말 많이 했는가
    """

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
        user_spoke = random.random()
        is_stranger = random.random()
        closeness = random.random()
        positivity = random.random()
        risk = random.random()
        noise = random.random()
        energy = random.random()
        talk_pressure = random.random()

        return torch.tensor([
            user_spoke,
            is_stranger,
            closeness,
            positivity,
            risk,
            noise,
            energy,
            talk_pressure
        ], dtype=torch.float32)

    def best_action(self, obs):
        user_spoke, is_stranger, closeness, positivity, risk, noise, energy, talk_pressure = obs.tolist()

        # 위험하면 회피
        if risk > 0.75:
            return 5

        # 말 너무 많이 했거나 에너지 낮으면 침묵
        if talk_pressure > 0.75 or energy < 0.2:
            return 0

        # 상대가 말했고 처음 보는 사람이고 분위기 괜찮으면 짧은 인사
        if user_spoke > 0.5 and is_stranger > 0.5 and positivity > 0.4:
            return 1

        # 친밀도 높고 분위기 긍정적이면 질문
        if closeness > 0.6 and positivity > 0.5:
            return 2

        # 분위기가 낮으면 공감
        if positivity < 0.35 and risk < 0.5:
            return 3

        # 조용하고 에너지 충분하면 설명
        if noise < 0.4 and energy > 0.6:
            return 4

        return 1

    def step(self, obs, action):
        target = self.best_action(obs)
        reward = 1.0 if action == target else -0.3

        # 위험한 상황에서 회피 안 하면 큰 패널티
        risk = obs[4].item()
        if risk > 0.75 and action != 5:
            reward -= 1.0

        # 말 압박 높은데 긴 설명하면 패널티
        talk_pressure = obs[7].item()
        if talk_pressure > 0.7 and action == 4:
            reward -= 0.7

        # 계산 비용 비슷한 개념: 복잡한 행동은 비용이 높음
        action_cost = {
            0: 0.02,
            1: 0.05,
            2: 0.12,
            3: 0.10,
            4: 0.25,
            5: 0.06,
        }[action]

        reward -= action_cost

        next_obs = self.sample_obs()
        return next_obs, reward, target


# =========================
# 2. Model Components
# =========================

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class SAGEv0(nn.Module):
    """
    SAGE-v0:
    - Perception MLP
    - Self-State Core
    - Energy Router
    - 4개 Organ
    - Integrator
    - Action Head
    - Reward/Cost 예측 Head
    """

    def __init__(
        self,
        obs_dim=8,
        state_dim=64,
        hidden_dim=128,
        num_organs=4,
        action_dim=6,
        top_k=2,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.num_organs = num_organs
        self.top_k = top_k
        self.action_dim = action_dim

        self.perception = MLP(obs_dim, hidden_dim, state_dim)

        self.self_update = MLP(state_dim * 2, hidden_dim, state_dim)

        self.router = MLP(state_dim * 2, hidden_dim, num_organs)

        self.organs = nn.ModuleList([
            MLP(state_dim * 2, hidden_dim, state_dim)
            for _ in range(num_organs)
        ])

        self.integrator = MLP(state_dim * 2, hidden_dim, state_dim)

        self.action_head = MLP(state_dim, hidden_dim, action_dim)
        self.reward_head = MLP(state_dim, hidden_dim, 1)
        self.cost_head = MLP(state_dim, hidden_dim, 1)

        # 기관 에너지: 학습 파라미터가 아니라 생태계 상태값
        self.register_buffer("organ_energy", torch.ones(num_organs))

    def forward(self, obs, self_state):
        obs_emb = self.perception(obs)

        # Self-State 업데이트
        self_input = torch.cat([obs_emb, self_state], dim=-1)
        delta = self.self_update(self_input)
        new_state = F.layer_norm(self_state + 0.1 * delta, [self.state_dim])

        # Router
        router_input = torch.cat([obs_emb, new_state], dim=-1)
        router_logits = self.router(router_input)

        # 에너지 높은 기관은 조금 더 선택되기 쉬움
        energy_bias = torch.log(self.organ_energy + 1e-6).unsqueeze(0)
        router_logits = router_logits + 0.02 * energy_bias

        router_probs = F.softmax(router_logits, dim=-1)
        top_vals, top_idx = torch.topk(router_probs, self.top_k, dim=-1)

        organ_input = torch.cat([obs_emb, new_state], dim=-1)

        organ_mix = torch.zeros_like(new_state)

        # v0는 단순 구현. 나중에 최적화 가능.
        for k in range(self.top_k):
            idx = top_idx[:, k]
            weight = top_vals[:, k].unsqueeze(-1)

            out = torch.zeros_like(new_state)

            for organ_id in range(self.num_organs):
                mask = idx == organ_id
                if mask.any():
                    out[mask] = self.organs[organ_id](organ_input[mask])

            organ_mix += out * weight

        integrated_input = torch.cat([new_state, organ_mix], dim=-1)
        integrated = self.integrator(integrated_input)
        organism_state = F.layer_norm(new_state + integrated, [self.state_dim])

        action_logits = self.action_head(organism_state)
        reward_pred = self.reward_head(organism_state).squeeze(-1)
        cost_pred = F.softplus(self.cost_head(organism_state)).squeeze(-1)

        return {
            "action_logits": action_logits,
            "reward_pred": reward_pred,
            "cost_pred": cost_pred,
            "new_state": organism_state,
            "router_probs": router_probs,
            "selected_organs": top_idx,
        }
    
    @torch.no_grad()
    def update_energy(self, selected_organs, correct, lr=0.02, homeostasis=0.08):
        """
        v0.4 energy update

        selected_organs: [batch, top_k]
        correct: [batch] 1.0이면 정답, 0.0이면 오답

        핵심:
        - 에너지는 기본적으로 1.0으로 회귀한다.
        - 정답에 참여한 기관은 강화된다.
        - 오답에 참여한 기관은 약화된다.
        - batch 안에서 기관별 평균 기여도를 계산해서 업데이트한다.
        - v0.2처럼 전부 과충전되거나 v0.3처럼 전부 방전되는 것을 막는다.
        """

        # 1. 전체 기관은 1.0으로 회귀
        self.organ_energy += homeostasis * (1.0 - self.organ_energy)

        # 2. 정답이면 +1, 오답이면 -1
        score = correct.detach().float() * 2.0 - 1.0

        organ_score = torch.zeros_like(self.organ_energy)
        organ_count = torch.zeros_like(self.organ_energy)

        batch_size = selected_organs.shape[0]
        top_k = selected_organs.shape[1]

        # 3. 선택된 기관별 평균 성과 계산
        for i in range(batch_size):
            for k in range(top_k):
                organ_id = int(selected_organs[i, k].item())
                organ_score[organ_id] += score[i]
                organ_count[organ_id] += 1.0

        # 4. 기관별 평균 score만 반영
        for organ_id in range(self.num_organs):
            if organ_count[organ_id] > 0:
                mean_score = organ_score[organ_id] / organ_count[organ_id]
                self.organ_energy[organ_id] += lr * mean_score

        # 5. 극단값 방지
        self.organ_energy.clamp_(0.5, 1.6)


# =========================
# 3. Training
# =========================

@dataclass
class Config:
    episodes: int = 3000
    batch_size: int = 64
    lr: float = 1e-3
    device: str = "cpu"


def make_batch(env, batch_size):
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

    obs = torch.stack(obs_list)
    actions = torch.tensor(action_list, dtype=torch.long)
    rewards = torch.tensor(reward_list, dtype=torch.float32)

    return obs, actions, rewards


def train():
    cfg = Config()
    env = ToySocialEnv()

    model = SAGEv0(
        obs_dim=env.obs_dim,
        state_dim=64,
        hidden_dim=128,
        num_organs=4,
        action_dim=env.action_dim,
        top_k=2,
    ).to(cfg.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    avg_acc = 0.0
    avg_reward = 0.0

    pbar = tqdm(range(cfg.episodes))

    for step in pbar:
        obs, target_actions, rewards = make_batch(env, cfg.batch_size)

        obs = obs.to(cfg.device)
        target_actions = target_actions.to(cfg.device)
        rewards = rewards.to(cfg.device)

        self_state = torch.zeros(cfg.batch_size, model.state_dim, device=cfg.device)

        out = model(obs, self_state)

        action_loss = F.cross_entropy(out["action_logits"], target_actions)
        reward_loss = F.mse_loss(out["reward_pred"], rewards)

        # 기관을 너무 넓게 쓰지 않게 entropy 약하게 감소
        probs = out["router_probs"]
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()

        mean_probs = probs.mean(dim=0)
        uniform = torch.full_like(mean_probs, 1.0 / model.num_organs)
        load_balance_loss = F.mse_loss(mean_probs, uniform)

        # cost는 너무 커지지 않게
        cost_loss = out["cost_pred"].mean()
        loss = (
            action_loss
            + 0.3 * reward_loss
            + 0.005 * entropy
            + 0.05 * load_balance_loss
            + 0.001 * cost_loss
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        with torch.no_grad():
            pred_actions = out["action_logits"].argmax(dim=-1)
            acc = (pred_actions == target_actions).float().mean().item()
            reward_mean = rewards.mean().item()

            correct = (pred_actions == target_actions).float()
            model.update_energy(out["selected_organs"], correct)
            selected = out["selected_organs"].flatten()
            usage = torch.bincount(selected, minlength=model.num_organs).float()
            usage = usage / usage.sum()

            avg_acc = 0.98 * avg_acc + 0.02 * acc
            avg_reward = 0.98 * avg_reward + 0.02 * reward_mean

            if step % 50 == 0:
                energies = [round(x, 2) for x in model.organ_energy.cpu().tolist()]
                usage_list = [round(x, 2) for x in usage.cpu().tolist()]
                pbar.set_description(
                    f"loss={loss.item():.3f} acc={avg_acc:.3f} reward={avg_reward:.3f} "
                    f"energy={energies} usage={usage_list}"
                )

    print("\nTraining finished.")
    print("Final organ energy:", model.organ_energy.cpu().tolist())

    test_model(model, env)


def test_model(model, env):
    print("\n=== SAGE-v0 Test ===")

    action_names = env.ACTIONS

    for i in range(10):
        obs = env.sample_obs().unsqueeze(0)
        self_state = torch.zeros(1, model.state_dim)

        with torch.no_grad():
            out = model(obs, self_state)
            action = out["action_logits"].argmax(dim=-1).item()
            target = env.best_action(obs.squeeze(0))
            organs = out["selected_organs"].squeeze(0).tolist()

        print(f"\nCase {i+1}")
        print("obs:", [round(x, 2) for x in obs.squeeze(0).tolist()])
        print("selected organs:", organs)
        print("pred action:", action_names[action])
        print("best action:", action_names[target])


if __name__ == "__main__":
    train()