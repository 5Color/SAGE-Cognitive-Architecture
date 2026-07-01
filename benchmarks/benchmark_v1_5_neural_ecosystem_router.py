# benchmark_v1_5_neural_ecosystem_router.py
# SAGE v1.5 - Neural Ecosystem Router
#
# 목표:
# - v1.4에서 hand-coded scaffold organ이 강력하다는 것은 확인했다.
# - v1.5에서는 "어떤 organ을 호출할지"를 neural router가 학습할 수 있는지 검증한다.
#
# 핵심:
# - memory_organ / algebra_organ / concept_organ / planner_organ은 scaffold로 유지
# - organ 선택은 Random / FamilyOracle / NeuralRouter / NeuralRouter+Energy로 비교
# - AGI claim이 아니라 "생태계형 organ routing" 진단 실험이다.
#
# 필요 파일:
# - benchmark_v1_4_rule_transfer_memory.py
#
# 실행 예:
# python benchmark_v1_5_neural_ecosystem_router.py --train-episodes 40 --eval-episodes 20
# python benchmark_v1_5_neural_ecosystem_router.py --train-episodes 120 --eval-episodes 120

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import benchmark_v1_4_rule_transfer_memory as v14


SEEDS = [0, 1, 2, 3, 4]

TASK_FAMILIES = list(v14.TASK_FAMILIES)
ORGANS = list(v14.ORGANS)

ORGAN_TO_ID = {organ: i for i, organ in enumerate(ORGANS)}
ID_TO_ORGAN = {i: organ for organ, i in ORGAN_TO_ID.items()}

ACTION_DIM = v14.ACTION_DIM


SPECIALIZED_ORACLE = {
    "episodic_memory": "memory_organ",
    "affine_rule": "algebra_organ",
    "threshold_rule": "concept_organ",
    "language_action": "concept_organ",
    "grid_planning": "planner_organ",
    "world_dynamics": "planner_organ",
}


@dataclass
class Config:
    train_episodes: int = 120
    eval_episodes: int = 120
    queries_per_episode: int = 18
    support_per_episode: int = 6
    hidden_dim: int = 96
    epochs: int = 8
    lr: float = 2e-3
    batch_size: int = 256
    seed: int = 0
    device: str = "cpu"
    energy_bias: float = 0.35
    energy_lr: float = 0.08
    energy_decay: float = 0.995
    min_energy: float = 0.05
    max_energy: float = 3.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def round_float(value: Optional[float], digits: int = 4) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


def safe_std(values: Sequence[float]) -> float:
    return stdev(values) if len(values) >= 2 else 0.0


def normalized_entropy(values: Sequence[float]) -> Optional[float]:
    total = sum(max(0.0, float(v)) for v in values)
    if total <= 0 or len(values) <= 1:
        return None
    probs = [max(0.0, float(v)) / total for v in values if v > 0]
    entropy = -sum(p * math.log(p) for p in probs)
    return entropy / math.log(len(values))


def action_entropy(actions: Sequence[int]) -> float:
    if not actions:
        return 0.0
    counts = [0 for _ in range(ACTION_DIM)]
    for action in actions:
        counts[int(action) % ACTION_DIM] += 1
    value = normalized_entropy(counts)
    return float(value) if value is not None else 0.0


def make_episode(cfg: Config, family: str) -> v14.Episode:
    # v1.4 Episode factory는 queries/support 값만 쓰므로 episodes 값은 무관하다.
    episode_cfg = v14.Config(
        episodes=1,
        queries_per_episode=cfg.queries_per_episode,
        support_per_episode=cfg.support_per_episode,
        seed=cfg.seed,
    )
    return v14.make_episode(episode_cfg, family)


def key_type(value: Any) -> str:
    if isinstance(value, str):
        if value.startswith("symbol:"):
            return "symbol"
        if value.startswith("x:"):
            return "x"
        if value.startswith("word:"):
            return "word"
        return "str_other"

    if isinstance(value, tuple):
        if len(value) == 6 and value[0] == "transition":
            return "transition"
        if len(value) == 5:
            return "grid_state"
        if len(value) == 4:
            return "world_query"
        return "tuple_other"

    return "other"


KEY_TYPES = [
    "symbol",
    "x",
    "word",
    "grid_state",
    "world_query",
    "transition",
    "str_other",
    "tuple_other",
    "other",
]
KEY_TO_ID = {name: i for i, name in enumerate(KEY_TYPES)}


def one_hot(index: int, size: int) -> List[float]:
    out = [0.0 for _ in range(size)]
    if 0 <= index < size:
        out[index] = 1.0
    return out


def numeric_hint(value: Any) -> float:
    if isinstance(value, str) and ":" in value:
        try:
            raw = value.split(":", 1)[1]
            if raw.isdigit():
                return min(1.0, int(raw) / 999.0)
        except Exception:
            return 0.0
    if isinstance(value, tuple):
        nums = [x for x in value if isinstance(x, int)]
        if nums:
            return min(1.0, sum(abs(x) for x in nums) / (len(nums) * 10.0))
    return 0.0


def support_type_ratios(support: Sequence[Tuple[Any, int]]) -> List[float]:
    counts = [0 for _ in KEY_TYPES]
    for key, _ in support:
        counts[KEY_TO_ID.get(key_type(key), KEY_TO_ID["other"])] += 1
    total = max(1, len(support))
    return [c / total for c in counts]


def extract_features(
    episode: v14.Episode,
    query: Any,
    organ_energy: Optional[Sequence[float]] = None,
) -> List[float]:
    q_type = key_type(query)
    q_type_id = KEY_TO_ID.get(q_type, KEY_TO_ID["other"])

    support_keys = [key for key, _ in episode.support]
    support_actions = [int(action) for _, action in episode.support]
    exact_memory_match = 1.0 if query in support_keys else 0.0

    support_size_norm = min(1.0, len(episode.support) / 12.0)
    support_action_diversity = len(set(support_actions)) / max(1, ACTION_DIM)
    support_act_entropy = action_entropy(support_actions)

    has_transition_support = any(key_type(key) == "transition" for key in support_keys)
    has_word_support = any(key_type(key) == "word" for key in support_keys)
    has_x_support = any(key_type(key) == "x" for key in support_keys)

    tuple_len_norm = 0.0
    if isinstance(query, tuple):
        tuple_len_norm = min(1.0, len(query) / 8.0)

    if organ_energy is None:
        energy = [1.0 for _ in ORGANS]
    else:
        total_energy = sum(max(0.0, float(x)) for x in organ_energy)
        if total_energy <= 0:
            energy = [1.0 / len(ORGANS) for _ in ORGANS]
        else:
            energy = [max(0.0, float(x)) / total_energy for x in organ_energy]

    features: List[float] = []
    features.append(support_size_norm)
    features.extend(one_hot(q_type_id, len(KEY_TYPES)))
    features.extend(support_type_ratios(episode.support))
    features.extend([
        support_action_diversity,
        support_act_entropy,
        exact_memory_match,
        numeric_hint(query),
        tuple_len_norm,
        1.0 if has_transition_support else 0.0,
        1.0 if has_word_support else 0.0,
        1.0 if has_x_support else 0.0,
    ])
    features.extend(energy)
    return features


class MemoryOrgan:
    name = "memory_organ"

    def __init__(self) -> None:
        self.memory: Dict[Any, int] = {}
        self.default_action = 0

    def fit(self, episode: v14.Episode) -> None:
        self.memory = {key: action for key, action in episode.support}
        actions = [action for _, action in episode.support]
        if actions:
            self.default_action = max(set(actions), key=actions.count)

    def predict(self, query: Any) -> int:
        return int(self.memory.get(query, self.default_action))


class AlgebraOrgan:
    name = "algebra_organ"

    def __init__(self) -> None:
        self.affine: Optional[Tuple[int, int]] = None
        self.default_action = 0

    def fit(self, episode: v14.Episode) -> None:
        self.affine = None
        actions = [action for _, action in episode.support]
        if actions:
            self.default_action = max(set(actions), key=actions.count)

        pairs = []
        for key, action in episode.support:
            if isinstance(key, str) and key.startswith("x:"):
                try:
                    pairs.append((int(key.split(":", 1)[1]), int(action)))
                except Exception:
                    pass

        if len(pairs) >= 2:
            x0, y0 = pairs[0]
            x1, y1 = pairs[1]
            dx = (x1 - x0) % ACTION_DIM
            dy = (y1 - y0) % ACTION_DIM

            # v1.4 affine task uses a in [1, 5], so dx in [1, 5] is safely invertible here.
            if dx in [1, 5]:
                a = dy if dx == 1 else (-dy) % ACTION_DIM
                b = (y0 - a * x0) % ACTION_DIM
                self.affine = (a, b)

    def predict(self, query: Any) -> int:
        if self.affine is not None and isinstance(query, str) and query.startswith("x:"):
            try:
                x = int(query.split(":", 1)[1])
                a, b = self.affine
                return int((a * x + b) % ACTION_DIM)
            except Exception:
                pass
        return int(self.default_action)


class ConceptOrgan:
    name = "concept_organ"

    def __init__(self) -> None:
        self.threshold: Optional[Tuple[int, int, int]] = None
        self.default_action = 0
        self.lexicon: Dict[str, int] = {}

    def fit(self, episode: v14.Episode) -> None:
        self.threshold = None
        self.lexicon = {}
        actions = [action for _, action in episode.support]
        if actions:
            self.default_action = max(set(actions), key=actions.count)

        # language-action grounding scaffold
        for action, words in v14.LANG_ACTIONS.items():
            for word in words:
                self.lexicon[v14.make_key("word", word)] = int(action)

        # threshold concept scaffold
        pairs = []
        for key, action in episode.support:
            if isinstance(key, str) and key.startswith("x:"):
                try:
                    pairs.append((int(key.split(":", 1)[1]), int(action)))
                except Exception:
                    pass

        pairs = sorted(pairs)
        if len(pairs) >= 2:
            values = [action for _, action in pairs]
            for idx in range(1, len(pairs)):
                if values[idx] != values[idx - 1]:
                    self.threshold = (pairs[idx][0], values[idx - 1], values[idx])
                    break

    def predict(self, query: Any) -> int:
        if isinstance(query, str) and query.startswith("word:"):
            return int(self.lexicon.get(query, self.default_action))

        if self.threshold is not None and isinstance(query, str) and query.startswith("x:"):
            try:
                x = int(query.split(":", 1)[1])
                threshold, low_action, high_action = self.threshold
                return int(low_action if x < threshold else high_action)
            except Exception:
                pass

        return int(self.default_action)


class PlannerOrgan:
    name = "planner_organ"

    def __init__(self) -> None:
        self.world_deltas: Optional[Dict[int, Tuple[int, int]]] = None
        self.default_action = 0

    def fit(self, episode: v14.Episode) -> None:
        self.world_deltas = None
        actions = [action for _, action in episode.support]
        if actions:
            self.default_action = max(set(actions), key=actions.count)

        deltas: Dict[int, Tuple[int, int]] = {}
        for key, _ in episode.support:
            if (
                isinstance(key, tuple)
                and len(key) == 6
                and key[0] == "transition"
            ):
                _, x, y, action, nx, ny = key
                deltas[int(action)] = (int(nx) - int(x), int(ny) - int(y))
        if len(deltas) == 4:
            self.world_deltas = deltas

    def predict(self, query: Any) -> int:
        if isinstance(query, tuple) and len(query) == 5:
            return int(v14.plan_step(query))

        if isinstance(query, tuple) and len(query) == 4 and self.world_deltas is not None:
            return int(v14.world_best_action(query, self.world_deltas))

        return int(self.default_action)


def fit_organs(episode: v14.Episode) -> Dict[str, Any]:
    organs = {
        "memory_organ": MemoryOrgan(),
        "algebra_organ": AlgebraOrgan(),
        "concept_organ": ConceptOrgan(),
        "planner_organ": PlannerOrgan(),
    }
    for organ in organs.values():
        organ.fit(episode)
    return organs


def organ_predictions(episode: v14.Episode, query: Any) -> Dict[str, int]:
    organs = fit_organs(episode)
    return {name: int(organ.predict(query)) for name, organ in organs.items()}


def choose_target_organ(
    episode: v14.Episode,
    query: Any,
    target_action: int,
    predictions: Dict[str, int],
) -> str:
    correct = [organ for organ, pred in predictions.items() if int(pred) == int(target_action)]

    preferred = SPECIALIZED_ORACLE.get(episode.family, "memory_organ")
    if preferred in correct:
        return preferred

    # 자연생태계형 전문화를 위해 memory collapse를 바로 target으로 삼지 않는다.
    # 단, 정말 memory만 맞힌 경우에는 memory를 허용한다.
    fallback_order = ["algebra_organ", "concept_organ", "planner_organ", "memory_organ"]
    for organ in fallback_order:
        if organ in correct:
            return organ

    # 어느 organ도 못 맞히면 task family에 맞는 specialized organ을 학습 target으로 둔다.
    return preferred


class RouterNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, organ_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, organ_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_router_dataset(cfg: Config, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    set_seed(seed)

    xs: List[List[float]] = []
    ys: List[int] = []

    for family in TASK_FAMILIES:
        for _ in range(cfg.train_episodes):
            episode = make_episode(cfg, family)
            for query, target_action in episode.queries:
                preds = organ_predictions(episode, query)
                target_organ = choose_target_organ(episode, query, target_action, preds)
                xs.append(extract_features(episode, query, organ_energy=[1.0] * len(ORGANS)))
                ys.append(ORGAN_TO_ID[target_organ])

    x_tensor = torch.tensor(xs, dtype=torch.float32)
    y_tensor = torch.tensor(ys, dtype=torch.long)
    return x_tensor, y_tensor


def train_neural_router(cfg: Config, seed: int) -> RouterNet:
    x, y = build_router_dataset(cfg, seed)
    input_dim = x.shape[-1]

    model = RouterNet(input_dim, cfg.hidden_dim, len(ORGANS)).to(cfg.device)
    x = x.to(cfg.device)
    y = y.to(cfg.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    n = x.shape[0]
    indices = torch.arange(n, device=cfg.device)

    model.train()
    for _ in range(cfg.epochs):
        perm = indices[torch.randperm(n, device=cfg.device)]
        for start in range(0, n, cfg.batch_size):
            batch_idx = perm[start:start + cfg.batch_size]
            logits = model(x[batch_idx])
            loss = F.cross_entropy(logits, y[batch_idx])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    return model


class RouterPolicy:
    name = "RouterPolicy"

    def choose(self, episode: v14.Episode, query: Any, predictions: Dict[str, int], organ_energy: List[float]) -> str:
        raise NotImplementedError

    def update_energy(self, chosen_organ: str, correct: bool, cfg: Config, organ_energy: List[float]) -> None:
        return None


class RandomRouter(RouterPolicy):
    name = "RandomRouter"

    def choose(self, episode, query, predictions, organ_energy):
        return random.choice(ORGANS)


class FamilyOracleRouter(RouterPolicy):
    name = "FamilyOracleRouter"

    def choose(self, episode, query, predictions, organ_energy):
        return SPECIALIZED_ORACLE.get(episode.family, "memory_organ")


class NeuralRouter(RouterPolicy):
    name = "NeuralRouter"

    def __init__(self, model: RouterNet, cfg: Config):
        self.model = model
        self.cfg = cfg

    @torch.no_grad()
    def choose(self, episode, query, predictions, organ_energy):
        self.model.eval()
        features = extract_features(episode, query, organ_energy=[1.0] * len(ORGANS))
        x = torch.tensor([features], dtype=torch.float32, device=self.cfg.device)
        logits = self.model(x)
        idx = int(torch.argmax(logits, dim=-1).item())
        return ID_TO_ORGAN[idx]


class NeuralRouterEnergy(RouterPolicy):
    name = "NeuralRouterEnergy"

    def __init__(self, model: RouterNet, cfg: Config):
        self.model = model
        self.cfg = cfg

    @torch.no_grad()
    def choose(self, episode, query, predictions, organ_energy):
        self.model.eval()
        features = extract_features(episode, query, organ_energy=organ_energy)
        x = torch.tensor([features], dtype=torch.float32, device=self.cfg.device)
        logits = self.model(x).squeeze(0)

        energy = torch.tensor(
            [max(self.cfg.min_energy, float(v)) for v in organ_energy],
            dtype=torch.float32,
            device=self.cfg.device,
        )
        energy_bias = self.cfg.energy_bias * torch.log(energy)
        idx = int(torch.argmax(logits + energy_bias, dim=-1).item())
        return ID_TO_ORGAN[idx]

    def update_energy(self, chosen_organ: str, correct: bool, cfg: Config, organ_energy: List[float]) -> None:
        chosen_idx = ORGAN_TO_ID[chosen_organ]

        for i in range(len(organ_energy)):
            organ_energy[i] *= cfg.energy_decay

        if correct:
            organ_energy[chosen_idx] += cfg.energy_lr
        else:
            organ_energy[chosen_idx] -= cfg.energy_lr * 0.5

        for i in range(len(organ_energy)):
            organ_energy[i] = max(cfg.min_energy, min(cfg.max_energy, organ_energy[i]))


def summarize_usage(organ_counts: Dict[str, int]) -> Dict[str, Any]:
    counts = [organ_counts.get(organ, 0) for organ in ORGANS]
    total = sum(counts)
    if total <= 0:
        vector = [0.0 for _ in ORGANS]
    else:
        vector = [count / total for count in counts]

    entropy = normalized_entropy(vector)
    return {
        "usage_vector": {organ: round_float(value) for organ, value in zip(ORGANS, vector)},
        "usage_entropy": round_float(entropy),
        "usage_max": round_float(max(vector) if vector else 0.0),
        "usage_min": round_float(min(vector) if vector else 0.0),
        "collapse_score": round_float(max(vector) if vector else 0.0),
    }


def specialization_score(organ_counts_by_family: Dict[str, Dict[str, int]]) -> float:
    entropies = []
    top_organs = []

    for family in TASK_FAMILIES:
        counts = [organ_counts_by_family[family].get(organ, 0) for organ in ORGANS]
        total = sum(counts)
        if total <= 0:
            continue

        vector = [count / total for count in counts]
        entropy = normalized_entropy(vector)
        if entropy is not None:
            entropies.append(entropy)

        top_organs.append(ORGANS[max(range(len(ORGANS)), key=lambda i: vector[i])])

    if not entropies or not top_organs:
        return 0.0

    entropy_component = 1.0 - mean(entropies)
    diversity_component = len(set(top_organs)) / min(len(ORGANS), len(TASK_FAMILIES))

    # one-organ collapse 방지:
    # 낮은 entropy만으로 specialization으로 보지 않고,
    # 서로 다른 task family가 서로 다른 top organ을 가져야 높은 점수.
    return max(0.0, min(1.0, (0.55 * entropy_component + 0.45 * diversity_component) * diversity_component))


def evaluate_router(router: RouterPolicy, cfg: Config, seed: int) -> Dict[str, Any]:
    set_seed(seed)

    correct_by_family = {family: 0 for family in TASK_FAMILIES}
    total_by_family = {family: 0 for family in TASK_FAMILIES}
    router_correct_by_family = {family: 0 for family in TASK_FAMILIES}
    oracle_correct_by_family = {family: 0 for family in TASK_FAMILIES}

    organ_counts = {organ: 0 for organ in ORGANS}
    organ_counts_by_family = {
        family: {organ: 0 for organ in ORGANS}
        for family in TASK_FAMILIES
    }

    chosen_correct_by_organ = {organ: 0 for organ in ORGANS}
    chosen_total_by_organ = {organ: 0 for organ in ORGANS}

    confusion = {
        target: {chosen: 0 for chosen in ORGANS}
        for target in ORGANS
    }

    organ_energy = [1.0 for _ in ORGANS]
    energy_history: List[List[float]] = []

    for family in TASK_FAMILIES:
        for _ in range(cfg.eval_episodes):
            episode = make_episode(cfg, family)

            for query, target_action in episode.queries:
                predictions = organ_predictions(episode, query)
                target_organ = choose_target_organ(episode, query, target_action, predictions)

                chosen_organ = router.choose(episode, query, predictions, organ_energy)
                if chosen_organ not in ORGAN_TO_ID:
                    chosen_organ = "memory_organ"

                pred_action = int(predictions[chosen_organ])
                correct = pred_action == int(target_action)
                oracle_correct = any(int(pred) == int(target_action) for pred in predictions.values())
                router_correct = chosen_organ == target_organ

                correct_by_family[family] += int(correct)
                total_by_family[family] += 1
                router_correct_by_family[family] += int(router_correct)
                oracle_correct_by_family[family] += int(oracle_correct)

                organ_counts[chosen_organ] += 1
                organ_counts_by_family[family][chosen_organ] += 1

                chosen_total_by_organ[chosen_organ] += 1
                chosen_correct_by_organ[chosen_organ] += int(correct)

                confusion[target_organ][chosen_organ] += 1

                router.update_energy(chosen_organ, correct, cfg, organ_energy)
                energy_history.append(list(organ_energy))

    family_accuracy = {
        family: correct_by_family[family] / max(1, total_by_family[family])
        for family in TASK_FAMILIES
    }
    family_router_accuracy = {
        family: router_correct_by_family[family] / max(1, total_by_family[family])
        for family in TASK_FAMILIES
    }
    family_oracle_accuracy = {
        family: oracle_correct_by_family[family] / max(1, total_by_family[family])
        for family in TASK_FAMILIES
    }

    task_acc = mean(family_accuracy.values())
    router_acc = mean(family_router_accuracy.values())
    oracle_task_acc = mean(family_oracle_accuracy.values())

    usage = summarize_usage(organ_counts)

    organ_usage_by_family: Dict[str, Dict[str, float]] = {}
    organ_entropy_by_family: Dict[str, Optional[float]] = {}
    organ_top_by_family: Dict[str, str] = {}

    for family in TASK_FAMILIES:
        counts = organ_counts_by_family[family]
        total = sum(counts.values())
        if total <= 0:
            vector = {organ: 0.0 for organ in ORGANS}
        else:
            vector = {organ: counts[organ] / total for organ in ORGANS}

        organ_usage_by_family[family] = {
            organ: round_float(value) for organ, value in vector.items()
        }
        organ_entropy_by_family[family] = round_float(normalized_entropy(list(vector.values())))
        organ_top_by_family[family] = max(ORGANS, key=lambda organ: vector[organ])

    chosen_organ_accuracy = {}
    for organ in ORGANS:
        total = chosen_total_by_organ[organ]
        chosen_organ_accuracy[organ] = (
            round_float(chosen_correct_by_organ[organ] / total)
            if total > 0
            else None
        )

    final_energy = energy_history[-1] if energy_history else organ_energy
    energy_mean = mean(final_energy)
    energy_std = safe_std(final_energy)
    energy_range = max(final_energy) - min(final_energy)

    return {
        "task_family_accuracy": {
            family: round_float(value) for family, value in family_accuracy.items()
        },
        "router_accuracy_by_family": {
            family: round_float(value) for family, value in family_router_accuracy.items()
        },
        "oracle_accuracy_by_family": {
            family: round_float(value) for family, value in family_oracle_accuracy.items()
        },
        "task_acc": round_float(task_acc),
        "router_acc": round_float(router_acc),
        "oracle_task_acc": round_float(oracle_task_acc),
        "oracle_gap": round_float(oracle_task_acc - task_acc),
        "random_gain": None,
        "organ_usage": usage,
        "organ_specialization": round_float(specialization_score(organ_counts_by_family)),
        "organ_usage_by_family": organ_usage_by_family,
        "organ_usage_entropy_by_family": organ_entropy_by_family,
        "organ_top_by_family": organ_top_by_family,
        "chosen_organ_accuracy": chosen_organ_accuracy,
        "router_confusion_matrix": confusion,
        "final_energy_vector": {
            organ: round_float(final_energy[i]) for i, organ in enumerate(ORGANS)
        },
        "energy_mean": round_float(energy_mean),
        "energy_std": round_float(energy_std),
        "energy_range": round_float(energy_range),
    }


def summarize_runs(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    metric_paths = {
        "task_acc": ("task_acc",),
        "router_acc": ("router_acc",),
        "oracle_gap": ("oracle_gap",),
        "random_gain": ("random_gain",),
        "organ_usage_entropy": ("organ_usage", "usage_entropy"),
        "organ_specialization": ("organ_specialization",),
        "collapse_score": ("organ_usage", "collapse_score"),
        "energy_std": ("energy_std",),
        "energy_range": ("energy_range",),
    }

    summary: Dict[str, Any] = {}

    for key, path in metric_paths.items():
        values = []
        for run in runs:
            value = run["metrics"]
            for item in path:
                value = value[item]
            if value is not None:
                values.append(float(value))

        if not values:
            summary[key] = {"mean": None, "std": None, "n": 0}
        else:
            summary[key] = {
                "mean": mean(values),
                "std": safe_std(values),
                "n": len(values),
            }

    return summary


def print_table(output: Dict[str, Any]) -> None:
    keys = [
        "task_acc",
        "router_acc",
        "oracle_gap",
        "random_gain",
        "organ_usage_entropy",
        "organ_specialization",
        "collapse_score",
        "energy_std",
    ]

    widths = [22, 18, 18, 18, 18, 24, 24, 18, 18]
    header = ["router"] + [f"{key} mean+/-std" for key in keys]

    print("\n=== SAGE v1.5 Neural Ecosystem Router ===")
    print(f"seeds: {output['seeds']}")
    print(" | ".join(value.ljust(width) for value, width in zip(header, widths)))
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))

    for router_name in output["routers"]:
        row = [router_name]
        for key in keys:
            item = output["summary"][router_name][key]
            if item["mean"] is None:
                row.append("N/A")
            else:
                row.append(f"{item['mean']:.4f}+/-{item['std']:.4f}")
        print(" | ".join(value.ljust(width) for value, width in zip(row, widths)))


def safe_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): safe_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [safe_jsonable(v) for v in obj]

    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()

    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    return repr(obj)


def run_seed(cfg: Config, seed: int) -> List[Dict[str, Any]]:
    set_seed(seed)

    print(f"\n[Seed {seed}] training NeuralRouter ...")
    neural_model = train_neural_router(cfg, seed)

    routers: List[RouterPolicy] = [
        RandomRouter(),
        FamilyOracleRouter(),
        NeuralRouter(neural_model, cfg),
        NeuralRouterEnergy(neural_model, cfg),
    ]

    seed_runs = []
    for router in routers:
        print(f"  evaluating {router.name} ...")
        metrics = evaluate_router(router, cfg, seed)
        seed_runs.append({
            "seed": seed,
            "router": router.name,
            "metrics": metrics,
            "config": asdict(cfg),
        })

        print(
            "    "
            f"task_acc={metrics['task_acc']:.4f}, "
            f"router_acc={metrics['router_acc']:.4f}, "
            f"oracle_gap={metrics['oracle_gap']:.4f}, "
            f"specialization={metrics['organ_specialization']:.4f}, "
            f"collapse={metrics['organ_usage']['collapse_score']:.4f}"
        )

    return seed_runs


def fill_random_gain(all_runs: List[Dict[str, Any]]) -> None:
    random_task_acc_by_seed = {}
    for run in all_runs:
        if run["router"] == "RandomRouter":
            random_task_acc_by_seed[run["seed"]] = run["metrics"]["task_acc"]

    for run in all_runs:
        baseline = random_task_acc_by_seed.get(run["seed"])
        if baseline is None:
            run["metrics"]["random_gain"] = None
        else:
            run["metrics"]["random_gain"] = round_float(run["metrics"]["task_acc"] - baseline)


def main() -> None:
    parser = argparse.ArgumentParser(description="SAGE v1.5 Neural Ecosystem Router")
    parser.add_argument("--train-episodes", type=int, default=120)
    parser.add_argument("--eval-episodes", type=int, default=120)
    parser.add_argument("--queries-per-episode", type=int, default=18)
    parser.add_argument("--support-per-episode", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--energy-bias", type=float, default=0.35)
    parser.add_argument("--energy-lr", type=float, default=0.08)
    parser.add_argument("--energy-decay", type=float, default=0.995)
    parser.add_argument("--out", type=str, default="results/v1_5_neural_ecosystem_router_benchmark.json")
    args = parser.parse_args()

    cfg = Config(
        train_episodes=args.train_episodes,
        eval_episodes=args.eval_episodes,
        queries_per_episode=args.queries_per_episode,
        support_per_episode=args.support_per_episode,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=args.device,
        energy_bias=args.energy_bias,
        energy_lr=args.energy_lr,
        energy_decay=args.energy_decay,
    )

    all_runs: List[Dict[str, Any]] = []
    for seed in SEEDS:
        seed_cfg = Config(**{**asdict(cfg), "seed": seed})
        all_runs.extend(run_seed(seed_cfg, seed))

    fill_random_gain(all_runs)

    routers = ["RandomRouter", "FamilyOracleRouter", "NeuralRouter", "NeuralRouterEnergy"]
    summary = {
        router: summarize_runs([run for run in all_runs if run["router"] == router])
        for router in routers
    }

    output = {
        "benchmark": "SAGE-v1.5-neural-ecosystem-router",
        "goal": (
            "Test whether a neural router can learn to call scaffold cognitive organs "
            "instead of relying on hand-coded family-specific organ routing."
        ),
        "interpretation_guardrail": (
            "This is not an AGI claim. The organs remain scaffolded; v1.5 only tests "
            "learned organ selection and energy-biased routing."
        ),
        "seeds": SEEDS,
        "task_families": TASK_FAMILIES,
        "organs": ORGANS,
        "routers": routers,
        "specialized_oracle": SPECIALIZED_ORACLE,
        "config": asdict(cfg),
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
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
