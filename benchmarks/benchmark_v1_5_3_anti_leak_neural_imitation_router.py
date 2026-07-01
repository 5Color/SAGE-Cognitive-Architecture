# benchmark_v1_5_3_anti_leak_neural_imitation_router.py
# SAGE v1.5.3 - Anti-Leak Neural Imitation Router
#
# 단어 뜻:
# - Anti-Leak = 정답 단서 누수 방지
# - Neural = 신경망 기반
# - Imitation = 모방
# - Evidence = 증거
# - Router = 어떤 organ을 쓸지 고르는 선택기
# - Teacher = 기준 선택을 만드는 선생 모델
# - Student = teacher의 선택을 배우는 학생 모델
#
# 목표:
# - v1.5.2 AntiLeakEvidenceRouter를 teacher로 둔다.
# - NeuralAntiLeakImitationRouter가 key type/family label 없이 organ selection policy를 배울 수 있는지 검증한다.
# - easy fingerprint를 줄이기 위해 모든 query/support key는 ("item", int) 형태로 통일한다.
# - neural student는 raw key type이 아니라 support evidence score vector를 입력으로 받는다.
#
# 실행:
# python benchmark_v1_5_3_anti_leak_neural_imitation_router.py --episodes 20 --queries-per-episode 8 --epochs 6
# python benchmark_v1_5_3_anti_leak_neural_imitation_router.py --episodes 120 --queries-per-episode 18 --epochs 10

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


SEEDS = [0, 1, 2, 3, 4]
ACTION_DIM = 6
GRID_SIZE = 5

TASK_FAMILIES = [
    "episodic_memory",
    "affine_rule",
    "threshold_rule",
    "modulo_concept_rule",
    "grid_planning",
    "world_dynamics",
]

ORGANS = [
    "memory_organ",
    "algebra_organ",
    "concept_organ",
    "planner_organ",
]

ORGAN_TO_ID = {organ: idx for idx, organ in enumerate(ORGANS)}
ID_TO_ORGAN = {idx: organ for organ, idx in ORGAN_TO_ID.items()}

FAMILY_ORACLE = {
    "episodic_memory": "memory_organ",
    "affine_rule": "algebra_organ",
    "threshold_rule": "concept_organ",
    "modulo_concept_rule": "concept_organ",
    "grid_planning": "planner_organ",
    "world_dynamics": "planner_organ",
}

GRID_OFFSET = 100_000
WORLD_QUERY_OFFSET = 200_000
TRANSITION_OFFSET = 300_000


@dataclass
class Config:
    episodes: int = 120
    queries_per_episode: int = 18
    support_per_episode: int = 8
    train_ratio: float = 0.70
    hidden_dim: int = 96
    epochs: int = 10
    batch_size: int = 256
    lr: float = 2e-3
    seed: int = 0
    device: str = "cpu"


@dataclass
class Episode:
    family: str
    support: List[Tuple[Any, int]]
    queries: List[Tuple[Any, int]]
    meta: Dict[str, Any]


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


def item(value: int) -> Tuple[str, int]:
    # 모든 task가 같은 key shape를 쓰도록 통일한다.
    return ("item", int(value))


def item_value(key: Any) -> Optional[int]:
    if isinstance(key, tuple) and len(key) == 2 and key[0] == "item" and isinstance(key[1], int):
        return int(key[1])
    return None


# ----------------------------------------------------------------------
# Anti-leak synthetic task ecology
# ----------------------------------------------------------------------

def make_episodic_memory_episode(cfg: Config) -> Episode:
    keys = random.sample(range(10_000, 99_999), cfg.support_per_episode)
    mapping = {key: random.randrange(ACTION_DIM) for key in keys}
    support = [(item(key), action) for key, action in mapping.items()]
    query_keys = random.choices(keys, k=cfg.queries_per_episode)
    queries = [(item(key), mapping[key]) for key in query_keys]
    return Episode("episodic_memory", support, queries, {"mapping_size": len(mapping)})


def make_affine_rule_episode(cfg: Config) -> Episode:
    # y = (a*x+b) mod ACTION_DIM, a in {1, 5} keeps invertible modular pattern.
    a = random.choice([1, 5])
    b = random.randrange(ACTION_DIM)
    fn = lambda x: (a * x + b) % ACTION_DIM

    xs = list(range(ACTION_DIM))
    support = [(item(x), fn(x)) for x in xs]
    query_xs = [random.randrange(ACTION_DIM) for _ in range(cfg.queries_per_episode)]
    queries = [(item(x), fn(x)) for x in query_xs]
    return Episode("affine_rule", support, queries, {"a": a, "b": b})


def make_threshold_rule_episode(cfg: Config) -> Episode:
    threshold = random.randint(1, ACTION_DIM - 2)
    low_action = random.randrange(ACTION_DIM)
    high_action = random.choice([x for x in range(ACTION_DIM) if x != low_action])
    fn = lambda x: low_action if x < threshold else high_action

    xs = list(range(ACTION_DIM))
    support = [(item(x), fn(x)) for x in xs]
    query_xs = [random.randrange(ACTION_DIM) for _ in range(cfg.queries_per_episode)]
    queries = [(item(x), fn(x)) for x in query_xs]
    return Episode(
        "threshold_rule",
        support,
        queries,
        {"threshold": threshold, "low_action": low_action, "high_action": high_action},
    )


def make_modulo_concept_rule_episode(cfg: Config) -> Episode:
    # concept organ용 간단한 category rule.
    # x % 3 category마다 action을 배정한다.
    category_actions = random.sample(range(ACTION_DIM), 3)
    fn = lambda x: category_actions[x % 3]

    xs = list(range(ACTION_DIM))
    support = [(item(x), fn(x)) for x in xs]
    query_xs = [random.randrange(ACTION_DIM) for _ in range(cfg.queries_per_episode)]
    queries = [(item(x), fn(x)) for x in query_xs]
    return Episode("modulo_concept_rule", support, queries, {"category_actions": category_actions})


def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def encode_grid_state(x: int, y: int, gx: int, gy: int, blocked_mask: int) -> int:
    # key shape는 동일하지만 value 내부에는 task state가 encoding된다.
    return GRID_OFFSET + (((((x * GRID_SIZE + y) * GRID_SIZE + gx) * GRID_SIZE + gy) << 25) + blocked_mask)


def decode_grid_state(value: int) -> Optional[Tuple[int, int, int, int, int]]:
    if value < GRID_OFFSET or value >= WORLD_QUERY_OFFSET:
        return None
    raw = value - GRID_OFFSET
    blocked_mask = raw & ((1 << 25) - 1)
    packed = raw >> 25
    gy = packed % GRID_SIZE
    packed //= GRID_SIZE
    gx = packed % GRID_SIZE
    packed //= GRID_SIZE
    y = packed % GRID_SIZE
    packed //= GRID_SIZE
    x = packed % GRID_SIZE
    return x, y, gx, gy, blocked_mask


def mask_has(mask: int, pos: Tuple[int, int]) -> bool:
    idx = pos[0] * GRID_SIZE + pos[1]
    return bool(mask & (1 << idx))


def random_grid_encoded() -> int:
    x, y = random.randrange(GRID_SIZE), random.randrange(GRID_SIZE)
    gx, gy = random.randrange(GRID_SIZE), random.randrange(GRID_SIZE)
    occupied = {(x, y), (gx, gy)}
    mask = 0
    for _ in range(random.randint(0, 4)):
        cell = (random.randrange(GRID_SIZE), random.randrange(GRID_SIZE))
        if cell not in occupied:
            idx = cell[0] * GRID_SIZE + cell[1]
            mask |= (1 << idx)
            occupied.add(cell)
    return encode_grid_state(x, y, gx, gy, mask)


def plan_step_from_grid_value(value: int) -> int:
    decoded = decode_grid_state(value)
    if decoded is None:
        return 0
    x, y, gx, gy, blocked_mask = decoded
    moves = {
        0: (x, max(0, y - 1)),
        1: (x, min(GRID_SIZE - 1, y + 1)),
        2: (max(0, x - 1), y),
        3: (min(GRID_SIZE - 1, x + 1), y),
    }
    candidates: List[Tuple[int, int]] = []
    for action, pos in moves.items():
        if pos == (x, y) or mask_has(blocked_mask, pos):
            continue
        candidates.append((manhattan(pos, (gx, gy)), action))
    if not candidates:
        return 4
    return min(candidates)[1]


def make_grid_planning_episode(cfg: Config) -> Episode:
    support_values = [random_grid_encoded() for _ in range(cfg.support_per_episode)]
    support = [(item(value), plan_step_from_grid_value(value)) for value in support_values]
    query_values = [random_grid_encoded() for _ in range(cfg.queries_per_episode)]
    queries = [(item(value), plan_step_from_grid_value(value)) for value in query_values]
    return Episode("grid_planning", support, queries, {"grid_size": GRID_SIZE})


def encode_world_query(x: int, y: int, gx: int, gy: int) -> int:
    return WORLD_QUERY_OFFSET + (((x * GRID_SIZE + y) * GRID_SIZE + gx) * GRID_SIZE + gy)


def decode_world_query(value: int) -> Optional[Tuple[int, int, int, int]]:
    if value < WORLD_QUERY_OFFSET or value >= TRANSITION_OFFSET:
        return None
    raw = value - WORLD_QUERY_OFFSET
    gy = raw % GRID_SIZE
    raw //= GRID_SIZE
    gx = raw % GRID_SIZE
    raw //= GRID_SIZE
    y = raw % GRID_SIZE
    raw //= GRID_SIZE
    x = raw % GRID_SIZE
    return x, y, gx, gy


def encode_transition(x: int, y: int, action: int, nx: int, ny: int) -> int:
    raw = (((((x * GRID_SIZE + y) * 4 + action) * GRID_SIZE + nx) * GRID_SIZE) + ny)
    return TRANSITION_OFFSET + raw


def decode_transition(value: int) -> Optional[Tuple[int, int, int, int, int]]:
    if value < TRANSITION_OFFSET:
        return None
    raw = value - TRANSITION_OFFSET
    ny = raw % GRID_SIZE
    raw //= GRID_SIZE
    nx = raw % GRID_SIZE
    raw //= GRID_SIZE
    action = raw % 4
    raw //= 4
    y = raw % GRID_SIZE
    raw //= GRID_SIZE
    x = raw % GRID_SIZE
    return x, y, action, nx, ny


def apply_delta(state: Tuple[int, int], action: int, deltas: Dict[int, Tuple[int, int]]) -> Tuple[int, int]:
    dx, dy = deltas[action]
    return (
        max(0, min(GRID_SIZE - 1, state[0] + dx)),
        max(0, min(GRID_SIZE - 1, state[1] + dy)),
    )


def world_best_action_from_query(value: int, deltas: Dict[int, Tuple[int, int]]) -> int:
    decoded = decode_world_query(value)
    if decoded is None:
        return 0
    x, y, gx, gy = decoded
    ranked = []
    for action in range(4):
        nx, ny = apply_delta((x, y), action, deltas)
        ranked.append((manhattan((nx, ny), (gx, gy)), action))
    return min(ranked)[1]


def make_world_dynamics_episode(cfg: Config) -> Episode:
    base_deltas = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    shuffled = random.sample(base_deltas, len(base_deltas))
    deltas = {action: shuffled[action] for action in range(4)}

    support: List[Tuple[Any, int]] = []
    center = (2, 2)
    for action in range(4):
        nx, ny = apply_delta(center, action, deltas)
        support.append((item(encode_transition(center[0], center[1], action, nx, ny)), action))

    # validation evidence를 위해 query-like support examples도 추가한다.
    while len(support) < cfg.support_per_episode:
        x, y = random.randrange(GRID_SIZE), random.randrange(GRID_SIZE)
        gx, gy = random.randrange(GRID_SIZE), random.randrange(GRID_SIZE)
        value = encode_world_query(x, y, gx, gy)
        support.append((item(value), world_best_action_from_query(value, deltas)))

    queries: List[Tuple[Any, int]] = []
    for _ in range(cfg.queries_per_episode):
        x, y = random.randrange(GRID_SIZE), random.randrange(GRID_SIZE)
        gx, gy = random.randrange(GRID_SIZE), random.randrange(GRID_SIZE)
        value = encode_world_query(x, y, gx, gy)
        queries.append((item(value), world_best_action_from_query(value, deltas)))

    return Episode("world_dynamics", support, queries, {"hidden_deltas": deltas})


EPISODE_FACTORIES = {
    "episodic_memory": make_episodic_memory_episode,
    "affine_rule": make_affine_rule_episode,
    "threshold_rule": make_threshold_rule_episode,
    "modulo_concept_rule": make_modulo_concept_rule_episode,
    "grid_planning": make_grid_planning_episode,
    "world_dynamics": make_world_dynamics_episode,
}


def make_episode(cfg: Config, family: str) -> Episode:
    return EPISODE_FACTORIES[family](cfg)


# ----------------------------------------------------------------------
# Scaffold organs
# ----------------------------------------------------------------------

class MemoryOrgan:
    name = "memory_organ"

    def __init__(self) -> None:
        self.memory: Dict[Any, int] = {}
        self.default_action = 0

    def fit(self, support: Sequence[Tuple[Any, int]]) -> None:
        self.memory = {key: int(action) for key, action in support}
        actions = [int(action) for _, action in support]
        if actions:
            self.default_action = max(set(actions), key=actions.count)

    def predict(self, query: Any) -> int:
        return int(self.memory.get(query, self.default_action))


class AlgebraOrgan:
    name = "algebra_organ"

    def __init__(self) -> None:
        self.affine: Optional[Tuple[int, int]] = None
        self.default_action = 0

    def fit(self, support: Sequence[Tuple[Any, int]]) -> None:
        self.affine = None
        actions = [int(action) for _, action in support]
        if actions:
            self.default_action = max(set(actions), key=actions.count)

        pairs: List[Tuple[int, int]] = []
        for key, action in support:
            value = item_value(key)
            if value is not None and 0 <= value < ACTION_DIM:
                pairs.append((value, int(action)))

        if len(pairs) >= 2:
            x0, y0 = pairs[0]
            for x1, y1 in pairs[1:]:
                dx = (x1 - x0) % ACTION_DIM
                dy = (y1 - y0) % ACTION_DIM
                if dx in [1, 5]:
                    a = dy if dx == 1 else (-dy) % ACTION_DIM
                    b = (y0 - a * x0) % ACTION_DIM
                    if all(((a * x + b) % ACTION_DIM) == y for x, y in pairs):
                        self.affine = (a, b)
                        return

    def predict(self, query: Any) -> int:
        value = item_value(query)
        if self.affine is not None and value is not None and 0 <= value < ACTION_DIM:
            a, b = self.affine
            return int((a * value + b) % ACTION_DIM)
        return int(self.default_action)


class ConceptOrgan:
    name = "concept_organ"

    def __init__(self) -> None:
        self.mode = "default"
        self.threshold: Optional[Tuple[int, int, int]] = None
        self.modulo_map: Dict[int, int] = {}
        self.default_action = 0

    def fit(self, support: Sequence[Tuple[Any, int]]) -> None:
        self.mode = "default"
        self.threshold = None
        self.modulo_map = {}
        actions = [int(action) for _, action in support]
        if actions:
            self.default_action = max(set(actions), key=actions.count)

        pairs: List[Tuple[int, int]] = []
        for key, action in support:
            value = item_value(key)
            if value is not None and 0 <= value < ACTION_DIM:
                pairs.append((value, int(action)))

        if not pairs:
            return

        candidates: List[Tuple[float, str, Any]] = []

        # threshold candidate
        sorted_pairs = sorted(pairs)
        values = [action for _, action in sorted_pairs]
        for idx in range(1, len(sorted_pairs)):
            if values[idx] != values[idx - 1]:
                threshold = sorted_pairs[idx][0]
                low_action = values[idx - 1]
                high_action = values[idx]
                correct = sum(
                    int((low_action if x < threshold else high_action) == y)
                    for x, y in pairs
                )
                candidates.append((correct / len(pairs), "threshold", (threshold, low_action, high_action)))
                break

        # modulo concept candidate
        groups: Dict[int, List[int]] = {0: [], 1: [], 2: []}
        for x, y in pairs:
            groups[x % 3].append(y)
        modulo_map: Dict[int, int] = {}
        consistent = True
        for r, ys in groups.items():
            if ys:
                action = max(set(ys), key=ys.count)
                modulo_map[r] = action
                if any(y != action for y in ys):
                    consistent = False
        if modulo_map:
            correct = sum(int(modulo_map.get(x % 3, self.default_action) == y) for x, y in pairs)
            # consistency bonus: modulo rule이 clean하면 concept으로 더 선호한다.
            score = correct / len(pairs) + (0.05 if consistent else 0.0)
            candidates.append((score, "modulo", modulo_map))

        if candidates:
            _score, mode, payload = max(candidates, key=lambda item: item[0])
            self.mode = mode
            if mode == "threshold":
                self.threshold = payload
            elif mode == "modulo":
                self.modulo_map = payload

    def predict(self, query: Any) -> int:
        value = item_value(query)
        if value is None or not (0 <= value < ACTION_DIM):
            return int(self.default_action)

        if self.mode == "threshold" and self.threshold is not None:
            threshold, low_action, high_action = self.threshold
            return int(low_action if value < threshold else high_action)

        if self.mode == "modulo" and self.modulo_map:
            return int(self.modulo_map.get(value % 3, self.default_action))

        return int(self.default_action)


class PlannerOrgan:
    name = "planner_organ"

    def __init__(self) -> None:
        self.world_deltas: Optional[Dict[int, Tuple[int, int]]] = None
        self.default_action = 0

    def fit(self, support: Sequence[Tuple[Any, int]]) -> None:
        self.world_deltas = None
        actions = [int(action) for _, action in support]
        if actions:
            self.default_action = max(set(actions), key=actions.count)

        deltas: Dict[int, Tuple[int, int]] = {}
        for key, _action in support:
            value = item_value(key)
            if value is None:
                continue
            decoded = decode_transition(value)
            if decoded is None:
                continue
            x, y, action, nx, ny = decoded
            deltas[int(action)] = (int(nx) - int(x), int(ny) - int(y))
        if len(deltas) == 4:
            self.world_deltas = deltas

    def predict(self, query: Any) -> int:
        value = item_value(query)
        if value is None:
            return int(self.default_action)

        if decode_grid_state(value) is not None:
            return int(plan_step_from_grid_value(value))

        if decode_world_query(value) is not None and self.world_deltas is not None:
            return int(world_best_action_from_query(value, self.world_deltas))

        return int(self.default_action)


def make_organs() -> Dict[str, Any]:
    return {
        "memory_organ": MemoryOrgan(),
        "algebra_organ": AlgebraOrgan(),
        "concept_organ": ConceptOrgan(),
        "planner_organ": PlannerOrgan(),
    }


def fit_organs(support: Sequence[Tuple[Any, int]]) -> Dict[str, Any]:
    organs = make_organs()
    for organ in organs.values():
        organ.fit(support)
    return organs


def organ_predictions(support: Sequence[Tuple[Any, int]], query: Any) -> Dict[str, int]:
    organs = fit_organs(support)
    return {name: int(organ.predict(query)) for name, organ in organs.items()}


# ----------------------------------------------------------------------
# Teacher routers
# ----------------------------------------------------------------------

def split_support(support: Sequence[Tuple[Any, int]]) -> Tuple[List[Tuple[Any, int]], List[Tuple[Any, int]]]:
    support = list(support)
    if len(support) <= 2:
        return support, support

    # world dynamics transition examples are needed for planner fit.
    transition_items = []
    other_items = []
    for pair in support:
        value = item_value(pair[0])
        if value is not None and decode_transition(value) is not None:
            transition_items.append(pair)
        else:
            other_items.append(pair)

    if transition_items:
        fit_part = transition_items + other_items[: max(1, len(other_items) // 2)]
        val_part = other_items[max(1, len(other_items) // 2):] or other_items or transition_items
        return fit_part, val_part

    shuffled = support[:]
    random.shuffle(shuffled)
    cut = max(1, int(len(shuffled) * 0.65))
    return shuffled[:cut], shuffled[cut:] or shuffled


def evidence_scores(support: Sequence[Tuple[Any, int]]) -> Dict[str, float]:
    fit_part, val_part = split_support(support)
    scores: Dict[str, float] = {}
    for organ_name in ORGANS:
        organs = fit_organs(fit_part)
        organ = organs[organ_name]
        correct = 0
        for key, target in val_part:
            pred = int(organ.predict(key))
            correct += int(pred == int(target))
        scores[organ_name] = correct / max(1, len(val_part))
    return scores


class RandomOrganRouter:
    name = "RandomOrganRouter"

    def choose(self, episode: Episode, query: Any) -> Tuple[str, float]:
        return random.choice(ORGANS), 1.0 / len(ORGANS)


class KeyTypeRouter:
    name = "KeyTypeRouter"

    def choose(self, episode: Episode, query: Any) -> Tuple[str, float]:
        # anti-leak setting에서는 모든 key가 ("item", int)이므로 사실상 약한 baseline이다.
        return "memory_organ", 1.0 / len(ORGANS)


class FamilyOracleRouter:
    name = "FamilyOracleRouter"

    def choose(self, episode: Episode, query: Any) -> Tuple[str, float]:
        return FAMILY_ORACLE[episode.family], 1.0


class AntiLeakEvidenceRouter:
    name = "AntiLeakEvidenceRouter"

    def choose(self, episode: Episode, query: Any) -> Tuple[str, float]:
        scores = evidence_scores(episode.support)
        chosen = max(ORGANS, key=lambda organ: scores[organ])
        sorted_scores = sorted(scores.values(), reverse=True)
        margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) >= 2 else sorted_scores[0]
        confidence = max(1.0 / len(ORGANS), margin)
        return chosen, float(confidence)


# ----------------------------------------------------------------------
# Neural anti-leak imitation student
# ----------------------------------------------------------------------

def score_entropy(scores: Sequence[float]) -> float:
    value = normalized_entropy(scores)
    return 0.0 if value is None else float(value)


def action_entropy(actions: Sequence[int]) -> float:
    if not actions:
        return 0.0
    counts = [0 for _ in range(ACTION_DIM)]
    for action in actions:
        counts[int(action) % ACTION_DIM] += 1
    value = normalized_entropy(counts)
    return 0.0 if value is None else float(value)


def extract_evidence_features(episode: Episode) -> List[float]:
    scores_dict = evidence_scores(episode.support)
    scores = [scores_dict[organ] for organ in ORGANS]
    sorted_scores = sorted(scores, reverse=True)
    top_score = sorted_scores[0]
    second_score = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
    margin = top_score - second_score

    actions = [int(action) for _, action in episode.support]
    support_size_norm = min(1.0, len(episode.support) / 12.0)
    action_diversity = len(set(actions)) / max(1, ACTION_DIM)

    features: List[float] = []
    features.extend(scores)
    features.extend([
        top_score,
        second_score,
        margin,
        score_entropy(scores),
        support_size_norm,
        action_diversity,
        action_entropy(actions),
    ])
    return features


class RouterNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, organ_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, organ_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def build_imitation_dataset(cfg: Config, seed: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    set_seed(seed)
    teacher = AntiLeakEvidenceRouter()

    xs: List[List[float]] = []
    ys: List[int] = []
    weights: List[float] = []

    for family in TASK_FAMILIES:
        for _ in range(cfg.episodes):
            episode = make_episode(cfg, family)
            chosen, confidence = teacher.choose(episode, episode.queries[0][0])
            feature = extract_evidence_features(episode)
            # 같은 episode의 query들은 같은 support evidence를 공유하므로 episode-level target으로 충분하다.
            xs.append(feature)
            ys.append(ORGAN_TO_ID[chosen])
            weights.append(float(confidence))

    x = torch.tensor(xs, dtype=torch.float32)
    y = torch.tensor(ys, dtype=torch.long)
    w = torch.tensor(weights, dtype=torch.float32).clamp(min=0.20, max=1.00)
    return x, y, w


def train_student_router(cfg: Config, seed: int) -> Tuple[RouterNet, Dict[str, float]]:
    x, y, w = build_imitation_dataset(cfg, seed)
    n = x.shape[0]
    input_dim = x.shape[1]

    indices = list(range(n))
    random.shuffle(indices)
    train_n = max(1, int(n * cfg.train_ratio))
    train_idx = torch.tensor(indices[:train_n], dtype=torch.long, device=cfg.device)
    val_idx = torch.tensor(indices[train_n:], dtype=torch.long, device=cfg.device)

    x = x.to(cfg.device)
    y = y.to(cfg.device)
    w = w.to(cfg.device)

    model = RouterNet(input_dim, cfg.hidden_dim, len(ORGANS)).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    model.train()
    for _ in range(cfg.epochs):
        perm = train_idx[torch.randperm(train_idx.numel(), device=cfg.device)]
        for start in range(0, perm.numel(), cfg.batch_size):
            batch = perm[start:start + cfg.batch_size]
            logits = model(x[batch])
            loss_vec = F.cross_entropy(logits, y[batch], reduction="none")
            loss = (loss_vec * w[batch]).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    model.eval()
    with torch.no_grad():
        train_pred = torch.argmax(model(x[train_idx]), dim=-1)
        train_acc = (train_pred == y[train_idx]).float().mean().item()
        if val_idx.numel() > 0:
            val_pred = torch.argmax(model(x[val_idx]), dim=-1)
            val_acc = (val_pred == y[val_idx]).float().mean().item()
        else:
            val_acc = train_acc

    stats = {
        "teacher_train_imitation_acc": train_acc,
        "teacher_val_imitation_acc": val_acc,
        "dataset_size": float(n),
    }
    return model, stats


class NeuralAntiLeakImitationRouter:
    name = "NeuralAntiLeakImitationRouter"

    def __init__(self, model: RouterNet, cfg: Config):
        self.model = model
        self.cfg = cfg

    @torch.no_grad()
    def choose(self, episode: Episode, query: Any) -> Tuple[str, float]:
        self.model.eval()
        features = torch.tensor([extract_evidence_features(episode)], dtype=torch.float32, device=self.cfg.device)
        logits = self.model(features)
        probs = F.softmax(logits, dim=-1).squeeze(0)
        idx = int(torch.argmax(probs).item())
        confidence = float(probs[idx].item())
        return ID_TO_ORGAN[idx], confidence


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------

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
        top_organs.append(ORGANS[max(range(len(ORGANS)), key=lambda idx: vector[idx])])

    if not entropies or not top_organs:
        return 0.0
    entropy_component = 1.0 - mean(entropies)
    diversity_component = len(set(top_organs)) / min(len(ORGANS), len(TASK_FAMILIES))
    return max(0.0, min(1.0, (0.55 * entropy_component + 0.45 * diversity_component) * diversity_component))


def evaluate_router(router: Any, cfg: Config, seed: int) -> Dict[str, Any]:
    # train과 eval episode가 동일해지는 것을 막기 위해 eval seed를 분리한다.
    set_seed(seed + 100_000)
    teacher = AntiLeakEvidenceRouter()
    keytype_baseline = KeyTypeRouter()
    oracle = FamilyOracleRouter()

    correct_by_family = {family: 0 for family in TASK_FAMILIES}
    keytype_correct_by_family = {family: 0 for family in TASK_FAMILIES}
    oracle_correct_by_family = {family: 0 for family in TASK_FAMILIES}
    imitate_by_family = {family: 0 for family in TASK_FAMILIES}
    total_by_family = {family: 0 for family in TASK_FAMILIES}

    organ_counts_by_family = {family: {organ: 0 for organ in ORGANS} for family in TASK_FAMILIES}
    route_confidences: List[float] = []

    for family in TASK_FAMILIES:
        for _ in range(cfg.episodes):
            episode = make_episode(cfg, family)
            for query, target_action in episode.queries:
                chosen, confidence = router.choose(episode, query)
                if chosen not in ORGAN_TO_ID:
                    chosen = "memory_organ"

                teacher_chosen, _ = teacher.choose(episode, query)
                keytype_chosen, _ = keytype_baseline.choose(episode, query)
                oracle_chosen, _ = oracle.choose(episode, query)

                preds = organ_predictions(episode.support, query)
                correct_by_family[family] += int(preds[chosen] == int(target_action))
                keytype_correct_by_family[family] += int(preds[keytype_chosen] == int(target_action))
                oracle_correct_by_family[family] += int(preds[oracle_chosen] == int(target_action))
                imitate_by_family[family] += int(chosen == teacher_chosen)
                total_by_family[family] += 1

                organ_counts_by_family[family][chosen] += 1
                route_confidences.append(float(confidence))

    family_accuracy = {
        family: correct_by_family[family] / max(1, total_by_family[family])
        for family in TASK_FAMILIES
    }
    keytype_accuracy = {
        family: keytype_correct_by_family[family] / max(1, total_by_family[family])
        for family in TASK_FAMILIES
    }
    oracle_accuracy = {
        family: oracle_correct_by_family[family] / max(1, total_by_family[family])
        for family in TASK_FAMILIES
    }
    imitation_accuracy = {
        family: imitate_by_family[family] / max(1, total_by_family[family])
        for family in TASK_FAMILIES
    }

    task_diversity = mean(family_accuracy.values())
    keytype_task = mean(keytype_accuracy.values())
    oracle_task = mean(oracle_accuracy.values())
    rule_generalization = mean([
        family_accuracy["affine_rule"],
        family_accuracy["threshold_rule"],
        family_accuracy["modulo_concept_rule"],
    ])
    planning = mean([
        family_accuracy["grid_planning"],
        family_accuracy["world_dynamics"],
    ])
    imitation_acc = mean(imitation_accuracy.values())
    spec = specialization_score(organ_counts_by_family)
    route_confidence = mean(route_confidences) if route_confidences else 0.0

    anti_leak_score = mean([
        task_diversity,
        rule_generalization,
        planning,
        spec,
        route_confidence,
        imitation_acc,
    ])

    organ_usage_by_family: Dict[str, Dict[str, float]] = {}
    organ_top_by_family: Dict[str, str] = {}
    for family in TASK_FAMILIES:
        counts = organ_counts_by_family[family]
        total = sum(counts.values())
        vector = {organ: counts[organ] / total if total else 0.0 for organ in ORGANS}
        organ_usage_by_family[family] = {organ: round_float(value) for organ, value in vector.items()}
        organ_top_by_family[family] = max(ORGANS, key=lambda organ: vector[organ])

    return {
        "family_accuracy": {k: round_float(v) for k, v in family_accuracy.items()},
        "family_imitation_acc": {k: round_float(v) for k, v in imitation_accuracy.items()},
        "anti_leak_score": round_float(anti_leak_score),
        "task_diversity": round_float(task_diversity),
        "rule_generalization": round_float(rule_generalization),
        "planning": round_float(planning),
        "organ_specialization": round_float(spec),
        "route_confidence": round_float(route_confidence),
        "imitation_acc": round_float(imitation_acc),
        "oracle_task_acc": round_float(oracle_task),
        "keytype_task_acc": round_float(keytype_task),
        "oracle_gap": round_float(oracle_task - task_diversity),
        "keytype_gain": round_float(task_diversity - keytype_task),
        "organ_usage_by_family": organ_usage_by_family,
        "organ_top_by_family": organ_top_by_family,
    }


def summarize_runs(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = [
        "anti_leak_score",
        "task_diversity",
        "rule_generalization",
        "planning",
        "organ_specialization",
        "route_confidence",
        "imitation_acc",
        "oracle_gap",
        "keytype_gain",
    ]
    out: Dict[str, Any] = {}
    for key in keys:
        values = [float(run["metrics"][key]) for run in runs]
        out[key] = {"mean": mean(values), "std": safe_std(values), "n": len(values)}
    return out


def print_table(output: Dict[str, Any]) -> None:
    keys = [
        "anti_leak_score",
        "task_diversity",
        "rule_generalization",
        "planning",
        "organ_specialization",
        "route_confidence",
        "imitation_acc",
        "oracle_gap",
        "keytype_gain",
    ]
    widths = [34, 24, 22, 28, 18, 24, 22, 18, 18, 18]
    header = ["router"] + [f"{key} mean+/-std" for key in keys]

    print("\n=== SAGE v1.5.3 Anti-Leak Neural Imitation Router ===")
    print(f"seeds: {output['seeds']}")
    print(" | ".join(text.ljust(width) for text, width in zip(header, widths)))
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))
    for router_name in output["routers"]:
        row = [router_name]
        for key in keys:
            item = output["summary"][router_name][key]
            row.append(f"{item['mean']:.4f}+/-{item['std']:.4f}")
        print(" | ".join(text.ljust(width) for text, width in zip(row, widths)))


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
    seed_cfg = Config(**{**asdict(cfg), "seed": seed})
    set_seed(seed)

    print(f"\n[Seed {seed}] training NeuralAntiLeakImitationRouter ...")
    student_model, train_stats = train_student_router(seed_cfg, seed)

    routers = [
        RandomOrganRouter(),
        KeyTypeRouter(),
        AntiLeakEvidenceRouter(),
        NeuralAntiLeakImitationRouter(student_model, seed_cfg),
        FamilyOracleRouter(),
    ]

    runs: List[Dict[str, Any]] = []
    for router in routers:
        print(f"  evaluating {router.name} ...")
        metrics = evaluate_router(router, seed_cfg, seed)
        runs.append({
            "seed": seed,
            "router": router.name,
            "config": asdict(seed_cfg),
            "train_stats": train_stats if router.name == "NeuralAntiLeakImitationRouter" else None,
            "metrics": metrics,
        })
        print(
            "    "
            f"score={metrics['anti_leak_score']:.4f}, "
            f"task={metrics['task_diversity']:.4f}, "
            f"rule={metrics['rule_generalization']:.4f}, "
            f"planning={metrics['planning']:.4f}, "
            f"imit={metrics['imitation_acc']:.4f}"
        )

    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description="SAGE v1.5.3 Anti-Leak Neural Imitation Router")
    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument("--queries-per-episode", type=int, default=18)
    parser.add_argument("--support-per-episode", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--out", type=str, default="results/v1_5_3_anti_leak_neural_imitation_router_benchmark.json")
    args = parser.parse_args()

    cfg = Config(
        episodes=args.episodes,
        queries_per_episode=args.queries_per_episode,
        support_per_episode=args.support_per_episode,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )

    all_runs: List[Dict[str, Any]] = []
    for seed in SEEDS:
        all_runs.extend(run_seed(cfg, seed))

    router_names = [
        "RandomOrganRouter",
        "KeyTypeRouter",
        "AntiLeakEvidenceRouter",
        "NeuralAntiLeakImitationRouter",
        "FamilyOracleRouter",
    ]
    summary = {
        router_name: summarize_runs([run for run in all_runs if run["router"] == router_name])
        for router_name in router_names
    }

    output = {
        "benchmark": "SAGE-v1.5.3-anti-leak-neural-imitation-router",
        "goal": (
            "Train a neural router to imitate AntiLeakEvidenceRouter organ selection under anti-leak keys, "
            "without task family labels or raw key-type fingerprints."
        ),
        "terminology": {
            "anti_leak": "정답 단서 누수 방지",
            "neural": "신경망 기반",
            "imitation": "모방",
            "evidence": "support에서 얻은 증거",
            "router": "어떤 organ을 사용할지 고르는 선택기",
            "teacher": "학습 target을 제공하는 기준 모델",
            "student": "teacher의 선택을 학습하는 neural model",
        },
        "interpretation_guardrail": (
            "This is not an AGI claim. It tests whether anti-leak evidence routing can be compressed "
            "into a neural router. The student still receives symbolic evidence-score features."
        ),
        "seeds": SEEDS,
        "task_families": TASK_FAMILIES,
        "organs": ORGANS,
        "routers": router_names,
        "family_oracle": FAMILY_ORACLE,
        "config": asdict(cfg),
        "runs": all_runs,
        "summary": summary,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(safe_jsonable(output), indent=2, ensure_ascii=False), encoding="utf-8")

    print_table(output)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
