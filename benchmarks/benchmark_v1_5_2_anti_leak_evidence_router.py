# benchmark_v1_5_2_anti_leak_evidence_router.py
# SAGE v1.5.2 - Anti-Leak Evidence Router
#
# English words:
# - Anti-Leak = 정답 단서 누수 방지
# - Evidence = 증거, support/query에서 얻는 단서
# - Router = 어떤 organ을 쓸지 고르는 선택기
# - Fingerprint = 지문, 너무 쉬운 식별 단서
# - Held-out = 학습/검증에 직접 쓰지 않은 새 예시
#
# Goal:
# - v1.5 / v1.5.1에서는 router가 family label 없이 organ을 고를 수 있음을 보았다.
# - 하지만 query type, key prefix 같은 쉬운 fingerprint가 남아 있을 수 있다.
# - v1.5.2는 모든 task가 같은 key shape를 쓰도록 만들어 leakage를 줄인다.
# - EvidenceRouter가 support evidence만으로 organ을 고를 수 있는지 더 엄격하게 검증한다.
#
# Run:
# python benchmark_v1_5_2_anti_leak_evidence_router.py --episodes 20 --queries-per-episode 8
# python benchmark_v1_5_2_anti_leak_evidence_router.py --episodes 120 --queries-per-episode 18

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Optional, Sequence, Tuple


SEEDS = [0, 1, 2, 3, 4]
ACTION_DIM = 6
GRID_SIZE = 5

TASK_FAMILIES = [
    "episodic_memory",
    "affine_rule",
    "threshold_rule",
    "grid_planning",
]

ORGANS = [
    "memory_organ",
    "algebra_organ",
    "concept_organ",
    "planner_organ",
]

FAMILY_ORACLE = {
    "episodic_memory": "memory_organ",
    "affine_rule": "algebra_organ",
    "threshold_rule": "concept_organ",
    "grid_planning": "planner_organ",
}

# All tasks use the same key shape.
# 모든 task가 같은 key 모양을 쓴다.
# 예: ("item", 12345)
# 이렇게 하면 "word:", "x:", tuple length 같은 쉬운 fingerprint를 줄일 수 있다.
Key = Tuple[str, int]


@dataclass
class Config:
    episodes: int = 120
    queries_per_episode: int = 18
    support_per_episode: int = 6
    seed: int = 0
    max_x: int = 72


@dataclass
class Episode:
    family: str
    support: List[Tuple[Key, int]]
    queries: List[Tuple[Key, int]]
    meta: Dict[str, Any]


def set_seed(seed: int) -> None:
    random.seed(seed)


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


def make_key(value: int) -> Key:
    return ("item", int(value))


def key_value(key: Key) -> int:
    # Every key has the same surface shape: ("item", int).
    # 그래서 router는 key prefix만 보고 family를 알아낼 수 없다.
    return int(key[1])


# ----------------------------------------------------------------------
# Grid encoding
# ----------------------------------------------------------------------

def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def plan_step(state: Tuple[int, int, int, int, Tuple[Tuple[int, int], ...]]) -> int:
    x, y, gx, gy, blocked_tuple = state
    blocked = set(blocked_tuple)
    moves = {
        0: (x, max(0, y - 1)),
        1: (x, min(GRID_SIZE - 1, y + 1)),
        2: (max(0, x - 1), y),
        3: (min(GRID_SIZE - 1, x + 1), y),
    }
    candidates: List[Tuple[int, int]] = []
    for action, pos in moves.items():
        if pos in blocked or pos == (x, y):
            continue
        candidates.append((manhattan(pos, (gx, gy)), action))
    if not candidates:
        return 4
    return min(candidates)[1]


def random_grid_state() -> Tuple[int, int, int, int, Tuple[Tuple[int, int], ...]]:
    x, y = random.randrange(GRID_SIZE), random.randrange(GRID_SIZE)
    gx, gy = random.randrange(GRID_SIZE), random.randrange(GRID_SIZE)
    occupied = {(x, y), (gx, gy)}
    blocked: List[Tuple[int, int]] = []
    for _ in range(random.randint(0, 4)):
        cell = (random.randrange(GRID_SIZE), random.randrange(GRID_SIZE))
        if cell not in occupied:
            blocked.append(cell)
            occupied.add(cell)
    return x, y, gx, gy, tuple(sorted(blocked))


def encode_grid_state(state: Tuple[int, int, int, int, Tuple[Tuple[int, int], ...]]) -> int:
    x, y, gx, gy, blocked = state
    mask = 0
    for bx, by in blocked:
        mask |= 1 << (bx * GRID_SIZE + by)

    # Put grid codes in the same integer field as other tasks.
    # 숫자 범위는 다르지만 surface key shape는 동일하다.
    code = x
    code = code * GRID_SIZE + y
    code = code * GRID_SIZE + gx
    code = code * GRID_SIZE + gy
    code = code * (1 << (GRID_SIZE * GRID_SIZE)) + mask
    return code


def decode_grid_state(code: int) -> Tuple[int, int, int, int, Tuple[Tuple[int, int], ...]]:
    mask_size = 1 << (GRID_SIZE * GRID_SIZE)
    mask = code % mask_size
    code //= mask_size
    gy = code % GRID_SIZE
    code //= GRID_SIZE
    gx = code % GRID_SIZE
    code //= GRID_SIZE
    y = code % GRID_SIZE
    code //= GRID_SIZE
    x = code % GRID_SIZE

    blocked = []
    for i in range(GRID_SIZE * GRID_SIZE):
        if mask & (1 << i):
            blocked.append((i // GRID_SIZE, i % GRID_SIZE))
    return x, y, gx, gy, tuple(sorted(blocked))


# ----------------------------------------------------------------------
# Episode generators
# ----------------------------------------------------------------------

def make_episodic_memory_episode(cfg: Config) -> Episode:
    # Exact recall task.
    # Query is always one of the support keys, so memory organ is appropriate.
    xs = random.sample(range(0, cfg.max_x * 20), cfg.support_per_episode)
    mapping = {x: random.randrange(ACTION_DIM) for x in xs}
    support = [(make_key(x), action) for x, action in mapping.items()]
    query_xs = random.choices(xs, k=cfg.queries_per_episode)
    queries = [(make_key(x), mapping[x]) for x in query_xs]
    return Episode("episodic_memory", support, queries, {"mapping": mapping})


def make_affine_rule_episode(cfg: Config) -> Episode:
    # Rule task: y = (a*x + b) % ACTION_DIM.
    # Query is mostly held-out x, so exact memory should not solve it.
    a = random.choice([1, 5])
    b = random.randrange(ACTION_DIM)
    fn = lambda x: (a * x + b) % ACTION_DIM

    xs = random.sample(range(0, cfg.max_x), cfg.support_per_episode)
    support = [(make_key(x), fn(x)) for x in xs]

    support_set = set(xs)
    query_pool = [x for x in range(0, cfg.max_x) if x not in support_set]
    query_xs = random.choices(query_pool, k=cfg.queries_per_episode)
    queries = [(make_key(x), fn(x)) for x in query_xs]
    return Episode("affine_rule", support, queries, {"a": a, "b": b})


def make_threshold_rule_episode(cfg: Config) -> Episode:
    # Concept task: low/high split by threshold.
    # Same key shape as affine_rule, so the router must use support evidence.
    threshold = random.randint(12, cfg.max_x - 12)
    low_action = random.randrange(ACTION_DIM)
    high_action = random.choice([x for x in range(ACTION_DIM) if x != low_action])
    fn = lambda x: low_action if x < threshold else high_action

    # Ensure support has both sides.
    low_xs = random.sample(range(0, threshold), max(1, cfg.support_per_episode // 2))
    high_xs = random.sample(range(threshold, cfg.max_x), cfg.support_per_episode - len(low_xs))
    xs = low_xs + high_xs
    random.shuffle(xs)
    support = [(make_key(x), fn(x)) for x in xs]

    support_set = set(xs)
    query_pool = [x for x in range(0, cfg.max_x) if x not in support_set]
    query_xs = random.choices(query_pool, k=cfg.queries_per_episode)
    queries = [(make_key(x), fn(x)) for x in query_xs]
    return Episode(
        "threshold_rule",
        support,
        queries,
        {"threshold": threshold, "low_action": low_action, "high_action": high_action},
    )


def make_grid_planning_episode(cfg: Config) -> Episode:
    # Planner task.
    # It still uses ("item", int) keys, so tuple length no longer leaks the family.
    support: List[Tuple[Key, int]] = []
    for _ in range(cfg.support_per_episode):
        state = random_grid_state()
        code = encode_grid_state(state)
        support.append((make_key(code), plan_step(state)))

    queries: List[Tuple[Key, int]] = []
    for _ in range(cfg.queries_per_episode):
        state = random_grid_state()
        code = encode_grid_state(state)
        queries.append((make_key(code), plan_step(state)))

    return Episode("grid_planning", support, queries, {"grid_size": GRID_SIZE})


EPISODE_FACTORIES = {
    "episodic_memory": make_episodic_memory_episode,
    "affine_rule": make_affine_rule_episode,
    "threshold_rule": make_threshold_rule_episode,
    "grid_planning": make_grid_planning_episode,
}


def make_episode(cfg: Config, family: str) -> Episode:
    return EPISODE_FACTORIES[family](cfg)


# ----------------------------------------------------------------------
# Scaffold organs
# ----------------------------------------------------------------------

class MemoryOrgan:
    name = "memory_organ"

    def __init__(self) -> None:
        self.memory: Dict[Key, int] = {}
        self.default_action = 0

    def fit(self, support: Sequence[Tuple[Key, int]]) -> None:
        self.memory = {key: int(action) for key, action in support}
        actions = [int(action) for _, action in support]
        if actions:
            self.default_action = max(set(actions), key=actions.count)

    def predict(self, query: Key) -> int:
        return int(self.memory.get(query, self.default_action))


class AlgebraOrgan:
    name = "algebra_organ"

    def __init__(self) -> None:
        self.affine: Optional[Tuple[int, int]] = None
        self.default_action = 0

    def fit(self, support: Sequence[Tuple[Key, int]]) -> None:
        self.affine = None
        actions = [int(action) for _, action in support]
        if actions:
            self.default_action = max(set(actions), key=actions.count)

        pairs = [(key_value(key), int(action)) for key, action in support]
        if len(pairs) >= 2:
            x0, y0 = pairs[0]
            x1, y1 = pairs[1]
            dx = (x1 - x0) % ACTION_DIM
            dy = (y1 - y0) % ACTION_DIM
            if dx in [1, 5]:
                a = dy if dx == 1 else (-dy) % ACTION_DIM
                b = (y0 - a * x0) % ACTION_DIM
                self.affine = (a, b)

    def predict(self, query: Key) -> int:
        if self.affine is not None:
            x = key_value(query)
            a, b = self.affine
            return int((a * x + b) % ACTION_DIM)
        return int(self.default_action)


class ConceptOrgan:
    name = "concept_organ"

    def __init__(self) -> None:
        self.threshold: Optional[Tuple[int, int, int]] = None
        self.default_action = 0

    def fit(self, support: Sequence[Tuple[Key, int]]) -> None:
        self.threshold = None
        actions = [int(action) for _, action in support]
        if actions:
            self.default_action = max(set(actions), key=actions.count)

        pairs = sorted((key_value(key), int(action)) for key, action in support)
        if len(pairs) < 2:
            return

        values = [action for _, action in pairs]
        best_idx = None
        # Find a one-split threshold that best explains support.
        best_acc = -1.0
        for idx in range(1, len(pairs)):
            left_actions = values[:idx]
            right_actions = values[idx:]
            low_action = max(set(left_actions), key=left_actions.count)
            high_action = max(set(right_actions), key=right_actions.count)
            correct = sum(int(a == low_action) for a in left_actions) + sum(int(a == high_action) for a in right_actions)
            acc = correct / len(pairs)
            if acc > best_acc:
                best_acc = acc
                best_idx = idx
                self.threshold = (pairs[idx][0], low_action, high_action)

    def predict(self, query: Key) -> int:
        if self.threshold is None:
            return int(self.default_action)
        x = key_value(query)
        threshold, low_action, high_action = self.threshold
        return int(low_action if x < threshold else high_action)


class PlannerOrgan:
    name = "planner_organ"

    def __init__(self) -> None:
        self.default_action = 0

    def fit(self, support: Sequence[Tuple[Key, int]]) -> None:
        actions = [int(action) for _, action in support]
        if actions:
            self.default_action = max(set(actions), key=actions.count)

    def predict(self, query: Key) -> int:
        # Decode every key as a possible grid state.
        # Non-grid numeric tasks will usually not match the true labels consistently,
        # so support evidence should reject planner_organ there.
        try:
            state = decode_grid_state(key_value(query))
            return int(plan_step(state))
        except Exception:
            return int(self.default_action)


def make_organs() -> Dict[str, Any]:
    return {
        "memory_organ": MemoryOrgan(),
        "algebra_organ": AlgebraOrgan(),
        "concept_organ": ConceptOrgan(),
        "planner_organ": PlannerOrgan(),
    }


def fit_organs(support: Sequence[Tuple[Key, int]]) -> Dict[str, Any]:
    organs = make_organs()
    for organ in organs.values():
        organ.fit(support)
    return organs


def organ_predictions(support: Sequence[Tuple[Key, int]], query: Key) -> Dict[str, int]:
    organs = fit_organs(support)
    return {name: int(organ.predict(query)) for name, organ in organs.items()}


# ----------------------------------------------------------------------
# Routers
# ----------------------------------------------------------------------

class RandomOrganRouter:
    name = "RandomOrganRouter"

    def choose(self, episode: Episode, query: Key) -> Tuple[str, float]:
        return random.choice(ORGANS), 1.0 / len(ORGANS)


class KeyTypeRouter:
    name = "KeyTypeRouter"

    def choose(self, episode: Episode, query: Key) -> Tuple[str, float]:
        # This router represents a leak/fingerprint baseline.
        # But all keys now have the same shape, so it cannot separate task families.
        # It always chooses memory_organ as a weak default.
        if isinstance(query, tuple) and len(query) == 2 and query[0] == "item":
            return "memory_organ", 0.25
        return "memory_organ", 0.25


class FamilyOracleRouter:
    name = "FamilyOracleRouter"

    def choose(self, episode: Episode, query: Key) -> Tuple[str, float]:
        return FAMILY_ORACLE.get(episode.family, "memory_organ"), 1.0


class AntiLeakEvidenceRouter:
    name = "AntiLeakEvidenceRouter"

    def score_organ(self, organ_name: str, episode: Episode, query: Key) -> float:
        support = list(episode.support)
        if not support:
            return 0.0

        # Query exact-match is valid evidence for memory.
        # support 안에 query가 있으면 exact memory가 실제로 쓸 수 있는 증거다.
        exact_match_bonus = 0.0
        if query in [key for key, _ in support] and organ_name == "memory_organ":
            exact_match_bonus = 0.35

        # Leave-one-out evidence.
        # support 중 하나를 숨기고 나머지로 organ을 fit한 뒤 맞히는지 본다.
        correct = 0
        total = 0
        for idx in range(len(support)):
            val_key, val_target = support[idx]
            fit_support = support[:idx] + support[idx + 1 :]
            if not fit_support:
                fit_support = support
            organs = fit_organs(fit_support)
            pred = int(organs[organ_name].predict(val_key))
            correct += int(pred == int(val_target))
            total += 1

        loo_acc = correct / max(1, total)

        # Full support consistency helps memory and simple organs, but not too much.
        full_organs = fit_organs(support)
        full_correct = 0
        for val_key, val_target in support:
            pred = int(full_organs[organ_name].predict(val_key))
            full_correct += int(pred == int(val_target))
        full_acc = full_correct / max(1, len(support))

        # Complexity prior prevents memory from winning every support-fitting case.
        # prior = 사전 가중치 / 기본 선호도.
        complexity_prior = {
            "memory_organ": -0.05,
            "algebra_organ": 0.02,
            "concept_organ": 0.02,
            "planner_organ": 0.02,
        }[organ_name]

        score = 0.70 * loo_acc + 0.30 * full_acc + exact_match_bonus + complexity_prior
        return float(score)

    def choose(self, episode: Episode, query: Key) -> Tuple[str, float]:
        scores = {organ: self.score_organ(organ, episode, query) for organ in ORGANS}
        chosen = max(ORGANS, key=lambda organ: scores[organ])
        ranked = sorted(scores.values(), reverse=True)
        confidence = ranked[0] - ranked[1] if len(ranked) >= 2 else ranked[0]
        confidence = max(1.0 / len(ORGANS), min(1.0, confidence))
        return chosen, float(confidence)


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
    set_seed(seed)

    correct_by_family = {family: 0 for family in TASK_FAMILIES}
    total_by_family = {family: 0 for family in TASK_FAMILIES}
    organ_counts_by_family = {
        family: {organ: 0 for organ in ORGANS}
        for family in TASK_FAMILIES
    }
    confidence_values: List[float] = []

    for family in TASK_FAMILIES:
        for _ in range(cfg.episodes):
            episode = make_episode(cfg, family)
            for query, target in episode.queries:
                chosen, confidence = router.choose(episode, query)
                if chosen not in ORGANS:
                    chosen = "memory_organ"
                preds = organ_predictions(episode.support, query)
                pred = int(preds[chosen])
                correct_by_family[family] += int(pred == int(target))
                total_by_family[family] += 1
                organ_counts_by_family[family][chosen] += 1
                confidence_values.append(float(confidence))

    family_accuracy = {
        family: correct_by_family[family] / max(1, total_by_family[family])
        for family in TASK_FAMILIES
    }

    task_diversity = mean(family_accuracy.values())
    exact_memory = family_accuracy["episodic_memory"]
    rule_generalization = mean([
        family_accuracy["affine_rule"],
        family_accuracy["threshold_rule"],
    ])
    planning = family_accuracy["grid_planning"]
    spec = specialization_score(organ_counts_by_family)
    route_confidence = mean(confidence_values) if confidence_values else 0.0

    organ_usage_by_family: Dict[str, Dict[str, float]] = {}
    organ_top_by_family: Dict[str, str] = {}
    organ_entropy_by_family: Dict[str, Optional[float]] = {}

    for family in TASK_FAMILIES:
        counts = organ_counts_by_family[family]
        total = sum(counts.values())
        if total <= 0:
            vector = {organ: 0.0 for organ in ORGANS}
        else:
            vector = {organ: counts[organ] / total for organ in ORGANS}
        organ_usage_by_family[family] = {organ: round_float(value) for organ, value in vector.items()}
        organ_top_by_family[family] = max(ORGANS, key=lambda organ: vector[organ])
        organ_entropy_by_family[family] = round_float(normalized_entropy(list(vector.values())))

    # anti_leak_score는 v1.5.2의 종합 점수.
    # 성능 + 규칙 일반화 + 계획 + 전문화 + confidence를 함께 본다.
    anti_leak_score = mean([
        task_diversity,
        exact_memory,
        rule_generalization,
        planning,
        spec,
        route_confidence,
    ])

    return {
        "family_accuracy": {k: round_float(v) for k, v in family_accuracy.items()},
        "anti_leak_score": round_float(anti_leak_score),
        "task_diversity": round_float(task_diversity),
        "exact_memory": round_float(exact_memory),
        "rule_generalization": round_float(rule_generalization),
        "planning": round_float(planning),
        "organ_specialization": round_float(spec),
        "route_confidence": round_float(route_confidence),
        "oracle_gap": None,
        "random_gain": None,
        "keytype_gain": None,
        "organ_usage_by_family": organ_usage_by_family,
        "organ_usage_entropy_by_family": organ_entropy_by_family,
        "organ_top_by_family": organ_top_by_family,
    }


def fill_relative_metrics(runs: List[Dict[str, Any]]) -> None:
    random_by_seed = {}
    keytype_by_seed = {}
    oracle_by_seed = {}

    for run in runs:
        seed = run["seed"]
        score = run["metrics"]["anti_leak_score"]
        if run["router"] == "RandomOrganRouter":
            random_by_seed[seed] = score
        elif run["router"] == "KeyTypeRouter":
            keytype_by_seed[seed] = score
        elif run["router"] == "FamilyOracleRouter":
            oracle_by_seed[seed] = score

    for run in runs:
        seed = run["seed"]
        score = run["metrics"]["anti_leak_score"]
        if seed in random_by_seed:
            run["metrics"]["random_gain"] = round_float(score - random_by_seed[seed])
        if seed in keytype_by_seed:
            run["metrics"]["keytype_gain"] = round_float(score - keytype_by_seed[seed])
        if seed in oracle_by_seed:
            run["metrics"]["oracle_gap"] = round_float(oracle_by_seed[seed] - score)


def summarize_runs(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = [
        "anti_leak_score",
        "task_diversity",
        "exact_memory",
        "rule_generalization",
        "planning",
        "organ_specialization",
        "route_confidence",
        "oracle_gap",
        "random_gain",
        "keytype_gain",
    ]
    summary: Dict[str, Any] = {}
    for key in keys:
        values = [float(run["metrics"][key]) for run in runs if run["metrics"].get(key) is not None]
        if not values:
            summary[key] = {"mean": None, "std": None, "n": 0}
        else:
            summary[key] = {"mean": mean(values), "std": safe_std(values), "n": len(values)}
    return summary


def print_table(output: Dict[str, Any]) -> None:
    keys = [
        "anti_leak_score",
        "task_diversity",
        "rule_generalization",
        "planning",
        "organ_specialization",
        "route_confidence",
        "oracle_gap",
        "keytype_gain",
    ]
    widths = [24, 24, 22, 28, 18, 26, 22, 18, 18]
    header = ["router"] + [f"{key} mean+/-std" for key in keys]

    print("\n=== SAGE v1.5.2 Anti-Leak Evidence Router ===")
    print(f"seeds: {output['seeds']}")
    print(" | ".join(text.ljust(width) for text, width in zip(header, widths)))
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))

    for router_name in output["routers"]:
        row = [router_name]
        for key in keys:
            item = output["summary"][router_name][key]
            if item["mean"] is None:
                row.append("N/A")
            else:
                row.append(f"{item['mean']:.4f}+/-{item['std']:.4f}")
        print(" | ".join(text.ljust(width) for text, width in zip(row, widths)))


def run_seed(cfg: Config, seed: int) -> List[Dict[str, Any]]:
    seed_cfg = Config(**{**asdict(cfg), "seed": seed})
    set_seed(seed)

    routers = [
        RandomOrganRouter(),
        KeyTypeRouter(),
        AntiLeakEvidenceRouter(),
        FamilyOracleRouter(),
    ]

    seed_runs: List[Dict[str, Any]] = []
    print(f"\n[Seed {seed}] evaluating anti-leak routers ...")
    for router in routers:
        print(f"  evaluating {router.name} ...")
        metrics = evaluate_router(router, seed_cfg, seed)
        seed_runs.append({
            "seed": seed,
            "router": router.name,
            "config": asdict(seed_cfg),
            "metrics": metrics,
        })
        print(
            "    "
            f"score={metrics['anti_leak_score']:.4f}, "
            f"task={metrics['task_diversity']:.4f}, "
            f"rule={metrics['rule_generalization']:.4f}, "
            f"planning={metrics['planning']:.4f}, "
            f"spec={metrics['organ_specialization']:.4f}"
        )

    return seed_runs


def safe_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): safe_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [safe_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)


def main() -> None:
    parser = argparse.ArgumentParser(description="SAGE v1.5.2 Anti-Leak Evidence Router")
    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument("--queries-per-episode", type=int, default=18)
    parser.add_argument("--support-per-episode", type=int, default=6)
    parser.add_argument("--max-x", type=int, default=72)
    parser.add_argument("--out", type=str, default="results/v1_5_2_anti_leak_evidence_router_benchmark.json")
    args = parser.parse_args()

    cfg = Config(
        episodes=args.episodes,
        queries_per_episode=args.queries_per_episode,
        support_per_episode=args.support_per_episode,
        max_x=args.max_x,
    )

    all_runs: List[Dict[str, Any]] = []
    for seed in SEEDS:
        all_runs.extend(run_seed(cfg, seed))

    fill_relative_metrics(all_runs)

    router_names = [
        "RandomOrganRouter",
        "KeyTypeRouter",
        "AntiLeakEvidenceRouter",
        "FamilyOracleRouter",
    ]

    summary = {
        router_name: summarize_runs([run for run in all_runs if run["router"] == router_name])
        for router_name in router_names
    }

    output = {
        "benchmark": "SAGE-v1.5.2-anti-leak-evidence-router",
        "goal": (
            "Reduce task-family fingerprint leakage by forcing all task families to use the same key shape, "
            "then test whether support evidence can still select useful cognitive organs."
        ),
        "terminology": {
            "anti_leak": "정답 단서 누수 방지",
            "evidence": "support/query에서 얻은 증거",
            "router": "어떤 organ을 사용할지 고르는 선택기",
            "fingerprint": "너무 쉬운 식별 단서",
            "held_out": "직접 보지 않은 새 예시",
        },
        "interpretation_guardrail": (
            "This is not an AGI claim. It is a stricter routing benchmark. "
            "If AntiLeakEvidenceRouter only beats weak baselines but remains far from oracle, "
            "organ routing still requires stronger evidence extraction."
        ),
        "anti_leak_design": {
            "shared_key_shape": "All task keys are represented as ('item', int).",
            "removed_fingerprints": ["word prefix", "x prefix", "tuple length", "direct family label"],
            "remaining_possible_leaks": ["numeric range", "support label pattern"],
        },
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
