# SAGE v1.6.2 - Anti-Leak Routing Task Plugin
#
# Goal:
# - Migrate a simplified anti-leak evidence-routing benchmark into the v1.6 core/config structure.
# - Keep the control loop in sage_core.engine.
# - Keep task-specific organs/routers/environment here.
#
# This is a migration benchmark, not a new AGI claim.

from __future__ import annotations

import random
from collections import Counter
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from sage_core import BaseEnvironment, BaseOrgan, BaseRouter, OrganResult, SAGEState
from sage_core.metrics import BasicRunMetric

ACTION_DIM = 4
FAMILIES = ["memory", "linear_rule", "threshold_rule", "planning"]
ORGANS = ["memory_organ", "linear_organ", "threshold_organ", "planner_organ"]

FAMILY_TO_ORGAN = {
    "memory": "memory_organ",
    "linear_rule": "linear_organ",
    "threshold_rule": "threshold_organ",
    "planning": "planner_organ",
}


def majority_action(support: Sequence[Tuple[Any, int]]) -> int:
    if not support:
        return 0
    actions = [int(action) for _, action in support]
    return Counter(actions).most_common(1)[0][0]


def as_item(value: int) -> Tuple[str, int]:
    # Anti-leak key shape: all tasks use the same visible key format.
    return ("item", int(value))


def parse_item(key: Any) -> int:
    if isinstance(key, tuple) and len(key) == 2 and key[0] == "item":
        return int(key[1])
    return 0


class Episode:
    def __init__(
        self,
        family: str,
        support: List[Tuple[Any, int]],
        query: Any,
        target: int,
    ) -> None:
        self.family = family
        self.support = support
        self.query = query
        self.target = int(target)


def make_memory_episode(rng: random.Random, support_size: int) -> Episode:
    # Query is always included in support. Correct organ should retrieve exact memory.
    keys = rng.sample(range(100, 999), support_size)
    actions = [rng.randrange(ACTION_DIM) for _ in keys]
    support = [(as_item(k), a) for k, a in zip(keys, actions)]
    idx = rng.randrange(len(support))
    query, target = support[idx]
    return Episode("memory", support, query, target)


def make_linear_episode(rng: random.Random, support_size: int) -> Episode:
    # y = (a*x + b) mod ACTION_DIM
    a = rng.choice([1, 3])
    b = rng.randrange(ACTION_DIM)
    xs = rng.sample(range(0, 30), support_size + 1)
    support_x = xs[:-1]
    query_x = xs[-1]
    support = [(as_item(x), (a * x + b) % ACTION_DIM) for x in support_x]
    target = (a * query_x + b) % ACTION_DIM
    return Episode("linear_rule", support, as_item(query_x), target)


def make_threshold_episode(rng: random.Random, support_size: int) -> Episode:
    # y = low_action if x < threshold else high_action
    threshold = rng.randrange(8, 22)
    low_action = rng.randrange(ACTION_DIM)
    high_action = (low_action + rng.randrange(1, ACTION_DIM)) % ACTION_DIM

    # Ensure support has examples on both sides.
    left = rng.sample(range(0, threshold), max(2, support_size // 2))
    right = rng.sample(range(threshold, 30), support_size - len(left))
    xs = left + right
    rng.shuffle(xs)

    support = [
        (as_item(x), low_action if x < threshold else high_action)
        for x in xs
    ]

    query_x = rng.randrange(0, 30)
    target = low_action if query_x < threshold else high_action
    return Episode("threshold_rule", support, as_item(query_x), target)


def make_planning_episode(rng: random.Random, support_size: int) -> Episode:
    # 1D planning: query asks which action moves position toward goal.
    # visible key is still ("item", encoded_int), not tuple state.
    # action 0 = left, action 1 = right, action 2 = stay-left-ish, action 3 = stay-right-ish.
    support: List[Tuple[Any, int]] = []
    for _ in range(support_size):
        pos = rng.randrange(0, 10)
        goal = rng.randrange(0, 10)
        encoded = pos * 10 + goal
        action = 1 if goal > pos else 0 if goal < pos else 2
        support.append((as_item(encoded), action))

    pos = rng.randrange(0, 10)
    goal = rng.randrange(0, 10)
    encoded = pos * 10 + goal
    target = 1 if goal > pos else 0 if goal < pos else 2
    return Episode("planning", support, as_item(encoded), target)


def make_episode(rng: random.Random, family: str, support_size: int) -> Episode:
    if family == "memory":
        return make_memory_episode(rng, support_size)
    if family == "linear_rule":
        return make_linear_episode(rng, support_size)
    if family == "threshold_rule":
        return make_threshold_episode(rng, support_size)
    if family == "planning":
        return make_planning_episode(rng, support_size)
    raise ValueError(f"unknown family: {family}")


# ----------------------------------------------------------------------
# Organs
# ----------------------------------------------------------------------

class MemoryOrgan(BaseOrgan):
    name = "memory_organ"

    def process(self, state: SAGEState, signal: Dict[str, Any]) -> OrganResult:
        support = signal["support"]
        query = signal["query"]
        table = {key: int(action) for key, action in support}
        default = majority_action(support)
        action = table.get(query, default)
        confidence = 0.95 if query in table else 0.30
        return OrganResult(self.name, action, confidence, {"exact_match": query in table})


class LinearOrgan(BaseOrgan):
    name = "linear_organ"

    def process(self, state: SAGEState, signal: Dict[str, Any]) -> OrganResult:
        support = signal["support"]
        query_x = parse_item(signal["query"])
        default = majority_action(support)

        pairs = [(parse_item(key), int(action)) for key, action in support]
        if len(pairs) < 2:
            return OrganResult(self.name, default, 0.20, {"fit": False})

        # Brute-force tiny modular affine fit.
        best_a, best_b, best_score = 0, default, -1
        for a in range(ACTION_DIM):
            for b in range(ACTION_DIM):
                score = sum(1 for x, y in pairs if (a * x + b) % ACTION_DIM == y)
                if score > best_score:
                    best_a, best_b, best_score = a, b, score

        action = (best_a * query_x + best_b) % ACTION_DIM
        confidence = best_score / max(1, len(pairs))
        return OrganResult(self.name, action, confidence, {"a": best_a, "b": best_b, "fit_score": confidence})


class ThresholdOrgan(BaseOrgan):
    name = "threshold_organ"

    def process(self, state: SAGEState, signal: Dict[str, Any]) -> OrganResult:
        support = signal["support"]
        query_x = parse_item(signal["query"])
        default = majority_action(support)

        pairs = sorted((parse_item(key), int(action)) for key, action in support)
        if len(pairs) < 3:
            return OrganResult(self.name, default, 0.20, {"fit": False})

        best_threshold, best_low, best_high, best_score = 0, default, default, -1
        xs = [x for x, _ in pairs]
        candidates = list(range(min(xs), max(xs) + 2))

        for threshold in candidates:
            lows = [y for x, y in pairs if x < threshold]
            highs = [y for x, y in pairs if x >= threshold]
            if not lows or not highs:
                continue
            low_action = Counter(lows).most_common(1)[0][0]
            high_action = Counter(highs).most_common(1)[0][0]
            score = sum(
                1
                for x, y in pairs
                if (low_action if x < threshold else high_action) == y
            )
            if score > best_score:
                best_threshold = threshold
                best_low = low_action
                best_high = high_action
                best_score = score

        if best_score < 0:
            return OrganResult(self.name, default, 0.20, {"fit": False})

        action = best_low if query_x < best_threshold else best_high
        confidence = best_score / max(1, len(pairs))
        return OrganResult(
            self.name,
            action,
            confidence,
            {"threshold": best_threshold, "low": best_low, "high": best_high, "fit_score": confidence},
        )


class PlannerOrgan(BaseOrgan):
    name = "planner_organ"

    def process(self, state: SAGEState, signal: Dict[str, Any]) -> OrganResult:
        encoded = parse_item(signal["query"])
        pos = encoded // 10
        goal = encoded % 10
        action = 1 if goal > pos else 0 if goal < pos else 2
        confidence = 0.90
        return OrganResult(self.name, action, confidence, {"pos": pos, "goal": goal})


# ----------------------------------------------------------------------
# Routers
# ----------------------------------------------------------------------

class RandomAntiLeakRouter(BaseRouter):
    name = "random_anti_leak_router"

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def route(self, state: SAGEState, signal: Dict[str, Any], organs: Mapping[str, BaseOrgan]) -> List[str]:
        return [self.rng.choice(list(organs.keys()))]

    def aggregate(self, state: SAGEState, signal: Dict[str, Any], outputs: Mapping[str, OrganResult]) -> Dict[str, Any]:
        if not outputs:
            return {"prediction": 0, "chosen_organ": None}
        result = next(iter(outputs.values()))
        return {"prediction": int(result.action), "chosen_organ": result.organ_name, "confidence": result.confidence}


class KeyTypeAntiLeakRouter(BaseRouter):
    name = "keytype_anti_leak_router"

    def route(self, state: SAGEState, signal: Dict[str, Any], organs: Mapping[str, BaseOrgan]) -> List[str]:
        # All visible keys have the same type in this anti-leak task.
        # This baseline should be weak because key type no longer identifies family.
        return ["memory_organ"] if "memory_organ" in organs else [next(iter(organs.keys()))]

    def aggregate(self, state: SAGEState, signal: Dict[str, Any], outputs: Mapping[str, OrganResult]) -> Dict[str, Any]:
        if not outputs:
            return {"prediction": 0, "chosen_organ": None}
        result = next(iter(outputs.values()))
        return {"prediction": int(result.action), "chosen_organ": result.organ_name, "confidence": result.confidence}


class FamilyOracleAntiLeakRouter(BaseRouter):
    name = "family_oracle_anti_leak_router"

    def route(self, state: SAGEState, signal: Dict[str, Any], organs: Mapping[str, BaseOrgan]) -> List[str]:
        # Oracle only: uses hidden family label from environment metadata.
        family = signal.get("_oracle_family", "memory")
        chosen = FAMILY_TO_ORGAN.get(family, "memory_organ")
        return [chosen] if chosen in organs else [next(iter(organs.keys()))]

    def aggregate(self, state: SAGEState, signal: Dict[str, Any], outputs: Mapping[str, OrganResult]) -> Dict[str, Any]:
        if not outputs:
            return {"prediction": 0, "chosen_organ": None}
        result = next(iter(outputs.values()))
        return {"prediction": int(result.action), "chosen_organ": result.organ_name, "confidence": result.confidence}


class EvidenceAntiLeakRouter(BaseRouter):
    name = "evidence_anti_leak_router"

    def route(self, state: SAGEState, signal: Dict[str, Any], organs: Mapping[str, BaseOrgan]) -> List[str]:
        # Activate all organs. Aggregate chooses the organ whose predicted support validation works best.
        return [name for name in ORGANS if name in organs]

    def aggregate(self, state: SAGEState, signal: Dict[str, Any], outputs: Mapping[str, OrganResult]) -> Dict[str, Any]:
        support = list(signal["support"])
        if not outputs:
            return {"prediction": 0, "chosen_organ": None, "confidence": 0.0}

        # For this migrated plugin, organ confidence is produced from its own fitting logic.
        # Tie-breaks prefer confidence, then stable organ order.
        best_name = max(outputs, key=lambda name: (outputs[name].confidence, -ORGANS.index(name) if name in ORGANS else -999))
        best = outputs[best_name]
        return {
            "prediction": int(best.action),
            "chosen_organ": best.organ_name,
            "confidence": float(best.confidence),
        }


# ----------------------------------------------------------------------
# Environment / Metric
# ----------------------------------------------------------------------

class AntiLeakRoutingEnvironment(BaseEnvironment):
    name = "anti_leak_routing_environment"

    def __init__(
        self,
        episodes_per_family: int = 40,
        support_size: int = 6,
        seed: int = 0,
        shuffle: bool = True,
    ) -> None:
        self.episodes_per_family = int(episodes_per_family)
        self.support_size = int(support_size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.rng = random.Random(seed)
        self.episodes: List[Episode] = []
        self.index = 0

    def _build_episodes(self) -> None:
        self.rng = random.Random(self.seed)
        episodes: List[Episode] = []
        for family in FAMILIES:
            for _ in range(self.episodes_per_family):
                episodes.append(make_episode(self.rng, family, self.support_size))
        if self.shuffle:
            self.rng.shuffle(episodes)
        self.episodes = episodes

    def _signal_from_episode(self, episode: Episode) -> Dict[str, Any]:
        return {
            "support": episode.support,
            "query": episode.query,
            "_oracle_family": episode.family,
            "meta": {
                "note": "All visible query/support keys use the same anti-leak shape.",
                "key_shape": "('item', int)",
            },
        }

    def reset(self) -> Dict[str, Any]:
        self._build_episodes()
        self.index = 0
        return self._signal_from_episode(self.episodes[self.index])

    def step(self, action: Dict[str, Any]):
        episode = self.episodes[self.index]
        pred = int(action.get("prediction", 0))
        correct = pred == episode.target
        reward = 1.0 if correct else -1.0

        self.index += 1
        done = self.index >= len(self.episodes)
        if done:
            next_signal = {"support": [], "query": ("item", 0), "_oracle_family": "done", "meta": {}}
        else:
            next_signal = self._signal_from_episode(self.episodes[self.index])

        return next_signal, reward, done, {
            "correct": correct,
            "target": episode.target,
            "prediction": pred,
            "family": episode.family,
            "chosen_organ": action.get("chosen_organ"),
            "confidence": action.get("confidence", 0.0),
        }


class AntiLeakRoutingMetric(BasicRunMetric):
    name = "anti_leak_routing_metric"

    def evaluate(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        base = super().evaluate(history)
        by_family: Dict[str, List[float]] = {}
        chosen_by_family: Dict[str, Counter] = {}

        for item in history:
            info = item.get("info", {})
            family = str(info.get("family", "unknown"))
            correct = 1.0 if info.get("correct", False) else 0.0
            chosen = str(info.get("chosen_organ", "none"))

            by_family.setdefault(family, []).append(correct)
            chosen_by_family.setdefault(family, Counter())[chosen] += 1

        family_accuracy = {
            family: sum(vals) / max(1, len(vals))
            for family, vals in sorted(by_family.items())
        }
        family_top_organ = {
            family: counts.most_common(1)[0][0]
            for family, counts in sorted(chosen_by_family.items())
            if counts
        }

        base["family_accuracy"] = family_accuracy
        base["family_top_organ"] = family_top_organ
        base["task_diversity"] = sum(family_accuracy.values()) / max(1, len(family_accuracy))
        base["families"] = list(sorted(family_accuracy.keys()))
        return base
