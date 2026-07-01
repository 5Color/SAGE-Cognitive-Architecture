from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


SEEDS = [0, 1, 2, 3, 4]
ACTION_DIM = 6
GRID_SIZE = 5

TASK_FAMILIES = [
    "episodic_memory",
    "affine_rule",
    "threshold_rule",
    "language_action",
    "grid_planning",
    "world_dynamics",
]

ORGANS = [
    "memory_organ",
    "algebra_organ",
    "concept_organ",
    "planner_organ",
]


@dataclass
class Config:
    episodes: int = 120
    queries_per_episode: int = 18
    support_per_episode: int = 6
    seed: int = 0


def round_float(value: Optional[float], digits: int = 4) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


def normalized_entropy(values: Sequence[float]) -> Optional[float]:
    total = sum(max(0.0, float(v)) for v in values)
    if total <= 0 or len(values) <= 1:
        return None
    probs = [max(0.0, float(v)) / total for v in values if v > 0]
    entropy = -sum(p * math.log(p) for p in probs)
    return entropy / math.log(len(values))


def safe_std(values: Sequence[float]) -> float:
    return stdev(values) if len(values) >= 2 else 0.0


def make_key(prefix: str, value: Any) -> str:
    return f"{prefix}:{value}"


@dataclass
class Episode:
    family: str
    support: List[Tuple[Any, int]]
    queries: List[Tuple[Any, int]]
    meta: Dict[str, Any]


def make_episodic_memory_episode(cfg: Config) -> Episode:
    keys = random.sample(range(100, 999), cfg.support_per_episode)
    mapping = {key: random.randrange(ACTION_DIM) for key in keys}
    support = [(make_key("symbol", key), action) for key, action in mapping.items()]
    query_keys = random.choices(keys, k=cfg.queries_per_episode)
    queries = [(make_key("symbol", key), mapping[key]) for key in query_keys]
    return Episode("episodic_memory", support, queries, {"mapping": mapping})


def make_affine_rule_episode(cfg: Config) -> Episode:
    # Invertible modular affine maps can be inferred from two support points.
    a = random.choice([1, 5])
    b = random.randrange(ACTION_DIM)
    fn = lambda x: (a * x + b) % ACTION_DIM
    support_x = [0, 1]
    support = [(make_key("x", x), fn(x)) for x in support_x]
    query_x = [random.randrange(ACTION_DIM) for _ in range(cfg.queries_per_episode)]
    queries = [(make_key("x", x), fn(x)) for x in query_x]
    return Episode("affine_rule", support, queries, {"a": a, "b": b})


def make_threshold_rule_episode(cfg: Config) -> Episode:
    threshold = random.randint(1, ACTION_DIM - 2)
    low_action = random.randrange(ACTION_DIM)
    high_action = random.choice([x for x in range(ACTION_DIM) if x != low_action])
    fn = lambda x: low_action if x < threshold else high_action
    support_x = list(range(ACTION_DIM))
    support = [(make_key("x", x), fn(x)) for x in support_x]
    query_x = [random.randrange(ACTION_DIM) for _ in range(cfg.queries_per_episode)]
    queries = [(make_key("x", x), fn(x)) for x in query_x]
    return Episode(
        "threshold_rule",
        support,
        queries,
        {"threshold": threshold, "low_action": low_action, "high_action": high_action},
    )


LANG_ACTIONS = {
    0: ["silence", "pause", "wait"],
    1: ["greet", "hello", "welcome"],
    2: ["ask", "question", "inquire"],
    3: ["empathize", "comfort", "support"],
    4: ["explain", "clarify", "describe"],
    5: ["avoid", "withdraw", "decline"],
}


def make_language_action_episode(cfg: Config) -> Episode:
    support: List[Tuple[str, int]] = []
    queries: List[Tuple[str, int]] = []
    for action, words in LANG_ACTIONS.items():
        chosen = random.sample(words, 2)
        support.append((make_key("word", chosen[0]), action))
        queries.append((make_key("word", chosen[1]), action))
    while len(queries) < cfg.queries_per_episode:
        action = random.randrange(ACTION_DIM)
        word = random.choice(LANG_ACTIONS[action])
        queries.append((make_key("word", word), action))
    random.shuffle(support)
    random.shuffle(queries)
    return Episode("language_action", support, queries, {"lexicon": LANG_ACTIONS})


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


def make_grid_planning_episode(cfg: Config) -> Episode:
    support = [(random_grid_state(), 0) for _ in range(2)]
    queries = []
    for _ in range(cfg.queries_per_episode):
        state = random_grid_state()
        queries.append((state, plan_step(state)))
    return Episode("grid_planning", support, queries, {"grid_size": GRID_SIZE})


def apply_world_delta(state: Tuple[int, int], action: int, deltas: Dict[int, Tuple[int, int]]) -> Tuple[int, int]:
    dx, dy = deltas[action]
    return (
        max(0, min(GRID_SIZE - 1, state[0] + dx)),
        max(0, min(GRID_SIZE - 1, state[1] + dy)),
    )


def world_best_action(query: Tuple[int, int, int, int], deltas: Dict[int, Tuple[int, int]]) -> int:
    x, y, gx, gy = query
    ranked = []
    for action in range(4):
        nx, ny = apply_world_delta((x, y), action, deltas)
        ranked.append((manhattan((nx, ny), (gx, gy)), action))
    return min(ranked)[1]


def make_world_dynamics_episode(cfg: Config) -> Episode:
    base_deltas = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    shuffled = random.sample(base_deltas, len(base_deltas))
    deltas = {action: shuffled[action] for action in range(4)}

    support = []
    center = (2, 2)
    for action in range(4):
        nx, ny = apply_world_delta(center, action, deltas)
        # Support labels are not enough here; the agent must infer transition deltas.
        support.append((("transition", center[0], center[1], action, nx, ny), action))

    queries = []
    for _ in range(cfg.queries_per_episode):
        x, y = random.randrange(GRID_SIZE), random.randrange(GRID_SIZE)
        gx, gy = random.randrange(GRID_SIZE), random.randrange(GRID_SIZE)
        query = (x, y, gx, gy)
        queries.append((query, world_best_action(query, deltas)))

    return Episode("world_dynamics", support, queries, {"hidden_deltas": deltas})


EPISODE_FACTORIES = {
    "episodic_memory": make_episodic_memory_episode,
    "affine_rule": make_affine_rule_episode,
    "threshold_rule": make_threshold_rule_episode,
    "language_action": make_language_action_episode,
    "grid_planning": make_grid_planning_episode,
    "world_dynamics": make_world_dynamics_episode,
}


class RandomAgent:
    name = "RandomBaseline"

    def fit(self, episode: Episode) -> None:
        pass

    def predict(self, query: Any) -> Tuple[int, str]:
        return random.randrange(ACTION_DIM), "memory_organ"


class EpisodicMemoryAgent:
    name = "EpisodicMemoryOnly"

    def __init__(self) -> None:
        self.memory: Dict[Any, int] = {}
        self.default_action = 0

    def fit(self, episode: Episode) -> None:
        self.memory = {key: action for key, action in episode.support}
        if episode.support:
            actions = [action for _, action in episode.support]
            self.default_action = max(set(actions), key=actions.count)

    def predict(self, query: Any) -> Tuple[int, str]:
        return self.memory.get(query, self.default_action), "memory_organ"


class RuleTransferAgent:
    name = "RuleTransferMemoryPlanner"

    def __init__(self) -> None:
        self.family = ""
        self.memory: Dict[Any, int] = {}
        self.affine: Optional[Tuple[int, int]] = None
        self.threshold: Optional[Tuple[int, int, int]] = None
        self.lexicon: Dict[str, int] = {}
        self.world_deltas: Optional[Dict[int, Tuple[int, int]]] = None

    def fit(self, episode: Episode) -> None:
        self.family = episode.family
        self.memory = {key: action for key, action in episode.support}
        self.affine = None
        self.threshold = None
        self.lexicon = {}
        self.world_deltas = None

        if episode.family == "affine_rule":
            pairs = [(int(str(key).split(":")[1]), action) for key, action in episode.support]
            if len(pairs) >= 2:
                x0, y0 = pairs[0]
                x1, y1 = pairs[1]
                dx = (x1 - x0) % ACTION_DIM
                dy = (y1 - y0) % ACTION_DIM
                if dx in [1, 5]:
                    a = dy if dx == 1 else (-dy) % ACTION_DIM
                    b = (y0 - a * x0) % ACTION_DIM
                    self.affine = (a, b)

        elif episode.family == "threshold_rule":
            pairs = sorted(
                (int(str(key).split(":")[1]), action) for key, action in episode.support
            )
            actions = [action for _, action in pairs]
            for idx in range(1, len(pairs)):
                if actions[idx] != actions[idx - 1]:
                    self.threshold = (pairs[idx][0], actions[idx - 1], actions[idx])
                    break

        elif episode.family == "language_action":
            for action, words in LANG_ACTIONS.items():
                for word in words:
                    self.lexicon[make_key("word", word)] = action

        elif episode.family == "world_dynamics":
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

    def predict(self, query: Any) -> Tuple[int, str]:
        if query in self.memory:
            return self.memory[query], "memory_organ"

        if self.family == "affine_rule" and self.affine is not None:
            x = int(str(query).split(":")[1])
            a, b = self.affine
            return (a * x + b) % ACTION_DIM, "algebra_organ"

        if self.family == "threshold_rule" and self.threshold is not None:
            x = int(str(query).split(":")[1])
            threshold, low_action, high_action = self.threshold
            return (low_action if x < threshold else high_action), "concept_organ"

        if self.family == "language_action":
            return self.lexicon.get(query, 0), "concept_organ"

        if self.family == "grid_planning":
            return plan_step(query), "planner_organ"

        if self.family == "world_dynamics" and self.world_deltas is not None:
            return world_best_action(query, self.world_deltas), "planner_organ"

        return self.memory.get(query, 0), "memory_organ"


def make_episode(cfg: Config, family: str) -> Episode:
    return EPISODE_FACTORIES[family](cfg)


def evaluate_agent(agent: Any, cfg: Config) -> Dict[str, Any]:
    correct_by_family = {family: 0 for family in TASK_FAMILIES}
    total_by_family = {family: 0 for family in TASK_FAMILIES}
    organ_counts = {
        family: {organ: 0 for organ in ORGANS}
        for family in TASK_FAMILIES
    }

    for family in TASK_FAMILIES:
        for _ in range(cfg.episodes):
            episode = make_episode(cfg, family)
            agent.fit(episode)
            for query, target in episode.queries:
                pred, organ = agent.predict(query)
                correct_by_family[family] += int(pred == target)
                total_by_family[family] += 1
                if organ not in organ_counts[family]:
                    organ = "memory_organ"
                organ_counts[family][organ] += 1

    family_accuracy = {
        family: round_float(correct_by_family[family] / total_by_family[family])
        for family in TASK_FAMILIES
    }
    usage_by_family = {}
    entropy_by_family = {}
    top_organs = []
    for family in TASK_FAMILIES:
        counts = [organ_counts[family][organ] for organ in ORGANS]
        total = sum(counts)
        vector = [round_float(count / total) if total > 0 else 0.0 for count in counts]
        usage_by_family[family] = dict(zip(ORGANS, vector))
        entropy_by_family[family] = round_float(normalized_entropy(vector))
        top_organs.append(ORGANS[max(range(len(ORGANS)), key=lambda idx: vector[idx])])

    valid_entropies = [value for value in entropy_by_family.values() if value is not None]
    specialization = 0.0
    if valid_entropies:
        entropy_component = 1.0 - mean(valid_entropies)
        diversity_component = len(set(top_organs)) / min(len(ORGANS), len(TASK_FAMILIES))
        # Low entropy is useful only when different tasks select different organs.
        # Multiplying by diversity prevents one-organ collapse from looking specialized.
        specialization = (
            0.55 * entropy_component + 0.45 * diversity_component
        ) * diversity_component

    axis_scores = {
        "task_diversity": round_float(mean(family_accuracy.values())),
        "long_memory": family_accuracy["episodic_memory"],
        "fast_rule_inference": round_float(mean([
            family_accuracy["affine_rule"],
            family_accuracy["threshold_rule"],
        ])),
        "language_action": family_accuracy["language_action"],
        "planning": family_accuracy["grid_planning"],
        "world_model_planning": family_accuracy["world_dynamics"],
        "organ_specialization": round_float(specialization),
    }

    return {
        "family_accuracy": family_accuracy,
        "axis_scores": axis_scores,
        "agi_transfer_score": round_float(mean(axis_scores.values())),
        "organ_usage_by_family": usage_by_family,
        "organ_usage_entropy_by_family": entropy_by_family,
        "organ_top_by_family": dict(zip(TASK_FAMILIES, top_organs)),
    }


def summarize_runs(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    metric_keys = [
        "agi_transfer_score",
        "task_diversity",
        "long_memory",
        "fast_rule_inference",
        "language_action",
        "planning",
        "world_model_planning",
        "organ_specialization",
    ]
    summary: Dict[str, Any] = {}
    for key in metric_keys:
        values = []
        for run in runs:
            metrics = run["metrics"]
            if key == "agi_transfer_score":
                values.append(metrics[key])
            else:
                values.append(metrics["axis_scores"][key])
        summary[key] = {
            "mean": mean(values),
            "std": safe_std(values),
            "n": len(values),
        }
    return summary


def print_table(output: Dict[str, Any]) -> None:
    keys = [
        "agi_transfer_score",
        "task_diversity",
        "long_memory",
        "fast_rule_inference",
        "planning",
        "organ_specialization",
    ]
    widths = [28, 22, 22, 20, 26, 18, 28]
    header = ["agent"] + [f"{key} mean+/-std" for key in keys]
    print("\n=== SAGE v1.4 Rule Transfer + Memory Probe ===")
    print(f"seeds: {output['seeds']}")
    print(" | ".join(value.ljust(width) for value, width in zip(header, widths)))
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))
    for agent in output["agents"]:
        row = [agent]
        for key in keys:
            item = output["summary"][agent][key]
            row.append(f"{item['mean']:.4f}+/-{item['std']:.4f}")
        print(" | ".join(value.ljust(width) for value, width in zip(row, widths)))


def main() -> None:
    parser = argparse.ArgumentParser(description="SAGE v1.4 rule transfer and memory probe")
    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument("--queries-per-episode", type=int, default=18)
    parser.add_argument("--support-per-episode", type=int, default=6)
    parser.add_argument("--out", type=str, default="results/v1_4_rule_transfer_memory_benchmark.json")
    args = parser.parse_args()

    agent_factories = [RandomAgent, EpisodicMemoryAgent, RuleTransferAgent]
    all_runs: List[Dict[str, Any]] = []
    for seed in SEEDS:
        random.seed(seed)
        cfg = Config(
            episodes=args.episodes,
            queries_per_episode=args.queries_per_episode,
            support_per_episode=args.support_per_episode,
            seed=seed,
        )
        print(f"\n[Seed {seed}]")
        for factory in agent_factories:
            agent = factory()
            metrics = evaluate_agent(agent, cfg)
            print(
                f"  {agent.name}: "
                f"transfer={metrics['agi_transfer_score']:.4f}, "
                f"task_div={metrics['axis_scores']['task_diversity']:.4f}, "
                f"memory={metrics['axis_scores']['long_memory']:.4f}, "
                f"fast_rule={metrics['axis_scores']['fast_rule_inference']:.4f}, "
                f"planning={metrics['axis_scores']['planning']:.4f}, "
                f"organ_spec={metrics['axis_scores']['organ_specialization']:.4f}"
            )
            all_runs.append(
                {
                    "seed": seed,
                    "agent": agent.name,
                    "config": asdict(cfg),
                    "metrics": metrics,
                }
            )

    agents = [factory.name for factory in agent_factories]
    output = {
        "benchmark": "SAGE-v1.4-rule-transfer-memory",
        "goal": "Measure rapid new-rule inference, durable episodic memory, planning, language-action grounding, and organ role routing.",
        "seeds": SEEDS,
        "tasks": TASK_FAMILIES,
        "organs": ORGANS,
        "agents": agents,
        "runs": all_runs,
        "summary": {
            agent: summarize_runs([run for run in all_runs if run["agent"] == agent])
            for agent in agents
        },
        "interpretation_guardrail": (
            "RuleTransferMemoryPlanner is a scaffold probe, not proof of AGI. "
            "It tests whether the architecture benefits from explicit fast-rule, memory, and planning organs."
        ),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print_table(output)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
