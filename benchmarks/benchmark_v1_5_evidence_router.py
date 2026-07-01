from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Optional, Sequence, Tuple

import benchmark_v1_4_rule_transfer_memory as v14


SEEDS = [0, 1, 2, 3, 4]
ORGANS = v14.ORGANS
TASK_FAMILIES = v14.TASK_FAMILIES


@dataclass
class Config:
    episodes: int = 120
    queries_per_episode: int = 18
    support_per_episode: int = 6
    seed: int = 0


class CandidateOrgan:
    name = "candidate"

    def fit(self, episode: v14.Episode) -> None:
        raise NotImplementedError

    def predict(self, query: Any) -> int:
        raise NotImplementedError


class MemoryOrgan(CandidateOrgan):
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
        return self.memory.get(query, self.default_action)


class AlgebraOrgan(CandidateOrgan):
    name = "algebra_organ"

    def __init__(self) -> None:
        self.memory = MemoryOrgan()
        self.affine: Optional[Tuple[int, int]] = None

    def fit(self, episode: v14.Episode) -> None:
        self.memory.fit(episode)
        self.affine = None
        pairs = []
        for key, action in episode.support:
            if isinstance(key, str) and key.startswith("x:"):
                try:
                    pairs.append((int(key.split(":", 1)[1]), int(action)))
                except ValueError:
                    pass
        pairs = sorted(set(pairs))
        for i in range(len(pairs)):
            for j in range(i + 1, len(pairs)):
                x0, y0 = pairs[i]
                x1, y1 = pairs[j]
                dx = (x1 - x0) % v14.ACTION_DIM
                dy = (y1 - y0) % v14.ACTION_DIM
                for a in [1, 5]:
                    if (a * dx) % v14.ACTION_DIM == dy:
                        b = (y0 - a * x0) % v14.ACTION_DIM
                        if all((a * x + b) % v14.ACTION_DIM == y for x, y in pairs):
                            self.affine = (a, b)
                            return

    def predict(self, query: Any) -> int:
        if self.affine is None:
            return self.memory.predict(query)
        if isinstance(query, str) and query.startswith("x:"):
            try:
                x = int(query.split(":", 1)[1])
                a, b = self.affine
                return (a * x + b) % v14.ACTION_DIM
            except ValueError:
                pass
        return self.memory.predict(query)


class ConceptOrgan(CandidateOrgan):
    name = "concept_organ"

    def __init__(self) -> None:
        self.memory = MemoryOrgan()
        self.threshold: Optional[Tuple[int, int, int]] = None
        self.lexicon: Dict[str, int] = {}

    def fit(self, episode: v14.Episode) -> None:
        self.memory.fit(episode)
        self.threshold = None
        self.lexicon = {}

        x_pairs = []
        word_support = False
        for key, action in episode.support:
            if isinstance(key, str) and key.startswith("x:"):
                try:
                    x_pairs.append((int(key.split(":", 1)[1]), int(action)))
                except ValueError:
                    pass
            elif isinstance(key, str) and key.startswith("word:"):
                word_support = True

        if x_pairs:
            pairs = sorted(set(x_pairs))
            for threshold in range(1, v14.ACTION_DIM):
                lows = [action for x, action in pairs if x < threshold]
                highs = [action for x, action in pairs if x >= threshold]
                if not lows or not highs:
                    continue
                low_action = max(set(lows), key=lows.count)
                high_action = max(set(highs), key=highs.count)
                if all((low_action if x < threshold else high_action) == action for x, action in pairs):
                    self.threshold = (threshold, low_action, high_action)
                    break

        if word_support:
            for action, words in v14.LANG_ACTIONS.items():
                for word in words:
                    self.lexicon[v14.make_key("word", word)] = action

    def predict(self, query: Any) -> int:
        if isinstance(query, str) and query.startswith("word:"):
            return self.lexicon.get(query, self.memory.predict(query))
        if self.threshold is not None and isinstance(query, str) and query.startswith("x:"):
            try:
                x = int(query.split(":", 1)[1])
                threshold, low_action, high_action = self.threshold
                return low_action if x < threshold else high_action
            except ValueError:
                pass
        return self.memory.predict(query)


class PlannerOrgan(CandidateOrgan):
    name = "planner_organ"

    def __init__(self) -> None:
        self.memory = MemoryOrgan()
        self.world_deltas: Optional[Dict[int, Tuple[int, int]]] = None

    def fit(self, episode: v14.Episode) -> None:
        self.memory.fit(episode)
        deltas: Dict[int, Tuple[int, int]] = {}
        for key, _ in episode.support:
            if isinstance(key, tuple) and len(key) == 6 and key[0] == "transition":
                _, x, y, action, nx, ny = key
                deltas[int(action)] = (int(nx) - int(x), int(ny) - int(y))
        self.world_deltas = deltas if len(deltas) == 4 else None

    def predict(self, query: Any) -> int:
        if (
            isinstance(query, tuple)
            and len(query) == 5
            and isinstance(query[4], tuple)
        ):
            return v14.plan_step(query)
        if self.world_deltas is not None and isinstance(query, tuple) and len(query) == 4:
            return v14.world_best_action(query, self.world_deltas)
        return self.memory.predict(query)


def make_config(cfg: Config) -> v14.Config:
    return v14.Config(
        episodes=cfg.episodes,
        queries_per_episode=cfg.queries_per_episode,
        support_per_episode=cfg.support_per_episode,
        seed=cfg.seed,
    )


def make_episode(cfg: Config, family: str) -> v14.Episode:
    return v14.make_episode(make_config(cfg), family)


def make_organs() -> List[CandidateOrgan]:
    return [MemoryOrgan(), AlgebraOrgan(), ConceptOrgan(), PlannerOrgan()]


def split_support(episode: v14.Episode) -> Tuple[List[Tuple[Any, int]], List[Tuple[Any, int]]]:
    support = list(episode.support)
    if len(support) <= 2:
        return support, support
    shuffled = support[:]
    random.shuffle(shuffled)
    split = max(1, len(shuffled) // 2)
    train = shuffled[:split]
    validate = shuffled[split:]
    if not validate:
        validate = train
    return train, validate


def support_score(organ: CandidateOrgan, episode: v14.Episode) -> float:
    train, validate = split_support(episode)
    probe = copy.deepcopy(organ)
    probe.fit(v14.Episode(episode.family, train, [], episode.meta))
    correct = 0
    for query, target in validate:
        correct += int(probe.predict(query) == target)
    if not validate:
        return 0.0
    return correct / len(validate)


def family_label_allowed_score(organ_name: str, family: str) -> float:
    oracle_map = {
        "episodic_memory": "memory_organ",
        "affine_rule": "algebra_organ",
        "threshold_rule": "concept_organ",
        "language_action": "concept_organ",
        "grid_planning": "planner_organ",
        "world_dynamics": "planner_organ",
    }
    return 1.0 if oracle_map[family] == organ_name else 0.0


class EvidenceRouterAgent:
    name = "EvidenceRouter"

    def __init__(self, *, allow_family_label: bool = False, random_router: bool = False) -> None:
        self.allow_family_label = allow_family_label
        self.random_router = random_router
        self.organs = make_organs()
        self.selected: CandidateOrgan = self.organs[0]
        self.selected_name = self.selected.name
        self.route_confidence = 0.0
        if random_router:
            self.name = "RandomOrganRouter"
        elif allow_family_label:
            self.name = "FamilyOracleRouter"

    def fit(self, episode: v14.Episode) -> None:
        self.organs = make_organs()
        for organ in self.organs:
            organ.fit(episode)

        if self.random_router:
            self.selected = random.choice(self.organs)
            self.selected_name = self.selected.name
            self.route_confidence = 0.25
            return

        scores = []
        for organ in self.organs:
            if self.allow_family_label:
                score = family_label_allowed_score(organ.name, episode.family)
            else:
                score = support_score(organ, episode)
            scores.append((score, organ.name, organ))

        scores.sort(key=lambda item: (item[0], item[1]), reverse=True)
        self.route_confidence = scores[0][0]
        self.selected = scores[0][2]
        self.selected_name = self.selected.name

    def predict(self, query: Any) -> Tuple[int, str]:
        return self.selected.predict(query), self.selected_name


def evaluate_agent(agent_factory: Any, cfg: Config) -> Dict[str, Any]:
    correct_by_family = {family: 0 for family in TASK_FAMILIES}
    total_by_family = {family: 0 for family in TASK_FAMILIES}
    route_counts = {
        family: {organ: 0 for organ in ORGANS}
        for family in TASK_FAMILIES
    }
    confidence_by_family = {family: [] for family in TASK_FAMILIES}

    for family in TASK_FAMILIES:
        for _ in range(cfg.episodes):
            episode = make_episode(cfg, family)
            agent = agent_factory()
            agent.fit(episode)
            confidence_by_family[family].append(float(getattr(agent, "route_confidence", 0.0)))
            for query, target in episode.queries:
                pred, organ = agent.predict(query)
                correct_by_family[family] += int(pred == target)
                total_by_family[family] += 1
                route_counts[family][organ] += 1

    family_accuracy = {
        family: v14.round_float(correct_by_family[family] / total_by_family[family])
        for family in TASK_FAMILIES
    }
    route_usage = {}
    route_entropy = {}
    route_top = {}
    for family, counts_by_organ in route_counts.items():
        counts = [counts_by_organ[organ] for organ in ORGANS]
        total = sum(counts)
        vector = [count / total if total else 0.0 for count in counts]
        route_usage[family] = {
            organ: v14.round_float(value)
            for organ, value in zip(ORGANS, vector)
        }
        route_entropy[family] = v14.round_float(v14.normalized_entropy(vector))
        route_top[family] = ORGANS[max(range(len(ORGANS)), key=lambda idx: vector[idx])]

    confidence = {
        family: v14.round_float(mean(values)) if values else None
        for family, values in confidence_by_family.items()
    }
    valid_entropies = [value for value in route_entropy.values() if value is not None]
    entropy_component = 1.0 - mean(valid_entropies) if valid_entropies else 0.0
    diversity_component = len(set(route_top.values())) / min(len(ORGANS), len(TASK_FAMILIES))
    specialization = (0.55 * entropy_component + 0.45 * diversity_component) * diversity_component

    axis_scores = {
        "task_diversity": v14.round_float(mean(family_accuracy.values())),
        "long_memory": family_accuracy["episodic_memory"],
        "fast_rule_inference": v14.round_float(mean([
            family_accuracy["affine_rule"],
            family_accuracy["threshold_rule"],
        ])),
        "language_action": family_accuracy["language_action"],
        "planning": family_accuracy["grid_planning"],
        "world_model_planning": family_accuracy["world_dynamics"],
        "organ_specialization": v14.round_float(specialization),
        "route_confidence": v14.round_float(mean(v for values in confidence_by_family.values() for v in values)),
    }

    return {
        "family_accuracy": family_accuracy,
        "axis_scores": axis_scores,
        "evidence_router_score": v14.round_float(mean(axis_scores.values())),
        "route_usage_by_family": route_usage,
        "route_entropy_by_family": route_entropy,
        "route_top_by_family": route_top,
        "route_confidence_by_family": confidence,
    }


def summarize_runs(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = [
        "evidence_router_score",
        "task_diversity",
        "long_memory",
        "fast_rule_inference",
        "language_action",
        "planning",
        "world_model_planning",
        "organ_specialization",
        "route_confidence",
    ]
    summary: Dict[str, Any] = {}
    for key in keys:
        values = []
        for run in runs:
            metrics = run["metrics"]
            if key == "evidence_router_score":
                values.append(metrics[key])
            else:
                values.append(metrics["axis_scores"][key])
        summary[key] = {
            "mean": mean(values),
            "std": stdev(values) if len(values) >= 2 else 0.0,
            "n": len(values),
        }
    return summary


def print_table(output: Dict[str, Any]) -> None:
    keys = [
        "evidence_router_score",
        "task_diversity",
        "fast_rule_inference",
        "planning",
        "organ_specialization",
        "route_confidence",
    ]
    widths = [24, 26, 22, 28, 18, 28, 22]
    header = ["agent"] + [f"{key} mean+/-std" for key in keys]
    print("\n=== SAGE v1.5 Evidence-Based Organ Router Probe ===")
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
    parser = argparse.ArgumentParser(description="SAGE v1.5 evidence-based organ router probe")
    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument("--queries-per-episode", type=int, default=18)
    parser.add_argument("--support-per-episode", type=int, default=6)
    parser.add_argument("--out", type=str, default="results/v1_5_evidence_router_benchmark.json")
    args = parser.parse_args()

    agent_factories = [
        ("RandomOrganRouter", lambda: EvidenceRouterAgent(random_router=True)),
        ("EvidenceRouter", lambda: EvidenceRouterAgent()),
        ("FamilyOracleRouter", lambda: EvidenceRouterAgent(allow_family_label=True)),
    ]
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
        for agent_name, factory in agent_factories:
            metrics = evaluate_agent(factory, cfg)
            print(
                f"  {agent_name}: "
                f"router={metrics['evidence_router_score']:.4f}, "
                f"task_div={metrics['axis_scores']['task_diversity']:.4f}, "
                f"fast_rule={metrics['axis_scores']['fast_rule_inference']:.4f}, "
                f"planning={metrics['axis_scores']['planning']:.4f}, "
                f"organ_spec={metrics['axis_scores']['organ_specialization']:.4f}, "
                f"confidence={metrics['axis_scores']['route_confidence']:.4f}"
            )
            all_runs.append(
                {
                    "seed": seed,
                    "agent": agent_name,
                    "config": asdict(cfg),
                    "metrics": metrics,
                }
            )

    agents = [name for name, _ in agent_factories]
    output = {
        "benchmark": "SAGE-v1.5-evidence-router",
        "goal": "Select cognitive organs from support evidence instead of hand-coded task-family labels.",
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
            "EvidenceRouter is still a symbolic router probe. It is progress toward neural organ selection, "
            "not proof that the neural SAGE core has learned this routing internally."
        ),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print_table(output)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
