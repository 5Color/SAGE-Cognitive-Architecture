# benchmark_v1_5_1_neural_imitation_router.py
# SAGE v1.5.1 - Neural Imitation of Evidence Router
#
# Terms:
# - Neural = 신경망 기반
# - Imitation = 모방
# - Evidence = 증거
# - Router = organ 선택기
# - Teacher = 기준 선택을 만드는 모델
# - Student = teacher를 따라 배우는 모델
#
# Goal:
# v1.5 EvidenceRouter가 family label 없이 support evidence만 보고 organ을 골랐다.
# v1.5.1은 EvidenceRouter를 teacher로 삼아 NeuralImitationRouter가 그 organ 선택 정책을 학습할 수 있는지 검증한다.
#
# Required:
#   benchmark_v1_4_rule_transfer_memory.py
#
# Smoke:
#   python benchmark_v1_5_1_neural_imitation_router.py --episodes 20 --queries-per-episode 8 --epochs 6
#
# Full:
#   python benchmark_v1_5_1_neural_imitation_router.py --episodes 120 --queries-per-episode 18 --epochs 10

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
ACTION_DIM = v14.ACTION_DIM

ORGAN_TO_ID = {organ: idx for idx, organ in enumerate(ORGANS)}
ID_TO_ORGAN = {idx: organ for organ, idx in ORGAN_TO_ID.items()}

FAMILY_ORACLE = {
    "episodic_memory": "memory_organ",
    "affine_rule": "algebra_organ",
    "threshold_rule": "concept_organ",
    "language_action": "concept_organ",
    "grid_planning": "planner_organ",
    "world_dynamics": "planner_organ",
}


@dataclass
class Config:
    episodes: int = 120
    queries_per_episode: int = 18
    support_per_episode: int = 6
    train_ratio: float = 0.70
    hidden_dim: int = 96
    epochs: int = 10
    batch_size: int = 256
    lr: float = 2e-3
    seed: int = 0
    device: str = "cpu"


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
    counts = [0 for _ in range(ACTION_DIM)]
    for a in actions:
        counts[int(a) % ACTION_DIM] += 1
    ent = normalized_entropy(counts)
    return 0.0 if ent is None else float(ent)


def make_v14_cfg(cfg: Config) -> v14.Config:
    return v14.Config(
        episodes=cfg.episodes,
        queries_per_episode=cfg.queries_per_episode,
        support_per_episode=cfg.support_per_episode,
        seed=cfg.seed,
    )


def make_episode(cfg: Config, family: str) -> v14.Episode:
    return v14.make_episode(make_v14_cfg(cfg), family)


# ---------------------------------------------------------------------
# Scaffold organs
# ---------------------------------------------------------------------

class MemoryOrgan:
    name = "memory_organ"

    def __init__(self) -> None:
        self.memory: Dict[Any, int] = {}
        self.default_action = 0

    def fit(self, support: Sequence[Tuple[Any, int]]) -> None:
        self.memory = {k: int(a) for k, a in support}
        actions = [int(a) for _, a in support]
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
        actions = [int(a) for _, a in support]
        if actions:
            self.default_action = max(set(actions), key=actions.count)

        pairs = []
        for key, action in support:
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
        self.lexicon: Dict[str, int] = {}
        self.default_action = 0

    def fit(self, support: Sequence[Tuple[Any, int]]) -> None:
        self.threshold = None
        self.lexicon = {}
        actions = [int(a) for _, a in support]
        if actions:
            self.default_action = max(set(actions), key=actions.count)

        for action, words in v14.LANG_ACTIONS.items():
            for word in words:
                self.lexicon[v14.make_key("word", word)] = int(action)

        pairs = []
        for key, action in support:
            if isinstance(key, str) and key.startswith("x:"):
                try:
                    pairs.append((int(key.split(":", 1)[1]), int(action)))
                except Exception:
                    pass

        pairs = sorted(pairs)
        for i in range(1, len(pairs)):
            if pairs[i][1] != pairs[i - 1][1]:
                self.threshold = (pairs[i][0], pairs[i - 1][1], pairs[i][1])
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

    def fit(self, support: Sequence[Tuple[Any, int]]) -> None:
        self.world_deltas = None
        actions = [int(a) for _, a in support]
        if actions:
            self.default_action = max(set(actions), key=actions.count)

        deltas: Dict[int, Tuple[int, int]] = {}
        for key, _ in support:
            if isinstance(key, tuple) and len(key) == 6 and key[0] == "transition":
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


def fit_organs(support: Sequence[Tuple[Any, int]]) -> Dict[str, Any]:
    organs = {
        "memory_organ": MemoryOrgan(),
        "algebra_organ": AlgebraOrgan(),
        "concept_organ": ConceptOrgan(),
        "planner_organ": PlannerOrgan(),
    }
    for organ in organs.values():
        organ.fit(support)
    return organs


def organ_predictions(support: Sequence[Tuple[Any, int]], query: Any) -> Dict[str, int]:
    organs = fit_organs(support)
    return {name: int(organ.predict(query)) for name, organ in organs.items()}


# ---------------------------------------------------------------------
# EvidenceRouter teacher
# ---------------------------------------------------------------------

def split_support(support: Sequence[Tuple[Any, int]]) -> Tuple[List[Tuple[Any, int]], List[Tuple[Any, int]]]:
    support = list(support)
    if len(support) <= 2:
        return support, support
    shuffled = support[:]
    random.shuffle(shuffled)
    cut = max(1, int(len(shuffled) * 0.65))
    return shuffled[:cut], shuffled[cut:] or shuffled


def evidence_score(organ_name: str, fit_part: Sequence[Tuple[Any, int]], val_part: Sequence[Tuple[Any, int]]) -> float:
    organ = fit_organs(fit_part)[organ_name]
    if not val_part:
        return 0.0
    correct = 0
    for key, target in val_part:
        correct += int(int(organ.predict(key)) == int(target))
    return correct / len(val_part)


class RandomOrganRouter:
    name = "RandomOrganRouter"

    def choose(self, episode: v14.Episode, query: Any) -> Tuple[str, float]:
        return random.choice(ORGANS), 1.0 / len(ORGANS)


class FamilyOracleRouter:
    name = "FamilyOracleRouter"

    def choose(self, episode: v14.Episode, query: Any) -> Tuple[str, float]:
        return FAMILY_ORACLE.get(episode.family, "memory_organ"), 1.0


class EvidenceRouter:
    name = "EvidenceRouter"

    def choose(self, episode: v14.Episode, query: Any) -> Tuple[str, float]:
        fit_part, val_part = split_support(episode.support)
        scores = {organ: evidence_score(organ, fit_part, val_part) for organ in ORGANS}

        # Small tie bonus from query/support shape. This still does not use family label.
        bonus = {organ: 0.0 for organ in ORGANS}
        if isinstance(query, str) and query.startswith("symbol:"):
            bonus["memory_organ"] = 0.03
        elif isinstance(query, str) and query.startswith("word:"):
            bonus["concept_organ"] = 0.03
        elif isinstance(query, str) and query.startswith("x:"):
            bonus["algebra_organ"] = 0.015
            bonus["concept_organ"] = 0.015
        elif isinstance(query, tuple):
            bonus["planner_organ"] = 0.03

        chosen = max(ORGANS, key=lambda o: scores[o] + bonus[o])
        sorted_scores = sorted(scores.values(), reverse=True)
        confidence = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) >= 2 else sorted_scores[0]
        confidence = max(float(confidence), 1.0 / len(ORGANS))
        return chosen, confidence


# ---------------------------------------------------------------------
# Neural imitation student
# ---------------------------------------------------------------------

KEY_TYPES = ["symbol", "x", "word", "grid_state", "world_query", "transition", "str_other", "tuple_other", "other"]
KEY_TO_ID = {name: idx for idx, name in enumerate(KEY_TYPES)}


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


def one_hot(index: int, size: int) -> List[float]:
    out = [0.0 for _ in range(size)]
    if 0 <= index < size:
        out[index] = 1.0
    return out


def numeric_hint(value: Any) -> float:
    if isinstance(value, str) and ":" in value:
        raw = value.split(":", 1)[1]
        if raw.isdigit():
            return min(1.0, int(raw) / 999.0)
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


def extract_features(episode: v14.Episode, query: Any) -> List[float]:
    support_keys = [key for key, _ in episode.support]
    support_actions = [int(a) for _, a in episode.support]
    q_type_id = KEY_TO_ID.get(key_type(query), KEY_TO_ID["other"])

    features: List[float] = []
    features.extend(one_hot(q_type_id, len(KEY_TYPES)))
    features.extend(support_type_ratios(episode.support))
    features.extend([
        min(1.0, len(episode.support) / 12.0),
        len(set(support_actions)) / max(1, ACTION_DIM),
        action_entropy(support_actions),
        1.0 if query in support_keys else 0.0,
        numeric_hint(query),
        min(1.0, len(query) / 8.0) if isinstance(query, tuple) else 0.0,
        1.0 if any(key_type(k) == "transition" for k in support_keys) else 0.0,
        1.0 if any(key_type(k) == "word" for k in support_keys) else 0.0,
        1.0 if any(key_type(k) == "x" for k in support_keys) else 0.0,
        1.0 if any(key_type(k) == "symbol" for k in support_keys) else 0.0,
    ])
    return features


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


def build_dataset(cfg: Config, seed: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    set_seed(seed)
    teacher = EvidenceRouter()

    xs: List[List[float]] = []
    ys: List[int] = []
    weights: List[float] = []

    for family in TASK_FAMILIES:
        for _ in range(cfg.episodes):
            episode = make_episode(cfg, family)
            for query, _target in episode.queries:
                chosen, confidence = teacher.choose(episode, query)
                xs.append(extract_features(episode, query))
                ys.append(ORGAN_TO_ID[chosen])
                weights.append(confidence)

    x = torch.tensor(xs, dtype=torch.float32)
    y = torch.tensor(ys, dtype=torch.long)
    w = torch.clamp(torch.tensor(weights, dtype=torch.float32), min=0.20, max=1.00)
    return x, y, w


def train_student(cfg: Config, seed: int) -> Tuple[RouterNet, Dict[str, float]]:
    x, y, w = build_dataset(cfg, seed)
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
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    model.train()
    for _ in range(cfg.epochs):
        perm = train_idx[torch.randperm(train_idx.numel(), device=cfg.device)]
        for start in range(0, perm.numel(), cfg.batch_size):
            batch = perm[start:start + cfg.batch_size]
            logits = model(x[batch])
            loss_vec = F.cross_entropy(logits, y[batch], reduction="none")
            loss = (loss_vec * w[batch]).mean()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    model.eval()
    with torch.no_grad():
        train_acc = (torch.argmax(model(x[train_idx]), dim=-1) == y[train_idx]).float().mean().item()
        if val_idx.numel() > 0:
            val_acc = (torch.argmax(model(x[val_idx]), dim=-1) == y[val_idx]).float().mean().item()
        else:
            val_acc = train_acc

    stats = {
        "teacher_train_imitation_acc": train_acc,
        "teacher_val_imitation_acc": val_acc,
        "dataset_size": float(n),
    }
    return model, stats


class NeuralImitationRouter:
    name = "NeuralImitationRouter"

    def __init__(self, model: RouterNet, cfg: Config):
        self.model = model
        self.cfg = cfg

    @torch.no_grad()
    def choose(self, episode: v14.Episode, query: Any) -> Tuple[str, float]:
        self.model.eval()
        x = torch.tensor([extract_features(episode, query)], dtype=torch.float32, device=self.cfg.device)
        probs = F.softmax(self.model(x), dim=-1).squeeze(0)
        idx = int(torch.argmax(probs).item())
        return ID_TO_ORGAN[idx], float(probs[idx].item())


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def specialization_score(counts_by_family: Dict[str, Dict[str, int]]) -> float:
    entropies = []
    top_organs = []

    for family in TASK_FAMILIES:
        counts = [counts_by_family[family].get(o, 0) for o in ORGANS]
        total = sum(counts)
        if total <= 0:
            continue
        vec = [c / total for c in counts]
        ent = normalized_entropy(vec)
        if ent is not None:
            entropies.append(ent)
        top_organs.append(ORGANS[max(range(len(ORGANS)), key=lambda i: vec[i])])

    if not entropies or not top_organs:
        return 0.0

    entropy_component = 1.0 - mean(entropies)
    diversity_component = len(set(top_organs)) / min(len(ORGANS), len(TASK_FAMILIES))
    return max(0.0, min(1.0, (0.55 * entropy_component + 0.45 * diversity_component) * diversity_component))


def evaluate_router(router: Any, cfg: Config, seed: int) -> Dict[str, Any]:
    set_seed(seed)
    teacher = EvidenceRouter()

    correct_by_family = {f: 0 for f in TASK_FAMILIES}
    total_by_family = {f: 0 for f in TASK_FAMILIES}
    imitation_by_family = {f: 0 for f in TASK_FAMILIES}
    oracle_correct_by_family = {f: 0 for f in TASK_FAMILIES}
    counts_by_family = {f: {o: 0 for o in ORGANS} for f in TASK_FAMILIES}
    confidences: List[float] = []

    for family in TASK_FAMILIES:
        for _ in range(cfg.episodes):
            episode = make_episode(cfg, family)
            for query, target in episode.queries:
                chosen, conf = router.choose(episode, query)
                if chosen not in ORGAN_TO_ID:
                    chosen = "memory_organ"

                teacher_chosen, _ = teacher.choose(episode, query)
                oracle_chosen = FAMILY_ORACLE.get(episode.family, "memory_organ")
                preds = organ_predictions(episode.support, query)

                correct_by_family[family] += int(preds[chosen] == int(target))
                total_by_family[family] += 1
                imitation_by_family[family] += int(chosen == teacher_chosen)
                oracle_correct_by_family[family] += int(preds[oracle_chosen] == int(target))

                counts_by_family[family][chosen] += 1
                confidences.append(float(conf))

    family_acc = {f: correct_by_family[f] / max(1, total_by_family[f]) for f in TASK_FAMILIES}
    family_imit = {f: imitation_by_family[f] / max(1, total_by_family[f]) for f in TASK_FAMILIES}
    family_oracle = {f: oracle_correct_by_family[f] / max(1, total_by_family[f]) for f in TASK_FAMILIES}

    usage_by_family = {}
    entropy_by_family = {}
    top_by_family = {}
    for family in TASK_FAMILIES:
        total = sum(counts_by_family[family].values())
        if total <= 0:
            vec_dict = {o: 0.0 for o in ORGANS}
        else:
            vec_dict = {o: counts_by_family[family][o] / total for o in ORGANS}
        usage_by_family[family] = {o: round_float(v) for o, v in vec_dict.items()}
        entropy_by_family[family] = round_float(normalized_entropy(list(vec_dict.values())))
        top_by_family[family] = max(ORGANS, key=lambda o: vec_dict[o])

    task_diversity = mean(family_acc.values())
    fast_rule_inference = mean([family_acc["affine_rule"], family_acc["threshold_rule"]])
    planning = mean([family_acc["grid_planning"], family_acc["world_dynamics"]])
    imitation_acc = mean(family_imit.values())
    oracle_acc = mean(family_oracle.values())
    spec = specialization_score(counts_by_family)
    route_confidence = mean(confidences) if confidences else 0.0

    evidence_router_score = mean([
        task_diversity,
        fast_rule_inference,
        planning,
        spec,
        route_confidence,
        imitation_acc,
    ])

    return {
        "family_accuracy": {k: round_float(v) for k, v in family_acc.items()},
        "family_imitation_acc": {k: round_float(v) for k, v in family_imit.items()},
        "task_diversity": round_float(task_diversity),
        "fast_rule_inference": round_float(fast_rule_inference),
        "planning": round_float(planning),
        "organ_specialization": round_float(spec),
        "route_confidence": round_float(route_confidence),
        "imitation_acc": round_float(imitation_acc),
        "oracle_task_acc": round_float(oracle_acc),
        "oracle_gap": round_float(oracle_acc - task_diversity),
        "evidence_gap": None,
        "evidence_router_score": round_float(evidence_router_score),
        "organ_usage_by_family": usage_by_family,
        "organ_usage_entropy_by_family": entropy_by_family,
        "organ_top_by_family": top_by_family,
    }


def fill_evidence_gap(runs: List[Dict[str, Any]]) -> None:
    teacher_score_by_seed = {}
    for run in runs:
        if run["router"] == "EvidenceRouter":
            teacher_score_by_seed[run["seed"]] = run["metrics"]["evidence_router_score"]

    for run in runs:
        teacher_score = teacher_score_by_seed.get(run["seed"])
        run["metrics"]["evidence_gap"] = (
            None if teacher_score is None else round_float(teacher_score - run["metrics"]["evidence_router_score"])
        )


def summarize_runs(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = [
        "evidence_router_score",
        "task_diversity",
        "fast_rule_inference",
        "planning",
        "organ_specialization",
        "route_confidence",
        "imitation_acc",
        "oracle_gap",
        "evidence_gap",
    ]

    out: Dict[str, Any] = {}
    for key in keys:
        values = [float(r["metrics"][key]) for r in runs if r["metrics"].get(key) is not None]
        out[key] = {
            "mean": mean(values) if values else None,
            "std": safe_std(values) if values else None,
            "n": len(values),
        }
    return out


def print_table(output: Dict[str, Any]) -> None:
    keys = [
        "evidence_router_score",
        "task_diversity",
        "fast_rule_inference",
        "planning",
        "organ_specialization",
        "route_confidence",
        "imitation_acc",
        "evidence_gap",
    ]

    widths = [24, 28, 22, 24, 18, 24, 22, 18, 18]
    header = ["router"] + [f"{k} mean+/-std" for k in keys]

    print("\n=== SAGE v1.5.1 Neural Imitation Router ===")
    print(f"seeds: {output['seeds']}")
    print(" | ".join(text.ljust(w) for text, w in zip(header, widths)))
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))

    for router in output["routers"]:
        row = [router]
        for key in keys:
            item = output["summary"][router][key]
            if item["mean"] is None:
                row.append("N/A")
            else:
                row.append(f"{item['mean']:.4f}+/-{item['std']:.4f}")
        print(" | ".join(text.ljust(w) for text, w in zip(row, widths)))


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

    print(f"\n[Seed {seed}] training NeuralImitationRouter ...")
    model, train_stats = train_student(seed_cfg, seed)

    routers = [
        RandomOrganRouter(),
        EvidenceRouter(),
        NeuralImitationRouter(model, seed_cfg),
        FamilyOracleRouter(),
    ]

    seed_runs = []
    for router in routers:
        print(f"  evaluating {router.name} ...")
        metrics = evaluate_router(router, seed_cfg, seed)
        seed_runs.append({
            "seed": seed,
            "router": router.name,
            "config": asdict(seed_cfg),
            "train_stats": train_stats if router.name == "NeuralImitationRouter" else None,
            "metrics": metrics,
        })

        print(
            "    "
            f"score={metrics['evidence_router_score']:.4f}, "
            f"task={metrics['task_diversity']:.4f}, "
            f"planning={metrics['planning']:.4f}, "
            f"spec={metrics['organ_specialization']:.4f}, "
            f"imit={metrics['imitation_acc']:.4f}"
        )

    return seed_runs


def main() -> None:
    parser = argparse.ArgumentParser(description="SAGE v1.5.1 Neural Imitation Router")
    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument("--queries-per-episode", type=int, default=18)
    parser.add_argument("--support-per-episode", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--out", type=str, default="results/v1_5_1_neural_imitation_router_benchmark.json")
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

    fill_evidence_gap(all_runs)

    router_names = ["RandomOrganRouter", "EvidenceRouter", "NeuralImitationRouter", "FamilyOracleRouter"]
    summary = {
        name: summarize_runs([run for run in all_runs if run["router"] == name])
        for name in router_names
    }

    output = {
        "benchmark": "SAGE-v1.5.1-neural-imitation-router",
        "goal": "Train a neural router to imitate EvidenceRouter organ selection without task family labels.",
        "terminology": {
            "neural": "신경망 기반",
            "imitation": "모방",
            "evidence": "support/query에서 얻은 증거",
            "router": "어떤 organ을 사용할지 고르는 선택기",
            "teacher": "학습 target을 제공하는 기준 모델",
            "student": "teacher의 선택을 학습하는 neural model",
        },
        "interpretation_guardrail": (
            "This is not an AGI claim. It tests whether symbolic evidence-based organ routing "
            "can be compressed into a neural router. High scores may still reflect easy feature fingerprints."
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
