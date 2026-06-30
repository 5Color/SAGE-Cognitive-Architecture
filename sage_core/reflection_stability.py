from __future__ import annotations
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List
import copy, json
from sage_runtime.emergent_reflection_loop import EmergentReflectionConfig, EmergentReflectionLoop

@dataclass
class PolicyVariant:
    name: str
    policy_path: str
    result_path: str
    selected_organ: str | None = None
    top_score: float | None = None
    top_second_gap: float | None = None
    mean_candidate_score: float | None = None
    passed_safety: bool = False
    def to_jsonable(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class StabilityProbeResult:
    version: str = "v2.0.3"
    benchmark: str = "SAGE-v2.0.3-reflection-stability-probe"
    base_policy_path: str = ""
    target_organ: str = "curiosity_organ"
    variant_count: int = 0
    selected_counts: Dict[str, int] = field(default_factory=dict)
    target_selected_count: int = 0
    target_selected_rate: float = 0.0
    min_top_second_gap: float | None = None
    max_top_second_gap: float | None = None
    mean_top_second_gap: float | None = None
    variants: List[Dict[str, Any]] = field(default_factory=list)
    interpretation: str = ""
    safety_policy: Dict[str, bool] = field(default_factory=lambda: {
        "network_actions": False,
        "shell_actions": False,
        "auto_delete_files": False,
        "auto_disable_organs": False,
        "auto_approve_memory": False,
        "create_memory_proposal": False,
        "git_actions": False,
        "human_approval_required": True,
    })
    passed: bool = False
    def to_jsonable(self) -> Dict[str, Any]:
        return asdict(self)

class ReflectionStabilityProbe:
    def __init__(
        self,
        base_policy_path: str = "configs/reflection_policy_exploratory.json",
        output_dir: str = "results/v2_0_3_stability_probe",
        variant_policy_dir: str = "configs/generated/stability_probe",
        target_organ: str = "curiosity_organ",
        novelty_deltas: List[float] | None = None,
        risk_deltas: List[float] | None = None,
        min_target_rate: float = 0.80,
    ) -> None:
        self.base_policy_path = base_policy_path
        self.output_dir = Path(output_dir)
        self.variant_policy_dir = Path(variant_policy_dir)
        self.target_organ = target_organ
        self.novelty_deltas = novelty_deltas or [-0.02, 0.0, 0.02]
        self.risk_deltas = risk_deltas or [0.0, 0.02]
        self.min_target_rate = min_target_rate

    def load_base_policy(self) -> Dict[str, Any]:
        p = Path(self.base_policy_path)
        if not p.exists():
            raise FileNotFoundError(f"Base policy not found: {self.base_policy_path}")
        return json.loads(p.read_text(encoding="utf-8"))

    def make_variants(self) -> List[PolicyVariant]:
        base = self.load_base_policy()
        base_weights = base.get("weights", {})
        self.variant_policy_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        variants: List[PolicyVariant] = []
        for n_delta in self.novelty_deltas:
            for r_delta in self.risk_deltas:
                variant = copy.deepcopy(base)
                weights = variant.setdefault("weights", {})
                weights["novelty"] = max(0.0, round(float(base_weights.get("novelty", 0.29)) + n_delta, 6))
                weights["risk"] = max(0.0, round(float(base_weights.get("risk", 0.08)) + r_delta, 6))
                name = f"novelty_{n_delta:+.2f}_risk_{r_delta:+.2f}".replace("+", "p").replace("-", "m").replace(".", "_")
                variant["name"] = f"stability_{name}"
                policy_path = self.variant_policy_dir / f"{variant['name']}.json"
                policy_path.write_text(json.dumps(variant, indent=2, ensure_ascii=False), encoding="utf-8")
                result_path = self.output_dir / f"{variant['name']}.json"
                variants.append(PolicyVariant(variant["name"], str(policy_path), str(result_path)))
        return variants

    def run_variant(self, variant: PolicyVariant) -> PolicyVariant:
        config = EmergentReflectionConfig(
            state_path="runtime_state/smoke_state.json",
            registry_path="registry/organ_registry.json",
            memory_root="memory",
            reflection_log_path=f"logs/v2_0_3_stability_{variant.name}.md",
            result_path=variant.result_path,
            create_memory_proposal=False,
            reflection_policy_path=variant.policy_path,
        )
        result = EmergentReflectionLoop(config).run_once()
        selected = result.get("selected") or {}
        metrics = result.get("emergence_metrics", {})
        safety = result.get("safety_policy", {})
        variant.selected_organ = selected.get("organ")
        variant.top_score = metrics.get("top_score")
        variant.top_second_gap = metrics.get("top_second_gap")
        variant.mean_candidate_score = metrics.get("mean_candidate_score")
        variant.passed_safety = (
            safety.get("network_actions") is False
            and safety.get("shell_actions") is False
            and safety.get("auto_delete_organs") is False
            and safety.get("auto_disable_organs") is False
            and safety.get("memory_approval_required") is True
        )
        return variant

    def run(self) -> StabilityProbeResult:
        variants = [self.run_variant(v) for v in self.make_variants()]
        counts: Dict[str, int] = {}
        gaps: List[float] = []
        for v in variants:
            key = v.selected_organ or "none"
            counts[key] = counts.get(key, 0) + 1
            if v.top_second_gap is not None:
                gaps.append(float(v.top_second_gap))
        target_count = counts.get(self.target_organ, 0)
        target_rate = target_count / max(1, len(variants))
        interpretation = (
            f"Reflection selection is stable enough for this stage: {self.target_organ} was selected in {target_count}/{len(variants)} variants."
            if target_rate >= self.min_target_rate
            else f"Reflection selection is unstable: {self.target_organ} was selected in {target_count}/{len(variants)} variants."
        )
        return StabilityProbeResult(
            base_policy_path=self.base_policy_path,
            target_organ=self.target_organ,
            variant_count=len(variants),
            selected_counts=counts,
            target_selected_count=target_count,
            target_selected_rate=round(target_rate, 6),
            min_top_second_gap=None if not gaps else round(min(gaps), 6),
            max_top_second_gap=None if not gaps else round(max(gaps), 6),
            mean_top_second_gap=None if not gaps else round(sum(gaps) / len(gaps), 6),
            variants=[v.to_jsonable() for v in variants],
            interpretation=interpretation,
            passed=(len(variants) >= 6 and target_rate >= self.min_target_rate and all(v.passed_safety for v in variants)),
        )
