from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sage_runtime.emergent_reflection_loop import EmergentReflectionConfig, EmergentReflectionLoop


def main() -> None:
    config = EmergentReflectionConfig(
        state_path="runtime_state/smoke_state.json",
        registry_path="registry/organ_registry.json",
        memory_root="memory",
        reflection_log_path="logs/v2_0_emergent_reflection_smoke.md",
        result_path="results/v2_0_emergent_reflection_smoke_detail.json",
        create_memory_proposal=True,
    )

    before = len(list(Path("memory/inbox").glob("*.json"))) if Path("memory/inbox").exists() else 0
    loop = EmergentReflectionLoop(config)
    detail = loop.run_once()
    after = len(list(Path("memory/inbox").glob("*.json"))) if Path("memory/inbox").exists() else 0

    metrics = detail.get("emergence_metrics", {})
    selected = detail.get("selected") or {}

    result = {
        "benchmark": "SAGE-v2.0-emergent-reflection-smoke",
        "version": "v2.0",
        "selected_organ": selected.get("organ"),
        "candidate_count": metrics.get("candidate_count"),
        "organ_diversity": metrics.get("organ_diversity"),
        "top_score": metrics.get("top_score"),
        "top_second_gap": metrics.get("top_second_gap"),
        "mean_novelty": metrics.get("mean_novelty"),
        "mean_reuse_value": metrics.get("mean_reuse_value"),
        "memory_inbox_count_before": before,
        "memory_inbox_count_after": after,
        "result_detail_path": config.result_path,
        "reflection_log_path": config.reflection_log_path,
        "safety_policy": detail.get("safety_policy", {}),
        "passed": (
            selected.get("organ") is not None
            and float(metrics.get("candidate_count", 0.0)) >= 5.0
            and float(metrics.get("organ_diversity", 0.0)) > 0.5
            and after >= before + 1
            and detail.get("safety_policy", {}).get("network_actions") is False
            and detail.get("safety_policy", {}).get("shell_actions") is False
            and detail.get("safety_policy", {}).get("auto_delete_organs") is False
        ),
    }

    out = Path("results/v2_0_emergent_reflection_smoke.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== SAGE v2.0 Emergent Reflection Smoke ===")
    print(f"selected_organ: {result['selected_organ']}")
    print(f"candidate_count: {result['candidate_count']}")
    print(f"organ_diversity: {result['organ_diversity']}")
    print(f"top_score: {result['top_score']}")
    print(f"memory_inbox: {before} -> {after}")
    print(f"passed: {result['passed']}")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
