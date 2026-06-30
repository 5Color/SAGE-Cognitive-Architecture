from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sage_core.autonomy_policy import AutonomyLevelPolicy


def main() -> None:
    policy = AutonomyLevelPolicy(active_level=2)

    checks = {
        "run_reflection_loop": policy.decide("run_reflection_loop").to_jsonable(),
        "run_cleanup_advisor": policy.decide("run_cleanup_advisor").to_jsonable(),
        "write_memory_proposal": policy.decide("write_memory_proposal").to_jsonable(),
        "approve_memory": policy.decide("approve_memory").to_jsonable(),
        "file_delete": policy.decide("file_delete").to_jsonable(),
        "git_push": policy.decide("git_push").to_jsonable(),
        "network_access": policy.decide("network_access").to_jsonable(),
        "unknown_action": policy.decide("unknown_action").to_jsonable(),
    }

    report = policy.report().to_jsonable()

    result = {
        "benchmark": "SAGE-v2.3-autonomy-level-policy-smoke",
        "version": "v2.3",
        "active_level": policy.active_level,
        "checks": checks,
        "report_passed": report.get("passed"),
        "passed": (
            checks["run_reflection_loop"]["allowed"] is True
            and checks["run_cleanup_advisor"]["allowed"] is True
            and checks["write_memory_proposal"]["allowed"] is True
            and checks["approve_memory"]["allowed"] is False
            and checks["approve_memory"]["requires_human_approval"] is True
            and checks["file_delete"]["allowed"] is False
            and checks["file_delete"]["forbidden"] is True
            and checks["git_push"]["allowed"] is False
            and checks["network_access"]["allowed"] is False
            and checks["unknown_action"]["allowed"] is False
            and checks["unknown_action"]["forbidden"] is True
            and report.get("passed") is True
        ),
    }

    out = Path("results/v2_3_autonomy_level_policy_smoke_result.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== SAGE v2.3 Autonomy Level Policy Smoke ===")
    print(f"active_level: {result['active_level']}")
    print(f"report_passed: {result['report_passed']}")
    print(f"passed: {result['passed']}")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
