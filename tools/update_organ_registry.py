from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sage_core.lifecycle import OrganLifecycleManager


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SAGE organ registry from v1.7.1 lifecycle result.")
    parser.add_argument(
        "--result",
        default="results/v1_7_1_lifecycle_calibration_multiseed.json",
        help="Path to v1.7.1 multiseed result JSON.",
    )
    parser.add_argument(
        "--variant",
        default="calibrated_safe",
        help="Summary variant to use. Examples: calibrated_balanced, calibrated_safe.",
    )
    parser.add_argument(
        "--out",
        default="registry/organ_registry.json",
        help="Output registry JSON path.",
    )
    args = parser.parse_args()

    result_path = Path(args.result)
    if not result_path.exists():
        raise FileNotFoundError(f"Missing result file: {result_path}")

    data = json.loads(result_path.read_text(encoding="utf-8"))

    manager = OrganLifecycleManager(core_organs=["memory_organ"])
    registry = manager.build_registry(
        result=data,
        variant=args.variant,
        source_result=str(result_path),
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(registry.to_jsonable(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=== SAGE v1.8 Organ Lifecycle Registry Update ===")
    print(f"source: {result_path}")
    print(f"variant: {args.variant}")
    print(f"saved: {out_path}")
    print()

    for name, record in sorted(registry.organs.items()):
        print(
            f"{name:18s} | "
            f"status={record.status:15s} | "
            f"recommendation={record.recommendation:32s} | "
            f"health={record.health_score:.3f} | "
            f"success={record.chosen_success_rate}"
        )


if __name__ == "__main__":
    main()
