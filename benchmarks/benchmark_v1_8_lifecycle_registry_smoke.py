from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def main() -> None:
    registry_path = Path("registry/organ_registry.json")
    if not registry_path.exists():
        registry_path = Path("registry/organ_registry.example.json")

    data = json.loads(registry_path.read_text(encoding="utf-8"))
    organs = data.get("organs", {})

    status_counts = Counter(item.get("status", "unknown") for item in organs.values())
    recommendation_counts = Counter(item.get("recommendation", "unknown") for item in organs.values())

    unsafe_auto_delete = [
        name for name, item in organs.items()
        if bool(item.get("can_auto_delete", False))
    ]
    unsafe_auto_disable = [
        name for name, item in organs.items()
        if bool(item.get("can_auto_disable", False))
    ]

    result = {
        "benchmark": "SAGE-v1.8-lifecycle-registry-smoke",
        "version": "v1.8",
        "registry_path": str(registry_path),
        "num_organs": len(organs),
        "status_counts": dict(status_counts),
        "recommendation_counts": dict(recommendation_counts),
        "unsafe_auto_delete_organs": unsafe_auto_delete,
        "unsafe_auto_disable_organs": unsafe_auto_disable,
        "passed": len(organs) > 0 and not unsafe_auto_delete and not unsafe_auto_disable,
    }

    out = Path("results/v1_8_lifecycle_registry_smoke.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== SAGE v1.8 Lifecycle Registry Smoke ===")
    print(f"registry: {registry_path}")
    print(f"num_organs: {result['num_organs']}")
    print(f"status_counts: {result['status_counts']}")
    print(f"recommendation_counts: {result['recommendation_counts']}")
    print(f"passed: {result['passed']}")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
