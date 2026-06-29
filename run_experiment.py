# SAGE v1.6.1 - Config Runner
#
# Goal:
# - Run SAGE experiments from JSON config.
# - Avoid copy-pasting the same benchmark loop.
# - Keep sage_core as the experiment subject and benchmarks as the experiment lab.
#
# 실행 예시:
# python run_experiment.py --config benchmarks/configs/parity_smoke.json

from __future__ import annotations

import argparse
import importlib
import json
import random
from pathlib import Path
from typing import Any, Dict, Mapping

from sage_core import SAGEEngine, SAGEState


def load_class(module_name: str, class_name: str) -> type:
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def build_component(spec: Mapping[str, Any]) -> Any:
    """Build one component from config.

    spec example:
    {
      "module": "benchmarks.tasks.parity_task",
      "class": "ParityEnvironment",
      "params": {"length": 32}
    }
    """
    cls = load_class(str(spec["module"]), str(spec["class"]))
    params = dict(spec.get("params", {}))
    return cls(**params)


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="SAGE config-based experiment runner")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON experiment config")
    parser.add_argument("--out", type=str, default=None, help="Optional output JSON path override")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))

    seed = int(config.get("seed", 0))
    set_seed(seed)

    organs = {}
    for organ_spec in config["organs"]:
        organ = build_component(organ_spec)
        organs[organ.name] = organ

    router = build_component(config["router"])
    environment = build_component(config["environment"])
    metric = build_component(config["metric"])

    state_params = dict(config.get("state", {}))
    state = SAGEState(**state_params)

    engine_params = dict(config.get("engine", {}))
    max_steps = int(config.get("max_steps", 64))

    engine = SAGEEngine(
        organs=organs,
        router=router,
        env=environment,
        metric=metric,
        state=state,
        **engine_params,
    )

    result = engine.run(max_steps=max_steps)

    output = {
        "benchmark": config.get("benchmark", config_path.stem),
        "goal": config.get("goal", ""),
        "version": "v1.6.1",
        "config_path": str(config_path),
        "seed": seed,
        "components": {
            "organs": [organ.name for organ in organs.values()],
            "router": router.name,
            "environment": environment.name,
            "metric": metric.name,
        },
        "interpretation_guardrail": (
            "This is a config-runner smoke/refactor test, not an intelligence or AGI benchmark."
        ),
        "result": result,
    }

    out_path = Path(args.out or config.get("output_path", "results/v1_6_1_config_runner_smoke.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== SAGE v1.6.1 Config Runner ===")
    print(f"benchmark={output['benchmark']}")
    print(f"seed={seed}")
    print(f"router={router.name}")
    print(f"environment={environment.name}")
    print(f"accuracy={result.get('accuracy', 0.0):.4f}")
    print(f"avg_reward={result.get('avg_reward', 0.0):.4f}")
    print(f"steps={result.get('steps', 0)}")
    print(f"final_energy={result.get('final_state', {}).get('energy', 0.0):.4f}")
    print(f"organ_usage={result.get('organ_usage', {})}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
