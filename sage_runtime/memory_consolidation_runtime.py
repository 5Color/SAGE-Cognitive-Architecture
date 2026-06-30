from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict
import json

from sage_core.memory_consolidation import (
    ConsolidationConfig,
    MemoryConsolidationOrgan,
)


@dataclass
class MemoryConsolidationRuntimeConfig:
    config_path: str = "configs/memory_consolidation.json"

    @classmethod
    def load(cls, path: str | Path) -> "MemoryConsolidationRuntimeConfig":
        p = Path(path)
        if not p.exists():
            return cls()
        data = json.loads(p.read_text(encoding="utf-8"))
        allowed = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in allowed}
        base = {f.name: getattr(cls(), f.name) for f in fields(cls)}
        return cls(**{**base, **filtered})


class MemoryConsolidationRuntime:
    def __init__(self, runtime_config: MemoryConsolidationRuntimeConfig) -> None:
        self.runtime_config = runtime_config

    def _load_consolidation_config(self) -> ConsolidationConfig:
        path = Path(self.runtime_config.config_path)
        if not path.exists():
            return ConsolidationConfig()
        data = json.loads(path.read_text(encoding="utf-8"))
        allowed = {f.name for f in fields(ConsolidationConfig)}
        filtered = {k: v for k, v in data.items() if k in allowed}
        base = {f.name: getattr(ConsolidationConfig(), f.name) for f in fields(ConsolidationConfig)}
        return ConsolidationConfig(**{**base, **filtered})

    def run_once(self) -> Dict[str, Any]:
        config = self._load_consolidation_config()
        organ = MemoryConsolidationOrgan(config)
        report = organ.run().to_jsonable()

        out = Path(config.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        report["output_path"] = str(out)
        return report
