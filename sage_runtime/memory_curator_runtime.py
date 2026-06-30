from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict
import json

from sage_core.memory_curator import MemoryCuratorOrgan


@dataclass
class MemoryCuratorRuntimeConfig:
    memory_root: str = "memory"
    output_path: str = "results/v2_5_memory_curator_report.json"

    @classmethod
    def load(cls, path: str | Path) -> "MemoryCuratorRuntimeConfig":
        p = Path(path)
        if not p.exists():
            return cls()

        data = json.loads(p.read_text(encoding="utf-8"))

        # Ignore documentation-only / future fields such as "note".
        allowed = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in allowed}

        base = {f.name: getattr(cls(), f.name) for f in fields(cls)}
        return cls(**{**base, **filtered})


class MemoryCuratorRuntime:
    def __init__(self, config: MemoryCuratorRuntimeConfig) -> None:
        self.config = config
        self.organ = MemoryCuratorOrgan(memory_root=config.memory_root)

    def run_once(self) -> Dict[str, Any]:
        report = self.organ.run().to_jsonable()
        out = Path(self.config.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        report["output_path"] = str(out)
        return report
