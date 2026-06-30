from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict
import json

from sage_core.cpu_language_core import CPULanguageCore


@dataclass
class CPULanguageCoreRuntimeConfig:
    memory_root: str = "memory"
    output_path: str = "results/v2_4_cpu_language_core_result.json"
    retrieve_memory: bool = True

    @classmethod
    def load(cls, path: str | Path) -> "CPULanguageCoreRuntimeConfig":
        p = Path(path)
        if not p.exists():
            return cls()

        data = json.loads(p.read_text(encoding="utf-8"))

        # Ignore documentation-only / future fields such as "note".
        allowed = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in allowed}

        base = {f.name: getattr(cls(), f.name) for f in fields(cls)}
        return cls(**{**base, **filtered})


class CPULanguageCoreRuntime:
    def __init__(self, config: CPULanguageCoreRuntimeConfig) -> None:
        self.config = config
        self.core = CPULanguageCore(memory_root=self.config.memory_root)

    def run_once(self, text: str) -> Dict[str, Any]:
        result = self.core.run(text, retrieve_memory=self.config.retrieve_memory).to_jsonable()

        out = Path(self.config.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        result["output_path"] = str(out)
        return result
