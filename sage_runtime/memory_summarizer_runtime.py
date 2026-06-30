from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict
import json

from sage_core.memory_summarizer import MemorySummarizer, MemorySummarizerConfig


@dataclass
class MemorySummarizerRuntimeConfig:
    config_path: str = "configs/memory_summarizer.json"

    @classmethod
    def load(cls, path: str | Path) -> "MemorySummarizerRuntimeConfig":
        p = Path(path)
        if not p.exists():
            return cls()
        data = json.loads(p.read_text(encoding="utf-8"))
        allowed = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in allowed}
        base = {f.name: getattr(cls(), f.name) for f in fields(cls)}
        return cls(**{**base, **filtered})


class MemorySummarizerRuntime:
    def __init__(self, runtime_config: MemorySummarizerRuntimeConfig) -> None:
        self.runtime_config = runtime_config

    def _load_config(self) -> MemorySummarizerConfig:
        path = Path(self.runtime_config.config_path)
        if not path.exists():
            return MemorySummarizerConfig()
        data = json.loads(path.read_text(encoding="utf-8"))
        allowed = {f.name for f in fields(MemorySummarizerConfig)}
        filtered = {k: v for k, v in data.items() if k in allowed}
        base = {f.name: getattr(MemorySummarizerConfig(), f.name) for f in fields(MemorySummarizerConfig)}
        return MemorySummarizerConfig(**{**base, **filtered})

    def run_once(self) -> Dict[str, Any]:
        config = self._load_config()
        summarizer = MemorySummarizer(config)
        report = summarizer.run().to_jsonable()
        report["output_path"] = config.output_path
        return report
