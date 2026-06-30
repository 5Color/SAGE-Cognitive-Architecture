from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict
import json

from sage_core.memory_context_manager import MemoryContextConfig, MemoryContextManager


@dataclass
class MemoryContextRuntimeConfig:
    config_path: str = "configs/memory_context_manager.json"

    @classmethod
    def load(cls, path: str | Path) -> "MemoryContextRuntimeConfig":
        p = Path(path)
        if not p.exists():
            return cls()
        data = json.loads(p.read_text(encoding="utf-8"))
        allowed = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in allowed}
        base = {f.name: getattr(cls(), f.name) for f in fields(cls)}
        return cls(**{**base, **filtered})


class MemoryContextRuntime:
    def __init__(self, runtime_config: MemoryContextRuntimeConfig) -> None:
        self.runtime_config = runtime_config

    def _load_config(self) -> MemoryContextConfig:
        path = Path(self.runtime_config.config_path)
        if not path.exists():
            return MemoryContextConfig()
        data = json.loads(path.read_text(encoding="utf-8"))
        allowed = {f.name for f in fields(MemoryContextConfig)}
        filtered = {k: v for k, v in data.items() if k in allowed}
        base = {f.name: getattr(MemoryContextConfig(), f.name) for f in fields(MemoryContextConfig)}
        return MemoryContextConfig(**{**base, **filtered})

    def build(self, query: str) -> Dict[str, Any]:
        config = self._load_config()
        manager = MemoryContextManager(config)
        bundle = manager.build_context(query).to_jsonable()
        bundle["output_path"] = config.output_path
        return bundle
