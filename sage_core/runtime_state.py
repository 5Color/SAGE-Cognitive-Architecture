from __future__ import annotations
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
import json

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

@dataclass
class SAGERuntimeState:
    """Persistent runtime state for SAGE v1.9."""
    version: str = "v1.9"
    node_name: str = "sage-runtime-node"
    mode: str = "safe_idle"
    step: int = 0
    energy: float = 1.0
    focus: str = "observe"
    last_tick_at: str | None = None
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    counters: Dict[str, int] = field(default_factory=lambda: {
        "ticks": 0,
        "memory_proposals": 0,
        "reflections": 0,
        "registry_reads": 0,
    })
    notes: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)

    def tick(self, focus: str = "observe") -> None:
        self.step += 1
        self.counters["ticks"] = self.counters.get("ticks", 0) + 1
        self.focus = focus
        self.last_tick_at = utc_now()
        self.updated_at = self.last_tick_at

    def to_jsonable(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def load(cls, path: str | Path) -> "SAGERuntimeState":
        p = Path(path)
        if not p.exists():
            return cls()
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls(**data)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_jsonable(), indent=2, ensure_ascii=False), encoding="utf-8")
