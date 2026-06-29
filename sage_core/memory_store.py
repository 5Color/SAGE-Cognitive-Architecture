from __future__ import annotations
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
import json
import uuid

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

@dataclass
class MemoryProposal:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: str = field(default_factory=utc_now)
    status: str = "inbox"
    source: str = "runtime"
    importance: float = 0.5
    content: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)
    human_review_required: bool = True

    def to_jsonable(self) -> Dict[str, Any]:
        return asdict(self)

class MemoryStore:
    """File-based memory inbox. Runtime proposes; human approves later."""
    def __init__(self, root: str | Path = "memory") -> None:
        self.root = Path(root)
        self.inbox = self.root / "inbox"
        self.approved = self.root / "approved"
        self.rejected = self.root / "rejected"
        for p in [self.inbox, self.approved, self.rejected]:
            p.mkdir(parents=True, exist_ok=True)

    def propose(self, content: str, importance: float = 0.5, source: str = "runtime", evidence: Dict[str, Any] | None = None) -> MemoryProposal:
        proposal = MemoryProposal(
            content=content,
            importance=max(0.0, min(1.0, float(importance))),
            source=source,
            evidence=evidence or {},
        )
        safe_time = proposal.created_at.replace(":", "-").replace(".", "-")
        path = self.inbox / f"{safe_time}_{proposal.id}.json"
        path.write_text(json.dumps(proposal.to_jsonable(), indent=2, ensure_ascii=False), encoding="utf-8")
        return proposal

    def count_inbox(self) -> int:
        return len(list(self.inbox.glob("*.json")))
