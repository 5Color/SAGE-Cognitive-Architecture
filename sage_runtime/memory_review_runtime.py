from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import json

from sage_core.memory_review import MemoryReviewTool


@dataclass
class MemoryReviewRuntimeConfig:
    memory_root: str = "memory"
    output_path: str = "results/v2_1_memory_review_report.json"

    @classmethod
    def load(cls, path: str | Path) -> "MemoryReviewRuntimeConfig":
        p = Path(path)
        if not p.exists():
            return cls()
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls(**{**cls().__dict__, **data})


class MemoryReviewRuntime:
    def __init__(self, config: MemoryReviewRuntimeConfig) -> None:
        self.config = config
        self.tool = MemoryReviewTool(memory_root=config.memory_root)

    def write_report(self, report: Dict[str, Any]) -> Dict[str, Any]:
        output = Path(self.config.output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        report["output_path"] = str(output)
        return report

    def list(self) -> Dict[str, Any]:
        report = self.tool.report().to_jsonable()
        return self.write_report(report)

    def show(self, candidate_id: str) -> Dict[str, Any]:
        payload = self.tool.show_candidate(candidate_id)
        output = Path(self.config.output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        payload["output_path"] = str(output)
        return payload

    def decide(
        self,
        candidate_id: str,
        action: str,
        reason: str,
        confirm: bool,
    ) -> Dict[str, Any]:
        decision = self.tool.decide(
            candidate_id_or_prefix=candidate_id,
            action=action,
            reason=reason,
            confirm=confirm,
        )
        report = self.tool.report(last_decision=decision).to_jsonable()
        return self.write_report(report)
