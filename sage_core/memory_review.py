from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import hashlib
import json
import shutil


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_id(text: str, length: int = 12) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def safe_read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "_load_error": f"{type(exc).__name__}: {exc}",
            "_raw_preview": path.read_text(encoding="utf-8", errors="replace")[:1000]
            if path.exists()
            else "",
        }


def extract_text_fields(data: Any, max_items: int = 12) -> List[str]:
    """Collect short useful text fragments from arbitrary nested JSON."""
    found: List[str] = []

    preferred_keys = {
        "claim",
        "proposal",
        "summary",
        "content",
        "text",
        "note",
        "final_note",
        "selected_organ",
        "organ",
        "reason",
        "rationale",
        "title",
        "question",
        "answer",
    }

    def walk(obj: Any, depth: int = 0) -> None:
        if len(found) >= max_items or depth > 4:
            return
        if isinstance(obj, dict):
            # Preferred keys first.
            for key in preferred_keys:
                if key in obj and isinstance(obj[key], (str, int, float, bool)):
                    value = str(obj[key]).strip()
                    if value and value not in found:
                        found.append(f"{key}: {value[:300]}")
                        if len(found) >= max_items:
                            return
            for key, value in obj.items():
                if key in preferred_keys:
                    continue
                walk(value, depth + 1)
                if len(found) >= max_items:
                    return
        elif isinstance(obj, list):
            for item in obj[:10]:
                walk(item, depth + 1)
                if len(found) >= max_items:
                    return
        elif isinstance(obj, (str, int, float, bool)):
            value = str(obj).strip()
            if value and len(value) > 2 and value not in found:
                found.append(value[:300])

    walk(data)
    return found


@dataclass
class MemoryCandidate:
    candidate_id: str
    filename: str
    path: str
    created_at: str
    size_bytes: int
    preview: List[str] = field(default_factory=list)
    load_error: Optional[str] = None

    def to_jsonable(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryReviewDecision:
    candidate_id: str
    action: str
    source_path: str
    target_path: str
    reason: str
    reviewer: str = "human"
    decided_at: str = field(default_factory=utc_now)
    destructive: bool = False
    auto_approved: bool = False
    requires_human_approval: bool = True

    def to_jsonable(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryReviewReport:
    version: str = "v2.1"
    created_at: str = field(default_factory=utc_now)
    tool_name: str = "memory_review_tool"
    memory_root: str = "memory"
    inbox_count: int = 0
    approved_count: int = 0
    rejected_count: int = 0
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    last_decision: Optional[Dict[str, Any]] = None
    safety_policy: Dict[str, bool] = field(default_factory=lambda: {
        "auto_approve_memory": False,
        "auto_reject_memory": False,
        "auto_delete_memory": False,
        "file_delete": False,
        "requires_human_confirmation_for_move": True,
        "human_approval_required": True,
        "proposal_only_until_commanded": True,
    })
    passed: bool = False

    def to_jsonable(self) -> Dict[str, Any]:
        return asdict(self)


class MemoryReviewTool:
    """Human-in-the-loop memory review.

    Default operations only inspect memory proposals.
    Approve/reject moves a file only when an explicit command and confirmation are supplied.
    It never deletes files and never auto-approves memory.
    """

    def __init__(self, memory_root: str | Path = "memory") -> None:
        self.memory_root = Path(memory_root)
        self.inbox_dir = self.memory_root / "inbox"
        self.approved_dir = self.memory_root / "approved"
        self.rejected_dir = self.memory_root / "rejected"
        self.review_dir = self.memory_root / "review"

        for path in [self.inbox_dir, self.approved_dir, self.rejected_dir, self.review_dir]:
            path.mkdir(parents=True, exist_ok=True)

    def _rel(self, path: Path) -> str:
        return str(path).replace("\\", "/")

    def _candidate_id_for_file(self, path: Path) -> str:
        # Stable enough across runs but not dependent on full absolute path.
        payload = f"{path.name}|{path.stat().st_size if path.exists() else 0}"
        return stable_id(payload)

    def list_candidates(self) -> List[MemoryCandidate]:
        files = sorted([p for p in self.inbox_dir.glob("*.json") if p.is_file()])
        candidates: List[MemoryCandidate] = []

        for path in files:
            data = safe_read_json(path)
            stat = path.stat()
            candidate_id = self._candidate_id_for_file(path)
            preview = extract_text_fields(data)
            load_error = data.get("_load_error") if isinstance(data, dict) else None
            candidates.append(
                MemoryCandidate(
                    candidate_id=candidate_id,
                    filename=path.name,
                    path=self._rel(path),
                    created_at=datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                    size_bytes=stat.st_size,
                    preview=preview,
                    load_error=load_error,
                )
            )

        return candidates

    def count_json(self, directory: Path) -> int:
        return len([p for p in directory.glob("*.json") if p.is_file()])

    def report(self, last_decision: Optional[MemoryReviewDecision] = None) -> MemoryReviewReport:
        candidates = self.list_candidates()
        return MemoryReviewReport(
            memory_root=self._rel(self.memory_root),
            inbox_count=len(candidates),
            approved_count=self.count_json(self.approved_dir),
            rejected_count=self.count_json(self.rejected_dir),
            candidates=[c.to_jsonable() for c in candidates],
            last_decision=last_decision.to_jsonable() if last_decision else None,
            passed=True,
        )

    def find_candidate_path(self, candidate_id_or_prefix: str) -> Path:
        candidate_id_or_prefix = candidate_id_or_prefix.strip()
        matches = []
        for candidate in self.list_candidates():
            if candidate.candidate_id == candidate_id_or_prefix or candidate.candidate_id.startswith(candidate_id_or_prefix):
                matches.append(candidate)

        if not matches:
            # Also allow exact filename or filename prefix.
            for p in self.inbox_dir.glob("*.json"):
                if p.name == candidate_id_or_prefix or p.name.startswith(candidate_id_or_prefix):
                    matches.append(MemoryCandidate(
                        candidate_id=self._candidate_id_for_file(p),
                        filename=p.name,
                        path=self._rel(p),
                        created_at=utc_now(),
                        size_bytes=p.stat().st_size,
                    ))

        if not matches:
            raise FileNotFoundError(f"No memory candidate matched: {candidate_id_or_prefix}")

        unique_paths = {m.path for m in matches}
        if len(unique_paths) > 1:
            ids = ", ".join([f"{m.candidate_id}:{m.filename}" for m in matches])
            raise ValueError(f"Ambiguous candidate id/prefix. Matches: {ids}")

        return Path(matches[0].path)

    def show_candidate(self, candidate_id_or_prefix: str) -> Dict[str, Any]:
        path = self.find_candidate_path(candidate_id_or_prefix)
        data = safe_read_json(path)
        candidate_id = self._candidate_id_for_file(path)
        return {
            "candidate_id": candidate_id,
            "filename": path.name,
            "path": self._rel(path),
            "preview": extract_text_fields(data, max_items=20),
            "raw": data,
        }

    def _unique_target(self, directory: Path, filename: str) -> Path:
        target = directory / filename
        if not target.exists():
            return target

        stem = target.stem
        suffix = target.suffix
        for idx in range(1, 10000):
            candidate = directory / f"{stem}.{idx}{suffix}"
            if not candidate.exists():
                return candidate
        raise RuntimeError(f"Could not create unique target for {target}")

    def decide(
        self,
        candidate_id_or_prefix: str,
        action: str,
        reason: str,
        confirm: bool = False,
        reviewer: str = "human",
    ) -> MemoryReviewDecision:
        action = action.strip().lower()
        if action not in {"approve", "reject"}:
            raise ValueError("action must be either 'approve' or 'reject'")

        if not confirm:
            raise PermissionError(
                "Memory move requires explicit confirmation. Re-run with --confirm."
            )

        source = self.find_candidate_path(candidate_id_or_prefix)
        if not source.exists():
            raise FileNotFoundError(str(source))

        target_dir = self.approved_dir if action == "approve" else self.rejected_dir
        target = self._unique_target(target_dir, source.name)

        shutil.move(str(source), str(target))

        decision = MemoryReviewDecision(
            candidate_id=self._candidate_id_for_file(target),
            action=action,
            source_path=self._rel(source),
            target_path=self._rel(target),
            reason=reason or "(no reason provided)",
            reviewer=reviewer,
        )

        audit_name = f"{utc_now().replace(':', '-').replace('.', '-')}_{decision.candidate_id}_{action}.json"
        audit_path = self.review_dir / audit_name
        audit_path.write_text(json.dumps(decision.to_jsonable(), indent=2, ensure_ascii=False), encoding="utf-8")

        return decision
