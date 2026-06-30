from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple
from collections import Counter
import json
import re


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def words(text: str) -> List[str]:
    return re.findall(r"[가-힣A-Za-z0-9_+#.\-]+", text.lower())


@dataclass
class MemorySummarizerConfig:
    memory_root: str = "memory"
    stages: List[str] = field(default_factory=lambda: ["approved", "validated", "provisional"])
    summary_dir: str = "memory/summaries"
    output_path: str = "results/v2_6_memory_summarizer_report.json"
    max_items_per_stage: int = 200
    max_representative_items: int = 8

    def to_jsonable(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryItem:
    stage: str
    path: str
    text: str
    terms: List[str]

    def to_jsonable(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "path": self.path,
            "preview": self.text[:300],
            "terms": self.terms[:20],
        }


@dataclass
class MemorySummaryReport:
    version: str = "v2.6"
    created_at: str = field(default_factory=utc_now)
    tool_name: str = "memory_summarizer"
    mode: str = "read_only_summarization"
    config: Dict[str, Any] = field(default_factory=dict)
    item_count: int = 0
    stage_counts: Dict[str, int] = field(default_factory=dict)
    top_terms: List[List[Any]] = field(default_factory=list)
    topic_counts: Dict[str, int] = field(default_factory=dict)
    representative_items: List[Dict[str, Any]] = field(default_factory=list)
    summary_markdown_path: str = ""
    summary_json_path: str = ""
    compression: Dict[str, float] = field(default_factory=dict)
    safety_policy: Dict[str, bool] = field(default_factory=lambda: {
        "read_only_source_memory": True,
        "file_delete": False,
        "file_move": False,
        "memory_auto_approve": False,
        "network_actions": False,
        "git_actions": False,
        "summaries_are_derived_memory": True,
    })
    passed: bool = False

    def to_jsonable(self) -> Dict[str, Any]:
        return asdict(self)


class MemorySummarizer:
    """Read-only memory summarizer.

    It closes the memory lifecycle:
    generate -> consolidate -> summarize -> retrieve.

    It does not mutate approved/validated/provisional source memory.
    """

    TOPIC_KEYWORDS: Dict[str, List[str]] = {
        "architecture": ["sage", "organ", "router", "runtime", "architecture", "cognitive"],
        "reflection": ["reflection", "critic", "curiosity", "observer", "planner", "반성"],
        "memory": ["memory", "approved", "validated", "provisional", "기억", "승인"],
        "safety": ["safety", "policy", "forbidden", "allowed", "stop", "approval", "안전", "금지"],
        "language_core": ["cpu", "language", "token", "chunk", "composer", "한국어", "언어"],
        "benchmark": ["benchmark", "passed", "true", "result", "score", "통과", "검증"],
        "autonomy": ["autonomy", "level", "safe_auto", "자율성"],
    }

    STOPWORDS = {
        "the", "and", "or", "to", "of", "in", "is", "are", "a", "an", "for",
        "with", "that", "this", "true", "false", "있습니다", "그리고", "하지만",
        "합니다", "있는", "없는", "것", "수", "등", "및", "한다", "된다",
    }

    def __init__(self, config: MemorySummarizerConfig) -> None:
        self.config = config
        self.memory_root = Path(config.memory_root)

    def run(self) -> MemorySummaryReport:
        items = self._load_items()
        stage_counts = Counter(item.stage for item in items)
        term_counter = Counter()
        topic_counts = Counter()

        for item in items:
            term_counter.update(item.terms)
            item_lower = item.text.lower()
            for topic, keys in self.TOPIC_KEYWORDS.items():
                if any(k.lower() in item_lower for k in keys):
                    topic_counts[topic] += 1

        representatives = self._select_representatives(items)
        markdown = self._build_markdown(items, stage_counts, term_counter, topic_counts, representatives)

        summary_dir = Path(self.config.summary_dir)
        summary_dir.mkdir(parents=True, exist_ok=True)
        md_path = summary_dir / "latest_memory_summary.md"
        json_path = summary_dir / "latest_memory_summary.json"

        md_path.write_text(markdown, encoding="utf-8")

        total_source_chars = sum(len(item.text) for item in items)
        summary_chars = len(markdown)
        ratio = round(summary_chars / max(1, total_source_chars), 4)

        report = MemorySummaryReport(
            config=self.config.to_jsonable(),
            item_count=len(items),
            stage_counts=dict(stage_counts),
            top_terms=[[term, count] for term, count in term_counter.most_common(20)],
            topic_counts=dict(topic_counts),
            representative_items=[item.to_jsonable() for item in representatives],
            summary_markdown_path=str(md_path).replace("\\", "/"),
            summary_json_path=str(json_path).replace("\\", "/"),
            compression={
                "source_chars": total_source_chars,
                "summary_chars": summary_chars,
                "summary_to_source_ratio": ratio,
            },
            passed=(
                md_path.exists()
                and summary_dir.exists()
                and len(markdown) > 0
                and (len(items) == 0 or len(representatives) >= 1)
            ),
        )

        json_path.write_text(json.dumps(report.to_jsonable(), indent=2, ensure_ascii=False), encoding="utf-8")

        out = Path(self.config.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report.to_jsonable(), indent=2, ensure_ascii=False), encoding="utf-8")

        return report

    def _load_items(self) -> List[MemoryItem]:
        items: List[MemoryItem] = []
        for stage in self.config.stages:
            stage_dir = self.memory_root / stage
            if not stage_dir.exists():
                continue

            files = [
                p for p in stage_dir.rglob("*")
                if p.is_file()
                and p.name != ".gitkeep"
                and p.suffix.lower() in {".json", ".md", ".txt"}
            ]
            files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

            for path in files[: self.config.max_items_per_stage]:
                try:
                    raw = path.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue

                text = self._extract_text(raw)
                text = normalize_text(text)
                if not text:
                    continue

                items.append(MemoryItem(
                    stage=stage,
                    path=str(path).replace("\\", "/"),
                    text=text,
                    terms=self._terms(text),
                ))
        return items

    def _extract_text(self, raw: str) -> str:
        try:
            data = json.loads(raw)
        except Exception:
            return raw

        parts: List[str] = []

        def walk(value: Any) -> None:
            if isinstance(value, str):
                parts.append(value)
            elif isinstance(value, dict):
                for k, v in value.items():
                    key = str(k).lower()
                    if key in {"content", "summary", "reflection", "proposal", "message", "text", "result", "reason"}:
                        walk(v)
                    elif isinstance(v, (dict, list)):
                        walk(v)
            elif isinstance(value, list):
                for item in value:
                    walk(item)

        walk(data)
        return "\n".join(parts) if parts else raw

    def _terms(self, text: str) -> List[str]:
        out: List[str] = []
        for w in words(text):
            if len(w) < 2:
                continue
            if w in self.STOPWORDS:
                continue
            out.append(w)
        return out

    def _select_representatives(self, items: List[MemoryItem]) -> List[MemoryItem]:
        scored: List[Tuple[float, MemoryItem]] = []
        for item in items:
            score = {"approved": 3.0, "validated": 2.0, "provisional": 1.0}.get(item.stage, 0.5)
            lower = item.text.lower()

            for keys in self.TOPIC_KEYWORDS.values():
                if any(k.lower() in lower for k in keys):
                    score += 0.5

            if any(k in lower for k in ["passed", "통과", "검증", "benchmark"]):
                score += 1.0
            if any(k in lower for k in ["forbidden", "safety", "policy", "금지", "안전"]):
                score += 0.8

            scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        reps: List[MemoryItem] = []
        seen = set()
        for _, item in scored:
            if item.path in seen:
                continue
            seen.add(item.path)
            reps.append(item)
            if len(reps) >= self.config.max_representative_items:
                break
        return reps

    def _build_markdown(
        self,
        items: List[MemoryItem],
        stage_counts: Counter,
        term_counter: Counter,
        topic_counts: Counter,
        representatives: List[MemoryItem],
    ) -> str:
        lines: List[str] = []
        lines.append("# SAGE Memory Summary")
        lines.append("")
        lines.append(f"Created: {utc_now()}")
        lines.append("")
        lines.append("## Scope")
        lines.append("")
        lines.append("This summary is derived from SAGE memory stages.")
        lines.append("")
        for stage in self.config.stages:
            lines.append(f"- `{stage}`: {stage_counts.get(stage, 0)} items")
        lines.append("")
        lines.append("## Top Topics")
        lines.append("")
        if topic_counts:
            for topic, count in topic_counts.most_common():
                lines.append(f"- {topic}: {count}")
        else:
            lines.append("- No dominant topics found.")
        lines.append("")
        lines.append("## Top Terms")
        lines.append("")
        if term_counter:
            for term, count in term_counter.most_common(15):
                lines.append(f"- {term}: {count}")
        else:
            lines.append("- No terms found.")
        lines.append("")
        lines.append("## Representative Memory Items")
        lines.append("")
        if representatives:
            for item in representatives:
                preview = item.text[:260].replace("\n", " ")
                lines.append(f"### {item.stage}: `{item.path}`")
                lines.append("")
                lines.append(preview)
                lines.append("")
        else:
            lines.append("No representative items.")
            lines.append("")
        lines.append("## Safety Note")
        lines.append("")
        lines.append("This summarizer is read-only with respect to source memory. It writes derived summaries only.")
        lines.append("")
        return "\n".join(lines)
