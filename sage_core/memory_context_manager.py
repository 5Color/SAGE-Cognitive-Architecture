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


def extract_terms(text: str) -> List[str]:
    raw = re.findall(r"[가-힣A-Za-z0-9_+#.\-]+", text.lower())
    stopwords = {
        "the", "and", "or", "to", "of", "in", "is", "are", "a", "an",
        "이", "그", "저", "것", "수", "등", "및", "좀", "더", "다음",
        "현재", "어떤", "있는", "없는", "하는", "하고", "해서", "으로",
    }
    return [w for w in raw if len(w) >= 2 and w not in stopwords]


@dataclass
class MemoryContextConfig:
    memory_root: str = "memory"
    memory_summary_path: str = "memory/summaries/latest_memory_summary.md"
    output_path: str = "results/v2_8_memory_context_bundle.json"

    stages: List[str] = field(default_factory=lambda: ["approved", "validated", "provisional"])
    max_items_per_stage: int = 200
    max_selected_items: int = 6
    max_summary_snippets: int = 6
    max_bundle_chars: int = 6000

    stage_weights: Dict[str, float] = field(default_factory=lambda: {
        "approved": 1.4,
        "validated": 1.1,
        "provisional": 0.75,
    })

    def to_jsonable(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SummarySnippet:
    score: float
    heading: str
    text: str
    reasons: List[str]

    def to_jsonable(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryContextItem:
    score: float
    stage: str
    path: str
    preview: str
    reasons: List[str]
    matched_terms: List[str]

    def to_jsonable(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryContextBundle:
    version: str = "v2.8"
    created_at: str = field(default_factory=utc_now)
    tool_name: str = "memory_context_manager"
    mode: str = "read_only_context_selection"

    query: str = ""
    query_terms: List[str] = field(default_factory=list)
    inferred_topics: List[str] = field(default_factory=list)

    selected_summary_snippets: List[Dict[str, Any]] = field(default_factory=list)
    selected_memory_items: List[Dict[str, Any]] = field(default_factory=list)
    context_text: str = ""

    stats: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)

    safety_policy: Dict[str, bool] = field(default_factory=lambda: {
        "read_only_memory": True,
        "source_memory_delete": False,
        "source_memory_move": False,
        "memory_auto_approve": False,
        "network_actions": False,
        "git_actions": False,
        "arbitrary_shell_actions": False,
        "writes_context_bundle_only": True,
    })

    passed: bool = False

    def to_jsonable(self) -> Dict[str, Any]:
        return asdict(self)


class MemoryContextManager:
    TOPIC_KEYWORDS: Dict[str, List[str]] = {
        "architecture": ["sage", "organ", "router", "runtime", "architecture", "cognitive", "구조", "아키텍처"],
        "reflection": ["reflection", "critic", "curiosity", "observer", "planner", "반성", "비판"],
        "memory": ["memory", "approved", "validated", "provisional", "inbox", "summary", "기억", "요약"],
        "safety": ["safety", "policy", "forbidden", "allowed", "stop", "approval", "안전", "금지", "정책"],
        "language_core": ["cpu", "language", "token", "chunk", "composer", "한국어", "언어"],
        "benchmark": ["benchmark", "passed", "true", "result", "score", "통과", "검증"],
        "autonomy": ["autonomy", "level", "safe_auto", "자율성", "level 2", "level 3"],
        "chat_loop": ["chat", "loop", "local", "대화", "채팅"],
    }

    def __init__(self, config: MemoryContextConfig) -> None:
        self.config = config
        self.memory_root = Path(config.memory_root)

    def build_context(self, query: str) -> MemoryContextBundle:
        query = normalize_text(query)
        query_terms = extract_terms(query)
        topics = self._infer_topics(query, query_terms)

        summary_snippets = self._select_summary_snippets(query_terms, topics)
        memory_items = self._select_memory_items(query_terms, topics)

        context_text = self._compose_context_text(query, topics, summary_snippets, memory_items)
        context_text = context_text[: self.config.max_bundle_chars]

        stats = {
            "query_char_count": len(query),
            "query_term_count": len(query_terms),
            "summary_snippet_count": len(summary_snippets),
            "memory_item_count": len(memory_items),
            "context_char_count": len(context_text),
            "stage_counts": dict(Counter(item.stage for item in memory_items)),
        }

        bundle = MemoryContextBundle(
            query=query,
            query_terms=query_terms,
            inferred_topics=topics,
            selected_summary_snippets=[s.to_jsonable() for s in summary_snippets],
            selected_memory_items=[m.to_jsonable() for m in memory_items],
            context_text=context_text,
            stats=stats,
            config=self.config.to_jsonable(),
            passed=bool(query and (summary_snippets or memory_items) and context_text),
        )

        out = Path(self.config.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(bundle.to_jsonable(), ensure_ascii=False, indent=2), encoding="utf-8")
        return bundle

    def _infer_topics(self, query: str, query_terms: List[str]) -> List[str]:
        haystack = " ".join([query.lower(), *query_terms])
        scored: List[Tuple[int, str]] = []
        for topic, keys in self.TOPIC_KEYWORDS.items():
            score = sum(1 for k in keys if k.lower() in haystack)
            if score > 0:
                scored.append((score, topic))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [topic for _, topic in scored]

    def _score_text(self, text: str, query_terms: List[str], topics: List[str]) -> Tuple[float, List[str], List[str]]:
        lower = text.lower()
        text_terms = set(extract_terms(text))

        matched_terms = sorted(set(query_terms) & text_terms)
        reasons: List[str] = []
        score = 0.0

        if matched_terms:
            score += 2.0 * len(matched_terms)
            reasons.append(f"matched_terms:{','.join(matched_terms[:8])}")

        for topic in topics:
            keys = self.TOPIC_KEYWORDS.get(topic, [])
            hits = [k for k in keys if k.lower() in lower]
            if hits:
                score += 1.5
                reasons.append(f"topic:{topic}")

        if any(k in lower for k in ["passed", "통과", "benchmark", "검증"]):
            score += 0.8
            reasons.append("verified_result_hint")

        if any(k in lower for k in ["safety", "forbidden", "policy", "금지", "안전", "정책"]):
            score += 0.7
            reasons.append("safety_policy_hint")

        if any(k in lower for k in ["summary", "context", "memory", "기억", "요약"]):
            score += 0.4
            reasons.append("memory_context_hint")

        return score, reasons, matched_terms

    def _select_summary_snippets(self, query_terms: List[str], topics: List[str]) -> List[SummarySnippet]:
        path = Path(self.config.memory_summary_path)
        if not path.exists():
            return []

        raw = path.read_text(encoding="utf-8", errors="replace")
        sections = self._split_markdown_sections(raw)

        snippets: List[SummarySnippet] = []
        for heading, text in sections:
            score, reasons, _ = self._score_text(f"{heading}\n{text}", query_terms, topics)
            if score <= 0:
                continue
            snippets.append(SummarySnippet(
                score=round(score, 4),
                heading=heading,
                text=normalize_text(text)[:900],
                reasons=reasons,
            ))

        snippets.sort(key=lambda s: s.score, reverse=True)
        return snippets[: self.config.max_summary_snippets]

    def _split_markdown_sections(self, text: str) -> List[Tuple[str, str]]:
        lines = text.splitlines()
        sections: List[Tuple[str, List[str]]] = []
        current_heading = "root"
        current_lines: List[str] = []

        def push() -> None:
            if current_lines:
                sections.append((current_heading, current_lines.copy()))

        for line in lines:
            if line.startswith("#"):
                push()
                current_heading = line.strip("# ").strip() or "section"
                current_lines = []
            else:
                if line.strip():
                    current_lines.append(line)
        push()

        if not sections and text.strip():
            return [("root", text.strip())]

        return [(h, "\n".join(body)) for h, body in sections]

    def _select_memory_items(self, query_terms: List[str], topics: List[str]) -> List[MemoryContextItem]:
        items: List[MemoryContextItem] = []

        for stage in self.config.stages:
            root = self.memory_root / stage
            if not root.exists():
                continue

            files = [
                p for p in root.rglob("*")
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

                text = normalize_text(self._extract_text(raw))
                if not text:
                    continue

                base_score, reasons, matched_terms = self._score_text(text, query_terms, topics)
                if base_score <= 0:
                    continue

                weight = self.config.stage_weights.get(stage, 0.5)
                final_score = base_score * weight
                reasons.append(f"stage_weight:{stage}={weight}")

                items.append(MemoryContextItem(
                    score=round(final_score, 4),
                    stage=stage,
                    path=str(path).replace("\\", "/"),
                    preview=text[:900],
                    reasons=reasons,
                    matched_terms=matched_terms,
                ))

        items.sort(key=lambda item: item.score, reverse=True)
        return items[: self.config.max_selected_items]

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
                for key, val in value.items():
                    k = str(key).lower()
                    if k in {"content", "summary", "reflection", "proposal", "message", "text", "result", "reason", "response_summary", "user_input"}:
                        walk(val)
                    elif isinstance(val, (dict, list)):
                        walk(val)
            elif isinstance(value, list):
                for item in value:
                    walk(item)

        walk(data)
        return "\n".join(parts) if parts else raw

    def _compose_context_text(
        self,
        query: str,
        topics: List[str],
        snippets: List[SummarySnippet],
        items: List[MemoryContextItem],
    ) -> str:
        lines: List[str] = []
        lines.append("# SAGE Memory Context Bundle")
        lines.append("")
        lines.append(f"Query: {query}")
        lines.append(f"Inferred topics: {', '.join(topics) if topics else 'none'}")
        lines.append("")
        lines.append("## Selected Summary Snippets")
        lines.append("")

        if snippets:
            for idx, snippet in enumerate(snippets, start=1):
                lines.append(f"### Summary {idx}: {snippet.heading} · score {snippet.score}")
                lines.append(snippet.text)
                lines.append("")
        else:
            lines.append("No relevant summary snippets selected.")
            lines.append("")

        lines.append("## Selected Memory Items")
        lines.append("")
        if items:
            for idx, item in enumerate(items, start=1):
                lines.append(f"### Memory {idx}: {item.stage} · score {item.score}")
                lines.append(f"Path: {item.path}")
                lines.append(f"Reasons: {', '.join(item.reasons)}")
                lines.append(item.preview)
                lines.append("")
        else:
            lines.append("No relevant memory items selected.")
            lines.append("")

        lines.append("## Safety")
        lines.append("")
        lines.append("This bundle is read-only context selection. It does not approve, delete, move, or execute memory.")
        return "\n".join(lines)
