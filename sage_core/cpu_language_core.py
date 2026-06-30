from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import re


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def is_hangul_syllable(ch: str) -> bool:
    return "\uac00" <= ch <= "\ud7a3"


def is_latin(ch: str) -> bool:
    return "a" <= ch.lower() <= "z"


def simple_words(text: str) -> List[str]:
    return re.findall(r"[가-힣A-Za-z0-9_+#.\-]+", text)


@dataclass
class TokenAnalysis:
    input_text: str
    char_count: int
    syllable_tokens: List[str]
    word_tokens: List[str]
    chunk_tokens: List[str]
    hangul_ratio: float
    latin_ratio: float
    compression: Dict[str, float]

    def to_jsonable(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SemanticState:
    intent: str
    topics: List[str]
    signals: Dict[str, Any]
    confidence: float

    def to_jsonable(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryHit:
    path: str
    score: float
    preview: str

    def to_jsonable(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LanguageCoreResult:
    version: str = "v2.4"
    created_at: str = field(default_factory=utc_now)
    engine_name: str = "cpu_language_core_architecture_probe"
    input_text: str = ""
    analysis: Dict[str, Any] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)
    memory_hits: List[Dict[str, Any]] = field(default_factory=list)
    response: str = ""
    safety_policy: Dict[str, bool] = field(default_factory=lambda: {
        "cpu_only": True,
        "external_model_required": False,
        "network_actions": False,
        "memory_read_only": True,
        "memory_auto_approve": False,
        "file_delete": False,
        "core_code_auto_modify": False,
    })
    passed: bool = False

    def to_jsonable(self) -> Dict[str, Any]:
        return asdict(self)


class KoreanChunkTokenizer:
    """Small CPU tokenizer/chunker for architecture experiments.

    This is not a production tokenizer.
    It is a transparent baseline for comparing char/syllable/word/chunk behavior.
    """

    CONNECTIVE_ENDINGS = (
        "해서", "하고", "하며", "지만", "는데", "니까", "으면", "면서",
        "려고", "도록", "거나", "라서", "인데", "다고", "다는",
    )

    def analyze(self, text: str) -> TokenAnalysis:
        text = normalize_text(text)
        chars = list(text)
        syllables = [ch for ch in chars if not ch.isspace()]
        words = simple_words(text)
        chunks = self.chunk(text, words)

        char_count = len([ch for ch in chars if not ch.isspace()])
        hangul_count = sum(1 for ch in syllables if is_hangul_syllable(ch))
        latin_count = sum(1 for ch in syllables if is_latin(ch))
        denom = max(1, len(syllables))

        compression = {
            "word_vs_syllable": round(len(words) / max(1, len(syllables)), 4),
            "chunk_vs_syllable": round(len(chunks) / max(1, len(syllables)), 4),
            "chunk_vs_word": round(len(chunks) / max(1, len(words)), 4),
        }

        return TokenAnalysis(
            input_text=text,
            char_count=char_count,
            syllable_tokens=syllables,
            word_tokens=words,
            chunk_tokens=chunks,
            hangul_ratio=round(hangul_count / denom, 4),
            latin_ratio=round(latin_count / denom, 4),
            compression=compression,
        )

    def chunk(self, text: str, words: Optional[List[str]] = None) -> List[str]:
        if words is None:
            words = simple_words(text)

        chunks: List[str] = []
        buffer: List[str] = []

        for word in words:
            buffer.append(word)
            if self._is_boundary(word):
                chunks.append(" ".join(buffer))
                buffer = []

        if buffer:
            chunks.append(" ".join(buffer))

        merged: List[str] = []
        for chunk in chunks:
            if merged and len(chunk) <= 2:
                merged[-1] = merged[-1] + " " + chunk
            else:
                merged.append(chunk)

        return merged

    def _is_boundary(self, word: str) -> bool:
        if word.endswith(self.CONNECTIVE_ENDINGS):
            return True
        if word in {"그리고", "하지만", "근데", "그래서", "즉", "다음", "결론"}:
            return True
        if word.endswith(("다", "요", "죠", "임", "함", "됨")) and len(word) >= 2:
            return True
        return False


class StateExtractor:
    """Converts user text into a small state representation."""

    TOPIC_KEYWORDS: Dict[str, List[str]] = {
        "SAGE": ["sage", "organ", "runtime", "reflection", "memory", "autonomy", "agi", "자율성", "기억", "반성"],
        "cpu_language_model": ["cpu", "언어모델", "language", "token", "chunk", "composer", "한국어"],
        "development": ["다음", "구현", "코드", "검증", "테스트", "커밋", "버전", "패치"],
        "safety": ["안전", "승인", "금지", "삭제", "네트워크", "통제", "policy"],
        "summary": ["요약", "정리", "피드백", "보고서"],
    }

    def extract(self, text: str, analysis: TokenAnalysis) -> SemanticState:
        lowered = text.lower()
        words = [w.lower() for w in analysis.word_tokens]

        topics: List[str] = []
        for topic, keys in self.TOPIC_KEYWORDS.items():
            if any(k.lower() in lowered for k in keys):
                topics.append(topic)

        intent = "statement"
        if "?" in text or any(k in text for k in ["뭐", "가능", "어때", "왜", "어떻게"]):
            intent = "question"
        if any(k in text for k in ["해줘", "ㄱㄱ", "가자", "만들", "다음 단계"]):
            intent = "request_next_step"
        if any(k in text for k in ["요약", "정리", "피드백"]):
            intent = "summary_request"

        signals = {
            "has_question_mark": "?" in text,
            "has_korean": analysis.hangul_ratio > 0.3,
            "has_technical_terms": any(w in words for w in ["cpu", "agi", "sage", "runtime", "token", "memory"]),
            "urgency_hint": any(k in text for k in ["ㄱㄱ", "빨리", "바로", "다음"]),
            "positive_momentum": any(k in text for k in ["성공", "순조", "좋", "맞다", "검증"]),
        }

        confidence = 0.55
        if topics:
            confidence += min(0.25, 0.05 * len(topics))
        if intent != "statement":
            confidence += 0.1
        if signals["has_technical_terms"]:
            confidence += 0.1

        return SemanticState(
            intent=intent,
            topics=topics,
            signals=signals,
            confidence=round(min(confidence, 0.95), 4),
        )


class ApprovedMemoryRetriever:
    """Read-only memory retriever for memory/approved."""

    def __init__(self, memory_root: str | Path = "memory") -> None:
        self.memory_root = Path(memory_root)

    def retrieve(self, query: str, limit: int = 3) -> List[MemoryHit]:
        approved = self.memory_root / "approved"
        if not approved.exists():
            return []

        q_terms = set(t.lower() for t in simple_words(query))
        hits: List[MemoryHit] = []

        for path in approved.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".json", ".md", ".txt"}:
                continue

            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            lower = text.lower()
            score = 0.0
            for term in q_terms:
                if len(term) <= 1:
                    continue
                if term in lower:
                    score += 1.0

            for term in ["sage", "runtime", "memory", "reflection", "agi", "organ"]:
                if term in query.lower() and term in lower:
                    score += 0.5

            if score > 0:
                preview = normalize_text(text[:500])
                hits.append(MemoryHit(
                    path=str(path).replace("\\", "/"),
                    score=round(score, 4),
                    preview=preview,
                ))

        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:limit]


class TemplateComposer:
    """A tiny CPU composer.

    It does not pretend to be a general LLM.
    It converts extracted state + memory hints into a stable Korean response.
    """

    def compose(self, text: str, analysis: TokenAnalysis, state: SemanticState, memory_hits: List[MemoryHit]) -> str:
        lines: List[str] = []

        if state.intent == "summary_request":
            lines.append("현재 요청은 SAGE 진행상황을 요약하고 다음 검증 단계로 넘기려는 것으로 보입니다.")
        elif state.intent == "request_next_step":
            lines.append("다음 단계로 진행할 수 있습니다. 현재 입력은 개발 진행 요청으로 분류됩니다.")
        elif state.intent == "question":
            lines.append("질문으로 인식했습니다. 구조적으로 검토한 답변을 생성합니다.")
        else:
            lines.append("입력 내용을 상태로 변환했습니다.")

        if state.topics:
            lines.append("감지된 주제: " + ", ".join(state.topics))

        lines.append(
            f"언어 분석: syllable {len(analysis.syllable_tokens)}개, "
            f"word {len(analysis.word_tokens)}개, chunk {len(analysis.chunk_tokens)}개."
        )

        if memory_hits:
            lines.append(f"승인된 memory에서 관련 항목 {len(memory_hits)}개를 참고할 수 있습니다.")
            for hit in memory_hits[:2]:
                lines.append(f"- {hit.path} · score {hit.score}")
        else:
            lines.append("관련 approved memory는 아직 찾지 못했습니다.")

        lines.append("추천 출력 전략: 먼저 state JSON을 만들고, 그다음 Composer가 짧은 한국어 응답을 생성합니다.")

        return "\n".join(lines)


class CPULanguageCore:
    """CPU-only language architecture probe.

    Pipeline:
    input text -> Korean chunk tokenizer -> semantic state -> approved memory retrieval -> template composer
    """

    def __init__(self, memory_root: str | Path = "memory") -> None:
        self.tokenizer = KoreanChunkTokenizer()
        self.extractor = StateExtractor()
        self.retriever = ApprovedMemoryRetriever(memory_root=memory_root)
        self.composer = TemplateComposer()

    def run(self, text: str, retrieve_memory: bool = True) -> LanguageCoreResult:
        text = normalize_text(text)
        analysis = self.tokenizer.analyze(text)
        state = self.extractor.extract(text, analysis)
        memory_hits = self.retriever.retrieve(text) if retrieve_memory else []
        response = self.composer.compose(text, analysis, state, memory_hits)

        passed = (
            bool(text)
            and analysis.char_count > 0
            and len(analysis.chunk_tokens) >= 1
            and state.confidence >= 0.5
            and bool(response)
        )

        return LanguageCoreResult(
            input_text=text,
            analysis=analysis.to_jsonable(),
            state=state.to_jsonable(),
            memory_hits=[h.to_jsonable() for h in memory_hits],
            response=response,
            passed=passed,
        )
