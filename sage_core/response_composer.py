from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import re


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def compact_text(text: str) -> str:
    return re.sub(r"\s+", "", text.lower().strip())


@dataclass
class ConceptCard:
    key: str
    label: str
    aliases: List[str]
    definition: str
    contrast: str = ""
    sage_relation: str = ""
    caution: str = ""

    def to_jsonable(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ResponseComposerConfig:
    style: str = "plain_korean"
    include_context_note: bool = True
    max_context_items: int = 3
    max_context_snippets: int = 2
    avoid_debug_dump: bool = True

    def to_jsonable(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ComposedResponse:
    version: str = "v2.10.1"
    created_at: str = field(default_factory=utc_now)
    intent: str = "general"
    concept_key: Optional[str] = None
    response: str = ""
    sentence_plan: List[str] = field(default_factory=list)
    context_used: bool = False
    context_summary: Dict[str, Any] = field(default_factory=dict)
    passed: bool = False

    def to_jsonable(self) -> Dict[str, Any]:
        return asdict(self)


class ResponseComposer:
    """Symbolic sentence composer.

    Not a neural LLM. It turns state + concept card + selected memory context
    into Korean answer sentences.
    """

    def __init__(self, config: Optional[ResponseComposerConfig] = None) -> None:
        self.config = config or ResponseComposerConfig()
        self.concepts = self._default_concepts()

    def compose(
        self,
        user_input: str,
        core_result: Dict[str, Any],
        context_bundle: Optional[Dict[str, Any]] = None,
    ) -> ComposedResponse:
        intent = self.detect_intent(user_input, core_result)
        concept = self.detect_concept(user_input)

        if intent == "greeting":
            response, plan = self._compose_greeting(context_bundle)
        elif intent == "smalltalk":
            response, plan = self._compose_smalltalk(context_bundle)
        elif intent == "identity":
            response, plan = self._compose_identity(context_bundle)
        elif intent == "capability":
            response, plan = self._compose_capability(context_bundle)
        elif intent == "definition" and concept is not None:
            response, plan = self._compose_definition(concept, context_bundle)
        elif intent in {"next_step", "summary"}:
            response, plan = self._compose_next_step(user_input, core_result, context_bundle)
        else:
            response, plan = self._compose_general(user_input, core_result, context_bundle)

        if self.config.include_context_note:
            response = self._append_context_note(response, context_bundle)

        return ComposedResponse(
            intent=intent,
            concept_key=concept.key if concept else None,
            response=response,
            sentence_plan=plan,
            context_used=self._context_is_relevant(context_bundle),
            context_summary=self._context_summary(context_bundle),
            passed=bool(response and len(response) >= 10),
        )

    def detect_intent(self, user_input: str, core_result: Dict[str, Any]) -> str:
        c = compact_text(user_input)

        if c in {"안녕", "안녕하세요", "ㅎㅇ", "하이", "hello", "hi"} or re.fullmatch(r"(ㅎㅇ)+", c.rstrip("?!.~ㅋㅎ")):
            return "greeting"

        if len(c) <= 3 and c in {"먕", "냥", "음", "흠", "ㅇ", "ㅋ", "ㅋㅋ", "ㅎㅎ"}:
            return "smalltalk"

        if any(k in c for k in ["넌누구", "너는누구", "정체가뭐", "너뭐야", "sage가뭐", "세이지가뭐"]):
            return "identity"

        if any(k in c for k in ["뭐할수", "무엇을할수", "기능", "명령어", "도움말", "help"]):
            return "capability"

        if any(k in c for k in ["뭔지설명", "무엇인지설명", "뭐인지설명", "뜻이뭐", "정의", "설명해줘", "설명해", "뭐야", "무엇이야", "what is", "define"]):
            return "definition"

        if any(k in c for k in ["다음단계", "다음으로", "우선순위", "뭘해야", "진행상황", "로드맵"]):
            return "next_step"

        state_intent = core_result.get("state", {}).get("intent", "")
        if state_intent == "summary_request":
            return "summary"
        if state_intent == "request_next_step":
            return "next_step"

        return "general"

    def detect_concept(self, user_input: str) -> Optional[ConceptCard]:
        c = compact_text(user_input)
        for card in self.concepts.values():
            if card.key in c:
                return card
            for alias in card.aliases:
                if compact_text(alias) in c:
                    return card
        return None

    def _topic_particle(self, label: str) -> str:
        if re.fullmatch(r"[A-Za-z0-9_+.\-]+", label):
            return "는"
        if not label:
            return "은"
        code = ord(label[-1])
        if 0xAC00 <= code <= 0xD7A3:
            return "은" if ((code - 0xAC00) % 28) else "는"
        return "는"

    def _compose_greeting(self, context_bundle: Optional[Dict[str, Any]]) -> tuple[str, List[str]]:
        return (
            "ㅎㅇ. 나는 SAGE의 로컬 대화 루프야.\n"
            "지금은 입력을 분석하고, 관련 기억을 고르고, Response Composer로 문장을 구성하는 실험용 인터페이스야.\n"
            "`/status`, `/context <질문>`, `/consolidate`, `/summarize`를 사용할 수 있어.",
            ["greeting", "identity_short", "available_next_action"],
        )

    def _compose_smalltalk(self, context_bundle: Optional[Dict[str, Any]]) -> tuple[str, List[str]]:
        return (
            "먕. 입력이 짧거나 의미가 애매해서 아직 특정 주제로 판단하긴 어려워.\n"
            "SAGE 설명, AGI 설명, 다음 단계 제안처럼 조금 더 구체적으로 말해주면 관련 memory context를 골라서 답변할게.",
            ["short_ack", "ask_for_clarity"],
        )

    def _compose_identity(self, context_bundle: Optional[Dict[str, Any]]) -> tuple[str, List[str]]:
        return (
            "나는 SAGE, Self-organizing Adaptive Generative Ecosystem의 local chat loop야.\n"
            "현재 SAGE는 AGI라고 주장하는 시스템이 아니라, memory, reflection, safety policy, CPU language core, "
            "memory context manager, response composer를 연결해서 인지 구조를 검증하는 프로토타입이야.",
            ["name", "not_agi_claim", "architecture_summary"],
        )

    def _compose_capability(self, context_bundle: Optional[Dict[str, Any]]) -> tuple[str, List[str]]:
        return (
            "지금 나는 로컬 대화, 입력 상태 분석, 관련 memory context 선택, response composition, "
            "response composer 기반 문장 구성을 할 수 있어.\n"
            "네트워크 접근, git 실행, 파일 삭제, 임의 shell 실행은 하지 않아.",
            ["capability_list", "supported_commands", "safety_boundary"],
        )

    def _compose_definition(self, concept: ConceptCard, context_bundle: Optional[Dict[str, Any]]) -> tuple[str, List[str]]:
        sentences: List[str] = []
        sentences.append(f"{concept.label}{self._topic_particle(concept.label)} {concept.definition}")
        if concept.contrast:
            sentences.append(f"핵심 차이는 {concept.contrast}")
        if concept.sage_relation:
            sentences.append(f"SAGE 관점에서는 {concept.sage_relation}")
        if concept.caution:
            sentences.append(concept.caution)

        context_sentence = self._context_sentence(context_bundle)
        if context_sentence:
            sentences.append(context_sentence)

        return "\n".join(sentences), ["term_definition", "contrast", "sage_relation", "caution"]

    def _compose_next_step(
        self,
        user_input: str,
        core_result: Dict[str, Any],
        context_bundle: Optional[Dict[str, Any]],
    ) -> tuple[str, List[str]]:
        response = (
            "지금 SAGE의 다음 병목은 더 큰 언어모델을 붙이는 것이 아니라, memory quality control이야.\n"
            "대화와 피드백을 전부 장기기억으로 쌓으면 memory가 비대해지고 context selection이 흔들릴 수 있어.\n"
            "따라서 다음 단계는 Response Composer stabilization로, 중요한 피드백은 강화하고 중복되거나 낮은 가치의 기억은 새로 저장하지 않고 약화/압축하는 구조야."
        )
        return response, ["current_state", "bottleneck", "response_composer"]

    def _compose_general(
        self,
        user_input: str,
        core_result: Dict[str, Any],
        context_bundle: Optional[Dict[str, Any]],
    ) -> tuple[str, List[str]]:
        state = core_result.get("state", {})
        topics = state.get("topics", [])
        topic_text = ", ".join(topics) if topics else "명확한 topic 없음"

        context_sentence = self._context_sentence(context_bundle)
        if context_sentence:
            tail = context_sentence
        else:
            tail = "관련 기억이 충분하지 않으니, 질문을 조금 더 구체화하거나 학습 피드백 형식으로 좋은 답변 예시를 줄 수 있어."

        return (
            f"입력은 `{topic_text}` 쪽으로 해석됐어.\n{tail}",
            ["acknowledge", "context_based_answer"],
        )

    def _context_is_relevant(self, context_bundle: Optional[Dict[str, Any]]) -> bool:
        if not context_bundle or not context_bundle.get("passed"):
            return False
        if context_bundle.get("inferred_topics", []):
            return True
        for item in context_bundle.get("selected_memory_items", []):
            if item.get("matched_terms"):
                return True
            if any(str(r).startswith(("topic:", "matched_terms:")) for r in item.get("reasons", [])):
                return True
        for snippet in context_bundle.get("selected_summary_snippets", []):
            if any(str(r).startswith(("topic:", "matched_terms:")) for r in snippet.get("reasons", [])):
                return True
        return False

    def _context_sentence(self, context_bundle: Optional[Dict[str, Any]]) -> str:
        if not self._context_is_relevant(context_bundle):
            return ""

        topics = context_bundle.get("inferred_topics", [])
        snippets = context_bundle.get("selected_summary_snippets", [])
        items = context_bundle.get("selected_memory_items", [])

        parts: List[str] = []
        if topics:
            parts.append(f"현재 context manager는 이 질문을 {', '.join(topics[:4])} 주제와 연결했어.")
        if snippets:
            text = snippets[0].get("text", "").strip()
            if text:
                parts.append(f"요약 기억에서는 `{text[:180]}` 부분이 관련 있다고 선택됐어.")
        if items:
            parts.append(f"추가로 {len(items)}개의 memory item이 선택됐어.")
        return " ".join(parts)

    def _append_context_note(self, response: str, context_bundle: Optional[Dict[str, Any]]) -> str:
        if not self._context_is_relevant(context_bundle):
            return response
        stats = context_bundle.get("stats", {})
        topics = context_bundle.get("inferred_topics", [])
        return response + f"\n\n[context] topics={topics if topics else 'none'}, summary={stats.get('summary_snippet_count', 0)}, memory={stats.get('memory_item_count', 0)}"

    def _context_summary(self, context_bundle: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not context_bundle:
            return {}
        return {
            "passed": context_bundle.get("passed"),
            "relevant": self._context_is_relevant(context_bundle),
            "inferred_topics": context_bundle.get("inferred_topics", []),
            "stats": context_bundle.get("stats", {}),
            "selected_memory_count": len(context_bundle.get("selected_memory_items", [])),
            "selected_summary_count": len(context_bundle.get("selected_summary_snippets", [])),
        }

    def _default_concepts(self) -> Dict[str, ConceptCard]:
        cards = [
            ConceptCard(
                key="agi",
                label="AGI",
                aliases=["Artificial General Intelligence", "인공일반지능", "범용 인공지능", "범용인공지능"],
                definition="Artificial General Intelligence, 즉 인공일반지능을 뜻해. 특정 작업 하나만 잘하는 AI가 아니라, 여러 분야의 문제를 이해하고 배운 지식을 다른 상황에 옮겨 적용할 수 있는 범용 지능을 목표로 하는 개념이야.",
                contrast="일반적인 좁은 AI는 번역, 이미지 분류, 코딩 보조처럼 특정 범위에 강하지만, AGI는 문제 영역이 바뀌어도 스스로 적응하고 지식을 전이하는 능력을 목표로 한다는 점이야.",
                sage_relation="SAGE가 AGI라는 뜻은 아니고, AGI에 필요할 수 있는 기억, 반성, 계획, 안전 정책, 언어 처리, context selection을 organ 단위로 쪼개서 검증하는 쪽에 가까워.",
                caution="현재 SAGE는 AGI라고 주장하는 단계가 아니라, AGI 지향 인지 아키텍처 프로토타입으로 보는 게 정확해.",
            ),
            ConceptCard(
                key="llm",
                label="LLM",
                aliases=["Large Language Model", "대규모 언어 모델", "대규모언어모델"],
                definition="Large Language Model, 즉 대규모 언어 모델을 뜻해. 많은 텍스트를 학습해 다음 token을 예측하면서 문장 생성, 요약, 번역, 질의응답을 수행하는 모델이야.",
                contrast="LLM은 거대한 행렬연산과 token 예측이 중심이고, SAGE는 memory, state, routing, reflection 같은 제어 구조를 따로 두려는 점이 달라.",
                sage_relation="SAGE에서 LLM은 전체 시스템 그 자체라기보다, 나중에 composer organ이나 language organ 일부를 강화하는 부품으로 들어갈 수 있어.",
            ),
            ConceptCard(
                key="sage",
                label="SAGE",
                aliases=["세이지", "Self-organizing Adaptive Generative Ecosystem"],
                definition="Self-organizing Adaptive Generative Ecosystem의 약자야. 여러 organ이 memory, reflection, planning, safety, language, context selection 역할을 나누는 인지 아키텍처 프로토타입이야.",
                contrast="거대한 모델 하나로 모든 것을 해결하려는 방식이 아니라, 작은 모듈들이 상태와 기억을 공유하며 협력하는 구조를 목표로 한다는 점이 특징이야.",
                sage_relation="현재 SAGE는 local runtime에서 memory lifecycle과 chat loop를 검증하는 단계야.",
            ),
        ]
        return {card.key: card for card in cards}

