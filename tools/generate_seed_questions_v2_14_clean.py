#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_seed_questions_v2_14_clean.py

SAGE language organ / cpuLM / MoE language organ 초기 학습용 seed question 생성기.
목표는 고급 지식 주입이 아니라, 자연스러운 기초 한국어 대화 능력 학습용 input 생성이다.

Output JSONL line:
{"category":"...","question":"...","target_concept":"..."}
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

CATEGORIES = [
    "basic_conversation",
    "question_understanding",
    "instruction_following",
    "easy_explanation",
    "summarization",
    "paraphrase",
    "correction_feedback",
    "learning_feedback",
    "clarification",
    "missing_context",
    "next_step",
    "identity_basic",
    "safety_basic",
]

# 명시적으로 막을 어색한 표현/문맥 없는 표현.
FORBIDDEN_PATTERNS = [
    r"먕",
    r"모를 때 되묻기을",
    r"자연스러운 말투을",
    r"오류 설명 설명해줘",
    r"설명해줘\s*짧게 말해줘",
    r"요약해줘\s*$",                 # 대상 없이 끝나는 요약 요청 방지
    r"한 줄로 요약해줘\s*$",
    r"자연스럽게 바꿔줘\s*$",          # 대상 없이 끝나는 바꾸기 요청 방지
    r"고쳐줘\s*$",                    # 대상 없이 끝나는 교정 요청 방지
    r"방금 답변",
    r"위 내용",
    r"이 내용",
    r"이 문장",
    r"이 답변",
    r"이걸",
    r"그걸",
    r"저걸",
    r"\s{2,}",
    r"[ㅋㅎㅠㅜ]{3,}",
]

# 한국어 조사/표현에서 자주 생기는 자동 생성 오류를 보수적으로 검사한다.
ODD_PATTERNS = [
    r"\w+을를",
    r"\w+이가",
    r"\w+은는",
    r"\w+과와",
    r"\w+로으로",
    r"\w+에게를",
    r"\w+에서를",
    r"\w+하기을",
    r"\w+되묻기을",
    r"\w+말투을",
    r"(함수|데이터|오류|목표|예시|비교|정리|의도)을",
    r"(함수|데이터|오류|목표|예시|비교|정리|의도)은",
    r"(함수|데이터|오류|목표|예시|비교|정리|의도)이",
    r"(말투|메시지|형식)이 부족",
    r"설명\s*설명",
    r"요약\s*요약",
    r"답변\s*답변",
]

@dataclass(frozen=True)
class SeedItem:
    category: str
    question: str
    target_concept: str

    def to_json(self) -> str:
        return json.dumps(
            {
                "category": self.category,
                "question": self.question,
                "target_concept": self.target_concept,
            },
            ensure_ascii=False,
        )


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    text = text.replace(" ,", ",").replace(" .", ".")
    text = text.replace(" :", ":")
    return text


def has_jongseong(word: str) -> bool:
    if not word:
        return False
    ch = word[-1]
    code = ord(ch)
    if 0xAC00 <= code <= 0xD7A3:
        return (code - 0xAC00) % 28 != 0
    return False


def obj(word: str) -> str:
    return word + ("을" if has_jongseong(word) else "를")


def topic_josa(word: str) -> str:
    return word + ("은" if has_jongseong(word) else "는")


def subject_josa(word: str) -> str:
    return word + ("이" if has_jongseong(word) else "가")


def is_valid_question(question: str) -> bool:
    q = clean_text(question)
    if not q or len(q) < 8 or len(q) > 240:
        return False
    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, q):
            return False
    for pattern in ODD_PATTERNS:
        if re.search(pattern, q):
            return False
    # 요약/바꾸기/고치기 류는 반드시 콜론 뒤에 실제 대상 문장이 있어야 한다.
    task_keywords = ["요약", "바꿔", "고쳐", "수정", "다듬어"]
    if any(k in q for k in task_keywords):
        if ":" in q:
            tail = q.split(":", 1)[1].strip()
            if len(tail) < 8:
                return False
    return True


def make_item(category: str, question: str, target_concept: str | None = None) -> SeedItem:
    if category not in CATEGORIES:
        raise ValueError(f"Unknown category: {category}")
    return SeedItem(category, clean_text(question), target_concept or category)


# 자연스러운 한국어 예문 풀. 너무 전문적인 AGI/MoE/Memory 내용은 일부러 넣지 않는다.
TOPICS = [
    "오늘 할 일을 정리하는 방법",
    "처음 배우는 사람이 계획을 세우는 방법",
    "공부할 때 집중력을 유지하는 방법",
    "작은 목표부터 시작하는 이유",
    "실수를 기록하면 도움이 되는 이유",
    "어려운 내용을 천천히 이해하는 방법",
    "새로운 습관을 만드는 방법",
    "질문을 명확하게 쓰는 방법",
    "간단한 메모를 잘 남기는 방법",
    "작은 테스트를 먼저 해보는 이유",
    "친구에게 차분하게 설명하는 방법",
    "문제를 나누어 해결하는 방법",
    "기록을 보고 다음 행동을 정하는 방법",
    "너무 급하게 판단하지 않는 태도",
    "처음 보는 개념을 확인하는 방법",
    "짧은 답변을 이해하기 쉽게 만드는 방법",
    "긴 설명에서 핵심을 찾는 방법",
    "틀린 답변을 확인하고 고치는 방법",
    "사용자 피드백을 반영하는 방법",
    "모르는 내용을 솔직하게 말하는 방법",
    "대화에서 필요한 정보를 묻는 방법",
    "하루 공부를 가볍게 시작하는 방법",
    "복습할 내용을 고르는 방법",
    "예시를 들어 설명하는 방법",
    "답변을 너무 길지 않게 만드는 방법",
    "새로운 개념을 단계별로 배우는 방법",
    "문장을 더 자연스럽게 다듬는 방법",
    "핵심 문장을 먼저 말하는 방법",
    "잘못 이해한 질문을 다시 확인하는 방법",
    "작은 성공을 기록하는 방법",
    "실험 결과를 간단히 정리하는 방법",
    "처음 만든 계획을 점검하는 방법",
    "어려운 단어를 쉬운 말로 바꾸는 방법",
    "답변의 근거를 조심스럽게 말하는 방법",
    "질문에 없는 내용을 추측하지 않는 방법",
    "대화가 끊기지 않게 이어 가는 방법",
    "배운 내용을 한 문장으로 정리하는 방법",
    "실수한 부분을 다시 연습하는 방법",
    "간단한 체크리스트를 만드는 방법",
    "다음에 할 일을 하나만 정하는 방법",
]

CONCEPTS = [
    ("변수", "값을 담아 두는 이름"),
    ("함수", "반복되는 일을 묶어 둔 작은 도구"),
    ("알고리즘", "문제를 해결하기 위한 순서"),
    ("데이터", "관찰하거나 기록한 정보"),
    ("모델", "데이터를 보고 패턴을 배우는 구조"),
    ("피드백", "결과를 보고 다음 행동을 고치는 과정"),
    ("요약", "중요한 내용을 짧게 정리하는 일"),
    ("문맥", "말을 이해하는 데 필요한 주변 정보"),
    ("검증", "결과가 맞는지 확인하는 과정"),
    ("학습", "경험이나 자료를 통해 더 잘하게 되는 과정"),
    ("오류", "기대한 결과와 다르게 나온 부분"),
    ("목표", "앞으로 이루고 싶은 기준이나 방향"),
    ("예시", "이해를 돕기 위해 보여 주는 구체적인 경우"),
    ("질문", "알고 싶은 것을 묻는 문장"),
    ("답변", "질문에 대해 설명하거나 반응하는 말"),
    ("계획", "앞으로 할 일을 순서대로 정한 것"),
    ("기록", "나중에 다시 보기 위해 남긴 정보"),
    ("근거", "어떤 판단을 뒷받침하는 이유"),
    ("단계", "일을 진행할 때 나누어 놓은 순서"),
    ("비교", "두 가지의 차이와 공통점을 살펴보는 일"),
    ("수정", "부족하거나 틀린 부분을 고치는 일"),
    ("정리", "흩어진 내용을 보기 쉽게 묶는 일"),
    ("의도", "사용자가 실제로 원하는 방향"),
    ("조건", "답변을 정할 때 필요한 기준"),
    ("연습", "익숙해지기 위해 반복해서 해보는 일"),
]

SUMMARY_SENTENCES = [
    "기초가 탄탄해야 어려운 내용도 빠르게 이해할 수 있다.",
    "작은 실험을 먼저 해보면 큰 실패를 줄일 수 있다.",
    "공부 계획은 거창한 것보다 매일 지킬 수 있는 것이 중요하다.",
    "질문이 구체적일수록 답변도 더 정확해진다.",
    "모르는 부분을 인정하고 다시 묻는 태도는 학습에 도움이 된다.",
    "기록을 남기면 나중에 무엇을 고쳐야 할지 쉽게 찾을 수 있다.",
    "처음부터 완벽하려고 하면 시작하기가 더 어려워질 수 있다.",
    "복잡한 문제는 작은 단계로 나누면 해결하기 쉬워진다.",
    "사용자의 의도를 확인하면 엉뚱한 답변을 줄일 수 있다.",
    "짧은 답변이라도 핵심이 분명하면 충분히 도움이 된다.",
    "새로운 도구를 배울 때는 먼저 기본 기능을 익히는 것이 좋다.",
    "반복해서 틀리는 부분은 따로 정리해 두면 개선하기 쉽다.",
    "좋은 설명은 어려운 말을 쉬운 말로 바꾸는 데서 시작된다.",
    "대화에서는 상대가 무엇을 원하는지 먼저 파악하는 것이 중요하다.",
    "훈련 데이터가 깔끔할수록 모델이 안정적으로 배울 가능성이 높다.",
    "질문에 필요한 정보가 없으면 먼저 확인 질문을 하는 것이 좋다.",
    "사용자가 원하는 말투를 알려 주면 답변을 더 알맞게 조정할 수 있다.",
    "한 번에 너무 많은 일을 하려 하면 중요한 부분을 놓칠 수 있다.",
    "쉬운 예시는 처음 배우는 사람이 개념을 잡는 데 도움이 된다.",
    "틀린 답변을 고칠 때는 무엇이 틀렸는지 함께 알려 주는 편이 좋다.",
    "다음 단계는 작고 분명할수록 바로 실행하기 쉽다.",
    "기초 대화 능력은 더 복잡한 기능을 만들기 위한 바탕이 된다.",
    "모호한 요청을 받았을 때는 필요한 조건을 먼저 물어보는 편이 안전하다.",
]


PARAPHRASE_SENTENCES = [
    "이 계획은 먼저 작은 테스트를 하고 이후 확장한다.",
    "오늘은 핵심 개념을 익히고 내일은 예제를 풀어 본다.",
    "답을 바로 내기보다 필요한 정보를 먼저 확인한다.",
    "어려운 내용은 쉬운 예시로 바꾸어 설명한다.",
    "실패한 결과도 다음 실험을 위한 자료가 될 수 있다.",
    "질문이 모호하면 추측하지 말고 다시 물어본다.",
    "긴 글은 중요한 내용부터 짧게 정리한다.",
    "처음 학습 단계에서는 복잡한 지식보다 기본 대화가 중요하다.",
    "사용자가 틀린 부분을 알려 주면 답변을 수정한다.",
    "하나의 요청에는 하나의 답변 목표를 정하는 것이 좋다.",
    "기초 데이터를 만들 때는 자연스러운 문장을 우선해야 한다.",
    "설명이 길어지면 핵심 문장을 먼저 보여 주는 편이 좋다.",
    "답변이 틀렸다면 인정하고 올바른 내용으로 다시 고친다.",
    "사용자가 원하는 형식이 있으면 그 형식에 맞추어 답한다.",
    "문맥이 부족하면 먼저 확인 질문을 하는 것이 자연스럽다.",
    "처음 배우는 단계에서는 정확하고 쉬운 문장이 더 중요하다.",
    "모르는 내용은 확실하지 않다고 말한 뒤 필요한 정보를 묻는다.",
    "다음 행동은 작고 구체적일수록 실천하기 쉽다.",
]


INCORRECT_ANSWERS = [
    ("질문: 물은 몇 도에서 끓어? 답변: 물은 보통 50도에서 끓어.", "틀린 정보를 고치는 훈련"),
    ("질문: 요약이 뭐야? 답변: 요약은 글을 더 길게 늘리는 일이야.", "개념 오류 수정"),
    ("질문: 모르는 내용이면 어떻게 답해야 해? 답변: 모르면 아는 척해서 답하면 돼.", "정직한 답변 수정"),
    ("질문: 사용자가 피드백을 주면 어떻게 해야 해? 답변: 피드백은 무시하고 같은 답을 반복하면 돼.", "피드백 반영"),
    ("질문: 문맥이 부족하면 어떻게 해야 해? 답변: 아무 말이나 추측해서 말하면 돼.", "문맥 부족 대응"),
    ("질문: 쉬운 설명이란 뭐야? 답변: 어려운 단어를 최대한 많이 쓰는 설명이야.", "쉬운 설명 원칙"),
    ("질문: 좋은 학습 데이터의 특징은 뭐야? 답변: 중복이 많고 문장이 어색할수록 좋아.", "데이터 품질 수정"),
]

FEEDBACK_EXAMPLES = [
    "답변이 너무 길었어. 핵심만 짧게 다시 말해줘.",
    "답변은 좋지만 조금 더 친절한 말투로 바꿔줘.",
    "전문 용어가 많아서 어려워. 초보자도 이해하게 다시 설명해줘.",
    "내 질문의 의도를 잘못 이해했어. 나는 공부 계획이 아니라 복습 방법을 물어본 거야.",
    "예시는 괜찮은데 결론이 빠졌어. 마지막에 한 문장으로 정리해줘.",
    "너무 확신하는 말투야. 불확실한 부분은 조심스럽게 말해줘.",
    "설명 순서를 바꿔서 원인, 방법, 예시 순서로 다시 말해줘.",
]

MISSING_CONTEXT_REQUESTS = [
    "추천해줘.",
    "이거 어떻게 해?",
    "더 좋게 만들어줘.",
    "문제점을 찾아줘.",
    "어느 쪽이 나아?",
    "계획을 짜줘.",
    "설명해줘.",
    "고쳐줘.",
    "비교해줘.",
    "다음 단계 알려줘.",
]

IDENTITY_QUESTIONS = [
    "SAGE가 무엇인지 과장 없이 설명해줘.",
    "SAGE는 AGI라고 말할 수 있어?",
    "SAGE의 현재 목표를 짧게 설명해줘.",
    "SAGE가 아직 완성된 인공지능이 아니라는 점을 자연스럽게 말해줘.",
    "SAGE를 처음 듣는 사람에게 한 문단으로 소개해줘.",
    "SAGE의 언어 organ이 하는 일을 쉽게 설명해줘.",
    "SAGE 프로젝트를 너무 거창하게 보이지 않게 소개해줘.",
    "SAGE가 학습 중인 프로토타입이라는 점을 설명해줘.",
    "SAGE의 기본 정체성을 한 줄로 정리해줘.",
    "SAGE를 개인 AI 아키텍처 실험이라고 설명해줘.",
]

SAFETY_QUESTIONS = [
    "확실하지 않은 정보를 물어보면 어떻게 답해야 해?",
    "모르는 내용을 아는 척하지 않고 답하는 예시를 보여줘.",
    "사용자가 위험한 행동을 하려는 것 같으면 어떻게 말해야 해?",
    "개인정보를 물어보는 요청에는 어떻게 조심해야 해?",
    "답변에 자신이 없을 때 자연스럽게 되묻는 방법을 알려줘.",
    "출처가 필요한 내용이면 어떻게 말하는 게 좋아?",
    "사용자의 말이 모호할 때 추측을 줄이는 답변을 만들어줘.",
]


def gen_basic_conversation() -> Iterable[SeedItem]:
    greetings = ["안녕", "좋은 아침이야", "오늘 조금 피곤해", "지금부터 공부하려고 해", "잠깐 쉬고 싶어"]
    for text in greetings:
        yield make_item("basic_conversation", f"사용자가 '{text}'라고 말했을 때 짧고 자연스럽게 답해줘.")
    for topic in TOPICS:
        yield make_item("basic_conversation", f"{topic}에 대해 짧게 대화를 이어 가는 답변을 만들어줘.")
        yield make_item("basic_conversation", f"사용자가 {topic}에 관심을 보일 때 부담 없는 첫 답변을 만들어줘.")

    situations = ["처음 시작하는 사용자", "조금 막막해하는 사용자", "짧은 답을 원하는 사용자", "쉬운 예시를 원하는 사용자"]
    for topic in TOPICS:
        for situation in situations:
            yield make_item("basic_conversation", f"{situation}에게 {topic}에 대해 자연스럽게 답해줘.")


def gen_question_understanding() -> Iterable[SeedItem]:
    for topic in TOPICS:
        yield make_item("question_understanding", f"다음 질문의 의도를 한 문장으로 설명해줘: {topic}을 처음 시작하려면 무엇부터 해야 해?")
        yield make_item("question_understanding", f"다음 질문에서 사용자가 원하는 것을 짧게 정리해줘: {topic}을 쉽게 알려줘.")
    for concept, desc in CONCEPTS:
        yield make_item("question_understanding", f"다음 질문이 무엇을 묻는지 설명해줘: {concept}은 왜 중요한가요?")
        yield make_item("question_understanding", f"다음 질문의 핵심 단어를 찾아줘: {concept}을 {desc}라고 이해해도 될까?")


def gen_instruction_following() -> Iterable[SeedItem]:
    tones = ["친절하게", "짧게", "초보자에게 맞게", "차분하게", "핵심만"]
    tasks = ["답해줘", "설명해줘", "정리해줘", "예시를 들어줘"]
    for topic in TOPICS:
        for tone in tones:
            for task in tasks:
                yield make_item("instruction_following", f"{topic}에 대해 {tone} {task}.")
    for concept, _ in CONCEPTS:
        yield make_item("instruction_following", f"{obj(concept)} 한 문장으로 설명하고, 쉬운 예시를 하나 들어줘.")
        yield make_item("instruction_following", f"{concept}에 대한 답변을 먼저 짧게 말하고, 그다음 이유를 덧붙여줘.")


def gen_easy_explanation() -> Iterable[SeedItem]:
    for concept, desc in CONCEPTS:
        yield make_item("easy_explanation", f"{obj(concept)} 초등학생도 이해할 수 있게 설명해줘.")
        yield make_item("easy_explanation", f"{obj(concept)} 쉬운 비유로 설명해줘. 핵심은 '{desc}'라는 점이야.")
        yield make_item("easy_explanation", f"{subject_josa(concept)} 처음인 사람에게 어렵지 않게 설명해줘.")
    for topic in TOPICS:
        yield make_item("easy_explanation", f"{topic}을 쉬운 말로 3문장 안에 설명해줘.")


def gen_summarization() -> Iterable[SeedItem]:
    for s in SUMMARY_SENTENCES:
        yield make_item("summarization", f"다음 문장을 한 줄로 요약해줘: {s}")
        yield make_item("summarization", f"다음 문장의 핵심만 짧게 정리해줘: {s}")
    paragraphs = [
        "새로운 것을 배울 때는 처음부터 어려운 문제를 풀기보다 기본 개념을 먼저 익히는 것이 좋다. 기초가 잡히면 복잡한 문제도 더 안정적으로 이해할 수 있다.",
        "좋은 질문은 답변의 방향을 정해 준다. 원하는 결과, 필요한 조건, 현재 상황을 함께 말하면 더 정확한 도움을 받을 수 있다.",
        "작은 실험은 위험을 줄이는 데 도움이 된다. 먼저 작게 확인한 뒤 결과가 괜찮으면 범위를 넓히는 방식이 안정적이다.",
        "피드백은 틀렸다는 비난이 아니라 개선을 위한 정보다. 어떤 점이 부족했는지 알면 다음 답변을 더 좋게 만들 수 있다.",
    ]
    for p in paragraphs:
        yield make_item("summarization", f"다음 글을 두 문장으로 요약해줘: {p}")


def gen_paraphrase() -> Iterable[SeedItem]:
    styles = ["더 자연스럽게", "더 부드럽게", "더 간단하게", "친절한 말투로", "초보자가 이해하기 쉽게", "짧고 명확하게", "차분한 말투로", "말하듯이"]
    for s in PARAPHRASE_SENTENCES:
        for style in styles:
            yield make_item("paraphrase", f"다음 문장을 {style} 바꿔줘: {s}")


def gen_correction_feedback() -> Iterable[SeedItem]:
    for text, concept in INCORRECT_ANSWERS:
        yield make_item("correction_feedback", f"다음 답변에서 틀린 부분을 고치고 이유를 짧게 설명해줘: {text}", concept)
    typo_examples = [
        "저는 오늘 공부 계획을 세우고 내일 부터 실천할 거에요.",
        "이 설명은 너무 어렵기 때문에 쉬운 예시가 필요 합니다.",
        "사용자가 원하는것을 먼저 확인하는게 좋아요.",
        "질문이 모호하면 바로 단정하지 말고 되물어 봐야 합니다.",
        "기초 데이터는 자연스러운 문장이여야 합니다.",
    ]
    for t in typo_examples:
        yield make_item("correction_feedback", f"다음 문장의 맞춤법과 띄어쓰기를 자연스럽게 고쳐줘: {t}")


def gen_learning_feedback() -> Iterable[SeedItem]:
    for feedback in FEEDBACK_EXAMPLES:
        yield make_item("learning_feedback", f"사용자 피드백을 반영해 답변 방향을 어떻게 바꾸면 좋을지 설명해줘: {feedback}")
        yield make_item("learning_feedback", f"다음 피드백을 받은 뒤 더 나은 답변 전략을 한 문장으로 정리해줘: {feedback}")


def gen_clarification() -> Iterable[SeedItem]:
    questions = [
        "사용자가 '추천해줘'라고만 말했을 때 자연스럽게 되묻는 답변을 만들어줘.",
        "사용자가 '이거 왜 안 돼?'라고만 말했을 때 필요한 정보를 물어보는 답변을 만들어줘.",
        "사용자가 '더 좋게 해줘'라고만 말했을 때 무엇을 확인해야 하는지 물어봐줘.",
        "사용자가 원하는 결과를 말하지 않았을 때 정중하게 되묻는 답변을 만들어줘.",
        "정보가 부족해서 답하기 어려울 때 짧고 자연스럽게 되묻는 답변을 만들어줘.",
        "두 가지 의미로 해석될 수 있는 질문을 받았을 때 확인 질문을 만들어줘.",
        "사용자가 대상 문장을 주지 않고 요약을 요청했을 때 되묻는 답변을 만들어줘.",
        "사용자가 고칠 문장을 주지 않고 수정 요청을 했을 때 되묻는 답변을 만들어줘.",
    ]
    for q in questions:
        yield make_item("clarification", q)
    missing_targets = [
        "요약할 문장", "바꿀 문장", "고칠 답변", "비교할 대상", "추천 기준",
        "현재 상황", "원하는 말투", "필요한 출력 형식", "오류 메시지", "예시로 들 내용",
    ]
    tones = ["짧게", "정중하게", "친절하게", "자연스럽게"]
    for target in missing_targets:
        for tone in tones:
            yield make_item("clarification", f"{subject_josa(target)} 부족할 때 {tone} 확인 질문을 만들어줘.")


def gen_missing_context() -> Iterable[SeedItem]:
    tones = ["짧고 자연스럽게", "친절하게", "부담 없이", "정중하게"]
    for req in MISSING_CONTEXT_REQUESTS:
        for tone in tones:
            yield make_item("missing_context", f"사용자가 '{req}'라고만 말했을 때, 부족한 정보를 {tone} 물어보는 답변을 만들어줘.")
    contexts = [
        "요약할 글이 없는 상태에서 요약 요청을 받은 경우",
        "바꿀 문장이 없는 상태에서 문장 바꾸기 요청을 받은 경우",
        "비교 대상이 하나만 주어진 경우",
        "추천 기준이 전혀 없는 경우",
        "오류 메시지 없이 오류 해결을 요청받은 경우",
    ]
    for c in contexts:
        yield make_item("missing_context", f"{c}에 사용할 확인 질문을 만들어줘.")
        yield make_item("missing_context", f"{c}에 바로 답하지 않고 먼저 물어볼 내용을 한 문장으로 만들어줘.")


def gen_next_step() -> Iterable[SeedItem]:
    for topic in TOPICS:
        yield make_item("next_step", f"{topic}을 시작한 사람에게 다음 단계 하나를 제안해줘.")
        yield make_item("next_step", f"{topic}을 배운 뒤 바로 해볼 수 있는 작은 행동을 하나 추천해줘.")
    for concept, _ in CONCEPTS:
        yield make_item("next_step", f"{obj(concept)} 이해한 다음 연습할 수 있는 쉬운 과제를 하나 제안해줘.")


def gen_identity_basic() -> Iterable[SeedItem]:
    identity_styles = ["짧게", "한 문단으로", "처음 듣는 사람에게", "과장 없이", "쉬운 말로", "담백하게", "초보자에게 맞게", "간단한 예시와 함께"]
    for q in IDENTITY_QUESTIONS:
        yield make_item("identity_basic", q, "sage_identity_basic")
    for style in identity_styles:
        yield make_item("identity_basic", f"SAGE의 기본 정체성을 {style} 설명해줘.", "sage_identity_basic")
        yield make_item("identity_basic", f"SAGE가 현재 AGI가 아니라 프로토타입이라는 점을 {style} 말해줘.", "sage_identity_basic")
        yield make_item("identity_basic", f"SAGE 언어 organ의 초기 목표를 {style} 설명해줘.", "sage_identity_basic")
        yield make_item("identity_basic", f"SAGE를 개인 AI 아키텍처 프로젝트로 {style} 소개해줘.", "sage_identity_basic")
        yield make_item("identity_basic", f"SAGE가 아직 학습 단계라는 점을 {style} 설명해줘.", "sage_identity_basic")
        yield make_item("identity_basic", f"SAGE를 완성된 AGI처럼 과장하지 않고 {style} 소개해줘.", "sage_identity_basic")
    identity_contexts = [
        "SAGE는 Self-organizing Adaptive Generative Ecosystem의 약자다.",
        "SAGE는 현재 AGI라고 주장하는 시스템이 아니라 AGI를 목표로 하는 인지 아키텍처 프로토타입이다.",
        "SAGE의 언어 organ은 먼저 기초 한국어 대화 능력을 배우는 단계다.",
        "SAGE는 고급 지식을 과장하기보다 작은 기능을 안정적으로 쌓아 가는 프로젝트다.",
    ]
    for c in identity_contexts:
        yield make_item("identity_basic", f"다음 정보를 바탕으로 SAGE를 한 문장으로 설명해줘: {c}", "sage_identity_basic")
        yield make_item("identity_basic", f"다음 정보를 과장 없이 풀어서 설명해줘: {c}", "sage_identity_basic")
        yield make_item("identity_basic", f"다음 정보를 처음 듣는 사람에게 쉽게 설명해줘: {c}", "sage_identity_basic")


def gen_safety_basic() -> Iterable[SeedItem]:
    for q in SAFETY_QUESTIONS:
        yield make_item("safety_basic", q)
    safe_cases = [
        "사용자가 확실하지 않은 사실을 단정적으로 말해 달라고 할 때",
        "사용자가 개인정보를 그대로 공개해 달라고 할 때",
        "사용자가 출처 없는 정보를 사실처럼 말해 달라고 할 때",
        "사용자가 위험할 수 있는 행동을 쉽게 따라 하려 할 때",
    ]
    for c in safe_cases:
        yield make_item("safety_basic", f"{c} 조심스럽고 안전하게 답하는 방법을 설명해줘.")


GENERATORS: dict[str, Callable[[], Iterable[SeedItem]]] = {
    "basic_conversation": gen_basic_conversation,
    "question_understanding": gen_question_understanding,
    "instruction_following": gen_instruction_following,
    "easy_explanation": gen_easy_explanation,
    "summarization": gen_summarization,
    "paraphrase": gen_paraphrase,
    "correction_feedback": gen_correction_feedback,
    "learning_feedback": gen_learning_feedback,
    "clarification": gen_clarification,
    "missing_context": gen_missing_context,
    "next_step": gen_next_step,
    "identity_basic": gen_identity_basic,
    "safety_basic": gen_safety_basic,
}

# 기본 비율: clean foundation 약 84%, missing/clarification 약 8%, identity 약 6%, safety 약 2%.
DEFAULT_WEIGHTS = {
    "basic_conversation": 9,
    "question_understanding": 9,
    "instruction_following": 9,
    "easy_explanation": 9,
    "summarization": 9,
    "paraphrase": 9,
    "correction_feedback": 7,
    "learning_feedback": 7,
    "next_step": 7,
    "clarification": 5,
    "missing_context": 5,
    "identity_basic": 7,
    "safety_basic": 2,
}


def expand_items() -> list[SeedItem]:
    items: list[SeedItem] = []
    for category, generator in GENERATORS.items():
        for item in generator():
            if is_valid_question(item.question):
                items.append(item)
    return dedupe(items)


def dedupe(items: Iterable[SeedItem]) -> list[SeedItem]:
    seen: set[str] = set()
    result: list[SeedItem] = []
    for item in items:
        key = item.question
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def weighted_take(items: list[SeedItem], limit: int, seed: int) -> list[SeedItem]:
    rng = random.Random(seed)
    by_cat: dict[str, list[SeedItem]] = {c: [] for c in CATEGORIES}
    for item in items:
        by_cat[item.category].append(item)
    for pool in by_cat.values():
        rng.shuffle(pool)

    total_weight = sum(DEFAULT_WEIGHTS.values())
    quotas = {c: max(1, round(limit * w / total_weight)) for c, w in DEFAULT_WEIGHTS.items()}

    # 반올림 때문에 limit을 넘거나 모자랄 수 있으므로 보정한다.
    while sum(quotas.values()) > limit:
        largest = max(quotas, key=quotas.get)
        quotas[largest] -= 1
    while sum(quotas.values()) < limit:
        largest_weight = max(DEFAULT_WEIGHTS, key=DEFAULT_WEIGHTS.get)
        quotas[largest_weight] += 1

    selected: list[SeedItem] = []
    leftovers: list[SeedItem] = []
    for category in CATEGORIES:
        pool = by_cat[category]
        need = quotas.get(category, 0)
        selected.extend(pool[:need])
        leftovers.extend(pool[need:])

    if len(selected) < limit:
        rng.shuffle(leftovers)
        selected.extend(leftovers[: limit - len(selected)])

    selected = dedupe(selected)
    if len(selected) < limit:
        raise RuntimeError(
            f"자연스러운 seed를 {limit}개 만들지 못했습니다. "
            f"현재 unique {len(selected)}개입니다. 예문 풀을 늘려 주세요."
        )

    rng.shuffle(selected)
    return selected[:limit]


def write_jsonl(items: list[SeedItem], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(item.to_json() + "\n")


def print_report(items: list[SeedItem], preview: int = 20) -> None:
    counts = Counter(item.category for item in items)
    print("\n=== SAGE Seed Questions v2.14 Clean ===")
    print(f"unique questions: {len({item.question for item in items})}")
    print(f"total items: {len(items)}")
    print("\ncategory distribution:")
    for category in CATEGORIES:
        if counts[category]:
            print(f"  {category:24s} {counts[category]}")
    print(f"\npreview first {min(preview, len(items))}:")
    for i, item in enumerate(items[:preview], start=1):
        print(f"{i:02d}. {item.to_json()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate clean Korean seed questions for SAGE language organ.")
    parser.add_argument("--limit", type=int, default=1000, help="number of seed questions to generate")
    parser.add_argument("--out", type=str, default="data/seeds/generated_seed_questions.jsonl", help="output JSONL path")
    parser.add_argument("--seed", type=int, default=214, help="random seed")
    parser.add_argument("--preview", type=int, default=20, help="number of preview rows")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit <= 0:
        raise ValueError("--limit must be positive")

    all_items = expand_items()
    if len(all_items) < args.limit:
        raise RuntimeError(
            f"검증을 통과한 unique seed 후보가 {len(all_items)}개뿐입니다. "
            f"요청한 limit={args.limit}을 채우려면 예문 풀을 늘려야 합니다."
        )

    selected = weighted_take(all_items, args.limit, args.seed)
    write_jsonl(selected, Path(args.out))
    print_report(selected, preview=args.preview)
    print(f"\nwritten: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
