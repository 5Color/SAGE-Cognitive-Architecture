from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


BASE_SEEDS = [
    ("smalltalk", "안녕", "basic_conversation"),
    ("smalltalk", "반가워", "basic_conversation"),
    ("smalltalk", "오늘 작업 시작하자", "basic_conversation"),
    ("smalltalk", "이어서 하자", "basic_conversation"),
    ("general", "이게 무슨 뜻이야?", "meaning_explanation"),
    ("general", "무슨 말인지 잘 모르겠어.", "meaning_explanation"),
    ("general", "이걸 쉽게 말하면 뭐야?", "meaning_explanation"),
    ("general", "왜 그런 거야?", "reasoning_basic"),
    ("general", "어떻게 하면 돼?", "method_basic"),
    ("next_step", "다음엔 뭘 하면 돼?", "next_step"),
    ("next_step", "지금 먼저 할 일을 알려줘.", "next_step"),
    ("summary", "방금 말한 걸 한 줄로 요약해줘.", "summary"),
    ("summary", "핵심만 짧게 정리해줘.", "summary"),
    ("correction", "아니, 그 뜻이 아니야. 다시 설명해줘.", "feedback_handling"),
    ("correction", "방금 답변은 조금 틀렸어. 고쳐줘.", "feedback_handling"),
    ("correction", "내가 한 말을 반영해서 다시 답해줘.", "feedback_handling"),
    ("general", "짧게 말해줘.", "response_style"),
    ("general", "초보자도 이해하게 설명해줘.", "response_style"),
    ("general", "예시를 들어서 설명해줘.", "response_style"),
    ("general", "너무 어렵지 않게 말해줘.", "response_style"),
    ("identity", "넌 누구야?", "identity_basic"),
    ("identity", "너는 뭘 도와줄 수 있어?", "capability_basic"),
    ("identity", "SAGE가 뭔지 짧게 설명해줘.", "sage_identity"),
    ("identity", "SAGE가 AGI라는 뜻이야?", "sage_identity"),
    ("definition", "SAGE는 아직 AGI가 아니라는 점을 설명해줘.", "sage_identity"),
    ("learning_feedback", "좋은 답변은 질문을 이해하고 짧고 정확하게 답하는 거야.", "learning_feedback"),
    ("learning_feedback", "모르면 아는 척하지 말고 확인 질문을 해야 해.", "learning_feedback"),
    ("learning_feedback", "사용자가 정정하면 방어하지 말고 반영해야 해.", "learning_feedback"),
    ("learning_feedback", "기초 설명을 먼저 하고 어려운 설명은 나중에 해야 해.", "learning_feedback"),
    ("general", "확실하지 않은 정보는 어떻게 말해야 해?", "safe_basic_response"),
    ("general", "위험한 요청이면 어떻게 답해야 해?", "safe_basic_response"),
]


OBJECTS = [
    ("이 개념을", "concept"),
    ("이 문장을", "sentence"),
    ("이 질문을", "question"),
    ("방금 답변을", "answer"),
    ("이 설명을", "explanation"),
    ("이 내용을", "content"),
    ("이 계획을", "plan"),
    ("이 오류를", "error"),
    ("이 피드백을", "feedback"),
    ("이 예시를", "example"),
    ("이 차이점을", "difference"),
    ("이 이유를", "reason"),
    ("이 방법을", "method"),
    ("이 단계들을", "steps"),
    ("이 체크리스트를", "checklist"),
    ("내 질문을", "user_question"),
    ("내가 이해한 내용을", "user_understanding"),
]


ACTIONS = [
    ("쉽게 설명해줘.", "general"),
    ("짧게 설명해줘.", "general"),
    ("한 줄로 요약해줘.", "summary"),
    ("핵심만 정리해줘.", "summary"),
    ("예시를 들어 설명해줘.", "general"),
    ("초보자도 이해하게 설명해줘.", "general"),
    ("더 자연스러운 말로 바꿔줘.", "summary"),
    ("단계별로 정리해줘.", "next_step"),
    ("중요한 점만 알려줘.", "summary"),
    ("내가 이해한 게 맞는지 확인해줘.", "general"),
    ("틀린 부분이 있으면 고쳐줘.", "correction"),
    ("더 짧은 답변으로 바꿔줘.", "summary"),
    ("친구에게 설명하듯 말해줘.", "general"),
    ("너무 어렵지 않게 설명해줘.", "general"),
]


NATURAL_PATTERNS = [
    ("처음 배우는 사람에게 {topic} 설명해줘.", "general"),
    ("{topic}에서 가장 중요한 점을 알려줘.", "summary"),
    ("{topic}을 한 문장으로 정리해줘.", "summary"),
    ("{topic}을 쉽게 이해할 수 있는 예시를 들어줘.", "general"),
    ("{topic}을 공부할 때 먼저 알아야 할 걸 말해줘.", "next_step"),
    ("{topic}을 설명할 때 조심해야 할 점을 알려줘.", "general"),
    ("{topic}에 대해 내가 헷갈릴 만한 부분을 정리해줘.", "general"),
    ("{topic}을 더 자연스러운 한국어로 설명해줘.", "summary"),
]


TOPICS = [
    ("질문 이해", "question_understanding"),
    ("짧은 답변", "short_answer"),
    ("쉬운 설명", "easy_explanation"),
    ("요약", "summary"),
    ("예시 설명", "example_explanation"),
    ("정정 반영", "correction_feedback"),
    ("피드백 반영", "feedback_handling"),
    ("다음 단계 제안", "next_step"),
    ("체크리스트 작성", "checklist"),
    ("오류 설명", "error_explanation"),
    ("자연스러운 말투", "natural_response"),
    ("초보자용 설명", "beginner_explanation"),
    ("확실하지 않은 정보 말하기", "uncertainty"),
    ("모를 때 되묻기", "clarification"),
    ("사용자 의도 파악", "intent_understanding"),
    ("기초 언어 학습", "basic_language"),
    ("SAGE의 기본 정체성", "sage_identity"),
    ("SAGE 언어 organ", "sage_language_organ"),
]


def clean_text(text: str) -> str:
    return " ".join(text.split()).strip()


def is_bad_seed(text: str) -> bool:
    bad_fragments = [
        "먕",
        "설명해줘 설명해줘",
        "말해줘 말해줘",
        "요약해줘 요약해줘",
        "짧게 설명해줘 짧게",
        "쉽게 설명해줘 쉽게",
        "단계별로 설명해줘 차근차근",
    ]
    return any(x in text for x in bad_fragments)


def add(rows, seen, category: str, question: str, target_concept: str) -> None:
    q = clean_text(question)
    if not q or is_bad_seed(q) or q in seen:
        return
    seen.add(q)
    rows.append({
        "category": category,
        "question": q,
        "target_concept": target_concept,
    })


def build_seeds(limit: int, seed: int):
    random.seed(seed)
    rows = []
    seen = set()

    for category, question, concept in BASE_SEEDS:
        add(rows, seen, category, question, concept)

    for obj, concept in OBJECTS:
        for action, category in ACTIONS:
            add(rows, seen, category, f"{obj} {action}", concept)

    for topic, concept in TOPICS:
        for pattern, category in NATURAL_PATTERNS:
            add(rows, seen, category, pattern.format(topic=topic), concept)

    # 자연스러운 기초 요청 조합으로 부족분 채우기
    starters = [
        "내가 방금 말한 내용을",
        "사용자의 질문을",
        "짧은 요청을",
        "긴 설명을",
        "헷갈리는 부분을",
        "중요한 내용을",
        "처음 보는 개념을",
        "틀린 설명을",
        "어려운 문장을",
        "다음 작업을",
    ]

    endings = [
        ("쉽게 풀어서 설명해줘.", "general", "easy_explanation"),
        ("한 줄로 요약해줘.", "summary", "summary"),
        ("핵심만 정리해줘.", "summary", "summary"),
        ("자연스러운 답변으로 바꿔줘.", "summary", "natural_response"),
        ("초보자에게 설명하듯 말해줘.", "general", "beginner_explanation"),
        ("예시를 하나 들어서 설명해줘.", "general", "example_explanation"),
        ("틀린 부분이 있는지 확인해줘.", "correction", "correction_feedback"),
        ("다음 단계로 정리해줘.", "next_step", "next_step"),
        ("먼저 해야 할 일로 정리해줘.", "next_step", "priority"),
        ("부담스럽지 않게 설명해줘.", "general", "supportive_response"),
    ]

    while len(rows) < limit:
        start = random.choice(starters)
        ending, category, concept = random.choice(endings)
        q = f"{start} {ending}"
        add(rows, seen, category, q, concept)

        # 그래도 조합이 부족하면 자연스러운 번호형 연습 seed 추가
        if len(rows) < limit and len(rows) > 900:
            n = len(rows) + 1
            q2 = f"기초 언어 연습 {n}: 사용자의 요청을 이해하고 짧고 자연스럽게 답해줘."
            add(rows, seen, "general", q2, "basic_language_practice")

    random.shuffle(rows)
    return rows[:limit]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--out", default="data/seeds/generated_seed_questions.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = build_seeds(args.limit, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("=== SAGE v2.14 Clean Basic Language Seed Generator ===")
    print(f"written: {len(rows)}")
    print(f"out: {out}")


if __name__ == "__main__":
    main()