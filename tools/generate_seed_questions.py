from __future__ import annotations

import argparse
import json
from pathlib import Path

BASE = [
    ("definition", "AGI가 뭐야?", "agi"),
    ("definition", "AGI가 뭔지 설명해줘", "agi"),
    ("definition", "SAGE가 뭐야?", "sage"),
    ("definition", "SAGE랑 AGI 차이 설명해줘", "sage_agi_relation"),
    ("definition", "Memory Context Manager가 뭐야?", "memory_context_manager"),
    ("definition", "Weighted Memory가 왜 필요해?", "weighted_memory"),
    ("definition", "Raw Archive랑 runtime memory 차이가 뭐야?", "raw_archive"),
    ("definition", "Training Dataset Builder가 뭐야?", "training_dataset"),
    ("next_step", "SAGE 다음 단계 뭐해야 해?", "roadmap"),
    ("next_step", "지금 SAGE 병목이 뭐야?", "bottleneck"),
    ("next_step", "CPU 언어모델 붙이기 전에 뭘 해야 해?", "roadmap"),
    ("correction", "정정: SAGE는 AGI가 아니라 AGI 지향 인지 아키텍처 프로토타입이야.", "correction"),
    ("learning_feedback", "좋은 답변 예시: 질문: AGI가 뭐야? 이상적인 답변: AGI는 인공일반지능이고 SAGE는 AGI가 아니다. 규칙: 정의 질문은 term definition부터 답한다. 중요도: high", "feedback"),
    ("smalltalk", "ㅎㅇ", "smalltalk"),
    ("smalltalk", "먕", "smalltalk"),
    ("identity", "넌 누구야?", "sage_identity"),
    ("capability", "너 지금 뭐 할 수 있어?", "capability"),
]

VARIANTS = [
    "{q}",
    "{q} 짧게 설명해줘",
    "{q} SAGE 관점으로 설명해줘",
    "{q} 초보자도 이해하게 설명해줘",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/seeds/generated_seed_questions.jsonl")
    parser.add_argument("--limit", type=int, default=80)
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for category, q, concept in BASE:
        for template in VARIANTS:
            rows.append({
                "category": category,
                "question": template.format(q=q),
                "target_concept": concept,
            })
            if len(rows) >= args.limit:
                break
        if len(rows) >= args.limit:
            break

    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote {len(rows)} seeds to {out}")


if __name__ == "__main__":
    main()
