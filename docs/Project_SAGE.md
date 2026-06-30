# Project SAGE

## 1. Project Name

**SAGE**는 **Self-organizing Adaptive Generative Ecosystem**의 약자이다.

SAGE는 하나의 거대한 모델에 모든 기능을 몰아넣는 방식이 아니라, 여러 개의 독립적인 Organ들이 상태, 기억, 반성, 계획, 검증, 실험을 통해 협력하는 **인공 생태계형 지능 아키텍처**를 실험하기 위한 프로젝트이다.

---

## 2. Why This Project Started

나는 솔직히 AGI를 만들고 싶어서 SAGE를 시작했다.

단순히 챗봇을 만들고 싶었던 것도 아니고, 기존 LLM을 감싸는 wrapper를 만들고 싶었던 것도 아니다.

내가 만들고 싶은 것은 하나의 거대한 모델이 모든 것을 처리하는 구조가 아니라, 여러 개의 Organ들이 각자의 역할을 가지고 상태, 기억, 반성, 계획, 검증을 통해 협력하는 인공 생태계형 지능 구조이다.

처음 목표는 분명히 AGI였다.

하지만 지금의 SAGE가 AGI라고 주장하지는 않는다.

현재의 SAGE는 AGI를 완성한 결과물이 아니라, AGI에 필요할 수 있는 조건들을 하나씩 실험하고 검증하기 위한 연구 플랫폼이다.

그래서 이 프로젝트의 방향은 다음과 같다.

```text
AGI를 목표로 한다.
하지만 AGI라고 성급하게 주장하지 않는다.

큰 목표를 가진다.
하지만 작은 실험으로 검증한다.

가능성을 믿는다.
하지만 결과로 판단한다.
```

SAGE는 내가 AGI를 만들 수 있는지 확인하기 위한 도전이자, 지능이 어떤 구조에서 만들어질 수 있는지 실험하는 개인 연구 프로젝트이다.

---

## 3. Core Philosophy

SAGE의 핵심 철학은 다음과 같다.

```text
목표는 크게 잡되, 주장은 작게 한다.
아이디어보다 실험 결과를 우선한다.
창발을 주장하기 전에 검증한다.
위험한 자율성보다 통제된 자율성을 먼저 만든다.
실패도 데이터로 기록한다.
```

SAGE는 “AGI를 만들었다”고 주장하기 위한 프로젝트가 아니다.

SAGE는 AGI에 필요한 조건들을 하나씩 실험하고, 그 결과를 기록하며, 더 나은 구조를 찾아가는 연구 프로젝트이다.

---

## 4. What SAGE Is

SAGE는 다음 요소들을 가진다.

```text
State
Memory
Router
Organ
Reflection
Critic
Planner
Lifecycle
Adaptive Compute
Experiment Planner
Safe Continuous Runtime
```

SAGE는 여러 Organ이 독립적으로 후보를 만들고, Aggregator나 Router가 그중 적절한 선택을 고르는 구조를 가진다.

이 과정에서 SAGE는 상태를 기록하고, 결과를 평가하고, 다음 실험을 제안하며, 안전한 범위 안에서 반복 실행될 수 있다.

---

## 5. What SAGE Is Not

SAGE는 아직 AGI가 아니다.

SAGE는 인간처럼 이해하거나 의식을 가진 존재가 아니다.

SAGE는 현재 단계에서 다음을 주장하지 않는다.

```text
SAGE는 AGI다.
SAGE는 자아를 가진다.
SAGE는 인간처럼 느낀다.
SAGE는 모든 문제를 일반화할 수 있다.
SAGE는 LLM을 완전히 대체했다.
```

현재 SAGE는 **AGI를 향한 modular autonomous architecture prototype**이다.

즉, AGI를 주장하는 결과물이 아니라, AGI에 필요한 구조를 실험하기 위한 플랫폼이다.

---

## 6. Research Questions

SAGE가 검증하고 싶은 질문은 다음과 같다.

```text
여러 Organ이 협력하면 단일 모델과 다른 방식의 지능이 만들어질 수 있는가?

상태와 기억이 다음 판단을 실제로 개선할 수 있는가?

시스템이 스스로 결과를 보고 다음 실험을 제안할 수 있는가?

제한된 안전 조건 안에서 자율적인 연구 루프를 만들 수 있는가?

창발처럼 보이는 행동이 실제로 반복 가능하고 재사용 가능한가?
```

---

## 7. Current Architecture Direction

SAGE는 다음 방향으로 발전하고 있다.

```text
v1.x
- Organ routing
- Evidence-based selection
- Sparse routing
- Adaptive compute
- Organ lifecycle

v2.0
- Emergent reflection loop
- Memory proposal
- Reflection policy calibration

v2.0.1
- Configurable reflection policy

v2.0.2
- Autonomous experiment planner

v2.0.3
- Reflection stability probe

v2.0.4
- Safe continuous runtime
```

현재 SAGE는 결과를 읽고, 반성하고, 다음 실험을 제안하고, 제한된 안전 조건 안에서 반복 실행되는 방향으로 발전하고 있다.

---

## 8. Safety and Autonomy Policy

SAGE의 자율성은 단계적으로 확장한다.

현재 허용되는 행동은 다음과 같다.

```text
결과 JSON 읽기
reflection 생성
다음 실험 제안
허용된 benchmark 실행
results/에 JSON 저장
logs/에 기록
experiments/inbox/에 제안 저장
configs/generated/에 임시 config 생성
STOP 파일 감지 후 안전 종료
```

현재 금지되는 행동은 다음과 같다.

```text
네트워크 접근
임의 shell 명령 실행
파일 삭제
core 코드 자동 수정
organ 자동 삭제
organ 자동 비활성화
memory 자동 승인
git commit 자동 실행
git push 자동 실행
```

SAGE는 중요한 변경 전에 사람의 승인을 받아야 한다.

---

## 9. Memory Policy

SAGE의 memory는 바로 장기 기억으로 확정되지 않는다.

SAGE는 먼저 memory proposal을 생성하고, 이를 `memory/inbox`에 저장한다.

```text
memory/inbox
- SAGE가 기억할 가치가 있다고 제안한 후보

memory/approved
- 사람이 승인한 기억

memory/rejected
- 사람이 거절한 기억
```

SAGE는 memory를 자동 승인하지 않는다.

기억은 반드시 사람의 검토를 거쳐야 한다.

---

## 10. Emergence Policy

SAGE는 흥미로운 행동이 나왔다고 해서 바로 창발이라고 부르지 않는다.

어떤 행동이 의미 있으려면 다음 조건을 만족해야 한다.

```text
반복 가능해야 한다.
다음 판단을 개선해야 한다.
다른 상황에서도 재사용 가능해야 한다.
단순한 랜덤 변화가 아니어야 한다.
결과가 기록되고 검증 가능해야 한다.
```

창발은 주장하는 것이 아니라 검증하는 것이다.

---

## 11. Personal Principle

이 프로젝트는 AI가 대신 만들어주는 프로젝트가 아니다.

AI의 도움을 받을 수는 있지만, 방향성, 판단, 실험 해석, 최종 결정은 내가 책임진다.

SAGE는 내가 직접 질문하고, 설계하고, 검증하면서 성장시키는 연구 프로젝트이다.

---

## 12. One-Sentence Definition

SAGE는 AGI를 성급하게 주장하기 위한 시스템이 아니라, 여러 Organ이 상태, 기억, 반성, 계획, 검증을 통해 협력하는 인공 생태계형 지능 구조를 실험하기 위한 modular autonomous architecture prototype이다.
