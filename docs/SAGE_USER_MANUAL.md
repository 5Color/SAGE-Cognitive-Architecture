# SAGE User Manual

## 0. 이 문서의 목적

이 문서는 SAGE를 실행하거나 개발하기 전에 보는 사용 설명서이다.

SAGE는 단순한 챗봇이 아니라, 여러 Organ이 상태, 기억, 반성, 계획, 검증을 통해 협력하는 인공 생태계형 지능 아키텍처 실험 프로젝트이다.

이 문서는 다음을 빠르게 다시 기억하기 위해 만든다.

```text
SAGE가 무엇인지
각 폴더가 무슨 역할인지
각 버전이 무엇을 했는지
어떤 명령어를 써야 하는지
memory를 어떻게 승인/거절하는지
계속 실행할 때 무엇을 조심해야 하는지
```

---

## 1. SAGE 한 줄 정의

SAGE는 AGI를 목표로 하지만, 현재 결과를 AGI라고 성급하게 주장하지 않고, 여러 Organ이 상태, 기억, 반성, 계획, 검증을 통해 협력하는 인공 생태계형 지능 구조를 실험하기 위한 modular autonomous architecture prototype이다.

---

## 2. SAGE의 핵심 원칙

```text
목표는 AGI다.
하지만 현재 SAGE가 AGI라고 주장하지 않는다.

큰 목표를 가진다.
하지만 작은 실험으로 검증한다.

창발처럼 보이는 행동이 나와도 바로 믿지 않는다.
반복 가능성, 재사용성, 다음 판단 개선 여부를 확인한다.

위험한 자율성보다 통제된 자율성을 먼저 만든다.
자동 삭제, 자동 승인, 자동 git 작업은 금지한다.
```

---

## 3. 주요 폴더 의미

```text
sage_core/
- SAGE의 핵심 로직이 들어가는 곳
- router, lifecycle, reflection, memory review 같은 핵심 모듈

sage_runtime/
- 실제 실행 루프가 들어가는 곳
- reflection loop, continuous runtime, memory review runtime 등

tools/
- 사람이 실행하는 CLI 명령어 모음
- python tools/xxx.py 형태로 실행

benchmarks/
- SAGE가 잘 작동하는지 확인하는 테스트/벤치마크

configs/
- 실행 설정 파일
- policy, runtime, memory review 설정 등이 들어감

configs/generated/
- stability probe나 runtime이 생성한 임시 config
- 보존 가치는 있지만 active config는 아님

docs/
- 프로젝트 설명, 버전 문서, 아이디어 노트, 사용 설명서

docs/versions/
- 버전별 README 모음

docs/logs/
- 사람이 보존하기로 한 실험 로그

docs/ideas/
- 아이디어 노트

results/
- 실행 결과 JSON 저장소
- benchmark 결과, runtime summary, cleanup proposal 등이 저장됨

experiments/inbox/
- SAGE가 제안한 다음 실험 후보
- 아직 사람이 승인하지 않은 실험 제안

memory/inbox/
- SAGE가 기억할 가치가 있다고 제안한 기억 후보
- 아직 승인된 장기 기억이 아님

memory/approved/
- 사람이 승인한 기억
- 장기적으로 SAGE가 참고할 수 있는 기억

memory/rejected/
- 사람이 거절한 기억

memory/review/
- 기억 승인/거절 감사 기록

runtime_state/
- SAGE 실행 상태 저장소
- 보통 git에 넣지 않음

runtime_control/
- STOP 파일 같은 제어 파일 저장소

logs/
- runtime이 생성한 로그
- 보통 자동 생성물이라 git에 직접 넣지 않음
```

---

## 4. 주요 개념 Glossary

### Organ

SAGE 내부의 전문 기능 단위이다.

예시:

```text
Critic Organ
- 성급한 주장이나 오류를 비판

Planner Organ
- 다음 행동이나 실험 방향 제안

Memory Organ
- 기억 후보 생성 또는 기억 관련 판단

Curiosity Organ
- 새로운 가능성이나 탐색적 아이디어 제안

Observer Organ
- 현재 상태나 결과를 관찰
```

---

### Router

여러 Organ 중 어떤 Organ을 사용할지 고르는 장치이다.

```text
입력 / 상태 / evidence
↓
Router
↓
선택된 Organ
```

---

### Aggregator

여러 Organ이 만든 후보를 점수화하고 최종 선택하는 장치이다.

```text
여러 후보
↓
score 계산
↓
best candidate 선택
```

---

### State

SAGE의 현재 상태 정보이다.

예시:

```text
현재 cycle
마지막 결과
runtime mode
organ registry 상태
failure count
```

---

### Memory

SAGE가 장기적으로 참고할 수 있는 정보이다.

중요한 점:

```text
memory/inbox는 기억 후보
memory/approved는 승인된 기억
memory/rejected는 거절된 기억
```

SAGE는 memory를 자동 승인하지 않는다.

---

### Reflection

SAGE가 자신의 결과나 상태를 보고 다시 생각하는 과정이다.

예시:

```text
이번 결과가 의미 있는가?
창발이라고 볼 수 있는가?
다음 실험이 필요한가?
어떤 Organ이 더 적절했는가?
```

---

### Emergence

창발처럼 보이는 행동을 의미한다.

하지만 SAGE에서는 바로 창발이라고 주장하지 않는다.

의미 있는 행동으로 보려면:

```text
반복 가능해야 한다.
다음 판단을 개선해야 한다.
다른 상황에서도 재사용 가능해야 한다.
단순한 랜덤 변화가 아니어야 한다.
```

---

### Lifecycle

Organ의 상태를 관리하는 시스템이다.

예시:

```text
core_active
active
active_monitor
monitor
```

중요한 점:

```text
Organ을 자동 삭제하지 않는다.
Organ을 자동 비활성화하지 않는다.
사람 승인 없이 lifecycle 변경을 강제하지 않는다.
```

---

### Safe Continuous Runtime

SAGE를 제한된 안전 조건 안에서 반복 실행하는 루프이다.

금지:

```text
네트워크 접근
임의 shell 실행
파일 삭제
core 코드 자동 수정
memory 자동 승인
git commit/push
```

---

### Runtime Guard

continuous runtime이 폭주하지 않도록 감시하는 장치이다.

감시 항목:

```text
cycle count
failure count
memory inbox growth
results file growth
experiments inbox growth
STOP file
```

---

### Cleanup Retention Policy

SAGE가 쌓인 결과 파일을 보고 정리 후보를 제안하는 도구이다.

중요한 점:

```text
삭제하지 않는다.
이동하지 않는다.
archive도 자동 실행하지 않는다.
정리 후보만 제안한다.
```

---

### Memory Review Tool

memory/inbox 후보를 사람이 보고 approve/reject 하는 도구이다.

```text
list
show
approve
reject
report
```

---

## 5. 버전별 의미

```text
v0.x
- 초기 organ routing / energy / baseline 실험

v1.6
- core refactor
- SAGE 구조를 모듈화하기 시작

v1.6.1
- config runner
- json config 기반 실행

v1.6.2
- anti-leak task plugin
- shortcut 방지 실험 시작

v1.6.3
- sparse evidence router
- 모든 organ을 쓰지 않고 일부만 선택

v1.6.4
- sparse routing multiseed validation
- seed 운빨인지 검증

v1.7
- adaptive compute router
- 상황에 따라 Top1 / Top2 / Full Evidence 선택

v1.7.1
- lifecycle calibration metrics
- organ 상태와 추천 정책 측정

v1.8
- organ lifecycle manager
- organ registry 기반 상태 관리

v1.9
- runtime node
- state 저장, memory proposal, safe idle 시작

v2.0
- emergent reflection loop
- 여러 reflection organ 후보 중 하나를 선택

v2.0.1
- reflection policy config
- 코드 수정 없이 config로 정책 변경

v2.0.2
- autonomous experiment planner
- 결과를 읽고 다음 실험 후보 제안

v2.0.3
- reflection stability probe
- selected organ이 작은 변화에도 안정적인지 확인

v2.0.4
- safe continuous runtime
- 제한된 whitelist 안에서 반복 실행

v2.0.5
- runtime guard and long-run monitor
- 장시간 실행을 위한 감시 장치

v2.0.6
- cleanup retention policy
- results/configs/experiments/memory/logs 정리 후보 제안

v2.1
- memory review tool
- memory/inbox 후보를 사람이 승인/거절
```

---

## 6. 자주 쓰는 명령어

### Git 상태 확인

```powershell
git status --short
git log --oneline --decorate -8
git diff --cached --name-status
```

---

### v2.0.5 Guarded Runtime 실행

```powershell
python tools/run_guarded_continuous_runtime.py --max-cycles 10 --interval 10
```

멈추기:

```powershell
New-Item -ItemType File runtime_control\STOP -Force
```

STOP 파일 제거:

```powershell
Remove-Item runtime_control\STOP
```

---

### v2.0.6 Cleanup Proposal 실행

```powershell
python tools/propose_cleanup_retention.py
python -m benchmarks.benchmark_v2_0_6_cleanup_retention_policy
```

---

### v2.1 Memory Review

목록 보기:

```powershell
python tools/review_memory.py list
```

자세히 보기:

```powershell
python tools/review_memory.py show --id <실제_candidate_id>
```

주의:

```text
<실제_candidate_id>를 그대로 치면 안 된다.
예: 46ecd8332db9
```

승인:

```powershell
python tools/review_memory.py approve --id 46ecd8332db9 --reason "approved by human" --confirm
```

거절:

```powershell
python tools/review_memory.py reject --id db060bedac04 --reason "duplicate" --confirm
```

보고서:

```powershell
python tools/review_memory.py report
```

테스트:

```powershell
python -m benchmarks.benchmark_v2_1_memory_review_tool
```

---

## 7. 실행 전 체크리스트

SAGE를 실행하기 전에 확인한다.

```text
1. git status --short 확인
2. STOP 파일이 남아 있지 않은지 확인
3. memory/inbox가 너무 많지 않은지 확인
4. results/가 너무 많이 쌓였는지 확인
5. 최근 benchmark가 passed: True인지 확인
6. git add . 사용하지 않기
```

---

## 8. 절대 하지 말아야 할 것

```text
git add .
git clean -fd
git reset --hard
memory/inbox 전체 삭제
results 전체 삭제
configs/generated 전체 삭제
STOP 파일 남겨둔 채 runtime 실행
자동 memory approve
자동 git commit/push
```

---

## 9. 권장 작업 순서

일반 개발 순서:

```text
1. 새 기능 patch 적용
2. smoke test 실행
3. 결과 확인
4. 필요한 memory review
5. docs 업데이트
6. 필요한 파일만 git add
7. git diff --cached --name-status 확인
8. git commit
9. git tag
```

---

## 10. 현재 추천 로드맵

```text
v2.1
- Memory Review Tool

v2.2
- Local Control Panel UI

v2.3
- Local Chat Runtime

v2.4
- Korean Tokenization Probe

v2.5
- Korean State-Based Understanding

v2.6
- Dialectic Reflection Loop
```

---

## 11. 한 문장으로 기억하기

```text
SAGE는 혼자 마음대로 움직이는 AI가 아니라,
사람 승인 아래 기억하고, 반성하고, 실험하고, 성장하는 AGI 지향 인공 인지 생태계 실험 플랫폼이다.
```
