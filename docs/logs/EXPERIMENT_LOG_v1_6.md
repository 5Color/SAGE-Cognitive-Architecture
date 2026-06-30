# SAGE Experiment Log v1.6

## Overview

SAGE v1.6 라인은 기존의 독립 benchmark 파일 중심 구조에서 벗어나, 실험 구조를 더 깔끔하게 분리하고 재사용 가능한 형태로 정리한 버전이다.

v1.6의 핵심 목표는 성능을 무작정 올리는 것이 아니라, 다음 세 가지를 검증하는 것이다.

```text
1. SAGE Core를 독립적인 실험 엔진으로 분리할 수 있는가?
2. config 기반으로 실험을 재현 가능하게 만들 수 있는가?
3. evidence routing을 sparse routing으로 확장하여 연산 효율성을 검증할 수 있는가?
```

v1.6 라인은 다음 단계들로 구성된다.

```text
v1.6   - Core Refactor Skeleton
v1.6.1 - Config Runner
v1.6.2 - Anti-Leak Task Plugin Migration
v1.6.3 - Sparse Evidence Router
v1.6.4 - Sparse Multi-seed Validation
```

## Guardrail

SAGE v1.6의 결과는 AGI 증명이 아니다.

현재 결과는 작은 synthetic benchmark에서 state, energy, memory, organ routing, sparse compute control이 작동할 가능성을 보여주는 architecture evidence다.

---

# v1.6 - Core Refactor Skeleton

## Goal

v1.6의 목표는 SAGE를 하나의 큰 benchmark 파일이 아니라, 재사용 가능한 core 구조로 분리하는 것이다.

기존에는 각 benchmark 파일 안에 환경, organ, router, metric, 실행 루프가 섞여 있었다.
v1.6에서는 이를 `sage_core`로 분리하여 이후 실험들이 같은 core engine 위에서 동작하도록 만드는 것을 목표로 했다.

## Implementation

추가된 핵심 구조:

```text
sage_core/
  __init__.py
  state.py
  base.py
  engine.py
  metrics.py
  registry.py
```

역할:

```text
state.py
= SAGEState, OrganResult 정의

base.py
= BaseOrgan, BaseRouter, BaseEnvironment, BaseMetric 인터페이스 정의

engine.py
= SAGEEngine 실행 루프 정의

metrics.py
= BasicRunMetric 정의

registry.py
= 이후 config 기반 component 등록을 위한 registry 구조
```

테스트 benchmark:

```text
benchmarks/benchmark_v1_6_core_refactor_smoke.py
```

## Result

실행 결과:

```text
accuracy=1.0000
avg_reward=1.0000
steps=32
final_energy=2.6302
organ_usage={'parity_organ': 32, 'bias_organ': 32}
```

## Interpretation

v1.6 smoke test는 성공했다.

SAGEEngine이 environment, router, organs, metric을 받아 하나의 공통 loop로 실행될 수 있음을 확인했다.
state transition, energy update, organ usage logging도 정상 작동했다.

특히 `final_energy=2.6302`는 reward에 따라 state energy가 변화하고 있음을 보여준다.

## Conclusion

v1.6은 SAGE를 실험 코드 더미에서 core framework 구조로 이동시킨 첫 단계다.

---

# v1.6.1 - Config Runner

## Goal

v1.6.1의 목표는 실험을 Python 코드 수정 없이 JSON config로 실행할 수 있게 만드는 것이다.

기존에는 새 실험을 만들 때마다 benchmark 파일을 복사하고 수정해야 했다.
v1.6.1에서는 `run_experiment.py`를 추가하여, config 파일만 바꿔도 organ, router, environment, metric을 바꿀 수 있도록 했다.

## Implementation

추가 파일:

```text
run_experiment.py

benchmarks/tasks/
  __init__.py
  parity_task.py

benchmarks/configs/
  parity_smoke.json
  parity_smoke_bias_only.json
```

실행 방식:

```powershell
python run_experiment.py --config benchmarks/configs/parity_smoke.json
python run_experiment.py --config benchmarks/configs/parity_smoke_bias_only.json
```

## Result

### Parity Smoke

```text
accuracy=1.0000
avg_reward=1.0000
steps=32
final_energy=2.6302
organ_usage={'parity_organ': 32, 'bias_organ': 32}
```

### Bias Only Negative Control

```text
accuracy=0.5000
avg_reward=0.0000
steps=6
final_energy=0.0366
organ_usage={'bias_organ': 6}
```

## Interpretation

`parity_smoke`는 specialist organ이 사용될 때 정확도 100%를 달성했다.

`bias_only`는 항상 0을 예측하는 약한 organ만 사용하므로 정확도 50%에 머물렀다.
또한 energy가 낮아져 `steps=6`에서 조기 종료되었다.

이는 두 가지를 확인한다.

```text
1. JSON config 변경만으로 실험 동작이 바뀐다.
2. SAGEEngine의 energy-based termination이 실제로 작동한다.
```

## Conclusion

v1.6.1은 config 기반 실험 실행 구조를 성공적으로 도입했다.

이후 실험은 task plugin과 config를 추가하는 방식으로 확장할 수 있게 되었다.

---

# v1.6.2 - Anti-Leak Task Plugin Migration

## Goal

v1.6.2의 목표는 기존 v1.5.x 계열 anti-leak evidence routing 실험을 새 core/config 구조로 이식하는 것이다.

v1.5.x에서는 anti-leak routing 실험이 독립 benchmark 파일로 존재했다.
v1.6.2에서는 이를 `benchmarks/tasks/anti_leak_routing_task.py`로 분리하고, config runner로 실행할 수 있도록 했다.

## Implementation

추가 파일:

```text
benchmarks/tasks/anti_leak_routing_task.py

benchmarks/configs/
  anti_leak_random.json
  anti_leak_keytype.json
  anti_leak_evidence.json
  anti_leak_oracle.json
```

비교 router:

```text
RandomAntiLeakRouter
KeyTypeAntiLeakRouter
EvidenceAntiLeakRouter
FamilyOracleAntiLeakRouter
```

Task families:

```text
memory
linear_rule
threshold_rule
planning
```

## Result

```text
Random        accuracy 0.6000
KeyType       accuracy 0.5750
Evidence      accuracy 0.9125
Oracle        accuracy 0.9625
```

Evidence Router family accuracy:

```text
linear_rule     0.975
memory          1.000
planning        0.825
threshold_rule  0.850
```

Evidence Router family top organ:

```text
linear_rule     -> linear_organ
memory          -> memory_organ
planning        -> planner_organ
threshold_rule  -> threshold_organ
```

## Interpretation

v1.6.2는 anti-leak evidence routing task가 새 core/config 구조에서도 정상 작동함을 보여주었다.

Evidence Router는 Random과 KeyType baseline보다 훨씬 높은 정확도를 달성했다.
또한 family별 top organ이 각 task family에 맞게 선택되었다.

이는 organ specialization이 새 core 구조에서도 유지된다는 신호다.

## Limitation

v1.6.2의 Evidence Router는 모든 step에서 모든 organ을 실행한다.

```text
memory_organ      160
linear_organ      160
threshold_organ   160
planner_organ     160
```

즉 성능은 좋지만 compute-efficient sparse routing은 아직 검증되지 않았다.

## Conclusion

v1.6.2는 새 core/config 구조에서 anti-leak routing을 재현한 successful migration 버전이다.

다음 목표는 모든 organ을 실행하지 않고 필요한 organ만 선택하는 sparse routing이다.

---

# v1.6.3 - Sparse Evidence Router

## Goal

v1.6.3의 목표는 Evidence Router가 모든 organ을 실행하는 문제를 줄이고, support evidence를 가볍게 분석한 뒤 상위 k개 organ만 실행하도록 만드는 것이다.

즉, 정확도뿐 아니라 organ call 수와 compute saving을 측정하기 시작한 첫 버전이다.

## Core Idea

기존 v1.6.2 구조:

```text
모든 organ 실행 -> 결과 중 가장 좋은 organ 선택
```

v1.6.3 구조:

```text
support evidence 분석 -> top-k organ 선택 -> 선택된 organ만 실행
```

비교 대상:

```text
Full Evidence
Sparse Top1
Sparse Top2
Oracle
```

## Metrics

추가된 효율 지표:

```text
organ_calls
avg_organs_per_step
compute_saving_vs_full
sparse_efficiency_score
```

## Result

```text
Full Evidence  accuracy 0.9125 | organ/step 4.0 | saving 0%
Sparse Top1    accuracy 0.8563 | organ/step 1.0 | saving 75%
Sparse Top2    accuracy 0.9000 | organ/step 2.0 | saving 50%
Oracle         accuracy 0.9625 | organ/step 1.0 | saving 75%
```

## Interpretation

Sparse Top1은 organ을 step마다 1개만 실행하면서 75%의 compute saving을 달성했다.
하지만 accuracy는 Full Evidence보다 낮아졌다.

Sparse Top2는 가장 좋은 trade-off를 보였다.

```text
Full Evidence accuracy: 0.9125
Sparse Top2 accuracy:   0.9000

정확도 감소: 0.0125
연산 절약:   50%
```

Sparse Top2는 Full Evidence의 정확도를 거의 유지하면서 organ call을 절반으로 줄였다.

## Family Accuracy - Sparse Top2

```text
linear_rule     0.975
memory          0.925
planning        0.925
threshold_rule  0.775
```

Family top organ도 모두 적절하게 유지되었다.

```text
linear_rule     -> linear_organ
memory          -> memory_organ
planning        -> planner_organ
threshold_rule  -> threshold_organ
```

## Conclusion

v1.6.3은 SAGE가 정확도 중심 routing에서 efficiency-aware sparse routing으로 이동한 첫 버전이다.

특히 Sparse Top2는 정확도와 연산 절약 사이에서 매우 좋은 균형을 보였다.

---

# v1.6.4 - Sparse Multi-seed Validation

## Goal

v1.6.3은 seed 0 하나에서 좋은 결과를 보였다.
v1.6.4의 목표는 이 결과가 특정 seed의 우연인지, 여러 seed에서도 안정적으로 반복되는지 확인하는 것이다.

## Implementation

추가 파일:

```text
benchmarks/benchmark_v1_6_4_sparse_multiseed.py
README_v1_6_4.md
```

실행 명령:

```powershell
python -m benchmarks.benchmark_v1_6_4_sparse_multiseed --seeds 0 1 2 3 4 5 6 7 8 9 --episodes-per-family 80
```

Output:

```text
results/v1_6_4_sparse_multiseed_validation.json
```

## Result

```text
full_evidence  | acc 0.9491±0.0143 | org/step 4.00±0.00 | saving 0.00±0.00 | eff 0.6643±0.0100
sparse_top1    | acc 0.8456±0.0137 | org/step 1.00±0.00 | saving 0.75±0.00 | eff 0.8169±0.0096
sparse_top2    | acc 0.8969±0.0124 | org/step 2.00±0.00 | saving 0.50±0.00 | eff 0.7778±0.0087
oracle         | acc 0.9734±0.0075 | org/step 1.00±0.00 | saving 0.75±0.00 | eff 0.9064±0.0053
```

## Interpretation

v1.6.4는 sparse routing 결과가 seed 0의 우연이 아님을 확인했다.

Sparse Top2는 여러 seed에서도 평균 accuracy 0.8969를 기록했고, organ/step 2.0을 유지했다.
즉 organ 호출량을 50% 줄이면서도 안정적인 성능을 보였다.

Sparse Top1은 accuracy는 더 낮지만, 75% compute saving 덕분에 non-oracle 중 가장 높은 efficiency score를 보였다.

## Mode Interpretation

제품 또는 온디바이스 관점에서는 다음과 같이 해석할 수 있다.

```text
Full Evidence = 고정확도 모드
Sparse Top2   = 균형 모드
Sparse Top1   = 저전력 모드
Oracle        = 이상적인 상한선
```

## Conclusion

v1.6.4는 v1.6.3의 sparse routing 결과를 multi-seed로 검증한 버전이다.

이 결과를 통해 SAGE는 단순히 정확도만 보는 구조에서 벗어나, 정확도와 연산량 사이의 trade-off를 측정하는 단계로 이동했다.

---

# Overall v1.6 Conclusion

SAGE v1.6 전체의 핵심 성과는 다음과 같다.

```text
1. SAGE Core를 독립 구조로 분리했다.
2. JSON config 기반 실험 runner를 만들었다.
3. anti-leak routing task를 새 구조로 이식했다.
4. Evidence Router가 새 구조에서도 정상 작동함을 확인했다.
5. Sparse Evidence Router를 통해 organ call을 줄이는 실험에 성공했다.
6. Multi-seed validation으로 sparse routing 결과가 우연이 아님을 확인했다.
```

v1.6의 가장 중요한 변화는 SAGE가 단순 성능 실험을 넘어서 compute-aware routing 실험으로 이동했다는 점이다.

## Key Result

```text
Sparse Top2:
accuracy 0.8969 ± 0.0124
avg organs per step 2.00
compute saving 50%
```

이 결과는 SAGE가 정확도를 어느 정도 유지하면서 연산량을 줄일 수 있음을 보여준다.

## Limitation

아직 SAGE는 스스로 Top1, Top2, Full Evidence mode를 선택하지 않는다.
현재는 사람이 config로 mode를 고정한다.

또한 현재 실험은 synthetic benchmark이며, 실제 환경이나 더 다양한 held-out task family에서 검증된 것은 아니다.

## Next Step - v1.7

v1.7의 목표는 Adaptive Compute Mode Router다.

현재 구조:

```text
Sparse Top1 = 항상 1개 organ 실행
Sparse Top2 = 항상 2개 organ 실행
Full        = 항상 4개 organ 실행
```

v1.7 목표:

```text
확신이 높을 때    -> Top1
확신이 중간일 때  -> Top2
확신이 낮을 때    -> Full Evidence
energy가 낮을 때  -> 더 sparse한 mode 선호
```

즉 v1.7은 SAGE가 현재 상태와 confidence에 따라 compute mode를 스스로 선택할 수 있는지 검증하는 단계다.

## One-line Summary

SAGE v1.6 transformed the project from a collection of benchmark scripts into a config-driven experimental framework and showed that sparse organ routing can reduce compute while preserving useful accuracy on synthetic anti-leak tasks.
