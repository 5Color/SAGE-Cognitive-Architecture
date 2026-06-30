# SAGE v1.7 Plan - Adaptive Compute Mode Router

## Status

SAGE v1.6.x에서 core 구조와 config 기반 실험 시스템을 정리했다.

v1.6.3에서는 Sparse Evidence Router를 추가하여 organ 호출량을 줄이는 실험을 진행했다.
v1.6.4에서는 multi-seed validation을 통해 sparse routing 결과가 seed 0에만 의존한 우연이 아님을 확인했다.

## v1.6.4 Summary

### Result

* Full Evidence

  * accuracy: 0.9491 ± 0.0143
  * avg organs per step: 4.00
  * compute saving: 0%

* Sparse Top1

  * accuracy: 0.8456 ± 0.0137
  * avg organs per step: 1.00
  * compute saving: 75%

* Sparse Top2

  * accuracy: 0.8969 ± 0.0124
  * avg organs per step: 2.00
  * compute saving: 50%

* Oracle

  * accuracy: 0.9734 ± 0.0075
  * avg organs per step: 1.00
  * compute saving: 75%

### Interpretation

v1.6.4 confirmed that sparse routing is stable across multiple seeds.

Sparse Top2 preserved most of the Full Evidence accuracy while reducing organ calls by 50%.
Sparse Top1 showed lower accuracy, but achieved the best non-oracle efficiency score due to 75% compute saving.

This means SAGE moved from accuracy-only routing toward efficiency-aware routing.

## v1.7 Goal

v1.7의 목표는 고정된 Top1 / Top2 / Full Evidence 모드를 사람이 선택하는 것이 아니라, SAGE가 현재 상태에 따라 compute mode를 스스로 선택하도록 만드는 것이다.

즉, v1.7은 Adaptive Compute Mode Router를 검증하는 버전이다.

## Core Idea

현재 구조:

```text
Full Evidence = 항상 4개 organ 실행
Sparse Top2   = 항상 2개 organ 실행
Sparse Top1   = 항상 1개 organ 실행
```

v1.7 목표 구조:

```text
확신이 높을 때    → Top1
확신이 중간일 때  → Top2
확신이 낮을 때    → Full Evidence
energy가 낮을 때  → 더 sparse하게 실행
```

## Proposed Router

### AdaptiveSparseEvidenceRouter

이 router는 support evidence를 가볍게 분석한 뒤, organ 후보들의 score gap과 confidence를 계산한다.

예상 규칙:

```text
1. best_score가 매우 높고 second_score와 차이가 크면 Top1 사용
2. best_score와 second_score가 비슷하면 Top2 사용
3. 전체 score가 낮거나 불확실성이 높으면 Full Evidence 사용
4. energy가 낮으면 가능한 한 Top1 또는 Top2를 선호
```

## Metrics

v1.7에서는 단순 accuracy만 보지 않는다.

### Main Metrics

* accuracy
* avg_reward
* task_diversity
* organ_calls
* avg_organs_per_step
* compute_saving_vs_full
* sparse_efficiency_score

### New Metrics

* mode_usage

  * Top1을 몇 번 썼는가
  * Top2를 몇 번 썼는가
  * Full Evidence를 몇 번 썼는가

* confidence_gap

  * 1등 organ score와 2등 organ score 차이

* adaptive_efficiency_score

  * accuracy와 compute saving을 함께 반영한 점수

## Expected Comparison

v1.7에서는 다음 router들을 비교한다.

```text
Full Evidence Router
Sparse Top1 Router
Sparse Top2 Router
Adaptive Sparse Evidence Router
Family Oracle Router
```

## Success Criteria

v1.7이 성공했다고 보려면 다음 조건을 만족해야 한다.

```text
1. Adaptive Router가 Full Evidence보다 organ calls를 줄인다.
2. Adaptive Router가 Sparse Top1보다 accuracy가 높다.
3. Adaptive Router가 Sparse Top2와 비슷하거나 더 좋은 efficiency score를 보인다.
4. mode_usage가 한쪽으로 collapse되지 않는다.
```

가장 이상적인 결과:

```text
Adaptive Router accuracy ≈ Sparse Top2 이상
Adaptive Router compute saving ≈ 50% 이상
Adaptive Router mode usage = Top1 / Top2 / Full을 상황별로 다르게 사용
```

## Research Meaning

v1.7은 SAGE가 단순히 organ을 선택하는 수준을 넘어서, 현재 상황에 맞게 연산량 자체를 조절할 수 있는지 확인하는 단계다.

이것은 온디바이스 AI, CPU-friendly AI, 저전력 decision engine 방향에서 중요한 의미를 가진다.

## Guardrail

v1.7은 AGI 증명이 아니다.

이 실험은 작은 synthetic benchmark에서 SAGE의 adaptive compute control 가능성을 검증하는 단계다.
현재 결과는 architecture evidence이며, 일반 지능이나 AGI 달성을 의미하지 않는다.

## One-line Summary

SAGE v1.7 tests whether the system can adaptively choose between Top1, Top2, and Full Evidence routing to balance accuracy and compute efficiency.
