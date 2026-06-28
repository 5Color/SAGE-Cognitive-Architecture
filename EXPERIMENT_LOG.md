# SAGE Experiment Log

## v0.1 - Initial Prototype

### Status

* Git 커밋으로는 남기지 못함
* 실행 결과 기록을 기반으로 회고 작성

### Result

* ToySocialEnv에서 첫 학습 성공
* acc 약 0.938
* reward 약 0.937
* Final organ energy: `[0.1, 10.0, 10.0, 0.1]`
* selected organs가 거의 `[1, 2]`로 고정됨
* 테스트 케이스 대부분 정답 예측 성공

### Interpretation

* SAGE-v0의 기본 학습 루프는 정상 작동함
* `best_action()` 규칙을 모델이 학습하는 데 성공함
* 하지만 Energy Router가 특정 기관만 계속 선택하는 문제가 발생함
* 초반에 선택된 기관이 reward를 얻고, 에너지가 증가하면서 다시 더 자주 선택되는 양의 피드백 루프가 생김

### Problem

* Organ collapse 발생
* 기관 1, 2만 살아남고 기관 0, 3은 거의 사용되지 않음
* 에너지 값이 `[0.1, 10.0, 10.0, 0.1]`처럼 극단적으로 벌어짐

### Conclusion

* 행동 학습은 성공
* 기관 생태계는 실패
* Router collapse / Organ collapse 문제가 실제로 확인됨
* 다음 버전에서는 energy bias 감소, load balancing loss, energy clamp 조정이 필요함

## v0.2 - Stabilized Router Energy Update

### Result

* acc 약 0.935
* selected organs가 이전보다 다양해짐
* Final organ energy: `[5.0, 5.0, 5.0, 5.0]`

### Interpretation

* v0.1의 organ collapse 문제는 일부 완화됨
* 특정 기관만 독점하던 현상은 줄어듦
* 하지만 모든 기관 에너지가 최대치로 포화됨

### Problem

* Energy overcharge 발생
* 모든 기관이 5.0에 도달하여 energy 값의 의미가 약해짐
* 생태계적 항상성보다는 전체 기관이 과충전되는 구조가 됨

### Conclusion

* 기관 다양화는 개선됨
* 에너지 시스템은 아직 불안정함
* 다음 버전에서는 energy를 1.0 근처로 회귀시키는 항상성 구조가 필요함

## v0.3 - Homeostatic Energy Update

### Result

* acc 약 0.937
* 테스트 케이스 10개 전부 정답
* selected organs는 다양하게 분산됨
* Final organ energy: `[0.4, 0.4, 0.4, 0.4]`

### Interpretation

* v0.1의 organ collapse는 완화됨
* v0.2의 energy overcharge 문제도 사라짐
* 하지만 모든 기관 에너지가 최저값으로 방전됨

### Problem

* Energy depletion 발생
* reward advantage가 거의 0에 가까워졌지만 activation cost는 계속 빠짐
* 성공해도 에너지를 충분히 얻지 못하고, 사용 비용만 누적되어 모든 기관이 0.4로 떨어짐

### Conclusion

* 행동 학습은 성공
* 기관 다양화도 성공
* 하지만 energy 항상성 설계는 아직 실패
* v0.4에서는 reward_mean이 아니라 정답/오답 기반으로 선택된 기관의 energy를 업데이트해야 함

### Next Plan

* reward 평균 기반 energy update 제거
* 정답/오답 기반 energy update 적용
* 정답에 기여한 기관은 강화
* 오답에 기여한 기관은 약화
* 전체 기관은 1.0 근처로 천천히 회귀
* 목표 energy 범위는 대략 0.8~1.3 사이

## v0.4 - Correctness-based Energy Homeostasis

### Result

* acc 약 0.930
* reward 약 0.935
* 테스트 케이스 10개 전부 정답
* Final organ energy: `[1.22, 1.23, 1.19, 1.21]`
* selected organs가 다양하게 분산됨

### Interpretation

* v0.1의 organ collapse 문제 완화
* v0.2의 energy overcharge 문제 해결
* v0.3의 energy depletion 문제 해결
* energy가 1.0 근처에서 안정적으로 유지됨

### Conclusion

* 행동 학습 성공
* 기관 선택 다양성 유지
* energy 항상성 안정화 성공
* SAGE-v0.4는 첫 번째 안정 버전으로 볼 수 있음

### Remaining Problem

* 기관별 전문화가 아직 강하게 검증되지는 않음
* 현재 ToySocialEnv가 단순해서 모든 기관이 비슷한 역할을 해도 높은 정확도가 나올 수 있음
* v0.5에서는 baseline 비교 또는 continual environment 실험이 필요함

## v0.5 - Planned Baseline Comparison

### Goal

* SAGE-v0.4가 단일 MLP보다 구조적으로 의미 있는지 비교한다.
* 단순히 acc만 보는 것이 아니라 reward per compute, 기관 사용률, 추론 비용까지 비교한다.

### Planned Models

* SingleMLP
* BasicMoE
* SAGE-v0.4

### Metrics

* accuracy
* average reward
* active organ count
* inference time
* reward per compute
* organ usage distribution

### Expected Question

SAGE는 단일 MLP나 일반 MoE보다 더 적은 계산으로 비슷하거나 더 좋은 적응 성능을 보이는가?


## v0.5 - Baseline Benchmark

### Result

- SingleMLP: acc 0.9294, reward 0.8413, ms/batch 0.4521, reward/Mparam 16.0090
- BasicMoE: acc 0.9244, reward 0.8413, ms/batch 2.1971, reward/Mparam 4.4894
- SAGE-v0.4: acc 0.9125, reward 0.8221, ms/batch 2.5151, reward/Mparam 2.6979

### Interpretation

- 정적 ToySocialEnv에서는 SingleMLP가 가장 효율적이었다.
- BasicMoE와 SAGE는 구조가 복잡해서 계산 비용이 증가했다.
- SAGE는 energy homeostasis와 organ usage는 유지했지만, 단순 규칙 분류 문제에서는 구조적 이점이 드러나지 않았다.

### Conclusion

- SAGE는 단순 정적 분류 문제에서는 비효율적이다.
- SAGE의 장점은 규칙 변화, 지속학습, 기억, 불확실성, 에너지 조절이 필요한 환경에서 검증해야 한다.
- v0.6에서는 ContinualEnv를 도입한다.


## v0.6 - Continual Benchmark

### Result

- TinyMLP: avg_acc 0.6714, reward 0.5306, adapt 0.1125
- SingleMLP: avg_acc 0.6742, reward 0.5346, adapt 0.0956
- BasicMoE: avg_acc 0.6707, reward 0.5286, adapt 0.0920
- SAGE-NoEnergy: avg_acc 0.6642, reward 0.5205, adapt 0.1045
- SAGE-v0.4: avg_acc 0.6716, reward 0.5309, adapt 0.0989

### Interpretation

- SAGE-v0.4는 v0.6 continual benchmark에서 SingleMLP를 넘지 못했다.
- Energy update를 켠 SAGE-v0.4는 SAGE-NoEnergy보다 약간 나았지만, 차이가 크지 않았다.
- TinyMLP와 SingleMLP가 더 효율적이었다.
- 현재 환경에서는 SAGE의 복잡한 organ routing 구조가 계산 비용 대비 뚜렷한 이점을 만들지 못했다.

### Problem

- Self-State가 매 batch마다 초기화되어 실제 지속 상태로 작동하지 않았다.
- Replay Memory가 없어 이전 phase를 보존하지 못했다.
- Phase 변화 감지 기능이 없었다.
- Energy는 안정화됐지만, 전문화나 적응 속도 향상으로 강하게 연결되지는 않았다.

### Conclusion

- v0.6은 SAGE의 한계를 드러낸 실험이다.
- 현재 SAGE-v0.4는 단순 continual 환경에서도 아직 구조적 우위를 보이지 못했다.
- v0.7에서는 Persistent Self-State와 Replay Memory를 추가해야 한다.



## v0.7 - Persistent State + Replay Memory

### Result

- SingleMLP-Replay: avg_acc 0.6908, reward 0.5529, adapt 0.1194
- BasicMoE-Replay: avg_acc 0.6831, reward 0.5437, adapt 0.1285
- SAGE-v0.4: avg_acc 0.6655, reward 0.5242, adapt 0.1070
- SAGE-StateOnly: avg_acc 0.6727, reward 0.5329, adapt 0.1457
- SAGE-ReplayOnly: avg_acc 0.6792, reward 0.5400, adapt 0.1265
- SAGE-v0.7-StateReplay: avg_acc 0.6802, reward 0.5385, adapt 0.1701

### Interpretation

- SAGE-v0.7은 평균 정확도와 reward에서는 SingleMLP-Replay보다 낮았다.
- 하지만 avg_adapt_gain은 전체 모델 중 가장 높았다.
- Persistent State와 Replay Memory를 함께 적용했을 때 SAGE의 적응 상승폭이 가장 크게 나타났다.
- 이는 SAGE 구조가 최종 정적 성능보다는 환경 변화 후 적응 과정에서 장점을 가질 가능성을 보여준다.

### Problem

- 현재 모델 입력에는 phase 정보가 없다.
- 같은 observation이라도 phase에 따라 정답 행동이 달라질 수 있어 모델 입장에서는 부분 관측 문제로 보인다.
- Persistent Self-State는 생겼지만, 현재 phase나 규칙 변화를 명시적으로 추론하는 Context State는 아직 부족하다.
- SAGE-v0.7은 적응력은 좋지만, 최종 성능과 계산 효율은 아직 부족하다.

### Conclusion

- v0.7은 부분 성공이다.
- SAGE가 평균 정확도 1등을 하지는 못했지만, adapt gain 1등을 달성했다.
- State + Replay 조합이 SAGE의 적응성을 강화한다는 실험적 신호가 나왔다.
- v0.8에서는 Context State, phase-change detection, uncertainty gate를 추가해야 한다.
