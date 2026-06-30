# SAGE Experiment Log

## v0.1 - Initial Prototype

### Status

* Git 커밋으로는 남기지 못했다.
* 실행 결과 기록을 기반으로 회고를 작성했다.

### Result

* ToySocialEnv에서 첫 학습에 성공했다.
* acc ≈ 0.938
* reward ≈ 0.937
* Final organ energy: `[0.1, 10.0, 10.0, 0.1]`
* selected organs가 거의 `[1, 2]`로 고정되었다.
* 테스트 케이스 대부분에서 정답 예측에는 성공했다.

### Interpretation

* SAGE-v0의 기본 학습 루프는 정상적으로 작동했다.
* `best_action()` 규칙을 모델이 학습하는 데에는 성공했다.
* 그러나 Energy Router가 특정 organ만 계속 선택하는 문제가 발생했다.
* 초반에 선택된 organ이 reward를 얻고, energy가 증가하면 다시 같은 organ이 선택되는 양의 피드백 루프가 생겼다.

### Problem

* Organ collapse가 발생했다.
* organ 1, 2만 살아남고 organ 0, 3은 거의 사용되지 않았다.
* energy 값이 `[0.1, 10.0, 10.0, 0.1]`처럼 극단적으로 벌어졌다.

### Conclusion

* 행동 학습은 성공했다.
* organ 생태계는 실패했다.
* Router collapse / Organ collapse 문제가 실제로 확인되었다.
* 다음 버전에서는 energy bias 감소, load balancing loss, energy clamp 조정이 필요하다.

## v0.2 - Stabilized Router Energy Update

### Result

* acc ≈ 0.935
* selected organs가 이전보다 다양해졌다.
* Final organ energy: `[5.0, 5.0, 5.0, 5.0]`

### Interpretation

* v0.1의 organ collapse 문제는 일부 완화되었다.
* 특정 organ만 독점하던 현상은 줄어들었다.
* 그러나 모든 organ energy가 최대치로 포화되었다.

### Problem

* Energy overcharge가 발생했다.
* 모든 organ이 5.0에 도달하여 energy 값의 의미가 약해졌다.
* 생태계적 항상성이라기보다는 전체 organ이 과충전되는 구조가 되었다.

### Conclusion

* organ 다양성은 개선되었다.
* energy 시스템은 아직 불안정하다.
* 다음 버전에서는 energy를 1.0 근처로 안정화하는 항상성 구조가 필요하다.

## v0.3 - Homeostatic Energy Update

### Result

* acc ≈ 0.937
* 테스트 케이스 10개 전부 정답을 예측했다.
* selected organs가 다양하게 분산되었다.
* Final organ energy: `[0.4, 0.4, 0.4, 0.4]`

### Interpretation

* v0.1의 organ collapse는 완화되었다.
* v0.2의 energy overcharge 문제는 사라졌다.
* 그러나 모든 organ energy가 최저값으로 방전되었다.

### Problem

* Energy depletion이 발생했다.
* reward advantage가 거의 0에 가까운 반면 activation cost는 계속 빠졌다.
* 성공해도 energy를 충분히 회복하지 못하고, 사용 비용만 누적되어 모든 organ이 0.4로 떨어졌다.

### Conclusion

* 행동 학습은 성공했다.
* organ 다양성도 성공했다.
* 그러나 energy 항상성 설계는 아직 실패했다.
* v0.4에서는 reward_mean이 아니라 정답/오답 기반으로 선택된 organ의 energy를 업데이트해야 한다.

### Next Plan

* reward 평균 기반 energy update를 제거한다.
* 정답/오답 기반 energy update를 적용한다.
* 정답에 기여한 organ은 강화한다.
* 오답에 기여한 organ은 약화한다.
* 전체 organ은 1.0 근처로 천천히 회복시킨다.
* 목표 energy 범위는 대략 0.8~1.3 사이다.

## v0.4 - Correctness-based Energy Homeostasis

### Result

* acc ≈ 0.930
* reward ≈ 0.935
* 테스트 케이스 10개 전부 정답을 예측했다.
* Final organ energy: `[1.22, 1.23, 1.19, 1.21]`
* selected organs가 다양하게 분산되었다.

### Interpretation

* v0.1의 organ collapse 문제가 완화되었다.
* v0.2의 energy overcharge 문제가 해결되었다.
* v0.3의 energy depletion 문제가 해결되었다.
* energy가 1.0 근처에서 안정적으로 유지되었다.

### Conclusion

* 행동 학습에 성공했다.
* organ 선택 다양성이 유지되었다.
* energy 항상성 안정화에 성공했다.
* SAGE-v0.4는 첫 번째 안정 버전으로 볼 수 있다.

### Remaining Problem

* organ별 전문화가 아직 강하게 검증되지 않았다.
* 현재 ToySocialEnv가 단순해서 모든 organ이 비슷한 일을 해도 높은 정확도가 나올 수 있다.
* v0.5에서는 baseline 비교 또는 continual environment 실험이 필요하다.

## v0.5 - Planned Baseline Comparison

### Goal

* SAGE-v0.4가 단일 MLP보다 구조적으로 의미가 있는지 비교한다.
* 단순히 acc만 보는 것이 아니라 reward per compute, organ usage, 추론 비용까지 비교한다.

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

SAGE가 단일 MLP나 일반 MoE보다 더 적은 계산으로 비슷하거나 더 좋은 적응 성능을 보이는가?

## v0.5 - Baseline Benchmark

### Result

- SingleMLP: acc 0.9294, reward 0.8413, ms/batch 0.4521, reward/Mparam 16.0090
- BasicMoE: acc 0.9244, reward 0.8413, ms/batch 2.1971, reward/Mparam 4.4894
- SAGE-v0.4: acc 0.9125, reward 0.8221, ms/batch 2.5151, reward/Mparam 2.6979

### Interpretation

- 정적 ToySocialEnv에서는 SingleMLP가 가장 효율적이었다.
- BasicMoE와 SAGE는 구조가 복잡해서 계산 비용이 증가했다.
- SAGE의 energy homeostasis와 organ usage는 유지되었지만, 단순 규칙 분류 문제에서는 구조적 이점이 드러나지 않았다.

### Conclusion

- SAGE는 단순 정적 분류 문제에서는 비효율적이다.
- SAGE의 강점은 규칙 변화, 지속학습, 기억, 불확실성, energy 조절이 필요한 환경에서 검증해야 한다.
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
- Energy update를 켠 SAGE-v0.4는 SAGE-NoEnergy보다 약간 낫지만 차이는 크지 않았다.
- TinyMLP와 SingleMLP가 가장 효율적이었다.
- 현재 환경에서는 SAGE의 복잡한 organ routing 구조가 계산 비용 대비 뚜렷한 이점을 만들지 못했다.

### Problem

- Self-State가 매 batch마다 초기화되어 실제 지속 상태로 작동하지 않았다.
- Replay Memory가 없어 이전 phase를 보존하지 못했다.
- phase 변화 감지 기능이 없었다.
- Energy는 안정화되었지만 전문화나 적응 속도 향상으로 강하게 연결되지 않았다.

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
- 그러나 avg_adapt_gain은 전체 모델 중 가장 높았다.
- Persistent State와 Replay Memory를 함께 적용했을 때 SAGE의 적응 상승폭이 가장 크게 나타났다.
- 이는 SAGE 구조가 최종 정적 성능보다 환경 변화에 적응하는 과정에서 강점을 가질 가능성을 보여준다.

### Problem

- 현재 모델 입력에는 phase 정보가 없다.
- 같은 observation이라도 phase에 따라 정답 행동이 달라질 수 있어, 모델 입장에서는 부분 관측 문제가 된다.
- Persistent Self-State는 생겼지만 현재 phase의 규칙 변화를 명시적으로 추론하는 Context State는 아직 부족하다.
- SAGE-v0.7은 적응성은 좋지만 최종 성능과 계산 효율은 아직 부족하다.

### Conclusion

- v0.7은 부분 성공이다.
- SAGE가 평균 정확도 1등을 하지는 못했지만 adapt gain 1등을 달성했다.
- State + Replay 조합은 SAGE의 적응성을 강화한다는 실험 신호가 나왔다.
- v0.8에서는 Context State, phase-change detection, uncertainty gate를 추가해야 한다.

## v0.8 - Raw Context Input

### Result

- SAGE-v0.7-StateReplay: avg_acc 0.6805, reward 0.5402, adapt 0.1620
- SAGE-v0.8-Context: avg_acc 0.6598, reward 0.5088, adapt 0.1613

### Interpretation

- raw context를 observation에 직접 붙였지만 평균 정확도와 reward가 하락했다.
- adapt gain은 거의 유지되었지만 전체 성능 개선으로 이어지지 않았다.
- phase_2에서는 성능이 개선되었지만 phase_0, phase_1, phase_3에서는 성능이 떨어졌다.

### Problem

- recent_acc, recent_loss, phase_shift_score, uncertainty가 현재 규칙을 명확히 알려주지 못했다.
- context가 action 판단단에 직접 섞이면서 noise로 작동했을 가능성이 있다.
- context를 observation에 단순 결합하기보다 router의 organ selection을 조절하는 gate로 사용해야 한다.

### Conclusion

- v0.8은 실패에 가까운 실험이다.
- 하지만 context를 어떤 방식으로 넣으면 안 되는지 확인했다.
- v0.9에서는 Context-Gated Router를 실험한다.

## v1.2 - Organ Specialization Metrics

### Goal

- SAGE organ들이 phase별로 실제 전문화되는지 측정한다.
- 이 버전의 목적은 avg_acc나 reward를 끌어올리는 것이 아니라 진단 지표를 만드는 것이다.
- `usage_mean`은 4개 organ 분포의 평균이 구조적으로 0.25 근처가 되므로 보조 지표로만 둔다.

### Implementation

- `benchmark_v1_2_organ_specialization.py`를 추가했다.
- 비교 대상:
  - SAGE-v0.7-StateReplay
  - SAGE-v1.1-AdaptiveGate-base0.10
- Seeds: `[0, 1, 2, 3, 4]`
- CLI controls:
  - `--train-steps`
  - `--eval-batches`
- Output:
  - `results/v1_2_organ_specialization_benchmark.json`

### Metrics

- 기존 성능/적응 지표를 유지했다.
  - avg_acc
  - reward
  - adapt
  - phase_accuracy
- organ usage 지표를 추가했다.
  - usage_vector
  - usage_std
  - usage_entropy
  - usage_max
  - usage_min
  - phase_usage_vector
  - phase_usage_entropy
- energy 지표를 추가했다.
  - final_energy_vector
  - energy_std
  - energy_max
  - energy_min
  - energy_range
- `[0, 1]` 범위의 `specialization_score`를 추가했다.
  - phase usage entropy가 낮을수록 점수가 오른다.
  - phase별 top organ이 다양할수록 점수가 오른다.
  - global one-organ collapse는 패널티를 받는다.

### Smoke Test

Command:

```bash
python benchmark_v1_2_organ_specialization.py --train-steps 400 --eval-batches 5
```

Local note: 프로젝트 `.venv` launcher가 사라진 Python 설치 경로를 가리키고 있어, 다른 local Python 3.11 실행 파일에 SAGE `.venv` site-packages와 torch DLL path를 붙여 실행했다.

Result:

- SAGE-v0.7-StateReplay: avg_acc 0.6616 +/- 0.0209, reward 0.5153 +/- 0.0280, adapt 0.0404 +/- 0.0048, usage_entropy 0.9189 +/- 0.0786, energy_std 0.0335 +/- 0.0115, specialization_score 0.1338 +/- 0.0162
- SAGE-v1.1-AdaptiveGate-base0.10: avg_acc 0.6664 +/- 0.0244, reward 0.5212 +/- 0.0327, adapt 0.0476 +/- 0.0091, usage_entropy 0.8316 +/- 0.0840, energy_std 0.0739 +/- 0.0213, specialization_score 0.1479 +/- 0.0115

### Full Run

Command:

```bash
python benchmark_v1_2_organ_specialization.py --train-steps 1200 --eval-batches 15
```

Result:

- SAGE-v0.7-StateReplay: avg_acc 0.6824 +/- 0.0101, reward 0.5444 +/- 0.0141, adapt 0.1352 +/- 0.0069, usage_entropy 0.9808 +/- 0.0128, energy_std 0.0152 +/- 0.0082, specialization_score 0.1414 +/- 0.0487
- SAGE-v1.1-AdaptiveGate-base0.10: avg_acc 0.6838 +/- 0.0061, reward 0.5440 +/- 0.0068, adapt 0.1395 +/- 0.0108, usage_entropy 0.8978 +/- 0.0780, energy_std 0.0280 +/- 0.0155, specialization_score 0.1600 +/- 0.0395

### Interpretation

- v1.1-AdaptiveGate-base0.10은 v0.7 대비 avg_acc/reward를 거의 그대로 유지하면서 adapt를 약간 개선했다.
- v1.1은 v0.7보다 usage_entropy가 낮고 energy_std가 높아서, adaptive gate가 더 강한 organ preference 신호를 만든 것으로 보인다.
- 그러나 대표 run의 phase usage vector는 phase별로 여전히 매우 비슷하다. 예시 seed 0 v1.1:
  - phase_0: `[0.1010, 0.1656, 0.2578, 0.4755]`
  - phase_1: `[0.1115, 0.1651, 0.2479, 0.4755]`
  - phase_2: `[0.1026, 0.1786, 0.2375, 0.4812]`
  - phase_3: `[0.1042, 0.1708, 0.2516, 0.4734]`
- 현재 결론은 SAGE v1.1에 약한 organ preference는 있지만, 강한 phase-specialized organ role은 아직 확인되지 않았다는 것이다.
- 다음 연구 방향은 one-organ collapse를 막으면서 명시적인 phase/role differentiation pressure 또는 context-diversity regularizer를 추가하는 것이다.

## v1.3 - AGI Readiness Probe

### Goal

- AGI 관련 architecture 축에 대한 구체적인 evidence board를 만든다.
- 이것은 AGI claim이 아니다. 현재 SAGE core에 어떤 구성 요소가 부족한지 측정한다.
- 다루는 축:
  - diverse tasks
  - long-term memory
  - self-goal setting
  - world model
  - planning
  - language-action grounding
  - fast rule inference
  - organ specialization

### Implementation

- `benchmark_v1_3_agi_readiness.py`를 추가했다.
- 두 agent를 평가했다.
  - `SAGE-v1.3-NeuralCore`: auxiliary next-observation/reward prediction을 가진 SAGE context-gated neural core.
  - `SAGE-v1.3-CognitiveScaffold`: 같은 neural core에 explicit memory, symbolic planner, language-action parser, two-example rule inference를 붙인 구조.
- Output:
  - `results/v1_3_agi_readiness_benchmark.json`

### Tasks

- `social_rules`: social/risk/action rule selection.
- `language_action`: command token to action grounding.
- `static_memory`: key-action recall.
- `planning`: one-step grid planning under blocked directions.
- `world_model`: choose action from predicted resource/threat/energy dynamics.
- `self_goal`: choose action from safety/energy/social goal priorities.

### Smoke Test

Command:

```bash
python benchmark_v1_3_agi_readiness.py --train-steps 200 --eval-batches 4
```

Result:

- SAGE-v1.3-NeuralCore: agi_readiness_score 0.3025 +/- 0.0433, task_diversity_score 0.3896 +/- 0.0410, long_memory_score 0.2453 +/- 0.0713, planning_score 0.3016 +/- 0.0727, language_action_score 0.2352 +/- 0.0596, organ_specialization_score 0.3067 +/- 0.1120
- SAGE-v1.3-CognitiveScaffold: agi_readiness_score 0.8149 +/- 0.0132, task_diversity_score 0.8561 +/- 0.0126, long_memory_score 1.0000 +/- 0.0000, planning_score 1.0000 +/- 0.0000, language_action_score 1.0000 +/- 0.0000, organ_specialization_score 0.3067 +/- 0.1120

### Full Run

Command:

```bash
python benchmark_v1_3_agi_readiness.py --train-steps 800 --eval-batches 10
```

Result:

- SAGE-v1.3-NeuralCore: agi_readiness_score 0.3553 +/- 0.0328, task_diversity_score 0.4644 +/- 0.0377, long_memory_score 0.2931 +/- 0.1271, planning_score 0.3672 +/- 0.1054, language_action_score 0.3056 +/- 0.0769, organ_specialization_score 0.3924 +/- 0.0109
- SAGE-v1.3-CognitiveScaffold: agi_readiness_score 0.8366 +/- 0.0087, task_diversity_score 0.8954 +/- 0.0162, long_memory_score 1.0000 +/- 0.0000, planning_score 1.0000 +/- 0.0000, language_action_score 1.0000 +/- 0.0000, organ_specialization_score 0.3924 +/- 0.0109

### Interpretation

- 현재 neural SAGE core만으로는 AGI 관련 축을 충분히 만족하지 못한다. long memory, planning, language-action grounding이 특히 약하다.
- explicit cognitive module을 추가하면 memory, planning, language grounding, fast rule inference 점수가 크게 개선된다.
- 이는 SAGE가 neural organ만 있는 구조가 아니라 memory, planner, world model, language/action interface를 포함한 multi-component organism으로 가야 한다는 방향을 지지한다.
- 이 결과는 다음 architecture path에 대한 evidence이지 AGI proof가 아니다. 다음 proof target은 더 넓은 open-ended task transfer와 hand-coded scaffolding 감소를 요구해야 한다.

## v1.4 - Rule Transfer + Memory Probe

### Goal

- 다음 AGI 관련 핵심 evidence인 rapid new-rule inference와 durable episodic memory를 검증한다.
- 고정 task accuracy를 넘어서, 각 episode에서 support examples를 제공한 뒤 query generalization을 평가한다.
- 이것은 scaffold probe이며 AGI claim이 아니다.

### Implementation

- `benchmark_v1_4_rule_transfer_memory.py`를 추가했다.
- Pure Python benchmark로 작성했다. torch dependency가 없으므로 현재 깨진 project `.venv` launcher 문제를 피할 수 있다.
- Output:
  - `results/v1_4_rule_transfer_memory_benchmark.json`
- 비교 대상:
  - `RandomBaseline`
  - `EpisodicMemoryOnly`
  - `RuleTransferMemoryPlanner`

### Task Families

- `episodic_memory`: support examples 이후 exact key-action recall.
- `affine_rule`: support examples로부터 modular affine rules를 추론한다.
- `threshold_rule`: support examples로부터 threshold/action split을 추론한다.
- `language_action`: command words를 actions에 매핑한다.
- `grid_planning`: blocked cells가 있는 grid에서 goal을 향한 one-step move를 선택한다.
- `world_dynamics`: support transitions로부터 action transition deltas를 추론하고 이를 이용해 planning한다.

### Organ Routing

- 네 개의 scaffold organ에 대한 role usage를 추적했다.
  - memory_organ
  - algebra_organ
  - concept_organ
  - planner_organ
- one-organ collapse를 penalize하도록 organ specialization scoring을 수정했다. 낮은 entropy만으로는 충분하지 않고, task families 전반의 top-organ diversity가 필요하다.

### Smoke Test

Command:

```bash
python benchmark_v1_4_rule_transfer_memory.py --episodes 20 --queries-per-episode 8
```

Result:

- RandomBaseline: agi_transfer_score 0.1578 +/- 0.0129, task_diversity 0.1548 +/- 0.0160, long_memory 0.1650 +/- 0.0267, fast_rule_inference 0.1443 +/- 0.0281, planning 0.1537 +/- 0.0206, organ_specialization 0.1656 +/- 0.0000
- EpisodicMemoryOnly: agi_transfer_score 0.4924 +/- 0.0076, task_diversity 0.5637 +/- 0.0085, long_memory 1.0000 +/- 0.0000, fast_rule_inference 0.6650 +/- 0.0092, planning 0.4175 +/- 0.0483, organ_specialization 0.1656 +/- 0.0000
- RuleTransferMemoryPlanner: agi_transfer_score 0.9916 +/- 0.0006, task_diversity 1.0000 +/- 0.0000, long_memory 1.0000 +/- 0.0000, fast_rule_inference 1.0000 +/- 0.0000, planning 1.0000 +/- 0.0000, organ_specialization 0.9413 +/- 0.0041

### Full Run

Command:

```bash
python benchmark_v1_4_rule_transfer_memory.py --episodes 120 --queries-per-episode 18
```

Result:

- RandomBaseline: agi_transfer_score 0.1670 +/- 0.0031, task_diversity 0.1669 +/- 0.0038, long_memory 0.1683 +/- 0.0078, fast_rule_inference 0.1652 +/- 0.0055, planning 0.1661 +/- 0.0065, organ_specialization 0.1656 +/- 0.0000
- EpisodicMemoryOnly: agi_transfer_score 0.5134 +/- 0.0029, task_diversity 0.5851 +/- 0.0032, long_memory 1.0000 +/- 0.0000, fast_rule_inference 0.6677 +/- 0.0047, planning 0.4043 +/- 0.0165, organ_specialization 0.1656 +/- 0.0000
- RuleTransferMemoryPlanner: agi_transfer_score 0.9890 +/- 0.0001, task_diversity 1.0000 +/- 0.0000, long_memory 1.0000 +/- 0.0000, fast_rule_inference 1.0000 +/- 0.0000, planning 0.9999 +/- 0.0002, organ_specialization 0.9227 +/- 0.0008

### Interpretation

- Exact memory만으로도 long-memory recall은 해결할 수 있지만, general cognitive architecture가 되기에는 부족하다. 이 방식은 하나의 organ으로 collapse되고 planning/general rule transfer에서 약하다.
- `RuleTransferMemoryPlanner` scaffold는 fast-rule induction, memory, planning, language-action grounding, world-dynamics inference가 explicit organ으로 있을 때 SAGE가 지향하는 방향이 가능하다는 점을 보여준다.
- v1.3보다 더 강한 architectural decomposition evidence다. world dynamics를 hidden episode metadata가 아니라 support transitions로부터 추론하기 때문이다.
- 여전히 AGI proof는 아니다. 이것은 다음 SAGE architecture가 이 organ들을 외부 hand-coded helper로 두는 대신 neural ecosystem 내부로 통합해야 한다는 controlled proof다.
- 다음 target은 neural SAGE router가 언제 어떤 organ을 호출할지 학습하게 만들고, scaffold가 prewritten family-specific rules에 의존할 수 없는 held-out task families에서 검증하는 것이다.

## v1.5 - Evidence-Based Organ Router Probe

### Goal

- v1.4에서는 `RuleTransferMemoryPlanner`가 task family를 알고 있는 상태에서 적절한 organ을 사용했다.
- v1.5의 목적은 family label 없이 support examples만 보고 어떤 organ을 호출할지 선택하는 것이다.
- 즉, 사람이 memory/algebra/concept/planner organ을 지정하는 scaffold에서 evidence-based organ selection으로 한 단계 이동한다.
- 이것은 neural SAGE router가 내부적으로 organ call policy를 학습하기 전의 중간 검증이다.

### Implementation

- `benchmark_v1_5_evidence_router.py`를 추가했다.
- 기존 `benchmark_v1_4_rule_transfer_memory.py`의 episode generator와 task family를 재사용했다.
- Output:
  - `results/v1_5_evidence_router_benchmark.json`
- 비교 대상:
  - `RandomOrganRouter`
  - `EvidenceRouter`
  - `FamilyOracleRouter`

### Router Design

- `RandomOrganRouter`는 organ을 무작위로 선택한다.
- `EvidenceRouter`는 hidden family label을 보지 않는다.
- `EvidenceRouter`는 각 candidate organ을 support subset에 fit한 뒤, 남은 support subset을 얼마나 잘 설명하는지로 organ을 고른다.
- `FamilyOracleRouter`는 task family label을 알고 있는 upper bound다.
- candidate organs:
  - memory_organ
  - algebra_organ
  - concept_organ
  - planner_organ

### Smoke Test

Command:

```bash
python benchmark_v1_5_evidence_router.py --episodes 20 --queries-per-episode 8
```

Result:

- RandomOrganRouter: evidence_router_score 0.5731 +/- 0.0240, task_diversity 0.6687 +/- 0.0150, fast_rule_inference 0.7419 +/- 0.0323, planning 0.5325 +/- 0.0323, organ_specialization 0.3956 +/- 0.1127, route_confidence 0.2500 +/- 0.0000
- EvidenceRouter: evidence_router_score 0.7776 +/- 0.0096, task_diversity 0.8069 +/- 0.0139, fast_rule_inference 0.6619 +/- 0.0244, planning 0.5175 +/- 0.0774, organ_specialization 0.5504 +/- 0.0141, route_confidence 0.6844 +/- 0.0093
- FamilyOracleRouter: evidence_router_score 1.0000 +/- 0.0000, task_diversity 1.0000 +/- 0.0000, fast_rule_inference 1.0000 +/- 0.0000, planning 1.0000 +/- 0.0000, organ_specialization 1.0000 +/- 0.0000, route_confidence 1.0000 +/- 0.0000

### Full Run

Command:

```bash
python benchmark_v1_5_evidence_router.py --episodes 120 --queries-per-episode 18
```

Result:

- RandomOrganRouter: evidence_router_score 0.5811 +/- 0.0239, task_diversity 0.6877 +/- 0.0160, fast_rule_inference 0.7515 +/- 0.0182, planning 0.5263 +/- 0.0176, organ_specialization 0.3367 +/- 0.1094, route_confidence 0.2500 +/- 0.0000
- EvidenceRouter: evidence_router_score 0.7750 +/- 0.0035, task_diversity 0.8059 +/- 0.0054, fast_rule_inference 0.6669 +/- 0.0048, planning 0.5013 +/- 0.0336, organ_specialization 0.5416 +/- 0.0071, route_confidence 0.6844 +/- 0.0054
- FamilyOracleRouter: evidence_router_score 1.0000 +/- 0.0000, task_diversity 1.0000 +/- 0.0000, fast_rule_inference 1.0000 +/- 0.0000, planning 1.0000 +/- 0.0000, organ_specialization 1.0000 +/- 0.0000, route_confidence 1.0000 +/- 0.0000

### Interpretation

- EvidenceRouter는 family label 없이도 RandomOrganRouter보다 높은 evidence_router_score를 달성했다.
- 이는 support examples만으로도 어떤 cognitive organ을 호출해야 하는지 일부 추론할 수 있음을 보여준다.
- 그러나 EvidenceRouter는 FamilyOracleRouter와 큰 격차가 있다. 즉 organ selection은 아직 완전하지 않다.
- 특히 planning score가 낮게 남아 있어, support evidence만으로 planner organ을 안정적으로 선택하거나 검증하는 방식이 부족하다.
- fast_rule_inference에서는 RandomOrganRouter가 높게 보이는데, 이는 일부 rule task에서 여러 organ이 support/query를 부분적으로 맞출 수 있기 때문이다. 따라서 v1.5는 단순 score뿐 아니라 route_confidence와 organ_specialization을 함께 해석해야 한다.
- 현재 결론은 “organ을 explicit module로 두는 것” 다음 단계로 “evidence 기반 organ routing”이 가능하다는 것이다.
- 다음 target은 이 symbolic EvidenceRouter를 neural SAGE router의 학습 target으로 바꾸고, support evidence에서 route confidence를 예측하도록 만드는 것이다.


## v1.5.1 - Neural Imitation Router

### Goal

- v1.5에서는 `EvidenceRouter`가 family label 없이 support evidence만 보고 organ을 선택할 수 있음을 확인했다.
- v1.5.1의 목표는 이 symbolic EvidenceRouter를 teacher로 사용해, neural router가 organ selection policy를 학습할 수 있는지 검증하는 것이다.
- 이것은 hand-coded evidence scoring에서 neural organ routing으로 이동하는 중간 단계다.

### Result

- RandomOrganRouter: evidence_router_score 0.4242 +/- 0.0150, task_diversity 0.5587 +/- 0.0039, fast_rule_inference 0.6274 +/- 0.0050, planning 0.5638 +/- 0.0072, organ_specialization 0.2927 +/- 0.0880, route_confidence 0.2500 +/- 0.0000, imitation_acc 0.2525 +/- 0.0020
- EvidenceRouter: evidence_router_score 0.7908 +/- 0.0035, task_diversity 0.8742 +/- 0.0058, fast_rule_inference 0.8695 +/- 0.0050, planning 0.7532 +/- 0.0139, organ_specialization 0.9198 +/- 0.0043, route_confidence 0.3945 +/- 0.0010, imitation_acc 0.9331 +/- 0.0026
- NeuralImitationRouter: evidence_router_score 0.8917 +/- 0.0100, task_diversity 0.8846 +/- 0.0111, fast_rule_inference 0.9455 +/- 0.0324, planning 0.7084 +/- 0.0045, organ_specialization 0.9702 +/- 0.0170, route_confidence 0.9310 +/- 0.0135, imitation_acc 0.9103 +/- 0.0127
- FamilyOracleRouter: evidence_router_score 0.9626 +/- 0.0009, task_diversity 1.0000 +/- 0.0000, fast_rule_inference 1.0000 +/- 0.0000, planning 1.0000 +/- 0.0000, organ_specialization 1.0000 +/- 0.0000, route_confidence 1.0000 +/- 0.0000, imitation_acc 0.7755 +/- 0.0052

### Interpretation

- NeuralImitationRouter는 EvidenceRouter의 organ selection policy를 약 91% 수준으로 모방했다.
- NeuralImitationRouter는 RandomOrganRouter보다 훨씬 높은 evidence_router_score와 task_diversity를 달성했다.
- 이는 symbolic evidence-based organ routing이 neural router로 압축될 수 있음을 보여준다.
- NeuralImitationRouter는 fast_rule_inference에서 EvidenceRouter보다 높았지만, planning에서는 낮았다.
- 따라서 rule routing은 개선되었지만 planner organ 선택은 아직 안정적이지 않다.
- route_confidence가 0.9310으로 매우 높아 overconfidence 가능성이 있다.
- 다음 단계에서는 confidence calibration, anti-leak feature setting, held-out task family 검증이 필요하다.

### Conclusion

- v1.5.1은 성공적인 중간 단계다.
- SAGE는 hand-coded organ selection에서 neural organ routing으로 이동할 수 있다는 실험 신호를 얻었다.
- 그러나 아직 일반적 organ routing 능력으로 해석하기에는 이르며, 더 엄격한 anti-leak benchmark가 필요하다.

## v1.5.2 - Anti-Leak Evidence Router

### Goal

* v1.5.1에서는 `NeuralImitationRouter`가 `EvidenceRouter`의 organ selection policy를 학습할 수 있음을 확인했다.
* 그러나 이전 실험들은 key type이나 query shape가 task family를 암시할 가능성이 있었다.
* v1.5.2의 목표는 이러한 easy fingerprint를 줄인 anti-leak setting에서도 evidence-based organ routing이 가능한지 검증하는 것이다.
* 이것은 SAGE가 단순히 key 모양을 외우는 것이 아니라, support evidence를 기반으로 적절한 organ을 고를 수 있는지 확인하는 실험이다.

### Terminology

* `Anti-Leak`: 정답 단서 누수 방지.
* `Evidence`: support/query에서 얻은 증거.
* `Fingerprint`: 너무 쉽게 task family를 알아낼 수 있는 표면적 단서.
* `KeyTypeRouter`: key 모양만 보고 organ을 고르는 baseline.
* `FamilyOracleRouter`: task family label을 알고 있는 upper bound.

### Result

* RandomOrganRouter: anti_leak_score 0.3900 +/- 0.0204, task_diversity 0.4493 +/- 0.0080, rule_generalization 0.3648 +/- 0.0147, planning 0.5266 +/- 0.0054, organ_specialization 0.2083 +/- 0.1484, route_confidence 0.2500 +/- 0.0000, oracle_gap 0.5214 +/- 0.0221, keytype_gain -0.0483 +/- 0.0211
* KeyTypeRouter: anti_leak_score 0.4383 +/- 0.0028, task_diversity 0.5059 +/- 0.0038, rule_generalization 0.3151 +/- 0.0086, planning 0.3934 +/- 0.0168, organ_specialization 0.1656 +/- 0.0000, route_confidence 0.2500 +/- 0.0000, oracle_gap 0.4731 +/- 0.0048, keytype_gain 0.0000 +/- 0.0000
* AntiLeakEvidenceRouter: anti_leak_score 0.7728 +/- 0.0329, task_diversity 0.8203 +/- 0.0109, rule_generalization 0.6441 +/- 0.0219, planning 0.9985 +/- 0.0020, organ_specialization 0.7537 +/- 0.1689, route_confidence 0.4256 +/- 0.0058, oracle_gap 0.1386 +/- 0.0291, keytype_gain 0.3345 +/- 0.0336
* FamilyOracleRouter: anti_leak_score 0.9114 +/- 0.0054, task_diversity 0.8228 +/- 0.0108, rule_generalization 0.6457 +/- 0.0215, planning 1.0000 +/- 0.0000, organ_specialization 1.0000 +/- 0.0000, route_confidence 1.0000 +/- 0.0000, oracle_gap 0.0000 +/- 0.0000, keytype_gain 0.4731 +/- 0.0048

### Interpretation

* AntiLeakEvidenceRouter는 KeyTypeRouter보다 anti_leak_score가 크게 높았다.
* 이는 key type이나 query shape 같은 쉬운 fingerprint 없이도 support evidence 기반 organ routing이 가능함을 보여준다.
* AntiLeakEvidenceRouter는 task_diversity에서 FamilyOracleRouter에 거의 접근했다.
* 특히 planning score가 0.9985로 매우 높아, planner organ 선택이 안정적으로 이루어진 것으로 보인다.
* rule_generalization은 FamilyOracleRouter와 거의 비슷하지만 전체적으로 0.64 수준이므로, rule 계열 task 자체는 아직 개선 여지가 있다.
* organ_specialization은 RandomOrganRouter와 KeyTypeRouter보다 크게 높았지만, FamilyOracleRouter보다는 낮았다.
* 현재 결론은 v1.5.2가 SAGE의 evidence-based organ routing 가능성을 가장 강하게 보여준 실험이라는 것이다.

### Conclusion

* v1.5.2는 성공적인 anti-leak routing benchmark다.
* SAGE는 단순 key fingerprint가 아니라 support evidence를 이용해 적절한 organ을 선택할 수 있다는 신호를 얻었다.
* 다음 단계는 이 anti-leak evidence routing을 neural router로 학습시키거나, 더 다양한 held-out task family에서 검증하는 것이다.





