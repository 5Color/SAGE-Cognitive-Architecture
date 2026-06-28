\# SAGE Experiment Log



\## v0.1 - Initial Prototype



\### Status



\* Git 커밋으로는 남기지 못함

\* 실행 결과 기록을 기반으로 회고 작성



\### Result



\* ToySocialEnv에서 첫 학습 성공

\* acc 약 0.938

\* reward 약 0.937

\* Final organ energy: `\[0.1, 10.0, 10.0, 0.1]`

\* selected organs가 거의 `\[1, 2]`로 고정됨

\* 테스트 케이스 대부분 정답 예측 성공



\### Interpretation



\* SAGE-v0의 기본 학습 루프는 정상 작동함

\* `best\_action()` 규칙을 모델이 학습하는 데 성공함

\* 하지만 Energy Router가 특정 기관만 계속 선택하는 문제가 발생함

\* 초반에 선택된 기관이 reward를 얻고, 에너지가 증가하면서 다시 더 자주 선택되는 양의 피드백 루프가 생김



\### Problem



\* Organ collapse 발생

\* 기관 1, 2만 살아남고 기관 0, 3은 거의 사용되지 않음

\* 에너지 값이 `\[0.1, 10.0, 10.0, 0.1]`처럼 극단적으로 벌어짐



\### Conclusion



\* 행동 학습은 성공

\* 기관 생태계는 실패

\* Router collapse / Organ collapse 문제가 실제로 확인됨

\* 다음 버전에서는 energy bias 감소, load balancing loss, energy clamp 조정이 필요함



\## v0.2 - Stabilized Router Energy Update



\### Result



\* acc 약 0.935

\* selected organs가 이전보다 다양해짐

\* Final organ energy: `\[5.0, 5.0, 5.0, 5.0]`



\### Interpretation



\* v0.1의 organ collapse 문제는 일부 완화됨

\* 특정 기관만 독점하던 현상은 줄어듦

\* 하지만 모든 기관 에너지가 최대치로 포화됨



\### Problem



\* Energy overcharge 발생

\* 모든 기관이 5.0에 도달하여 energy 값의 의미가 약해짐

\* 생태계적 항상성보다는 전체 기관이 과충전되는 구조가 됨



\### Conclusion



\* 기관 다양화는 개선됨

\* 에너지 시스템은 아직 불안정함

\* 다음 버전에서는 energy를 1.0 근처로 회귀시키는 항상성 구조가 필요함



\## v0.3 - Homeostatic Energy Update



\### Result



\* acc 약 0.937

\* 테스트 케이스 10개 전부 정답

\* selected organs는 다양하게 분산됨

\* Final organ energy: `\[0.4, 0.4, 0.4, 0.4]`



\### Interpretation



\* v0.1의 organ collapse는 완화됨

\* v0.2의 energy overcharge 문제도 사라짐

\* 하지만 모든 기관 에너지가 최저값으로 방전됨



\### Problem



\* Energy depletion 발생

\* reward advantage가 거의 0에 가까워졌지만 activation cost는 계속 빠짐

\* 성공해도 에너지를 충분히 얻지 못하고, 사용 비용만 누적되어 모든 기관이 0.4로 떨어짐



\### Conclusion



\* 행동 학습은 성공

\* 기관 다양화도 성공

\* 하지만 energy 항상성 설계는 아직 실패

\* v0.4에서는 reward\_mean이 아니라 정답/오답 기반으로 선택된 기관의 energy를 업데이트해야 함



\## v0.4 - Planned Fix



\### Goal



\* acc 0.90 이상 유지

\* selected organs 다양성 유지

\* energy가 0.5 또는 5.0에 고정되지 않도록 안정화

\* 기관 에너지가 대략 0.8\~1.3 사이에서 움직이도록 설계



\### Planned Change



\* reward 평균 기반 energy update 제거

\* 정답/오답 기반 energy update 적용

\* 정답에 기여한 기관은 강화

\* 오답에 기여한 기관은 약화

\* 전체 기관은 1.0 근처로 천천히 회귀



\### Expected Result



\* 행동 정확도 유지

\* 기관 선택 다양성 유지

\* 에너지 포화/방전 방지

\* SAGE의 organ ecosystem이 더 안정적으로 작동하는지 확인



