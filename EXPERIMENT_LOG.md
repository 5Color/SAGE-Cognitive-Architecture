\## v0.3 - homeostatic energy update



\### Result

\- acc 약 0.937

\- 테스트 10개 전부 정답

\- selected organs는 다양하게 분산됨

\- Final organ energy: \[0.4, 0.4, 0.4, 0.4]



\### Interpretation

\- v0.1의 organ collapse는 완화됨

\- v0.2의 energy overcharge 문제도 사라짐

\- 하지만 activation cost가 reward advantage보다 커서 모든 기관 에너지가 최저값으로 방전됨



\### Conclusion

\- 행동 학습은 성공

\- 기관 다양화도 성공

\- energy 항상성 설계는 아직 실패

\- v0.4에서는 reward\_mean이 아니라 정답/오답 기반으로 선택된 기관의 energy를 업데이트해야 함

