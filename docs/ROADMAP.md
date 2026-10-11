# 로드맵

지금까지 한 것: ETH/UCY 데이터 파이프라인(Social-GAN과 같은 샘플), ADE/FDE, 등속 모델 기준선, Vanilla LSTM encoder-decoder(증강 없음 · 회전 증강, 시드 3개), A단계 오차 분석, B단계 grid pooling. 자세한 내용은 [README](../README.md)에 있다.

모든 단계에 공통으로 적용하는 규칙은 다음과 같다.

- 평가는 같은 leave-one-out 5개 분할, 같은 ADE/FDE로 한다. AVG는 5개 장면의 단순 평균이다.
- 학습형 모델은 시드 0, 1, 2로 돌려 평균 ± 표준편차로 적는다.
- README와 docs에는 실제로 실행한 값만 적는다. 확인한 것과 추측은 나눠 쓴다.
- 단계가 끝나면 `docs/notes/<단계>.md`에 공부 노트를 쓴다(개념, 코드 위치, 결과 해석, 면접 질문 3개).

예상 소요는 이 PC(RTX 4070 Laptop 8 GB)에서 코드 작성, 학습, 정리를 합친 대략적인 작업 시간이다. 학습 시간은 지금 LSTM 기준(한 실행 5개 분할 약 7분)으로 어림했다.

| 단계 | 내용 | 상태 |
| :-: | :--- | :--- |
| A | 오차 분석: 오차 분해, heading 정규화, residual | 완료 ([분석](analysis.md), [노트](notes/A_analysis.md)) |
| B | 상호작용: 주변 보행자 정보, Social Pooling 간단 버전 | 완료 ([분석](interaction.md), [노트](notes/B_interaction.md)) |
| C | Transformer: 시간축 + 보행자 간 attention | 예정 |
| D | Multimodal: K=20 예측, best-of-20, 논문 표와 비교 | 예정 |
| E | Argoverse 2로 차량 확장 | 예정 |

---

## A. 오차 분석

**목표**: LSTM이 등속 모델을 언제 이기고 언제 지는지를 오차 분해로 확인한다. 진행 방향 정규화(heading frame)와 residual 예측이라는 두 가지 간단한 설계가 차이를 얼마나 줄이는지 잰다.

**할 일**

1. 오차 분해(등속 / LSTM 증강 없음 / LSTM 회전 증강, 시드 0): 예측 스텝별 오차, 방향 전환 각도별 ADE, 관측 속력별 ADE. hotel과 전체를 나란히 본다.
2. `--frame heading`: 마지막 관측 속도가 +x를 향하도록 돌려 입력하고, 예측을 원래 좌표로 되돌린다.
3. `--residual`: 등속 예측에 LSTM 보정량을 더한다. heading과 조합한 실험도 한다.

**완료 기준**

- [x] `docs/analysis.md`에 그림 3종(스텝별, 각도별, 속력별) × (hotel, 전체)과 해석이 있다
- [x] heading, residual, heading + residual을 각각 시드 3개로 돌린 결과가 README 표에 열로 들어가 있다
- [x] "hotel의 남은 차이가 어디서 오는가"에 대해 확인한 것과 추측이 나뉘어 적혀 있다
- [x] `docs/notes/A_analysis.md`

**예상 소요**: 4~6시간(학습 9회 약 30~40분 포함). 실제로 학습 9회는 3개씩 동시에 돌려 약 35분 걸렸다.

---

## B. 상호작용 (주변 보행자 정보, Social Pooling 간단 버전)

**목표**: 같은 프레임의 주변 보행자 정보를 넣어 단일 보행자 LSTM보다 나아지는지 확인한다. 특히 A에서 찾은 "등속 모델에 지는 상황"이 줄어드는지 본다.

**할 일**

1. 데이터: 같은 윈도우의 보행자들을 한 장면 배치로 묶는 collate를 만든다. `WindowMeta`(source_file, window_id)로 묶을 수 있다.
2. 간단한 기준선: 가장 가까운 이웃 k명의 상대 위치 · 상대 속도를 고정 길이 벡터로 붙인다.
3. Social Pooling 간단 버전: Social LSTM(Alahi et al., CVPR 2016)처럼 대상 보행자 주변 격자(예: 4×4, 칸 크기 0.5~1 m)에 이웃의 은닉 상태를 더해 넣는다. 먼저 인코더 끝에서 한 번만 pooling하는 버전으로 시작한다.
4. A와 같은 오차 분해를 다시 돌려, 이웃이 가까운 샘플과 혼자인 샘플로 나눠 비교한다.

**실제로 한 것** (2026-10-11, 요청 범위: heading + 회전 증강 기본, 이웃 위치 · 속도 grid pooling)

- 1번 대신, 샘플마다 같은 프레임 이웃의 위치 · 속도를 미리 뽑아 두는 방식(`extract_neighbors`)을 썼다. 장면 배치 collate는 이웃의 **은닉 상태**를 모을 때 필요하므로 C단계로 미룬다.
- 3번은 은닉 상태 대신 이웃의 위치 · 속도를 격자에 모으는 더 간단한 버전으로 했다.
- 결과: 이웃 없는 모델과 같은 수준(AVG ADE 0.531). 15° 이상 꺾는 샘플에서도 개선이 없었다. 자세한 내용은 [interaction.md](interaction.md)에 있다.

**완료 기준**

- [ ] 장면 배치 데이터 로더와 그 테스트 → C단계로 미룸
- [ ] 이웃 k명 기준선 → 하지 않음. grid pooling만 시드 3개
- [x] Social Pooling(grid pooling) 모델 시드 3개의 결과표, A단계 기준(각도 · 속력) 재분해
- [x] 이웃 수별 ADE 분석 그림
- [x] `docs/notes/B_interaction.md`

**예상 소요**: 8~12시간. 실제로는 과적합 진단과 설정 비교(시드 0, 4개 설정)를 포함해 학습에 약 1시간이 걸렸다.

---

## C. Transformer (시간축 + 보행자 간 attention)

**목표**: 시간축 self-attention 인코더로 LSTM을 대체하고, 보행자 간 attention을 더해 B의 pooling과 비교한다.

**할 일**

1. 시간축 Transformer: 관측 변위 7개 + 위치 인코딩 → Transformer encoder → 12스텝 변위. 디코더는 한 번에 12개를 내는 방식(non-autoregressive)과 autoregressive 방식을 비교한다.
2. 보행자 간 attention: 같은 프레임 보행자들의 시간축 임베딩끼리 attention하고, 거리가 먼 이웃은 마스킹한다(STAR, AgentFormer 계열의 간단화).
3. A에서 효과가 확인된 설계(heading frame, residual)를 그대로 가져가 공정하게 비교한다.

**완료 기준**

- [ ] 시간축만 쓰는 Transformer와 보행자 간 attention을 더한 버전, 각각 시드 3개의 결과표
- [ ] 파라미터 수 · 학습 시간 · ADE/FDE를 LSTM · Social Pooling과 한 표로 비교
- [ ] attention 가중치 시각화 예시 1~2개
- [ ] `docs/notes/C_transformer.md`

**예상 소요**: 10~14시간

---

## D. Multimodal (K=20 예측, best-of-20)

**목표**: 하나가 아니라 K개의 미래를 예측하고, 논문들이 쓰는 best-of-20 ADE/FDE로 Social GAN 등과 같은 기준에서 비교한다.

**할 일**

1. 다중 예측 방법 하나를 고른다. 후보: (a) K개 출력 head + winner-takes-all 손실, (b) CVAE(잠재 변수 샘플링), (c) GAN(Social GAN 방식, variety loss).
2. 평가: 샘플마다 K=20개 중 실제와 가장 가까운 예측의 ADE/FDE(minADE₂₀ / minFDE₂₀). 비교 기준선으로 등속 모델에 방향 · 속도 잡음을 넣어 20개를 만든 것도 잰다.
3. Social GAN, Trajectron++ 등의 논문 표 값을 **원 논문 표에서 직접 확인해** 옮기고, 평가 프로토콜 차이(샘플 규칙, K, 정규화)를 적는다.
4. best-of-K 지표의 한계(K가 크면 무작위 예측도 좋아 보임)를 실험으로 보여 준다. 예: K를 1, 5, 20으로 바꿔 등속 + 잡음 기준선과 비교한다.

**완료 기준**

- [ ] minADE₂₀ / minFDE₂₀ 평가 코드와 테스트
- [ ] 다중 예측 모델 시드 3개의 결과와 논문 표 비교(출처 표 번호 명시)
- [ ] K에 따른 지표 변화 그림
- [ ] `docs/notes/D_multimodal.md`

**예상 소요**: 10~15시간

---

## E. Argoverse 2 Motion Forecasting 일부로 차량 확장

**목표**: 보행자에서 만든 파이프라인을 차량 데이터로 넓힌다. 8 GB GPU와 노트북 디스크로 감당할 수 있는 부분집합만 쓴다.

**할 일**

1. 데이터: Argoverse 2 Motion Forecasting에서 train · val 일부(예: 수천~1만 시나리오)만 받는다. 받기 전에 라이선스와 이용 조건을 확인해 README에 적는다. 이 저장소에는 데이터를 넣지 않는다.
2. 문제 설정은 AV2 표준(과거 5 s, 미래 6 s, 10 Hz)을 따른다. focal agent만 예측한다.
3. 기준선: 등속 모델, A~C 중 가장 좋은 모델(지도 정보 없이). 지표는 AV2 표준(minADE₆, minFDE₆, miss rate)과 K=1 ADE/FDE.
4. 지도 정보(차선 중심선)를 간단히 넣는 실험: 가까운 차선 중심선 몇 개를 벡터로 인코딩해 붙인다(VectorNet 계열의 아주 작은 버전). 메모리가 허용하는 범위에서만 한다.
5. 보행자 대비 차량에서 등속 모델이 얼마나 강하고 약한지를 A와 같은 오차 분해로 비교한다.

**완료 기준**

- [ ] 부분집합 다운로드 스크립트(시나리오 수 · 디스크 사용량 명시)와 데이터 로더
- [ ] 등속, 지도 없는 학습 모델, (가능하면) 간단한 지도 인코딩 모델의 결과표. 공식 리더보드 값과는 데이터 규모가 달라 직접 비교하지 않는다고 명시
- [ ] 차량 vs 보행자 오차 분해 비교
- [ ] `docs/notes/E_argoverse2.md`

**예상 소요**: 15~25시간(데이터 준비에 따라 크게 달라짐)
