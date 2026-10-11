# B단계 공부 노트: 주변 보행자 정보 (grid pooling)

> 한 줄 요약: 같은 프레임 이웃의 위치 · 속도를 격자로 넣었지만, 평균 성능은 이웃 없는 모델과 같았다(AVG ADE 0.531). 기대했던 "꺾는 보행자"에서도 좋아지지 않았다. 첫 설정은 학습 데이터를 외워 버렸다. "정보를 더 넣었는데 왜 안 좋아지나"를 진단하는 법을 배운 단계다.

자세한 수치와 그림은 [docs/interaction.md](../interaction.md)에 있다.

---

## 1. 개념을 쉬운 말로

### 왜 이웃 정보를 넣나

사람은 혼자 걷지 않는다. 앞사람을 피해 돌고, 일행과 나란히 걷고, 마주 오는 사람과 비켜 간다. A단계에서 LSTM이 등속 모델을 이긴 곳이 "15° 이상 꺾는 보행자"였으므로, 꺾는 이유가 주변 사람이라면 이웃 정보가 그 부분을 더 좋게 만들 것이라고 기대했다.

### grid pooling (격자 모으기)

이웃의 수는 샘플마다 다르다(0명~30명). 신경망은 고정 길이 입력을 원하므로, 대상 보행자 주변에 바둑판(격자)을 깔고 칸마다 숫자를 센다.

- 칸마다 "몇 명이 있나", "그 사람들이 나보다 평균적으로 어느 쪽으로 얼마나 빨리 움직이나(상대 속도)"를 넣는다.
- 바둑판을 일렬로 펴서(4×4×3 = 48개 숫자) 작은 신경망에 넣고, 그 결과를 내 움직임 임베딩 옆에 붙여 LSTM에 넣는다.
- heading frame을 쓰므로 바둑판도 내 진행 방향에 맞춰 돈다. "오른쪽 앞 칸에 사람이 있다"가 어느 장면에서나 같은 뜻이 된다.

원조인 **Social LSTM**(Alahi et al., 2016)은 이웃 사람의 위치 대신 이웃 사람의 **LSTM 은닉 상태**(그 사람의 과거 움직임 요약)를 칸에 더한다. 이웃끼리 서로의 정보를 주고받는 셈이라 더 강력하지만, 같은 장면 사람들을 한 배치로 묶어야 해서 구현이 복잡하다. 이번에는 위치 · 속도를 직접 쓰는 간단한 버전으로 했다.

### 과적합과 "지문" 효과

**과적합**은 모델이 일반 규칙 대신 학습 데이터 자체를 외우는 것이다. 신호는 두 가지다. train loss는 계속 내려가는데 val loss는 오른다. early stopping이 아주 이른 에폭을 고른다.

이번에 본 것은 입력이 고차원이고 샘플마다 고유하면 그 자체가 **지문**이 된다는 점이다. 8×8×3 = 192개 숫자로 된 격자가 7스텝이면 샘플마다 거의 유일한 패턴이 된다. 모델은 "이 패턴이면 이 미래"를 외울 수 있다.

**섞음 대조 실험(shuffle control)**: 이웃 정보를 다른 무작위 샘플의 것으로 바꿔 뜻을 없애고 똑같이 학습했다.

- 뜻 없는 입력으로도 train loss가 조금 내려간다 → 고유한 입력만으로도 외우기가 쉬워진다(지문 효과).
- 진짜 이웃은 훨씬 더 빨리 내려간다 → 진짜 이웃에는 학습 데이터 안에서만 통하는 추가 단서가 있다.
- 둘 다 val은 좋아지지 않는다 → 그 단서는 다른 시간 구간에 일반화되지 않는다.

**누설(leakage)과 구분하는 법**: 이웃 정보가 미래를 몰래 알려 준다면 같은 방식으로 만든 val에서도 성능이 좋아져야 한다. val이 오히려 나빠졌으므로 누설이 아니라 외우기다.

### 설정을 val로만 고르기

과적합을 줄이려고 격자 크기와 dropout을 바꿔 봤다. 이때 **test 값을 보고 고르면 안 된다.** 그러면 test가 사실상 검증 데이터가 되어 결과가 낙관적으로 부풀려진다. 이번에는 5개 분할의 val ADE 평균으로만 골랐고, test 값은 참고용으로 같이 적었다.

### 분포 차이 (distribution shift)

leave-one-out에서는 학습 장면과 test 장면이 다르다. 이웃 밀도도 크게 다르다. univ를 test로 빼면 학습 데이터는 한산하고(3 m 안 평균 3.3명), test는 붐빈다(8.6명). 모델은 학습 때 거의 못 본 붐비는 격자를 test에서 처음 만난다.

---

## 2. 코드에서 핵심이 되는 부분

| 내용 | 위치 |
| :--- | :--- |
| 이웃 뽑기: 샘플마다 관측 프레임별 같은 프레임의 다른 보행자 | `src/trajpred/data.py:47` |
| 이웃 속도: 직전 프레임에서 같은 ped_id를 찾아 차이 | `src/trajpred/data.py:80` |
| 자기 자신 제외, 가까운 순 정렬 | `src/trajpred/data.py:95`–`96` |
| 데이터셋에서 이웃 텐서 만들기(`max_neighbors > 0`) | `src/trajpred/data.py:171`, `183` |
| 격자 만들기: 칸 번호 계산, 격자 밖 제외 | `src/trajpred/models/lstm.py:52`, `62`–`63` |
| 같은 칸끼리 더하기(`scatter_add_`), 속도는 칸 평균 | `src/trajpred/models/lstm.py:68`, `70` |
| 격자 임베딩(dropout 포함) | `src/trajpred/models/lstm.py:91` |
| heading frame에서 이웃도 같은 회전으로 | `src/trajpred/models/lstm.py:117` |
| 인코더 입력에 격자 임베딩 붙이기 | `src/trajpred/models/lstm.py:130`–`131` |
| 회전 증강에서 이웃도 같은 각도로 | `scripts/train_lstm.py:47`, `64` |
| 섞음 대조 실험(`--neighbor-control shuffle`) | `scripts/train_lstm.py:83` |
| 이웃 수 기준(마지막 관측 프레임, 3 m 안) | `scripts/analyze_errors.py:51`, `102` |
| 시드별 오차 분해를 평균 ± 표준편차로 | `scripts/compare_bins.py:51` |

---

## 3. 결과 해석

1. **평균**: social pooling AVG ADE 0.531 ± 0.002로 이웃 없는 heading(0.531 ± 0.009)과 같다. 등속 모델(0.520)은 여전히 넘지 못했다.
2. **장면별**: eth에서 가장 좋고(0.909 ± 0.011), univ · zara2에서는 이웃 없는 모델보다 나쁘다(0.546, 0.360).
3. **꺾는 샘플**: 15–45°는 −0.008, 45°+는 +0.011로 시드 편차 안이다. 개선이 없다. 이번 단계의 핵심 질문에 대한 답은 "아니오"다.
4. **이웃 수별**: "이웃이 많을수록 좋아진다"는 경향이 없다. 혼자인 샘플(−0.021)도 좋아진 것을 보면, 개선이 이웃 정보 덕분인지도 확실하지 않다.
5. **과적합**: 첫 설정(8×8)은 val 기준 최악이었다(test AVG 0.570). 격자를 4×4로 줄이고 dropout 0.5를 넣으니 이웃 없는 모델 수준으로 돌아왔다. 그래도 best 에폭 중앙값이 9로, 이웃 없는 모델(27)보다 훨씬 이르다.
6. **시드 잡음 주의**: heading frame에서는 회전 증강이 거의 아무것도 바꾸지 않는데도, "heading"과 "heading + 회전 증강"의 AVG가 0.531 대 0.543으로 달랐다. 대부분 eth(181개) 한 시드의 차이다. 작은 test 장면에서 시드 하나가 평균을 크게 흔들 수 있다.

**배운 점**

- 정보를 더 넣는다고 좋아지지 않는다. 표현이 고차원이고 학습 · test 분포가 다르면 오히려 외우기 통로가 된다.
- "섞음 대조 실험"은 새 입력이 진짜로 쓰이는지 확인하는 싸고 강력한 방법이다.
- 꺾는 이유가 "곧 마주칠 사람"이라면 이웃의 과거 위치만으로는 부족할 수 있다. 이웃끼리 정보를 주고받는 구조(Social LSTM의 은닉 상태 pooling, C단계 attention)에서 다시 시험해야 한다.

---

## 4. 면접 대비 질문

### Q1. Social Pooling을 넣었는데 성능이 좋아지지 않았습니다. 어떻게 원인을 찾았나요?

**답변 (한국어)**
먼저 학습 곡선을 봤습니다. 이웃을 넣은 모델은 train loss가 이웃 없는 모델보다 훨씬 빨리 내려갔는데(8에폭에 0.18 대 0.31), val ADE는 에폭 1이 가장 좋고 그 뒤로 나빠졌습니다. 과적합 신호입니다. 정보 누설이라면 같은 방식으로 만든 val도 좋아져야 하므로, 누설보다는 외우기라고 판단했습니다. 이를 확인하려고 이웃 정보를 다른 샘플의 것과 섞는 대조 실험을 했습니다. 뜻 없는 이웃으로도 train loss가 조금 내려가서, 고차원 격자 자체가 샘플을 구분하는 지문 역할을 한다는 것을 알았습니다. 그래서 격자를 8×8에서 4×4로 줄이고 dropout을 넣는 설정을 val 기준으로만 비교해 골랐습니다. 그 결과 이웃 없는 모델과 같은 수준까지 회복했습니다.

**Answer (English)**
I started with the learning curves. With neighbors, training loss dropped much faster than without (0.18 vs 0.31 after 8 epochs), but validation ADE was best at epoch 1 and got worse afterwards, which is a clear overfitting signal. If the neighbor features leaked future information, the validation set, built the same way, should have improved too. Since it got worse, I concluded the model was memorizing rather than benefiting from a leak. To check, I ran a shuffle control that replaced each sample's neighbors with another random sample's. Even meaningless neighbors lowered training loss somewhat, so the high-dimensional grid itself acted as a per-sample fingerprint. I then compared smaller grids (4×4 instead of 8×8) and dropout, selecting only on validation ADE. That brought performance back to the level of the model without neighbors.

### Q2. Social LSTM의 social pooling과 이번에 구현한 grid pooling은 어떻게 다른가요?

**답변 (한국어)**
둘 다 대상 보행자 주변을 격자로 나누고 칸마다 이웃 정보를 모은다는 점은 같습니다. Social LSTM은 이웃의 **LSTM 은닉 상태**를 칸에 더합니다. 은닉 상태는 그 사람의 과거 움직임 전체를 요약한 것이고, 매 스텝 이웃끼리 서로의 상태를 주고받습니다. 제 구현은 이웃의 **현재 상대 위치와 상대 속도**만 칸에 모읍니다. 같은 장면 사람들을 한 배치로 묶을 필요가 없어 구현과 계산이 단순하지만, 이웃이 어떤 의도로 움직이는지는 담지 못합니다. 또 디코더에는 이웃 정보를 넣지 않았습니다. 예측 구간에서 이웃이 어떻게 움직일지 반영하지 못하는 한계가 있습니다. 결과에서 꺾는 샘플이 개선되지 않은 것도 이 한계와 관련 있을 수 있다고 봅니다.

**Answer (English)**
Both divide the space around the target pedestrian into a grid and aggregate neighbor information per cell. Social LSTM sums the neighbors' **LSTM hidden states** into each cell. A hidden state summarizes that person's whole motion history, and neighbors exchange states at every step. My implementation only aggregates the neighbors' **current relative position and relative velocity**. It doesn't need scene-level batching, so it is simpler and cheaper, but it cannot capture where the neighbors intend to go. I also feed the grid only to the encoder, not the decoder, so the model cannot account for how neighbors move during the prediction horizon. I suspect this limitation is related to why turning pedestrians did not improve.

### Q3. leave-one-out 평가에서 학습 장면과 test 장면의 분포가 다르면 어떤 문제가 생기고, 어떻게 확인했나요?

**답변 (한국어)**
leave-one-out은 한 장면을 통째로 test로 빼므로, 학습 데이터와 test 데이터의 특성이 다를 수 있습니다. 실제로 마지막 관측 프레임에서 3 m 안의 이웃 수를 세어 보니, univ를 test로 뺄 때 학습 데이터는 평균 3.3명인데 test는 8.6명이었습니다. 다른 장면은 반대로 학습 데이터가 더 붐볐습니다. 이웃 정보를 쓰는 모델은 학습 때 거의 못 본 밀도의 격자를 test에서 만나게 되고, univ에서 이웃 없는 모델보다 나빠졌습니다(0.546 대 0.517). 또 early stopping에 쓰는 val도 학습과 같은 장면이라, val이 좋아도 test 장면에서 좋다는 보장이 없습니다. 그래서 결과를 장면별 · 상황별로 나눠 보고, 시드 3개의 편차와 비교해 차이가 의미 있는지 판단했습니다.

**Answer (English)**
Leave-one-out holds out an entire scene for testing, so training and test data can differ systematically. I counted the neighbors within 3 m at the last observed frame. When univ is held out, the training data averages 3.3 neighbors but the test set averages 8.6. For the other scenes it is the reverse: the training data is more crowded. A model that uses neighbor features meets grid densities at test time that it rarely saw in training, and on univ it did worse than the model without neighbors (0.546 vs 0.517). The validation split used for early stopping also comes from the training scenes, so good validation performance does not guarantee good test performance on the held-out scene. To handle this, I broke results down by scene and by situation, and compared every difference against the standard deviation over three seeds before calling it meaningful.
