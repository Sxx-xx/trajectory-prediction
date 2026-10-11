# D단계 공부 노트: 다중 예측과 best-of-20

> 한 줄 요약: 미래를 20개 내는 Transformer를 winner-takes-all로 학습해 AVG minADE₂₀ / minFDE₂₀ 0.216 / 0.371을 얻었다. 그런데 **방향을 무작위로 20번 뽑은 등속 모델도 FDE가 결정적 등속보다 낮게 나왔다.** best-of-K가 얼마나 관대한 지표인지 직접 확인한 단계다. 논문 표와 비교할 때는 K, 샘플 규칙, k = 1 성능을 같이 봐야 한다.

자세한 수치와 그림은 [docs/multimodal.md](../multimodal.md)에 있다.

---

## 1. 개념을 쉬운 말로

### 왜 여러 개를 예측하나

갈림길에 선 보행자는 왼쪽으로도, 오른쪽으로도 갈 수 있다. 예측을 하나만 내고 MSE로 학습하면 모델은 두 답의 **평균**, 즉 가운데로 가는 예측을 낸다. 실제로는 아무도 그 길로 가지 않는다. 그래서 여러 개(K개)의 그럴듯한 미래를 내게 한다.

### minADE_K / minFDE_K (best-of-K)

K개 예측 중 실제와 가장 가까운 것의 오차만 본다. "20번 찍어서 하나라도 맞으면 된다"는 지표다. 여러 미래를 덮는 능력은 재지만, 20개 중 어느 것이 맞을지 모델이 아는지는 재지 않는다. ADE와 FDE의 최솟값은 따로 고른다(Social GAN 방식).

### winner-takes-all (WTA) 손실

샘플마다 K개 예측 중 실제와 가장 가까운 하나(winner)만 정답 쪽으로 당긴다. 나머지는 그 샘플에서 배우지 않는다. 그러면 모드마다 "자기가 잘 맞히는 종류의 미래"를 맡게 된다. Social GAN의 variety loss도 같은 꼴이다. 위험은 **모드 붕괴**다. 처음에 한 모드가 늘 이기면 나머지가 영영 학습되지 않는다. 그래서 K=20에서는 출력층을 0으로 초기화하지 않았다(20개가 같은 예측에서 출발하지 않도록).

### 모드 점수

각 모드가 winner일 확률을 따로 예측한다. 이것으로 k < 20개만 고를 수 있다(k = 1 = 가장 그럴듯한 하나). 점수 MLP는 `h.detach()`를 입력으로 받아, 점수 학습이 궤적 예측 쪽을 흔들지 않게 했다.

### 등속 + 잡음 기준선

Schöller et al. 2020의 CVM-S다. 등속 예측의 방향에 N(0, 25°²) 잡음을 넣어 20개를 만든다. 학습이 전혀 없는데 best-of-20으로 재면 꽤 좋아 보인다. 다중 예측 모델이 "잡음 넣은 등속보다 나은가"를 보는 최소 기준이다.

---

## 2. 코드에서 핵심이 되는 부분

| 내용 | 위치 |
| :--- | :--- |
| minADE: 모드별 ADE 후 최솟값 | `src/trajpred/metrics.py:39`–`41` |
| minFDE: 모드별 FDE 후 최솟값 (ADE와 따로) | `src/trajpred/metrics.py:44`–`46` |
| 점수 상위 k개 고르기 | `src/trajpred/metrics.py:49` |
| 지표 테스트 | `tests/test_metrics.py` |
| 출력을 K × 12 × 2로 늘림 | `src/trajpred/models/transformer.py:57` |
| 모드 점수 MLP (h.detach() 입력) | `src/trajpred/models/transformer.py:59`, `113` |
| K > 1이면 0 초기화 안 함 | `src/trajpred/models/transformer.py:60` |
| 모드마다 등속 변위 + 보정량 | `src/trajpred/models/transformer.py:112` |
| WTA 손실: winner 고르기, winner MSE, 점수 cross-entropy | `scripts/train_multimodal.py:42`–`47` |
| early stopping 기준 = val minADE₂₀ | `scripts/train_multimodal.py:92` |
| test에서 k = 1..20 곡선 | `scripts/train_multimodal.py:161` |
| 등속 + 방향 회전으로 K개 만들기 | `scripts/eval_cv_sampling.py:33` |
| 잡음 분포(정규 / 균등) | `scripts/eval_cv_sampling.py:44` |
| σ를 train 데이터에서 고르기 | `scripts/eval_cv_sampling.py:51` |
| 모드 사용(winner 비율) 진단 | `scripts/summarize_multimodal.py:83`, `101` |

---

## 3. 결과 해석

1. **K=20 Transformer**: AVG 0.216 / 0.371(이웃 없음), 0.221 / 0.392(이웃 attention). σ=25° 잡음 등속(0.404 / 0.851)의 약 절반이고, k = 1~20 모든 구간에서 기준선보다 낮다.
2. **지표가 관대하다**: 관측 방향을 버린 무작위 방향 예측도 k = 20에서 FDE 1.090으로 결정적 등속(1.141)보다 낮다. σ=25° 잡음 등속은 k = 1에서 0.911로 가장 나쁜 축이지만, k = 5부터 등속보다 좋다. 같은 예측기의 순위가 k에 따라 뒤집힌다.
3. **논문 비교**: 같은 샘플 규칙인 Social GAN 논문 값(20V-20: 0.58 / 1.18)은 잡음 등속보다도 나쁘다. 2018년 표 값이 약하다고 읽는 것이 맞다. Trajectron++(0.19 / 0.41)와는 비슷한 수준이다. ADE는 Trajectron++가, FDE는 이 저장소가 조금 낮다. 학습 설정이 달라 우열을 말하기는 어렵다.
4. **k = 1 손해**: 점수 1위 하나는 0.529~0.531 / 1.152~1.154로, 단일 예측 전용 모델(0.505 / 1.094)보다 나쁘다. 여러 개를 덮도록 배우면 "가장 그럴듯한 하나"는 약해진다.
5. **이웃 정보의 역할**: k = 5에서는 도움이 되고(0.356 대 0.380), k = 20에서는 아니다. 20개면 이웃 없이도 가능한 방향을 거의 다 덮는다. 이웃은 순위를 정하는 데 쓰이는 것으로 보인다(추측).
6. **모드 붕괴 없음**: 20개 중 15~18개 모드가 test 샘플의 1% 이상에서 winner다. 최다 모드는 약 30%다.

**배운 점**

- 지표를 믿기 전에 "아무 생각 없는 예측"으로 그 지표를 재 봐야 한다. 무작위 방향 20개가 등속을 이기는 것을 보면 best-of-20 숫자만으로 모델을 말할 수 없다.
- 논문 표를 옮길 때는 K, 샘플 규칙, 어느 열(20V-20인지 20VP-20인지)인지까지 확인해야 한다. 같은 "S-GAN"이라도 논문마다 값이 다르다(Schöller et al.는 공개 모델을 자기 규칙으로 다시 쟀다).
- 다중 예측은 k = 1 지표나 확률 지표(NLL)와 같이 보고해야 공정하다.

---

## 4. 면접 대비 질문

### Q1. minADE₂₀ / minFDE₂₀ 지표의 장점과 한계는 무엇인가요?

**답변 (한국어)**
보행자의 미래는 본질적으로 여러 갈래라서, 예측 하나의 오차만 보면 평균으로 뭉개진 예측이 유리합니다. minADE₂₀는 20개 중 가장 가까운 것만 보므로 여러 미래를 덮는 능력을 평가할 수 있습니다. 한계는 너무 관대하다는 점입니다. 제 실험에서 관측 방향을 버리고 방향을 무작위로 20번 뽑은 예측도 minFDE₂₀가 1.090으로, 결정적 등속 모델(1.141)보다 낮았습니다. 등속에 25° 방향 잡음만 넣어도 0.404 / 0.851로 Social GAN 논문 값보다 좋았습니다. 그래서 저는 best-of-20과 함께 점수 1위 하나의 오차(k = 1), k에 따른 곡선, 그리고 잡음 넣은 등속 같은 단순 기준선을 같이 보고합니다.

**Answer (English)**
Pedestrian futures are inherently multimodal, so judging a single prediction favors blurry predictions that average the possible futures. minADE₂₀ only scores the closest of 20 predictions, so it measures whether the model covers the plausible futures. The limitation is that it is very lenient. In my experiments, a predictor that ignores the observed heading and samples 20 uniformly random directions reached a minFDE₂₀ of 1.090, lower than the deterministic constant-velocity model (1.141). Constant velocity with just 25° heading noise reached 0.404 / 0.851, better than the numbers in the Social GAN paper. So I report best-of-20 alongside the top-1 error (k = 1), the curve over k, and simple baselines such as noisy constant velocity.

### Q2. winner-takes-all 손실은 어떻게 동작하고, 어떤 문제가 생길 수 있나요?

**답변 (한국어)**
샘플마다 K개 예측의 오차를 재고, 가장 가까운 하나(winner)의 오차만 줄입니다. 그러면 각 출력이 서로 다른 종류의 미래를 맡게 됩니다. 문제는 모드 붕괴입니다. 처음에 모든 출력이 같으면 늘 같은 출력만 이겨서 나머지가 학습되지 않습니다. 저는 단일 예측 모델에서는 등속에서 출발하도록 출력층을 0으로 초기화했지만, K=20에서는 이 초기화를 빼서 출력들이 서로 다르게 시작하게 했습니다. 학습 뒤 test 샘플마다 winner를 세어 보니 20개 중 15~18개 모드가 1% 이상에서 이겼고, 최다 모드도 약 30%라 붕괴는 없었습니다. 또 하나의 대가는 점수 1위 하나의 성능이 단일 예측 전용 모델보다 나쁘다는 점입니다(ADE 0.529 대 0.505).

**Answer (English)**
For each sample I compute the error of all K predictions and only minimize the error of the closest one, the winner. This lets each output specialize in a different kind of future. The risk is mode collapse: if all outputs start identical, the same one always wins and the others never learn. For the single-prediction model I zero-initialized the output layer so it starts from constant velocity, but for K=20 I removed that initialization so the outputs start out different. After training I counted the winner for every test sample: 15 to 18 of the 20 modes won at least 1% of samples and the most frequent mode won about 30%, so there was no collapse. Another cost is that the top-scored single prediction is worse than a dedicated single-prediction model (ADE 0.529 vs 0.505).

### Q3. 논문의 best-of-20 결과와 자신의 결과를 비교할 때 무엇을 확인했나요?

**답변 (한국어)**
세 가지를 원 논문에서 직접 확인했습니다. 첫째, 샘플 규칙입니다. 제 데이터 로더는 Social GAN 로더와 샘플이 같은 것을 확인했지만, Schöller et al.는 미래가 12스텝보다 짧은 샘플도 포함해서 결정적 등속부터 값이 작습니다. 둘째, 어느 열을 비교하는지입니다. Social GAN 표에는 20V-20과 20VP-20이 따로 있고, Trajectron++ 표의 "S-GAN"은 20V-20 값입니다. 셋째, K와 최솟값 계산 방식입니다. 모두 20개 중 최선이고, ADE와 FDE를 따로 고릅니다. 이렇게 맞춘 뒤에도 학습 설정과 증강이 다르므로, 저는 Trajectron++와 비슷한 수준이라고만 말하고 우열은 주장하지 않았습니다. 대신 같은 조건에서 잰 잡음 등속 기준선과의 차이(0.216 대 0.404)를 주된 근거로 썼습니다.

**Answer (English)**
I checked three things directly in the original papers. First, the sample rules: my data loader produces the same samples as the Social GAN loader, but Schöller et al. include windows with fewer than 12 future steps, so even their deterministic constant-velocity numbers are lower. Second, which column is being compared: the Social GAN table has separate 20V-20 and 20VP-20 columns, and the "S-GAN" column in the Trajectron++ table is the 20V-20 one. Third, K and how the minimum is taken: all use the best of 20, with ADE and FDE minimized separately. Even after aligning these, training setups and augmentation differ, so I only claim to be at a similar level to Trajectron++, not better. My main evidence is instead the gap to a noisy constant-velocity baseline measured under identical conditions (0.216 vs 0.404).
