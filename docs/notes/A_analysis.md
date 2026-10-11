# A단계 공부 노트: LSTM 오차 분석

> 한 줄 요약: LSTM은 **방향을 바꾸는 보행자**에서는 등속 모델을 이기지만, **직진 · 정지 보행자**에서는 진다. ETH/UCY는 직진 · 정지가 대부분이라 평균으로는 등속 모델을 넘지 못했다. hotel에서 크게 졌던 이유의 대부분은 "좌표축 방향을 외운 것"이었다. heading 정규화나 회전 증강으로 고칠 수 있었지만, 그 뒤에도 "평균적인 보행자처럼 예측하는 습관"이 남는다.

자세한 수치와 그림은 [docs/analysis.md](../analysis.md)에 있다.

---

## 1. 개념을 쉬운 말로

### 오차 분해 (error decomposition)

ADE 하나만 보면 "LSTM이 0.04 m 나쁘다"는 것만 알 수 있다. 샘플을 상황별로 나눠(예측 몇 스텝째인지, 얼마나 꺾었는지, 얼마나 빨랐는지) 각 묶음의 ADE를 따로 보면 **어디서 이기고 어디서 지는지**가 보인다.

- **기여(contribution)**로 나누는 방법: 샘플마다 (LSTM 오차 − 등속 오차)를 구해 묶음별로 더한 뒤 전체 샘플 수로 나눈다. 묶음별 기여를 다 더하면 전체 차이와 정확히 같다. "그 묶음이 얼마나 나쁜가"뿐 아니라 "그 묶음이 얼마나 많은가"까지 반영된다.
- **장면 동일 비중**: 5개 장면을 합칠 때 그냥 합치면 샘플이 많은 univ(72%)가 결과를 다 정한다. 샘플마다 가중치 1/(그 장면 샘플 수)를 주면 장면 다섯 개가 같은 비중이 되고, 전체 평균이 README의 AVG와 같아진다.

### 진행 방향 정규화 (heading-aligned frame)

"북쪽으로 걷는 사람"과 "동쪽으로 걷는 사람"은 좌표만 다를 뿐 움직임은 같다. 그런데 신경망은 숫자 (x, y)를 그대로 보므로, 학습 데이터에 동쪽으로 걷는 사람이 많으면 "동쪽으로 걷는 사람"에 특화된다. 이게 hotel에서 생긴 일이다(학습은 x 방향 76%, hotel은 y 방향 94%).

해결책은 두 가지다.

1. **회전 증강**: 학습 때 궤적을 무작위로 돌려서 모든 방향을 보여 준다. 모델이 방향에 둔감해지도록 "가르치는" 방법이다.
2. **heading frame**: 입력을 항상 "진행 방향 = +x"가 되게 돌려서 넣는다. 그러면 모델은 "앞으로 얼마나, 옆으로 얼마나"만 본다. 방향을 외우는 것이 구조적으로 불가능하다. 예측은 반대로 돌려서 원래 좌표로 되돌린다.

heading frame은 수학적으로 **회전 등변성(rotation equivariance)**을 보장한다. 입력을 θ만큼 돌리면 출력도 정확히 θ만큼 돈다. 회전 증강은 이 성질을 근사적으로만 배운다(hotel에서 90° 단위로 돌렸을 때 ADE가 0.377~0.415로 조금씩 달랐다).

### residual 예측

모델이 "다음 변위" 전체를 처음부터 맞히는 대신, **좋은 기준선(등속 예측)에 더할 보정량**만 배우게 한다. 출력층을 0으로 초기화하면 학습 시작 시점의 모델이 정확히 등속 모델이다. 나쁜 출발점에서 시작하지 않으니 학습이 쉬워질 것이라는 기대다.

실제로는 residual만으로 hotel 문제가 풀리지 않았다. 출발점이 등속이라도 학습 데이터가 방향 편향을 갖고 있으면, 보정량도 방향 편향을 배운다. "출발점"은 바꿔도 "학습 데이터가 가르치는 것"은 바뀌지 않는다.

### 회귀 기울기로 본 "평균으로 당기기"

4.8 s 동안의 실제 이동 거리를 등속 이동 거리(12 × 마지막 변위)에 직선으로 맞췄을 때의 기울기를 봤다.

- 기울기 1: 빠르게 관측된 사람은 그만큼 멀리 간다(속도 유지).
- 기울기 < 1: 빠르게 관측된 사람도 생각보다 덜 간다. 측정 잡음 때문에 "빨라 보였던" 경우가 섞이거나 실제로 감속하기 때문이다. 통계에서는 이를 **평균으로의 회귀**라고 한다.

학습 장면은 약 0.85인데 hotel은 1.08이다. LSTM은 학습 장면의 0.85를 배워서 hotel 사람의 속도를 덜 믿는다.

### early stopping과 val 분포

early stopping은 val 성능이 가장 좋을 때의 모델을 고른다. 그런데 leave-one-out에서 val은 train과 같은 4개 장면이다. **test 장면과 분포가 다르다.** val ADE가 에폭에 따라 거의 변하지 않으면 어느 에폭이 뽑힐지는 잡음이 정한다. residual 시드 2가 에폭 1을 골라 hotel에서 혼자 좋게 나온 것(0.452 대 0.57)이 그 예다.

---

## 2. 코드에서 핵심이 되는 부분

| 내용 | 위치 |
| :--- | :--- |
| 진행 방향 회전 행렬. 마지막 변위 각도 θ를 구하고 −θ만큼 돌린다 | `src/trajpred/models/lstm.py:32` |
| 정지(1 mm 미만)면 돌리지 않는 처리 | `src/trajpred/models/lstm.py:39` |
| heading frame: 돌려 넣고, 예측을 `pred @ rot`으로 되돌린다 | `src/trajpred/models/lstm.py:69`–`73` |
| residual: 출력층 0 초기화 | `src/trajpred/models/lstm.py:61`–`62` |
| residual: 등속 변위 + 보정량, 그 값을 다음 스텝 입력으로 | `src/trajpred/models/lstm.py:92` |
| 변위 누적합으로 위치 복원 | `src/trajpred/models/lstm.py:96` |
| 체크포인트에서 설정까지 읽어 모델 만들기 | `src/trajpred/models/lstm.py:99` |
| 회전 증강(학습 배치에만) | `scripts/train_lstm.py:45`, `98` |
| early stopping(val ADE 기준, patience) | `scripts/train_lstm.py:116`, `122` |
| 실행 이름 규칙 `<augment>[_heading][_residual]_seed<seed>` | `scripts/train_lstm.py:178` |
| 방향 전환 각도와 "정해지지 않음" 판정 | `scripts/analyze_errors.py:91` |
| 장면 동일 비중 가중치 | `scripts/analyze_errors.py:112` |
| hotel 오차 차이를 묶음별 기여로 분해 | `scripts/analyze_errors.py:198`, `210` |
| along / cross 분해 | `scripts/analyze_errors.py:217` |
| 이동 거리 회귀 기울기 | `scripts/analyze_errors.py:240`, `258` |
| 속도를 k배 줄인 등속 모델 | `scripts/analyze_errors.py:298` |
| 시드 간 표본 표준편차(n − 1) | `scripts/summarize_lstm.py:59` |

---

## 3. 결과 해석

1. **LSTM이 이기는 곳**: 15° 이상 꺾는 보행자(all scenes 15–45°: 0.95~0.98 대 1.00), 아주 빠른 보행자, eth · univ 장면. 등속 가정이 깨지는 곳에서는 학습한 "사람은 이렇게 꺾는다"는 지식이 도움이 된다.
2. **등속이 이기는 곳**: 직진, 정지 · 저속, 첫 몇 스텝. 등속 가정이 거의 맞는 곳에서 LSTM은 평균 쪽으로 당기는 습관 때문에 오히려 손해를 본다.
3. **평균은 왜 등속이 이기나**: 15° 이상 꺾는 샘플이 약 25%뿐이다. 나머지 75%에서 잃는 것이 25%에서 얻는 것보다 크다.
4. **hotel**: 처음 +0.244의 차이는 대부분 방향 편향이었고, heading frame이나 회전 증강으로 +0.07~0.09까지 줄었다. 남은 차이의 확인된 원인은 두 가지다. (a) 정지한 사람을 움직이게 예측한다(실제 7 cm, LSTM 27~37 cm). (b) 걷는 사람의 속도를 줄여 예측한다(기울기 0.88 대 실제 1.08). 속도 줄이기만 따로 흉내 내 보면 +0.016 정도라, 남은 차이 전부를 설명하지는 못한다.
5. **설계 비교**: heading ≈ 회전 증강 ≈ heading + residual(AVG 0.528~0.531). residual만 쓰면 효과가 없다.

**배운 점**: "학습형 모델이 기준선보다 나쁘다"는 결과를 받았을 때, 평균 하나로 결론 내지 말고 상황별로 쪼개 보면 원인과 고칠 방향이 보인다. 이번에는 방향 편향(고칠 수 있었음)과 평균으로 당기는 습관(아직 남음)으로 나뉘었다.

---

## 4. 면접 대비 질문

### Q1. 학습한 LSTM이 단순한 등속 모델보다 ADE가 나빴습니다. 왜 그렇고, 어떻게 분석했나요?

**답변 (한국어)**
ADE 평균만 보지 않고 상황별로 오차를 나눴습니다. 예측 스텝, 미래 방향 전환 각도, 관측 속력으로 묶어 보니, LSTM은 15° 이상 꺾는 보행자에서는 등속보다 낫지만 직진 · 정지 보행자와 앞쪽 스텝에서는 나빴습니다. ETH/UCY는 꺾는 샘플이 약 25%뿐이라 평균으로는 등속이 이깁니다. 특히 hotel 장면에서 크게 졌는데, 진행 방향 분포를 비교해 보니 학습 장면은 x 방향, hotel은 y 방향이 대부분이었습니다. 입력을 90° 돌려 넣으면 오차가 0.57에서 0.37로 줄어서, 모델이 좌표축 방향을 외웠다는 것을 확인했습니다.

**Answer (English)**
Instead of relying on the average ADE, I decomposed the error by prediction step, by how much the pedestrian turned in the future, and by observed speed. The LSTM beat the constant velocity model on pedestrians who turned more than 15°, but lost on straight-walking and standing pedestrians and on the first few steps. Only about 25% of ETH/UCY samples involve such turns, so on average constant velocity wins. The largest gap was on the hotel scene. The training scenes are dominated by motion along x, while hotel pedestrians mostly walk along y. Rotating the hotel inputs by 90° reduced the LSTM's ADE from 0.57 to 0.37, which confirmed that the model had memorized the coordinate-axis direction.

### Q2. 회전 증강과 heading-aligned frame의 차이는 무엇이고, 어느 쪽을 선택하겠습니까?

**답변 (한국어)**
회전 증강은 학습 때 궤적을 무작위로 돌려서 모델이 모든 방향을 보게 하는 방법입니다. 방향에 둔감해지는 것을 데이터로 가르치므로 근사적입니다. heading frame은 입력을 항상 진행 방향이 +x가 되도록 돌려 넣고 출력을 되돌리는 결정적 변환이라, 회전 등변성이 구조적으로 보장됩니다. 실험에서는 두 방법의 성능이 비슷했습니다(AVG ADE 0.528 대 0.531, hotel 0.418 대 0.387). 저는 heading frame을 기본으로 하겠습니다. 증강 없이도 등변성이 보장되고, 모델이 "앞 · 옆"이라는 의미 있는 좌표를 보기 때문입니다. 다만 정지한 보행자는 진행 방향이 정의되지 않으므로 따로 처리해야 합니다. 상호작용 모델로 넘어가면 이웃 좌표도 같은 frame으로 돌려야 합니다.

**Answer (English)**
Rotation augmentation randomly rotates trajectories during training so that the model sees every direction. It teaches rotation invariance through data, so the result is only approximate. A heading-aligned frame is a deterministic transform: it rotates each input so that the last observed velocity points along +x, then rotates the prediction back. This makes the model rotation-equivariant by construction. In my experiments the two performed similarly (average ADE 0.528 vs 0.531, hotel 0.418 vs 0.387). I would default to the heading frame because it guarantees equivariance without augmentation and gives the model meaningful forward/lateral coordinates. Its weak point is that heading is undefined for standing pedestrians, so they need special handling. For interaction models, the neighbors must be rotated into the same frame as well.

### Q3. 등속 예측에 대한 residual을 학습하면 왜 도움이 될 거라 예상했고, 실제로는 어땠나요?

**답변 (한국어)**
등속 모델이 이미 강한 기준선이니, 출력층을 0으로 초기화해서 학습 시작점을 등속 모델로 두고 보정량만 배우게 하면 최소한 등속보다 나빠지지는 않을 것으로 기대했습니다. 실제로는 residual만으로는 hotel 문제가 풀리지 않았습니다(0.565 → 0.534 ± 0.071). 출발점이 등속이어도 학습 데이터의 방향 편향을 보정량이 그대로 배웠기 때문입니다. 게다가 시드 간 편차가 컸습니다. 한 시드는 early stopping이 에폭 1을 골라 거의 등속인 모델이 선택됐습니다. val이 test 장면과 다른 분포라 val ADE가 평평했기 때문입니다. heading frame과 조합하면 heading 단독과 비슷했고, 정지 보행자 오차가 조금 줄었습니다. 출발점을 바꾸는 것보다 데이터와 입력 표현의 편향을 없애는 것이 더 중요하다는 것을 배웠습니다.

**Answer (English)**
Constant velocity is already a strong baseline. I initialized the output layer to zero so that the model starts out exactly as constant velocity and only learns a correction, and I expected it to do at least as well as the baseline. In practice the residual alone did not fix the hotel problem (0.565 → 0.534 ± 0.071), because the correction itself learned the directional bias in the training data. It also had high variance across seeds. For one seed, early stopping picked epoch 1, which is still almost constant velocity. This happened because the validation split comes from different scenes than the test scene, so validation ADE was nearly flat across epochs. Combined with the heading frame, it performed about the same as the heading frame alone, with slightly lower error on standing pedestrians. The lesson was that removing bias from the data and the input representation matters more than choosing a good starting point.
