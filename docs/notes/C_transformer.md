# C단계 공부 노트: Transformer와 이웃 attention

> 한 줄 요약: 이웃 없는 Transformer는 LSTM과 비슷했다. 대상 보행자가 이웃들에게 cross-attention하게 하자 **FDE가 처음으로 등속 모델을 확실히 이겼다**(1.059 대 1.141). 꺾는 보행자가 크게 좋아졌고, 섞은 이웃으로는 이 효과가 나오지 않았다. 그런데 attention 가중치를 보면 가까운 이웃을 특별히 보지 않았다. "attention 가중치 ≠ 설명"을 직접 확인한 단계다.

자세한 수치와 그림은 [docs/transformer.md](../transformer.md)에 있다.

---

## 1. 개념을 쉬운 말로

### self-attention (시간축)

LSTM은 과거 7스텝을 **순서대로 하나씩** 읽으며 기억을 갱신한다. self-attention은 7개 스텝을 **한꺼번에** 놓고, 스텝마다 "다른 스텝들 중 누구를 얼마나 참고할지"를 계산해 섞는다.

- 각 토큰이 query(내가 찾는 것), key(내가 가진 것의 이름표), value(내 내용)를 만든다.
- query와 key가 잘 맞을수록 그 value를 많이 가져온다. 가중치는 softmax라 합이 1이다.
- attention 자체는 순서를 모르므로 "몇 번째 스텝인지"를 알려 주는 **위치 임베딩**을 더한다.
- 여러 head는 서로 다른 기준으로 동시에 참고하는 것이다(예: 한 head는 최근 속도, 다른 head는 방향 변화).

### non-autoregressive 디코더

LSTM 디코더는 1스텝 예측 → 그걸 다시 넣어 2스텝 → ... 식으로 12번 돈다(autoregressive). 이번 디코더는 요약 벡터 하나에서 **12스텝을 한 번에** 낸다.

- 장점: 빠르고, 앞 스텝의 실수가 뒤로 누적되지 않는다.
- 단점: 첫 스텝처럼 "마지막 속도를 그대로 이으면 되는" 쉬운 부분도 직접 맞혀야 한다. 실제로 첫 스텝 오차가 등속의 두 배 가까웠다.

### cross-attention과 null 토큰

cross-attention은 query와 key/value가 서로 다른 집합에서 온다. 여기서는 query = 대상 보행자 요약, key/value = 이웃들이다. "나는 이런 상황인데, 이웃 중 누구 정보를 얼마나 가져올까"를 계산한다.

- 이웃 수가 샘플마다 다르므로 빈 칸은 **마스크**로 가려 가중치 0이 되게 한다.
- 이웃이 0명이면 softmax할 대상이 없어 계산이 깨진다. 그래서 항상 존재하는 **null 토큰**을 하나 넣었다. null에 준 가중치는 "이웃을 참고하지 않는 정도"로 읽을 수 있다(평균 약 0.2).
- attention은 집합 연산이라 이웃 순서를 바꿔도 결과가 같다(코드로 확인).

### 섞은 이웃 대조와 이웃 지우기 시험

- **섞은 이웃**: 이웃 정보를 다른 샘플의 것으로 바꿔 학습 · 평가한다. 그래도 성능이 같다면 모델은 이웃의 "뜻"이 아니라 입력이 있다는 사실 자체(지문)를 쓰는 것이다. 이번에는 섞은 이웃의 FDE(1.114)가 이웃 없는 모델(1.117)과 같고, 진짜 이웃(1.059)만 좋았다 → 진짜 정보를 쓴다.
- **이웃 지우기 시험(ablation)**: 학습은 그대로 두고, 평가 때 특정 거리 구간의 이웃만 가려 오차 변화를 본다. attention 가중치가 아니라 **실제로 예측이 얼마나 그 입력에 기대는지**를 잰다.

### attention 가중치는 설명이 아니다

attention 출력은 Σ(가중치 × value)다. 가중치가 작아도 value 벡터가 크거나 특이하면 결과에 큰 영향을 줄 수 있다. 그래서 "가중치가 큰 이웃 = 중요한 이웃"은 보장되지 않는다(Jain & Wallace, 2019 등의 논의). 이번 결과가 그 예다. 가중치는 가까운 이웃에 몰리지 않았지만, 지우기 시험에서는 이웃 정보에 대한 의존이 분명했다.

---

## 2. 코드에서 핵심이 되는 부분

| 내용 | 위치 |
| :--- | :--- |
| 시점 위치 임베딩(학습하는 파라미터) | `src/trajpred/models/transformer.py:39` |
| null 토큰 | `src/trajpred/models/transformer.py:46` |
| 마지막 관측 프레임 이웃만 사용 | `src/trajpred/models/transformer.py:65` |
| 토큰 = [변위, 위치], 위치 임베딩 더해 인코더 통과 | `src/trajpred/models/transformer.py:80`–`81` |
| 마지막 시점 출력을 요약으로 | `src/trajpred/models/transformer.py:82` |
| 이웃 상대 속도 계산 | `src/trajpred/models/transformer.py:86` |
| key = [null, 이웃들], 빈 칸 마스크(null은 절대 안 가림) | `src/trajpred/models/transformer.py:88`–`89` |
| cross-attention과 가중치 저장 | `src/trajpred/models/transformer.py:90`–`92` |
| 잔차 연결 + LayerNorm | `src/trajpred/models/transformer.py:93` |
| 12스텝 한 번에 예측 | `src/trajpred/models/transformer.py:95` |
| 체크포인트에서 Transformer 불러오기 | `src/trajpred/models/lstm.py:157` |
| `--model transformer` 분기, 실행 이름 `tf_` | `scripts/train_lstm.py:118`, `235` |
| 섞은 이웃 모델은 평가 때도 같은 방식으로 섞기 | `scripts/analyze_errors.py:90` |
| closing speed(다가옴) 계산 | `scripts/analyze_attention.py:66` |
| 가장 많이 본 이웃 = 가장 가까운 이웃? | `scripts/analyze_attention.py:87` |
| 이웃 지우기 시험 | `scripts/analyze_attention.py:100`–`104` |

---

## 3. 결과 해석

1. **이웃 없는 Transformer ≈ LSTM**: AVG 0.529 / 1.117 대 LSTM heading 0.531 / 1.112. 파라미터는 절반 이하(79k 대 199k)다. 과거 8스텝 단일 보행자 예측에서는 인코더 구조가 큰 차이를 만들지 않는다.
2. **이웃 attention은 FDE를 크게 줄였다**: AVG FDE 1.059 ± 0.005로 등속(1.141)보다 0.08 낮다. 이 저장소에서 처음으로 등속을 확실히 이긴 지표다. 반면 ADE(0.537)는 등속(0.520)보다 높다. 앞쪽 스텝에서 잃고 뒤쪽 스텝에서 얻기 때문이다.
3. **꺾는 샘플**: all scenes 15–45° 0.972 → 0.901, 45°+ 1.556 → 1.398. A · B단계에서 풀지 못한 질문("이웃이 꺾는 이유를 알려 주는가")에 처음으로 "그렇다"는 결과가 나왔다.
4. **grid pooling(B)과 차이가 난 이유(추측)**: grid는 고정된 192/48차원 벡터라 샘플 지문이 되기 쉬웠다. attention은 이웃 하나하나를 같은 가중치로 임베딩하고 가중 평균하므로, 개별 샘플을 외우기 어렵고 일반화되는 규칙을 배우기 쉽다. 또 grid는 칸 경계에서 정보가 끊기지만, attention은 연속적인 상대 위치를 그대로 쓴다.
5. **손해**: 정지 · 저속(+0.040), 혼자인 샘플(+0.077), 6명 이상(+0.049, 주로 univ). 이웃에 기대도록 배운 모델은 이웃이 없거나(혼자) 학습 때와 밀도가 다르면(univ) 나빠진다.
6. **attention이 보는 것**: 가장 많이 본 이웃이 가장 가까운 이웃인 비율은 무작위 수준(약 0.12)이다. 먼 이웃(4 m+)의 가중치가 조금 크다. 지우기 시험에서도 all scenes에서는 먼 이웃을 지울 때 FDE가 가장 나빠졌다. 반대로 hotel에서는 4 m 안 이웃만 영향을 줬다. 개별 이웃 회피보다 "주변 사람들 배치 = 걸을 수 있는 길과 흐름"을 단서로 쓰는 것으로 보인다(추측).

**배운 점**

- 같은 "이웃 정보"라도 표현(고정 격자 vs 이웃 집합 attention)에 따라 결과가 완전히 달랐다.
- 대조 실험(섞은 이웃)은 "개선이 진짜인가"를 확인하는 가장 싼 방법이다.
- attention 시각화는 그럴듯한 이야기를 만들기 쉽다. 가중치만 보면 "먼 이웃을 본다"였고, 지우기 시험을 해야 장면마다 다르다는 것까지 보였다. 설명을 주장하려면 개입 실험이 필요하다.

---

## 4. 면접 대비 질문

### Q1. 시간축 Transformer 인코더와 LSTM 인코더의 차이는 무엇이고, 이 문제에서는 어땠나요?

**답변 (한국어)**
LSTM은 과거 스텝을 순서대로 읽으며 은닉 상태 하나에 정보를 압축합니다. Transformer는 모든 스텝을 한꺼번에 보고, self-attention으로 스텝 간 관계를 직접 계산합니다. 순서 정보는 위치 임베딩으로 넣습니다. 이론적으로 Transformer는 먼 스텝 간 관계를 잡기 쉽고 병렬화가 잘 되지만, 입력이 8스텝뿐이라 이 장점이 크지 않습니다. 실제로 이웃 없이 비교했을 때 Transformer(AVG ADE 0.529, FDE 1.117)와 LSTM(0.531, 1.112)은 같은 수준이었고, Transformer의 파라미터가 절반 이하였습니다. 차이를 만든 것은 인코더 구조가 아니라 이웃 정보를 attention으로 넣은 것이었습니다.

**Answer (English)**
An LSTM reads the past steps sequentially and compresses everything into a single hidden state. A Transformer looks at all steps at once and computes the relationships between them directly with self-attention, with order supplied by positional embeddings. In theory the Transformer captures long-range dependencies more easily and parallelizes better, but with only 8 observed steps that advantage is small. Without neighbors, the Transformer (average ADE 0.529, FDE 1.117) and the LSTM (0.531, 1.112) performed the same, and the Transformer had less than half the parameters. What made the difference was not the encoder but adding neighbor information through attention.

### Q2. 이웃 attention 모델이 정말 이웃 정보를 쓰는지 어떻게 확인했나요?

**답변 (한국어)**
세 가지로 확인했습니다. 첫째, 섞은 이웃 대조 실험입니다. 이웃 정보를 다른 샘플의 것으로 바꿔 똑같이 학습했더니 AVG FDE가 1.114로 이웃 없는 모델(1.117)과 같았고, 진짜 이웃(1.059)만 좋았습니다. 그래서 개선이 입력 형식 때문이 아니라 진짜 이웃 정보 때문이라고 판단했습니다. 둘째, A단계 기준으로 오차를 나눠 보니 15° 이상 꺾는 샘플에서 크게 좋아졌습니다. 셋째, 평가 때 이웃을 거리 구간별로 지우는 시험을 했습니다. 모두 지우면 FDE가 1.06에서 1.28로 나빠져서 의존이 분명했고, 속도만 지울 때보다 위치를 지울 때 훨씬 나빠져 위치가 정보의 대부분을 나른다는 것도 알았습니다.

**Answer (English)**
I checked in three ways. First, a shuffle control: I trained the same model with each sample's neighbors replaced by another sample's. Its average FDE was 1.114, the same as the model without neighbors (1.117), while only real neighbors gave 1.059. So the gain comes from real neighbor information, not the input format. Second, breaking errors down as in the earlier analysis showed large gains on pedestrians who turned more than 15°. Third, I ran ablations at evaluation time that removed neighbors by distance band. Removing all of them raised FDE from 1.06 to 1.28, so the dependence is clear. Removing positions hurt far more than zeroing velocities, which showed that positions carry most of the information.

### Q3. attention 가중치를 시각화하면 모델이 무엇을 보는지 알 수 있나요?

**답변 (한국어)**
조심해야 합니다. 이번 실험에서 가중치만 보면, 가장 많이 본 이웃이 가장 가까운 이웃인 비율이 약 12%로 무작위 수준이었습니다. 다가오는 이웃에도 특별히 더 주지 않았고, 오히려 먼 이웃의 가중치가 조금 컸습니다. 섞은 이웃 모델도 비슷한 패턴이라 가중치만으로는 둘을 구분하기 어려웠습니다. attention 출력은 가중치와 value의 곱의 합이라, 가중치가 작아도 value가 크면 영향이 클 수 있습니다. 그래서 이웃을 실제로 지워 보는 개입 실험을 같이 했습니다. 전체적으로는 먼 이웃을 지울 때 가장 나빠졌지만, hotel에서는 4 m 안의 이웃만 영향을 줬습니다. attention 시각화는 가설을 만드는 데 쓰고, 결론은 ablation 같은 개입 실험으로 확인해야 한다고 생각합니다.

**Answer (English)**
You have to be careful. In my experiment, the most-attended neighbor was the nearest neighbor only about 12% of the time, which is chance level. Approaching neighbors did not get extra weight either, and distant neighbors actually got slightly more. A model trained on shuffled neighbors showed a similar pattern, so the weights alone could not tell the two apart. Attention output is a weighted sum of value vectors, so a neighbor with a small weight but a large value can still have a big effect. That is why I also ran interventions that actually removed neighbors. Overall, removing distant neighbors hurt the most, but on the hotel scene only neighbors within 4 m mattered. I treat attention visualizations as a source of hypotheses, and confirm conclusions with interventions such as ablations.
