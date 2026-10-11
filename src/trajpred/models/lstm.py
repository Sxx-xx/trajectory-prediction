"""Vanilla LSTM encoder-decoder (autoregressive). 주변 보행자를 보지 않는 단일 보행자 모델.

입력과 출력은 위치가 아니라 '한 스텝 변위'(속도 × 0.4 s)다.
    관측 위치 obs[0..7] -> 변위 d[t] = obs[t] - obs[t-1]   (7개)
위치 대신 변위를 쓰면 장면마다 다른 좌표 범위에 덜 민감하고, 등속 모델과 같은 양을 다룬다.

    인코더: 관측 변위 7개를 차례로 LSTM 에 넣어 마지막 상태 (h, c) 를 얻는다
    디코더: (h, c) 에서 시작해 한 스텝씩 다음 변위를 낸다.
            스텝 k 의 입력은 스텝 k-1 의 '자기 예측' 변위다 (autoregressive).
            첫 스텝의 입력은 마지막 관측 변위 d[7] 이다.
    위치  : 마지막 관측점 + 예측 변위의 누적합

학습 때도 정답 대신 자기 예측을 다음 입력으로 넣는다 (teacher forcing 없음).
그래야 학습과 평가의 입력 분포가 같다.
"""

import torch
from torch import nn


class LSTMEncoderDecoder(nn.Module):
    def __init__(self, pred_len=12, embed_dim=64, hidden_dim=128):
        super().__init__()
        self.pred_len = pred_len
        # 변위 (2) -> 임베딩 (embed_dim). 인코더 · 디코더가 같은 임베딩을 쓴다.
        self.embed = nn.Sequential(nn.Linear(2, embed_dim), nn.ReLU())
        self.encoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        # 디코더는 한 스텝씩 돌리므로 LSTMCell 을 쓴다.
        self.decoder = nn.LSTMCell(embed_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 2)     # 은닉 상태 -> 다음 변위

    def forward(self, obs):
        """obs: [N, obs_len, 2] -> 예측 위치 [N, pred_len, 2]

        obs 는 절대 좌표든 상대 좌표든 상관없다. 변위만 쓰고, 마지막 관측점에 더해 위치로 돌린다.
        """
        disp = obs[:, 1:] - obs[:, :-1]                     # [N, obs_len-1, 2]

        # 인코더
        _, (h, c) = self.encoder(self.embed(disp))          # h, c: [1, N, hidden]
        h, c = h[0], c[0]                                   # [N, hidden]

        # 디코더: 자기 예측을 다음 입력으로
        step_in = disp[:, -1]                               # [N, 2] 마지막 관측 변위
        preds = []
        for _ in range(self.pred_len):
            h, c = self.decoder(self.embed(step_in), (h, c))  # [N, hidden]
            step_in = self.head(h)                          # [N, 2] 다음 변위
            preds.append(step_in)
        pred_disp = torch.stack(preds, dim=1)               # [N, pred_len, 2]

        return obs[:, -1:, :] + pred_disp.cumsum(dim=1)     # [N, pred_len, 2]
