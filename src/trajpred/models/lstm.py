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

선택 옵션 (둘을 함께 써도 된다)
    frame='heading' : 진행 방향 정규화. 샘플마다 마지막 관측 변위가 +x 를 향하도록 궤적을
                      마지막 관측점 중심으로 돌려 넣고, 예측을 원래 좌표로 되돌린다.
                      모델은 '앞 · 옆' 좌표만 보므로 장면의 좌표축 방향을 외울 수 없다.
                      마지막 변위가 MIN_HEADING_STEP 보다 작으면(거의 서 있음) 방향이
                      정해지지 않으므로 돌리지 않는다.
    residual=True   : 디코더가 변위 자체 대신 '등속 변위(마지막 관측 변위)에 더할 보정량'을 낸다.
                      출력층을 0 으로 초기화하므로 학습 시작 시점의 예측은 등속 모델과 같다.
"""

import torch
from torch import nn

MIN_HEADING_STEP = 1e-3   # m. 좌표가 1e-4 m 단위로 반올림돼 있어 이보다 작은 변위는 사실상 정지다


def heading_rotation(obs):
    """마지막 관측 변위를 +x 로 보내는 회전 행렬.

    obs: [N, obs_len, 2] -> R: [N, 2, 2]   (점 p 는 p @ R^T 로 돌린다)
    """
    v = obs[:, -1] - obs[:, -2]                              # [N, 2]
    theta = torch.atan2(v[:, 1], v[:, 0])                    # [N] 진행 방향 각도
    theta = torch.where(v.norm(dim=-1) < MIN_HEADING_STEP, torch.zeros_like(theta), theta)
    cos, sin = torch.cos(theta), torch.sin(theta)
    # -theta 만큼 돌리는 행렬 [[cos, sin], [-sin, cos]]
    return torch.stack([torch.stack([cos, sin], -1),
                        torch.stack([-sin, cos], -1)], -2)   # [N, 2, 2]


class LSTMEncoderDecoder(nn.Module):
    def __init__(self, pred_len=12, embed_dim=64, hidden_dim=128, frame='world', residual=False):
        super().__init__()
        if frame not in ('world', 'heading'):
            raise ValueError(f"frame 은 'world' | 'heading': {frame}")
        self.pred_len = pred_len
        self.frame = frame
        self.residual = residual
        # 변위 (2) -> 임베딩 (embed_dim). 인코더 · 디코더가 같은 임베딩을 쓴다.
        self.embed = nn.Sequential(nn.Linear(2, embed_dim), nn.ReLU())
        self.encoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        # 디코더는 한 스텝씩 돌리므로 LSTMCell 을 쓴다.
        self.decoder = nn.LSTMCell(embed_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 2)     # 은닉 상태 -> 다음 변위 (residual 이면 보정량)
        if residual:
            nn.init.zeros_(self.head.weight)     # 시작할 때 보정량 0 = 등속 모델
            nn.init.zeros_(self.head.bias)

    def forward(self, obs):
        """obs: [N, obs_len, 2] -> 예측 위치 [N, pred_len, 2]

        obs 는 절대 좌표든 상대 좌표든 상관없다. 변위만 쓰고, 마지막 관측점에 더해 위치로 돌린다.
        """
        if self.frame == 'heading':
            last = obs[:, -1:, :]                             # [N, 1, 2]
            rot = heading_rotation(obs)                       # [N, 2, 2]
            pred = self._predict((obs - last) @ rot.transpose(1, 2))   # 진행 방향 좌표에서 예측
            return pred @ rot + last                          # 원래 좌표로: p @ (R^T)^-1 = p @ R
        return self._predict(obs)

    def _predict(self, obs):
        """obs: [N, obs_len, 2] -> [N, pred_len, 2]. 좌표계 변환 없는 본체."""
        disp = obs[:, 1:] - obs[:, :-1]                     # [N, obs_len-1, 2]

        # 인코더
        _, (h, c) = self.encoder(self.embed(disp))          # h, c: [1, N, hidden]
        h, c = h[0], c[0]                                   # [N, hidden]

        # 디코더: 자기 예측을 다음 입력으로
        last_disp = disp[:, -1]                             # [N, 2] 마지막 관측 변위 = 등속 변위
        step_in = last_disp
        preds = []
        for _ in range(self.pred_len):
            h, c = self.decoder(self.embed(step_in), (h, c))  # [N, hidden]
            step_in = self.head(h)                          # [N, 2] 다음 변위
            if self.residual:
                step_in = last_disp + step_in               # 등속 변위 + 보정량
            preds.append(step_in)
        pred_disp = torch.stack(preds, dim=1)               # [N, pred_len, 2]

        return obs[:, -1:, :] + pred_disp.cumsum(dim=1)     # [N, pred_len, 2]


def load_lstm(path, device='cpu'):
    """scripts/train_lstm.py 가 저장한 체크포인트에서 모델을 만든다 (eval 모드)."""
    ckpt = torch.load(path, map_location=device)
    args = ckpt['args']
    model = LSTMEncoderDecoder(embed_dim=args['embed_dim'], hidden_dim=args['hidden_dim'],
                               frame=args.get('frame', 'world'), residual=args.get('residual', False))
    model.load_state_dict(ckpt['state_dict'])
    return model.to(device).eval()
