"""Transformer 예측 모델. 시간축 self-attention 인코더 + 한 번에 12스텝을 내는 디코더(non-autoregressive).

입력과 출력은 LSTM 과 같이 '한 스텝 변위'다.
    토큰 t (t = 1..7) = [변위 d[t] = obs[t] - obs[t-1],  위치 obs[t] - obs[-1]]  (4)
                      -> Linear -> d_model,  + 학습하는 위치 임베딩 (시점마다 하나)
    TransformerEncoder (self-attention 이 7개 토큰 전체를 서로 본다. 과거만 보므로 마스크는 없다)
    마지막 토큰(t = 7)의 출력 = 대상 보행자 요약 h  [N, d_model]
    h -> MLP -> 12 x 2 변위를 한 번에 -> 누적합으로 위치

social=True 이면 이웃 cross-attention 을 더한다.
    이웃 토큰 = 마지막 관측 프레임의 이웃마다 [상대 위치 (2), 속도 (2), 상대 속도 (2)] -> Linear -> d_model
    query = h,  key/value = [null 토큰, 이웃 토큰들]
        null 토큰: 학습하는 벡터 하나. 이웃이 0명이어도 attention 할 곳이 있게 하고,
                   'null 에 준 가중치' = '이웃을 안 보는 정도' 로 해석할 수 있다
        없는 이웃(빈 칸)은 key_padding_mask 로 가린다 -> 이웃 수가 샘플마다 달라도 된다
    h <- LayerNorm(h + attention 출력),  그다음 같은 MLP 로 예측

frame='heading' 이면 LSTM 과 같은 진행 방향 정규화를 쓴다 (이웃도 같이 돈다).
"""

import torch
from torch import nn

from .lstm import heading_rotation


class TransformerPredictor(nn.Module):
    def __init__(self, obs_len=8, pred_len=12, d_model=64, n_heads=4, n_layers=2, ff_dim=128,
                 dropout=0.1, frame='heading', social=False):
        super().__init__()
        if frame not in ('world', 'heading'):
            raise ValueError(f"frame 은 'world' | 'heading': {frame}")
        self.pred_len = pred_len
        self.frame = frame
        self.social = social
        steps = obs_len - 1                                         # 변위 토큰 수 (7)

        self.input_proj = nn.Linear(4, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, steps, d_model))   # 시점 임베딩 [1, 7, d]
        nn.init.normal_(self.pos_embed, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, ff_dim, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)

        if social:
            self.nbr_proj = nn.Linear(6, d_model)
            self.null_token = nn.Parameter(torch.zeros(1, 1, d_model))   # [1, 1, d]
            nn.init.normal_(self.null_token, std=0.02)
            self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            self.cross_norm = nn.LayerNorm(d_model)

        self.head = nn.Sequential(nn.Linear(d_model, ff_dim), nn.ReLU(), nn.Linear(ff_dim, pred_len * 2))
        self.last_attention = None    # 마지막 forward 의 cross-attention 가중치 [N, 1 + K] (시각화용)

    def forward(self, obs, nbr=None, nbr_mask=None):
        """obs: [N, obs_len, 2] -> 예측 위치 [N, pred_len, 2]

        social 이면 nbr [N, obs_len, K, 4] (상대 위치, 속도), nbr_mask [N, obs_len, K] 가 필요하다.
        마지막 관측 프레임(t = obs_len - 1)의 이웃만 쓴다.
        """
        if self.social and nbr is None:
            raise ValueError('social 모델은 nbr, nbr_mask 가 필요하다 (load_split(..., max_neighbors=K))')
        last = obs[:, -1:, :]                                       # [N, 1, 2]
        x = obs - last                                              # 마지막 관측점 기준 [N, obs_len, 2]
        if self.social:
            nbr, nbr_mask = nbr[:, -1], nbr_mask[:, -1]             # [N, K, 4], [N, K]
        if self.frame == 'heading':
            rot_t = heading_rotation(obs).transpose(1, 2)           # [N, 2, 2] (R^T)
            x = x @ rot_t
            if self.social:
                n, k, _ = nbr.shape
                nbr = (nbr.reshape(n, k * 2, 2) @ rot_t).reshape(n, k, 4)
        pred = self._predict(x, nbr, nbr_mask)                      # [N, pred_len, 2] (원점 기준)
        if self.frame == 'heading':
            pred = pred @ rot_t.transpose(1, 2)                     # 원래 방향으로: p @ R
        return pred + last

    def _predict(self, x, nbr, nbr_mask):
        """x: [N, obs_len, 2] (마지막 관측점 = 원점) -> [N, pred_len, 2]"""
        disp = x[:, 1:] - x[:, :-1]                                 # [N, 7, 2]
        tokens = torch.cat([disp, x[:, 1:]], dim=-1)                # [N, 7, 4] 변위, 위치
        h = self.encoder(self.input_proj(tokens) + self.pos_embed)  # [N, 7, d]
        h = h[:, -1]                                                # [N, d] 마지막 시점 요약

        if self.social:
            n = x.shape[0]
            rel_vel = nbr[..., 2:] - disp[:, -1:, :]                # [N, K, 2] 이웃 속도 - 대상 속도
            nbr_tokens = self.nbr_proj(torch.cat([nbr, rel_vel], dim=-1))     # [N, K, d]
            keys = torch.cat([self.null_token.expand(n, -1, -1), nbr_tokens], dim=1)   # [N, 1 + K, d]
            pad = torch.cat([torch.zeros(n, 1, dtype=torch.bool, device=x.device), ~nbr_mask], dim=1)  # [N, 1 + K]
            ctx, weights = self.cross_attn(h[:, None, :], keys, keys, key_padding_mask=pad,
                                           need_weights=True, average_attn_weights=True)  # [N, 1, d], [N, 1, 1 + K]
            self.last_attention = weights[:, 0].detach()            # [N, 1 + K]
            h = self.cross_norm(h + ctx[:, 0])                      # [N, d]

        pred_disp = self.head(h).reshape(-1, self.pred_len, 2)      # [N, 12, 2] 한 번에
        return pred_disp.cumsum(dim=1)                              # [N, 12, 2]
