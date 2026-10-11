"""등속 모델 (constant velocity). 학습이 필요 없는 기준선.

마지막 두 관측점의 차이를 한 스텝 속도로 보고, 그 속도로 계속 걷는다고 가정한다.
    v       = obs[-1] - obs[-2]
    pred[k] = obs[-1] + (k + 1) * v      (k = 0 .. pred_len-1)

좌표를 평행이동해도 결과가 같으므로 절대 좌표 · 상대 좌표 어느 쪽에도 쓸 수 있다.
학습형 모델과 같은 방식으로 부를 수 있게 nn.Module 로 만든다.
"""

import torch
from torch import nn


class ConstantVelocity(nn.Module):
    def __init__(self, pred_len=12):
        super().__init__()
        self.pred_len = pred_len

    def forward(self, obs):
        """obs: [N, obs_len, 2] -> 예측 [N, pred_len, 2]"""
        velocity = obs[:, -1] - obs[:, -2]                                    # [N, 2]
        steps = torch.arange(1, self.pred_len + 1, dtype=obs.dtype, device=obs.device)  # [pred_len]
        return obs[:, -1, None, :] + steps[None, :, None] * velocity[:, None, :]       # [N, pred_len, 2]
