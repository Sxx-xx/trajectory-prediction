"""LSTM 이 hotel 에서 등속 모델보다 나쁜 이유를 확인하는 진단.

1. 진행 방향 분포: 20스텝 동안 1 m 이상 움직인 보행자 중 x 방향 이동이 더 큰 비율
   (train = 나머지 4개 장면, test = 평가 장면)
2. 회전 시험: 학습한 best 모델에 test 입력을 90° 단위로 돌려 넣고, 예측을 되돌려 ADE 를 잰다.
   모델이 특정 진행 방향에 맞춰져 있다면 각도에 따라 ADE 가 달라진다.

    .venv/bin/python scripts/diag_lstm_direction.py   # scripts/train_lstm.py 를 먼저 실행할 것
"""

import torch

from trajpred.data import load_split
from trajpred.metrics import ade
from trajpred.models import LSTMEncoderDecoder


def x_dominant_fraction(ds, min_dist=1.0):
    """반환: (x 방향 이동이 더 큰 비율, 대상 보행자 수)"""
    d = ds.fut[:, -1] - ds.obs[:, 0]                 # [N, 2] 20스텝 동안의 이동
    moving = d.norm(dim=-1) > min_dist               # [N]
    return (d[moving, 0].abs() > d[moving, 1].abs()).float().mean().item(), int(moving.sum())


def main():
    rot90 = torch.tensor([[0., -1.], [1., 0.]])      # 반시계 90° 회전 행렬
    for scene in ['eth', 'hotel']:
        train = load_split('data/raw', scene, 'train', normalize=True)
        test = load_split('data/raw', scene, 'test', normalize=True)
        print(f'[{scene}] x 방향 우세 비율  train {x_dominant_fraction(train)}  test {x_dominant_fraction(test)}')

        ckpt = torch.load(f'checkpoints/lstm_{scene}.pt')
        model = LSTMEncoderDecoder(embed_dim=ckpt['args']['embed_dim'], hidden_dim=ckpt['args']['hidden_dim'])
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
        with torch.no_grad():
            for k in range(4):
                r = torch.linalg.matrix_power(rot90, k)          # [2, 2]
                pred = model(test.obs @ r.T) @ r                # 돌려 넣고 [N, 12, 2] 를 되돌린다
                print(f'[{scene}]   입력 회전 {90 * k:3d}°: test ADE {ade(pred, test.fut).mean():.3f}')


if __name__ == '__main__':
    main()
