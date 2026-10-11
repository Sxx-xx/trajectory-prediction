"""D단계 기준선: 등속 모델에 잡음을 넣어 K개 예측을 만들고 best-of-K 로 잰다. 학습 없음.

방법 (모두 마지막 관측 변위 v 의 크기는 그대로 두고 방향만 바꾼다)
    cvs25   : Schöller et al. 2020 의 CVM-S. 방향 각도에 N(0, 25°^2) 잡음 (논문 TABLE I 의 OUR-S)
    cvs_fit : 같은 잡음인데 σ 를 각 분할의 train 데이터에서 minADE_20 이 가장 작게 고른 것
              (test 장면은 고를 때 쓰지 않는다)
    uniform : 방향을 0~360° 에서 균등하게 뽑는다. 관측 방향을 버리는 '거의 무작위' 예측.
              best-of-K 지표가 K 가 클 때 얼마나 관대한지 보여 주는 용도
예측 k 번째 샘플 = obs[-1] + (스텝 1..12) x R(θ_k) v.  K개는 서로 독립이므로 앞 k개만 쓰면 best-of-k 다.

출력
    results/multimodal/cv_sampling.csv : method, sigma_deg, seed, scene, k, min_ade, min_fde  (k = 1..20)
    화면                               : 고른 σ 와 k = 1, 5, 20 표

    .venv/bin/python scripts/eval_cv_sampling.py
"""

import argparse
import csv
import math
from pathlib import Path

import torch

from trajpred.data import SCENES, load_split
from trajpred.metrics import min_ade, min_fde

K_MAX = 20
SEEDS = (0, 1, 2)
SIGMA_GRID = list(range(0, 65, 5))      # σ 후보 (도)


def sample_cv(obs, angles, pred_len=12):
    """obs: [N, 8, 2], angles: [N, K] (라디안, 관측 방향에 더할 회전) -> [N, K, pred_len, 2]"""
    v = obs[:, -1] - obs[:, -2]                                   # [N, 2]
    cos, sin = torch.cos(angles), torch.sin(angles)               # [N, K]
    vx = cos * v[:, None, 0] - sin * v[:, None, 1]                # [N, K]  R(θ) v
    vy = sin * v[:, None, 0] + cos * v[:, None, 1]
    rv = torch.stack([vx, vy], dim=-1)                            # [N, K, 2]
    steps = torch.arange(1, pred_len + 1, dtype=obs.dtype)        # [12]
    return obs[:, None, -1:, :] + steps[None, None, :, None] * rv[:, :, None, :]   # [N, K, 12, 2]


def draw_angles(method, sigma_deg, n, generator):
    """-> [n, K_MAX] 라디안"""
    if method == 'uniform':
        return torch.rand(n, K_MAX, generator=generator) * 2 * math.pi
    return torch.randn(n, K_MAX, generator=generator) * math.radians(sigma_deg)


def fit_sigma(root, scene):
    """scene 을 뺀 train 데이터에서 minADE_20 이 가장 작은 σ (시드 0)."""
    ds = load_split(root, scene, 'train', normalize=True)
    scores = {}
    for sigma in SIGMA_GRID:
        g = torch.Generator().manual_seed(0)
        preds = sample_cv(ds.obs, draw_angles('normal', sigma, len(ds), g))
        scores[sigma] = min_ade(preds, ds.fut).mean().item()
    return min(scores, key=scores.get), scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/raw')
    parser.add_argument('--out', default='results/multimodal')
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    fitted = {}
    for scene in SCENES:
        fitted[scene], scores = fit_sigma(args.data, scene)
        print(f'[{scene}] train minADE20 by σ: ' + ' '.join(f'{s}°:{v:.3f}' for s, v in scores.items())
              + f'  -> σ = {fitted[scene]}°')

    rows = []
    for scene in SCENES:
        ds = load_split(args.data, scene, 'test', normalize=True)
        for method, sigma in (('cvs25', 25), ('cvs_fit', fitted[scene]), ('uniform', None)):
            for seed in SEEDS:
                g = torch.Generator().manual_seed(seed)
                preds = sample_cv(ds.obs, draw_angles('uniform' if method == 'uniform' else 'normal',
                                                      sigma, len(ds), g))      # [N, 20, 12, 2]
                for k in range(1, K_MAX + 1):
                    rows.append({'method': method, 'sigma_deg': '' if sigma is None else sigma, 'seed': seed,
                                 'scene': scene, 'k': k,
                                 'min_ade': min_ade(preds[:, :k], ds.fut).mean().item(),
                                 'min_fde': min_fde(preds[:, :k], ds.fut).mean().item()})

    with open(out / 'cv_sampling.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: (f'{v:.6f}' if isinstance(v, float) else v) for k, v in r.items()} for r in rows)

    # 화면: 시드 평균, 장면 단순 평균
    for method in ('cvs25', 'cvs_fit', 'uniform'):
        line = []
        for k in (1, 5, 20):
            sel = [r for r in rows if r['method'] == method and r['k'] == k]
            a = sum(r['min_ade'] for r in sel) / len(sel)
            f_ = sum(r['min_fde'] for r in sel) / len(sel)
            line.append(f'k={k} {a:.3f}/{f_:.3f}')
        print(f'{method:8s} AVG  ' + '  '.join(line))
    print(f'saved {out}/cv_sampling.csv')


if __name__ == '__main__':
    main()
