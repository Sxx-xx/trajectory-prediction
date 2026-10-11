"""README 요약 · 포트폴리오용 대표 그림 3장을 results/figures/ 에 만든다. 모두 이미 저장된 결과와 체크포인트에서 만든다.

    results/figures/model_comparison.png : 단일 예측 모델별 AVG ADE / FDE (시드 3개 평균 ± 표준편차, 등속 참조선)
    results/figures/turning_examples.png : 꺾는 보행자 예측 예시 (등속 / LSTM / Transformer + 이웃 + residual, 시드 0)
    results/figures/best_of_k.png        : best-of-k 지표의 한계 (무작위 방향 20개도 등속 FDE 를 이긴다)

입력: results/lstm/summary.csv, results/cv.csv, results/multimodal/*, checkpoints/lstm_*_seed0_<scene>.pt

    .venv/bin/python scripts/make_portfolio_figures.py
"""

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_multimodal import load_curves, plot_k_curve  # noqa: E402

from trajpred.data import SCENES, load_split  # noqa: E402
from trajpred.metrics import ade  # noqa: E402
from trajpred.models import ConstantVelocity, load_lstm  # noqa: E402

OUT = Path('results/figures')
INK, MUTED, GRID = '#2b2a27', '#6f6e69', '#e4e3df'
REF = '#8a8983'        # 등속 참조 (회색 점선)
BASE = '#2a78d6'       # 학습 모델 (한 색)
HIGHLIGHT = '#eb6834'  # 최종 모델 강조

# (summary.csv 의 condition, 그림에 쓸 이름)  위에서 아래로 단계 순서
MODELS = [('none', 'LSTM'),
          ('rotate', 'LSTM + rotation aug'),
          ('none_heading', 'LSTM + heading frame'),
          ('rotate_heading_social_g4d5', 'LSTM + grid pooling'),
          ('tf_none_heading', 'Transformer'),
          ('tf_none_heading_social', 'Transformer + nbr attention'),
          ('tf_none_heading_residual_social', 'Transformer + nbr attention + residual')]
FINAL = 'tf_none_heading_residual_social'


def model_comparison(path):
    stats = {}
    for cond, _ in MODELS:   # summarize_lstm.py 와 같이 시드별 AVG 에서 평균 · 표본 표준편차 (README 표와 같은 반올림)
        vals = np.array([[float(r['ade_m']), float(r['fde_m'])] for s in (0, 1, 2)
                         for r in csv.DictReader(open(f'results/lstm/{cond}_seed{s}.csv')) if r['scene'] == 'AVG'])
        stats[cond] = {'ade_mean': vals[:, 0].mean(), 'ade_std': vals[:, 0].std(ddof=1),
                       'fde_mean': vals[:, 1].mean(), 'fde_std': vals[:, 1].std(ddof=1)}
    cv = {r['scene']: (float(r['ade_m']), float(r['fde_m'])) for r in csv.DictReader(open('results/cv.csv'))}['AVG']
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    y = np.arange(len(MODELS))[::-1]                                   # 위에서 아래로
    for ax, key, title, ref in ((axes[0], 'ade', 'AVG ADE (m)', cv[0]), (axes[1], 'fde', 'AVG FDE (m)', cv[1])):
        ax.axvline(ref, color=REF, lw=1.5, ls='--')
        ax.text(ref, len(MODELS) - 0.25, f'constant velocity {ref:.3f}', ha='center', va='bottom', fontsize=8,
                color=MUTED)
        for yi, (cond, _) in zip(y, MODELS):
            m, s = float(stats[cond][f'{key}_mean']), float(stats[cond][f'{key}_std'])
            color = HIGHLIGHT if cond == FINAL else BASE
            ax.errorbar(m, yi, xerr=s, fmt='o', color=color, ms=8, elinewidth=2, capsize=0)
            ax.text(m, yi + 0.18, f'{m:.3f}', va='bottom', ha='center', fontsize=8, color=INK)
        ax.set_title(title, fontsize=10, color=INK)
        ax.grid(True, axis='x', color=GRID)
        ax.set_ylim(-0.6, len(MODELS))
        for side in ('top', 'right', 'left'):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis='y', length=0)
    axes[0].set_xlim(0.48, 0.60)
    axes[1].set_xlim(1.02, 1.24)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([name for _, name in MODELS], fontsize=9, color=INK)
    fig.suptitle('Single-prediction models, ETH/UCY leave-one-out (mean ± std over 3 seeds, lower is better)',
                 fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def turning_examples(path, seed=0):
    """장면마다 관측 속력 >= 0.5 m/s, 실제 미래가 관측 방향에서 30° 넘게 꺾고, 4 m 안에 이웃이 있는 샘플을
    무작위로 하나 고른다 (모델 성능으로 고르지 않는다)."""
    runs = [('none_heading_seed0', 'LSTM + heading', '#2a78d6'),
            ('tf_none_heading_residual_social_seed0', 'Transformer + nbr attn + residual', HIGHLIGHT)]
    gen = torch.Generator().manual_seed(seed)
    cv_model = ConstantVelocity(pred_len=12)
    fig, axes = plt.subplots(1, len(SCENES), figsize=(3.3 * len(SCENES), 3.9))
    for ax, scene in zip(axes, SCENES):
        ds = load_split('data/raw', scene, 'test', normalize=True, max_neighbors=32)
        obs, fut = ds.obs, ds.fut                                           # [N, 8, 2], [N, 12, 2]
        v = obs[:, -1] - obs[:, -2]
        d = fut[:, -1] - obs[:, -1]
        speed = torch.linalg.norm(obs[:, 1:] - obs[:, :-1], dim=-1).mean(-1) / 0.4
        cos = (v * d).sum(-1) / (torch.linalg.norm(v, dim=-1) * torch.linalg.norm(d, dim=-1) + 1e-9)
        nbr_last, mask_last = ds.nbr[:, -1, :, :2], ds.nbr_mask[:, -1]      # [N, K, 2], [N, K]
        near = ((torch.linalg.norm(nbr_last, dim=-1) < 4.0) & mask_last).any(-1)
        cand = torch.nonzero((speed >= 0.5) & (cos < np.cos(np.radians(30))) & near).squeeze(1)
        i = cand[torch.randint(len(cand), (1,), generator=gen)].item()

        o, f = obs[i:i + 1], fut[i]
        preds = [('constant velocity', REF, cv_model(o)[0])]
        for run, label, color in runs:
            model = load_lstm(Path(f'checkpoints/lstm_{run}_{scene}.pt'))
            preds.append((label, color, model(o, ds.nbr[i:i + 1], ds.nbr_mask[i:i + 1])[0]))

        nb = nbr_last[i][mask_last[i]]
        ax.scatter(nb[:, 0], nb[:, 1], s=14, color='#c9c8c3', zorder=1)
        ax.plot(o[0, :, 0], o[0, :, 1], '-o', color=INK, lw=2, ms=3, zorder=3)
        full = torch.cat([o[0, -1:], f])
        ax.plot(full[:, 0], full[:, 1], '-', color='#1baf7a', lw=2.5, zorder=3)
        lines = []
        for label, color, p in preds:
            p = torch.cat([o[0, -1:], p])
            ls = '--' if label == 'constant velocity' else '-'
            ax.plot(p[:, 0], p[:, 1], ls, color=color, lw=2, zorder=4)
            lines.append(f'{ade(p[None, 1:], f[None]).item():.2f}')
        ax.set_title(f'{scene}   ADE: CV {lines[0]} / LSTM {lines[1]} / TF {lines[2]} m', fontsize=8, color=INK)
        ax.set_aspect('equal', adjustable='box')
        lim = torch.cat([o[0], f, *[p for _, _, p in preds]])
        cx, cy = lim.mean(0).tolist()
        half = max((lim.max(0).values - lim.min(0).values).max().item() / 2 + 1.0, 2.5)
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)
        ax.grid(True, color=GRID)
        ax.tick_params(labelsize=7, colors=MUTED)
    handles = [plt.Line2D([], [], color=INK, lw=2, marker='o', ms=3),
               plt.Line2D([], [], color='#1baf7a', lw=2.5),
               plt.Line2D([], [], color=REF, lw=2, ls='--')] + \
              [plt.Line2D([], [], color=c, lw=2) for _, _, c in runs] + \
              [plt.Line2D([], [], color='#c9c8c3', marker='o', lw=0, ms=4)]
    labels = ['observed 3.2 s', 'actual future 4.8 s', 'constant velocity'] + [l for _, l, _ in runs] + \
             ['neighbors (last observed frame)']
    fig.legend(handles, labels, loc='upper center', ncol=6, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model_comparison(OUT / 'model_comparison.png')
    turning_examples(OUT / 'turning_examples.png')
    curves = load_curves(Path('results/multimodal'), [0, 1, 2])
    cv = {r['scene']: (float(r['ade_m']), float(r['fde_m'])) for r in csv.DictReader(open('results/cv.csv'))}
    plot_k_curve(curves, cv, OUT / 'best_of_k.png')
    print(f'saved {OUT}/model_comparison.png, turning_examples.png, best_of_k.png')


if __name__ == '__main__':
    main()
