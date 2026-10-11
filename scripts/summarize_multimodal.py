"""D단계 결과를 모은다.

입력
    results/multimodal/<cond>_seed<s>.csv, <cond>_seed<s>_kcurve.csv   (train_multimodal.py)
    results/multimodal/cv_sampling.csv                                  (eval_cv_sampling.py)
    results/cv.csv                                                      (등속, 결정적)
    checkpoints/mm_<cond>_seed<s>_<scene>.pt                           (모드 사용 진단용)

출력
    화면 (README · docs 에 그대로 붙일 표)
        1. 장면별 minADE_20 / minFDE_20 (시드 평균 ± 표본 표준편차)
        2. k = 1, 5, 20 의 AVG
        3. 모드 사용: test 에서 winner 가 된 비율이 1% 이상인 모드 수, 가장 많이 이긴 모드의 비율
    results/multimodal/summary.csv
    docs/figures/D_k_curve.png : k 에 따른 AVG minADE_k / minFDE_k
    docs/figures/D_examples.png : 장면마다 꺾는 샘플 하나의 20개 예측 (이웃 attention 모델, 시드 0)

    .venv/bin/python scripts/summarize_multimodal.py
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from trajpred.data import SCENES, load_split  # noqa: E402
from trajpred.models import load_lstm  # noqa: E402

MODELS = [('mm20_tf_heading_residual', 'Transformer K=20'),
          ('mm20_tf_heading_residual_social', 'Transformer K=20 + 이웃 attention')]
BASELINES = [('cvs25', '등속 + 방향 잡음 σ=25°'), ('cvs_fit', '등속 + 방향 잡음 (σ train 에서 고름)'),
             ('uniform', '등속 속력 + 무작위 방향')]
FIG_LABELS = {'cv': 'constant velocity (1 prediction)', 'cvs25': 'CV + heading noise σ=25°',
              'uniform': 'CV speed + uniform random heading',
              'mm20_tf_heading_residual': 'Transformer K=20',
              'mm20_tf_heading_residual_social': 'Transformer K=20 + nbr attention'}
# 범주형 색 (고정 순서, analyze_errors.py 와 같은 팔레트). 등속은 회색 참조선.
FIG_COLORS = {'cvs25': '#2a78d6', 'uniform': '#eb6834', 'mm20_tf_heading_residual': '#1baf7a',
              'mm20_tf_heading_residual_social': '#eda100'}
REF_COLOR, GRID_COLOR = '#8a8983', '#e4e3df'
ROWS = SCENES + ['AVG']


def fmt(values):
    values = np.asarray(values, dtype=float)
    return f'{values.mean():.3f} ± {values.std(ddof=1):.3f}' if len(values) > 1 else f'{values.mean():.3f}'


def load_curves(folder, seeds):
    """-> curves[name][seed][scene] = (ade [20], fde [20])  (k = 1..20)"""
    curves = defaultdict(dict)
    for cond, _ in MODELS:
        for s in seeds:
            path = folder / f'{cond}_seed{s}_kcurve.csv'
            if not path.exists():
                continue
            per = defaultdict(lambda: ([], []))
            for r in csv.DictReader(open(path)):
                per[r['scene']][0].append(float(r['min_ade']))
                per[r['scene']][1].append(float(r['min_fde']))
            curves[cond][s] = {sc: (np.array(a), np.array(f)) for sc, (a, f) in per.items()}
    per = defaultdict(lambda: defaultdict(lambda: ([], [])))
    for r in csv.DictReader(open(folder / 'cv_sampling.csv')):
        per[(r['method'], int(r['seed']))][r['scene']][0].append(float(r['min_ade']))
        per[(r['method'], int(r['seed']))][r['scene']][1].append(float(r['min_fde']))
    for (method, s), scenes in per.items():
        curves[method][s] = {sc: (np.array(a), np.array(f)) for sc, (a, f) in scenes.items()}
    for name in curves:                       # AVG = 장면 단순 평균
        for s in curves[name]:
            c = curves[name][s]
            c['AVG'] = (np.mean([c[sc][0] for sc in SCENES], axis=0), np.mean([c[sc][1] for sc in SCENES], axis=0))
    return curves


@torch.no_grad()
def mode_usage(cond, seeds, data_root, device):
    """test 에서 각 모드가 winner(minADE 모드)가 된 비율. -> {seed: (1% 이상 모드 수 평균, 최다 모드 비율 평균)}"""
    out = {}
    for s in seeds:
        n_used, top = [], []
        for scene in SCENES:
            path = Path(f'checkpoints/mm_{cond}_seed{s}_{scene}.pt')
            if not path.exists():
                continue
            model = load_lstm(path, device)
            social = model.social
            ds = load_split(data_root, scene, 'test', normalize=True, max_neighbors=32 if social else 0)
            obs, fut = ds.obs.to(device), ds.fut.to(device)
            wins = torch.zeros(model.num_modes, device=device)
            for idx in torch.arange(len(obs)).split(2048):
                nbr = (ds.nbr[idx].to(device), ds.nbr_mask[idx].to(device)) if social else (None, None)
                preds, _ = model(obs[idx], *nbr)
                err = torch.linalg.norm(preds - fut[idx][:, None], dim=-1).mean(-1)     # [B, K]
                wins += torch.bincount(err.argmin(1), minlength=model.num_modes)
            share = wins / wins.sum()
            n_used.append((share >= 0.01).sum().item())
            top.append(share.max().item())
        if n_used:
            out[s] = (np.mean(n_used), np.mean(top))
    return out


@torch.no_grad()
def plot_examples(cond, data_root, path, device, seed=0):
    """장면마다 관측 속력 >= 0.5 m/s 이고 실제 미래가 관측 방향에서 30° 넘게 꺾는 샘플을 하나 무작위로 골라
    20개 예측을 그린다. 선 굵기 · 진하기 = 모드 점수(softmax)."""
    gen = torch.Generator().manual_seed(seed)
    fig, axes = plt.subplots(1, len(SCENES), figsize=(3.2 * len(SCENES), 3.6))
    for ax, scene in zip(axes, SCENES):
        model = load_lstm(Path(f'checkpoints/mm_{cond}_seed{seed}_{scene}.pt'), device)
        ds = load_split(data_root, scene, 'test', normalize=True, max_neighbors=32 if model.social else 0)
        v = ds.obs[:, -1] - ds.obs[:, -2]                                   # [N, 2]
        speed = torch.linalg.norm(ds.obs[:, 1:] - ds.obs[:, :-1], dim=-1).mean(-1) / 0.4
        d = ds.fut[:, -1] - ds.obs[:, -1]                                   # [N, 2] 실제 4.8 s 변위
        cos = (v * d).sum(-1) / (torch.linalg.norm(v, dim=-1) * torch.linalg.norm(d, dim=-1) + 1e-9)
        cand = torch.nonzero((speed >= 0.5) & (cos < np.cos(np.radians(30)))).squeeze(1)
        i = cand[torch.randint(len(cand), (1,), generator=gen)].item()
        nbr = (ds.nbr[i:i + 1].to(device), ds.nbr_mask[i:i + 1].to(device)) if model.social else (None, None)
        preds, logits = model(ds.obs[i:i + 1].to(device), *nbr)
        preds, prob = preds[0].cpu(), torch.softmax(logits[0], -1).cpu()   # [20, 12, 2], [20]
        for k in prob.argsort():                                            # 점수 낮은 것부터 (높은 것이 위에)
            w = (prob[k] / prob.max()).item()
            ax.plot(preds[k, :, 0], preds[k, :, 1], '-', color='#e0393e', lw=0.6 + 1.6 * w, alpha=0.25 + 0.75 * w)
        obs, fut = ds.obs[i], ds.fut[i]
        ax.plot(obs[:, 0], obs[:, 1], '-o', color='#8a8983', lw=2, ms=3)
        ax.plot(torch.cat([obs[-1:], fut])[:, 0], torch.cat([obs[-1:], fut])[:, 1], '-', color='#008300', lw=2)
        err = torch.linalg.norm(preds - fut[None], dim=-1).mean(-1)         # [20]
        ax.set_title(f'{scene}: minADE {err.min():.2f} m\ntop-1 ADE {err[prob.argmax()]:.2f} m', fontsize=9)
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(True, color=GRID_COLOR)
        ax.tick_params(labelsize=7)
    handles = [plt.Line2D([], [], color='#8a8983', lw=2, marker='o', ms=3), plt.Line2D([], [], color='#008300', lw=2),
               plt.Line2D([], [], color='#e0393e', lw=1.5)]
    fig.legend(handles, ['observed (3.2 s)', 'actual future (4.8 s)', '20 predictions (thicker = higher score)'],
               loc='upper center', ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_k_curve(curves, cv, path):
    ks = np.arange(1, 21)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, j, title in ((axes[0], 0, 'AVG minADE$_k$ (m)'), (axes[1], 1, 'AVG minFDE$_k$ (m)')):
        ax.axhline(cv['AVG'][j], color=REF_COLOR, lw=1.5, ls='--')
        ax.text(8.5, cv['AVG'][j], f'CV {cv["AVG"][j]:.3f}', va='bottom', ha='center', fontsize=8, color='#555')
        for name, color in FIG_COLORS.items():
            if name not in curves:
                continue
            arr = np.array([curves[name][s]['AVG'][j] for s in sorted(curves[name])])   # [시드, 20]
            mean = arr.mean(0)
            ax.plot(ks, mean, '-', color=color, lw=2, marker='o', ms=4, markevery=[0, 4, 19])
            if len(arr) > 1:
                ax.fill_between(ks, arr.min(0), arr.max(0), color=color, alpha=0.15, lw=0)
        ax.set_xlabel('k (number of predictions kept)')
        ax.set_title(title, fontsize=10)
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.set_ylim(bottom=0)
        ax.grid(True, color=GRID_COLOR)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    handles = [plt.Line2D([], [], color=REF_COLOR, lw=1.5, ls='--')] + \
              [plt.Line2D([], [], color=c, lw=2, marker='o', ms=4) for n, c in FIG_COLORS.items() if n in curves]
    labels = [FIG_LABELS['cv']] + [FIG_LABELS[n] for n in FIG_COLORS if n in curves]
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default='results/multimodal')
    parser.add_argument('--data', default='data/raw')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--no-usage', action='store_true', help='모드 사용 진단(체크포인트 필요) 생략')
    args = parser.parse_args()
    folder = Path(args.dir)
    curves = load_curves(folder, args.seeds)
    cv = {r['scene']: (float(r['ade_m']), float(r['fde_m'])) for r in csv.DictReader(open('results/cv.csv'))}
    names = [(n, l) for n, l in BASELINES + MODELS if n in curves]

    def cell(name, scene, k):
        a = [curves[name][s][scene][0][k - 1] for s in sorted(curves[name])]
        f = [curves[name][s][scene][1][k - 1] for s in sorted(curves[name])]
        return a, f

    # 1. 장면별 k = 20
    print('minADE_20 / minFDE_20 (m), 시드 [0, 1, 2] 평균 ± 표본 표준편차 (등속은 결정적 예측 1개)\n')
    print('| 장면 | 등속 (1개) | ' + ' | '.join(l for _, l in names) + ' |')
    print('| :--- | :-: | ' + ' | '.join(':-:' for _ in names) + ' |')
    rows = []
    for scene in ROWS:
        cells = []
        for n, _ in names:
            a, f = cell(n, scene, 20)
            cells.append(f'{fmt(a)} / {fmt(f)}')
            rows.append({'name': n, 'scene': scene, 'k': 20, 'ade_mean': np.mean(a), 'ade_std': np.std(a, ddof=1),
                         'fde_mean': np.mean(f), 'fde_std': np.std(f, ddof=1)})
        label = '**AVG**' if scene == 'AVG' else scene
        print(f'| {label} | {cv[scene][0]:.3f} / {cv[scene][1]:.3f} | ' + ' | '.join(cells) + ' |')

    # 2. k = 1, 5, 20 AVG
    print('\nAVG minADE_k / minFDE_k (m)\n')
    print('| 방법 | k = 1 | k = 5 | k = 20 |')
    print('| :--- | :-: | :-: | :-: |')
    print(f'| 등속 (결정적) | {cv["AVG"][0]:.3f} / {cv["AVG"][1]:.3f} | — | — |')
    for n, label in names:
        cells = []
        for k in (1, 5, 20):
            a, f = cell(n, 'AVG', k)
            cells.append(f'{fmt(a)} / {fmt(f)}')
            if k != 20:
                rows.append({'name': n, 'scene': 'AVG', 'k': k, 'ade_mean': np.mean(a), 'ade_std': np.std(a, ddof=1),
                             'fde_mean': np.mean(f), 'fde_std': np.std(f, ddof=1)})
        print(f'| {label} | ' + ' | '.join(cells) + ' |')
    print('\n(모델의 k < 20 은 모드 점수가 높은 k개, 기준선은 독립 샘플 앞 k개)')

    with open(folder / 'summary.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: (f'{v:.6f}' if isinstance(v, float) else v) for k, v in r.items()} for r in rows)

    # 3. 모드 사용
    if not args.no_usage:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('\n모드 사용 (test, 장면 5개 평균): winner 비율 1% 이상인 모드 수 / 가장 많이 이긴 모드의 비율\n')
        for n, label in MODELS:
            usage = mode_usage(n, args.seeds, args.data, device)
            print(f'- {label}: ' + ', '.join(f'시드 {s} {u:.1f}개 / {t:.3f}' for s, (u, t) in usage.items()))

    figures = Path('docs/figures')
    plot_k_curve(curves, cv, figures / 'D_k_curve.png')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    plot_examples('mm20_tf_heading_residual_social', args.data, figures / 'D_examples.png', device)
    print(f'\nsaved {folder}/summary.csv, {figures}/D_k_curve.png, {figures}/D_examples.png')


if __name__ == '__main__':
    main()
