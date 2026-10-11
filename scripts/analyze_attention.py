"""Transformer 이웃 cross-attention 이 어떤 이웃을 보는지 확인한다.

모든 test 샘플에서 마지막 관측 프레임의 attention 가중치 [null, 이웃 1..K] 를 모아 다음을 잰다.
    1. null 토큰에 준 가중치 평균 (= 이웃을 안 보는 정도)
    2. 이웃이 2명 이상인 샘플에서, 가중치가 가장 큰 이웃이 가장 가까운 이웃인 비율
       (무작위로 고를 때의 기대값 = 평균 1 / 이웃 수 와 비교)
    3. 이웃 한 명당 가중치 평균을 거리 · 다가옴/멀어짐 · 앞/뒤로 나눠서
       - 가중치는 null 을 뺀 이웃들끼리 다시 합이 1 이 되게 나눈 값에, 그 샘플의 이웃 수를 곱한다
         (= 균등하게 나눌 때 1). 1 보다 크면 '평균보다 더 본다'
       - 다가옴: 상대 속도가 대상 쪽을 향함 (closing speed = -(p . v_rel) / |p| > 0.1 m/s)
       - 앞: 대상의 진행 방향(마지막 관측 변위) 쪽에 있음
    4. 예시 그림: 15° 이상 꺾고 4 m 안에 이웃이 3명 이상인 샘플
    5. 이웃 지우기 시험 (평가 때만): 마지막 관측 프레임의 이웃 중 거리 구간 하나를 가리고(mask)
       ADE / FDE 가 얼마나 변하는지 잰다. 가중치가 아니라 '실제로 예측에 얼마나 쓰이는지' 를 본다.
       학습 때 본 적 없는 이웃 분포가 되므로 대략적인 의존도로만 읽는다

'all scenes' 값은 장면별 값을 단순 평균한 것이다 (README AVG 와 같은 관례).

    .venv/bin/python scripts/analyze_attention.py --runs tf_none_heading_social_seed0 \\
        tf_none_heading_social_shuffle_seed0
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from trajpred.data import SCENES, load_split  # noqa: E402
from trajpred.models import load_lstm  # noqa: E402

DT = 0.4
DIST_BINS = [('<1 m', 0, 1), ('1-2 m', 1, 2), ('2-4 m', 2, 4), ('4+ m', 4, 1e9)]
APPROACH = 0.1            # m/s. closing speed 가 이보다 크면 '다가옴', -APPROACH 보다 작으면 '멀어짐'
PAST_COLOR, FUTURE_COLOR, PRED_COLOR = '#8a8983', '#008300', '#e0393e'


def collect(root, scene, run, shuffle):
    """장면 하나의 attention 과 이웃 특징.

    반환 dict
        null [N], w [N, K] (null 을 뺀 이웃 가중치, 이웃끼리 합 1, 이웃 없으면 0), mask [N, K]
        dist, closing [N, K] (m, m/s), ahead [N, K] bool, n_nbr [N]
        obs [N, 8, 2], fut [N, 12, 2], pred [N, 12, 2], rel [N, K, 2], vel [N, K, 2] (마지막 관측 프레임)
    """
    ds = load_split(root, scene, 'test', normalize=True, max_neighbors=32)
    nbr, mask = ds.nbr, ds.nbr_mask                                    # [N, 8, K, 4], [N, 8, K]
    if shuffle:
        # 학습 때와 같은 방식으로 섞는다 (train_lstm.py 의 split_tensors 와 같은 시드)
        seed = int(run.rsplit('_seed', 1)[1])
        perm = torch.randperm(len(nbr), generator=torch.Generator().manual_seed(seed))
        nbr, mask = nbr[perm], mask[perm]
    model = load_lstm(f'checkpoints/lstm_{run}_{scene}.pt')
    with torch.no_grad():
        pred = model(ds.obs, nbr, mask)                                # [N, 12, 2]
    att = model.last_attention                                         # [N, 1 + K]
    m = mask[:, -1]                                                    # [N, K]
    rel, vel = nbr[:, -1, :, :2], nbr[:, -1, :, 2:]                    # [N, K, 2]
    target_v = (ds.obs[:, -1] - ds.obs[:, -2])                         # [N, 2] m / 스텝
    rel_v = (vel - target_v[:, None]) / DT                             # [N, K, 2] m/s
    dist = rel.norm(dim=-1)                                            # [N, K]
    closing = -(rel * rel_v).sum(-1) / dist.clamp_min(1e-6)            # [N, K] + 면 다가옴
    ahead = (rel * target_v[:, None]).sum(-1) > 0                      # [N, K]
    nbr_w = att[:, 1:] * m                                             # [N, K]
    nbr_w = nbr_w / nbr_w.sum(-1, keepdim=True).clamp_min(1e-12)
    return {'null': att[:, 0].numpy(), 'w': nbr_w.numpy(), 'mask': m.numpy(), 'dist': dist.numpy(),
            'closing': closing.numpy(), 'ahead': ahead.numpy(), 'n_nbr': m.sum(-1).numpy(),
            'obs': ds.obs.numpy(), 'fut': ds.fut.numpy(), 'pred': pred.numpy(),
            'rel': rel.numpy(), 'vel': vel.numpy()}


def relative_weight(d, select):
    """select [N, K] 인 이웃들의 '이웃 수 x 정규화 가중치' 평균. 1 = 균등."""
    scaled = d['w'] * d['n_nbr'][:, None]                              # [N, K]
    sel = select & d['mask'] & (d['n_nbr'][:, None] >= 2)
    return float(scaled[sel].mean()) if sel.any() else float('nan'), int(sel.sum())


def scene_stats(d):
    multi = d['n_nbr'] >= 2
    dist = np.where(d['mask'], d['dist'], np.inf)
    nearest = dist.argmin(-1)
    top = d['w'].argmax(-1)
    stats = {'null': float(d['null'].mean()),
             'top_is_nearest': float((top == nearest)[multi].mean()),
             'chance': float((1.0 / d['n_nbr'][multi]).mean())}
    for name, lo, hi in DIST_BINS:
        stats[f'dist {name}'] = relative_weight(d, (d['dist'] >= lo) & (d['dist'] < hi))
    stats['approaching'] = relative_weight(d, d['closing'] > APPROACH)
    stats['receding'] = relative_weight(d, d['closing'] < -APPROACH)
    stats['ahead'] = relative_weight(d, d['ahead'])
    stats['behind'] = relative_weight(d, ~d['ahead'])
    return stats


ABLATIONS = [('all neighbors', None), ('drop <2 m', (0, 2)), ('drop 2-4 m', (2, 4)), ('drop 4+ m', (4, 1e9)),
             ('drop all', (0, 1e9)), ('keep positions, zero velocities', 'zero_vel')]


def ablation(root, run):
    """-> {조건: {'hotel': (ADE, FDE), 'all scenes': (ADE, FDE)}}  all scenes = 장면 평균"""
    from trajpred.metrics import ade, fde
    per_scene = {name: {} for name, _ in ABLATIONS}
    for scene in SCENES:
        ds = load_split(root, scene, 'test', normalize=True, max_neighbors=32)
        model = load_lstm(f'checkpoints/lstm_{run}_{scene}.pt')
        dist = ds.nbr[:, -1, :, :2].norm(dim=-1)                       # [N, K]
        for name, band in ABLATIONS:
            mask, nbr = ds.nbr_mask.clone(), ds.nbr                    # [N, 8, K], [N, 8, K, 4]
            if band == 'zero_vel':
                nbr = nbr.clone()
                nbr[..., 2:] = 0                                        # 이웃 속도만 지운다 (위치는 그대로)
            elif band is not None:
                mask[:, -1] &= ~((dist >= band[0]) & (dist < band[1]))  # 모델은 마지막 프레임만 쓴다
            with torch.no_grad():
                pred = model(ds.obs, nbr, mask)
            per_scene[name][scene] = (ade(pred, ds.fut).mean().item(), fde(pred, ds.fut).mean().item())
    return {name: {'hotel': v['hotel'], 'all scenes': tuple(np.mean([v[s] for s in SCENES], axis=0))}
            for name, v in per_scene.items()}


def print_ablation(results):
    print('\n### 이웃 지우기 시험 (평가 때만, 마지막 관측 프레임). ADE / FDE (m), all scenes = 장면 평균')
    runs = list(results)
    print('| 조건 | ' + ' | '.join(f'{r} hotel | {r} all' for r in runs) + ' |')
    print('| :--- |' + ' --: | --: |' * len(runs))
    for name, _ in ABLATIONS:
        cells = []
        for r in runs:
            for g in ('hotel', 'all scenes'):
                a, f = results[r][name][g]
                cells.append(f'{a:.3f} / {f:.3f}')
        print(f'| {name} | ' + ' | '.join(cells) + ' |')


def print_tables(all_stats, runs):
    print('### null 가중치와 "가장 많이 본 이웃 = 가장 가까운 이웃" 비율 (이웃 2명 이상)')
    print('| 모델 | 장면 | null 가중치 | top = nearest | 무작위 기대값 |')
    print('| :--- | :--- | --: | --: | --: |')
    for run in runs:
        for scene in SCENES + ['all scenes']:
            s = all_stats[run][scene]
            print(f"| {run} | {scene} | {s['null']:.3f} | {s['top_is_nearest']:.3f} | {s['chance']:.3f} |")
    keys = [f'dist {b[0]}' for b in DIST_BINS] + ['approaching', 'receding', 'ahead', 'behind']
    print('\n### 이웃 한 명당 상대 가중치 (1 = 균등하게 나눌 때). all scenes = 장면 평균, 괄호는 이웃 수 합')
    print('| 모델 | ' + ' | '.join(keys) + ' |')
    print('| :--- |' + ' --: |' * len(keys))
    for run in runs:
        cells = []
        for k in keys:
            vals = [all_stats[run][s][k][0] for s in SCENES]
            count = sum(all_stats[run][s][k][1] for s in SCENES)
            cells.append(f'{np.nanmean(vals):.2f} ({count:,})')
        print(f'| {run} | ' + ' | '.join(cells) + ' |')


def plot_examples(data, run, path, n_examples=6):
    """15° 이상 꺾고 4 m 안 이웃 3명 이상인 샘플 중, 장면을 돌아가며 고른다 (각 장면에서 무작위)."""
    rng = np.random.default_rng(0)
    picks = []
    for scene in SCENES:
        d = data[scene]
        v = d['obs'][:, -1] - d['obs'][:, -2]
        end = d['fut'][:, -1]
        cos = (v * end).sum(-1) / np.maximum(np.linalg.norm(v, axis=-1) * np.linalg.norm(end, axis=-1), 1e-9)
        turn = np.degrees(np.arccos(np.clip(cos, -1, 1)))
        near = ((d['dist'] < 4) & d['mask']).sum(-1)
        ok = np.nonzero((np.linalg.norm(v, axis=-1) >= 0.08) & (turn >= 15) & (near >= 3))[0]
        if len(ok):
            picks.append((scene, int(rng.choice(ok))))
    picks = picks[:n_examples]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.6))
    cmap = plt.get_cmap('Blues')
    for ax, (scene, i) in zip(axes.flat, picks):
        d = data[scene]
        obs, fut, pred = d['obs'][i], d['fut'][i], d['pred'][i]
        ax.plot(obs[:, 0], obs[:, 1], '-o', color=PAST_COLOR, ms=3, lw=2)
        ax.plot(np.r_[obs[-1:, 0], fut[:, 0]], np.r_[obs[-1:, 1], fut[:, 1]], '-o', color=FUTURE_COLOR, ms=3, lw=2)
        ax.plot(np.r_[obs[-1:, 0], pred[:, 0]], np.r_[obs[-1:, 1], pred[:, 1]], '--o', color=PRED_COLOR, ms=3, lw=2)
        m = d['mask'][i]
        rel, vel, w = d['rel'][i][m], d['vel'][i][m], d['w'][i][m]
        order = np.argsort(w)                                           # 가중치 큰 이웃을 위에 그린다
        sc = ax.scatter(rel[order, 0], rel[order, 1], c=w[order], cmap=cmap, vmin=0, vmax=max(w.max(), 1e-6),
                        s=40 + 400 * w[order] / max(w.max(), 1e-6), edgecolors='#3d3d3a', linewidths=0.6, zorder=3)
        ax.quiver(rel[:, 0], rel[:, 1], vel[:, 0] / DT, vel[:, 1] / DT, angles='xy', scale_units='xy', scale=1,
                  width=0.004, color='#3d3d3a', alpha=0.6, zorder=2)
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=7)
        ax.set_title(f'{scene} #{i}  null weight {d["null"][i]:.2f}', fontsize=10)
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(True, color='#e4e3df')
        ax.tick_params(labelsize=8)
    for ax in list(axes.flat)[len(picks):]:
        ax.axis('off')
    handles = [plt.Line2D([], [], color=c, lw=2, ls=ls, marker='o', ms=3)
               for c, ls in ((PAST_COLOR, '-'), (FUTURE_COLOR, '-'), (PRED_COLOR, '--'))]
    handles.append(plt.Line2D([], [], ls='', marker='o', ms=8, mfc=cmap(0.8), mec='#3d3d3a'))
    fig.legend(handles, ['past (8 steps)', 'future (ground truth)', 'prediction',
                         'neighbor at last observed frame (color, size = attention among neighbors; arrow = velocity m/s)'],
               loc='upper center', ncol=2, frameon=False, fontsize=9)
    fig.supxlabel('x (m), relative to last observed point', fontsize=10)
    fig.supylabel('y (m)', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/raw')
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--figures', default='docs/figures')
    args = parser.parse_args()
    Path(args.figures).mkdir(parents=True, exist_ok=True)

    all_stats = {}
    for run in args.runs:
        shuffle = '_shuffle' in run
        data = {scene: collect(args.data, scene, run, shuffle) for scene in SCENES}
        stats = {scene: scene_stats(data[scene]) for scene in SCENES}
        avg = {}
        for k in stats['eth']:
            if isinstance(stats['eth'][k], tuple):
                avg[k] = (float(np.nanmean([stats[s][k][0] for s in SCENES])), sum(stats[s][k][1] for s in SCENES))
            else:
                avg[k] = float(np.mean([stats[s][k] for s in SCENES]))
        stats['all scenes'] = avg
        all_stats[run] = stats
        plot_examples(data, run, Path(args.figures) / f'C_attention_{run}.png')
    print_tables(all_stats, args.runs)
    print_ablation({run: ablation(args.data, run) for run in args.runs if '_shuffle' not in run})
    print(f'\nsaved {args.figures}/C_attention_*.png')


if __name__ == '__main__':
    main()
