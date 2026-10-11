"""등속 모델과 LSTM 의 오차를 상황별로 나눠 본다 (docs/analysis.md 용).

모든 test 샘플에 대해 모델별 예측을 만들고, 다음 기준으로 묶어 평균을 낸다.
    1. 예측 스텝 (1~12) 별 평균 거리 오차
    2. 미래 구간의 방향 전환 각도별 ADE
         각도 = 마지막 관측 변위 방향과 '마지막 관측점 -> 실제 마지막 미래점' 방향 사이의 각
         마지막 관측 변위가 0.08 m(0.2 m/s) 미만이거나 4.8 s 동안 0.5 m 미만 움직이면
         방향이 정해지지 않으므로 'slow/undefined' 로 따로 묶는다
    3. 관측 속력(관측 8점의 평균 속력) 구간별 ADE
    4. 마지막 스텝 오차를 진행 방향 성분(along: +면 예측이 더 멀리 감)과 옆 성분(cross)으로 분해
    5. 이동 거리 반응: 4.8 s 동안의 예측 이동 거리를 등속 이동 거리(12 × 마지막 관측 변위)에
       직선 회귀한 기울기. 1 이면 관측 속도를 그대로 따르고, 1 보다 작으면 평균 쪽으로 줄인다
       (실제 미래는 test 와 train 분할 둘 다 잰다)
    6. 속도를 k 배로 줄인 등속 모델의 test ADE. 학습 없이 '속도 줄이기'만의 효과를 본다

'hotel' 은 hotel test 샘플만, 'all scenes' 는 5개 장면을 같은 비중으로 합친 값이다
(샘플 가중치 1 / 장면 샘플 수). 그래서 'all scenes' 의 전체 ADE 는 README 표의 AVG 와 같다.

출력 (<tag> 는 --tag)
    docs/figures/<tag>_step_error.png, <tag>_turn_ade.png, <tag>_speed_ade.png
    results/analysis/<tag>_bins.csv      : 묶음별 샘플 수 · ADE (장면별 · hotel · all)
    results/analysis/<tag>_steps.csv     : 스텝별 평균 오차
    화면                                 : 위 표 + hotel 오차 차이 분해 + along/cross 분해

    .venv/bin/python scripts/analyze_errors.py                       # 등속 / 증강 없음 / 회전 증강 (시드 0)
    .venv/bin/python scripts/analyze_errors.py --tag all --runs none_seed0 rotate_seed0 \\
        none_heading_seed0 none_residual_seed0 none_heading_residual_seed0
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from trajpred.data import SCENES, load_split  # noqa: E402
from trajpred.metrics import displacement  # noqa: E402
from trajpred.models import ConstantVelocity, load_lstm  # noqa: E402

DT = 0.4
SLOW_STEP = 0.08          # m / 스텝 (= 0.2 m/s). 이보다 느리면 진행 방향을 정하지 않는다
MIN_FUTURE_DIST = 0.5     # m. 4.8 s 동안 이보다 덜 움직이면 전환 각도를 정하지 않는다
TURN_BINS = [('slow/undefined', None, None), ('0-15°', 0, 15), ('15-45°', 15, 45), ('45°+', 45, 181)]
SPEED_BINS = [('<0.2 m/s', 0, 0.2), ('0.2-0.8', 0.2, 0.8), ('0.8-1.5', 0.8, 1.5), ('1.5+ m/s', 1.5, 99)]
# 범주형 색 (고정 순서). 등속은 항상 첫 색이다.
COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#6250d6']
LABELS = {'cv': 'constant velocity', 'none': 'LSTM', 'rotate': 'LSTM + rotation aug',
          'none_heading': 'LSTM heading', 'none_residual': 'LSTM residual',
          'none_heading_residual': 'LSTM heading + residual'}
GRID_COLOR = '#e4e3df'


def label_of(run):
    return LABELS.get(run.rsplit('_seed', 1)[0], run)


def scene_samples(root, scene, runs):
    """장면 하나의 샘플별 특징과 모델별 오차.

    반환 dict (N = 샘플 수)
        speed [N], turn [N] (정해지지 않으면 nan), heading [N, 2] (단위 벡터, 정지면 0)
        err   {모델: [N, 12]}  스텝별 거리 오차
        final {모델: [N, 2]}   마지막 스텝 오차 벡터 (예측 - 실제)
        end   {모델: [N, 2]}   마지막 예측점 (마지막 관측점 기준), 'gt' 는 실제 마지막 미래점
    """
    ds = load_split(root, scene, 'test', normalize=True)
    obs, fut = ds.obs, ds.fut                                         # [N, 8, 2], [N, 12, 2]
    models = {'cv': ConstantVelocity(pred_len=12)}
    models.update({run: load_lstm(f'checkpoints/lstm_{run}_{scene}.pt') for run in runs})

    out = {'err': {}, 'final': {}, 'end': {'gt': fut[:, -1].numpy()}}
    with torch.no_grad():
        for name, model in models.items():
            pred = model(obs)                                         # [N, 12, 2]
            out['err'][name] = displacement(pred, fut).numpy()        # [N, 12]
            out['final'][name] = (pred[:, -1] - fut[:, -1]).numpy()   # [N, 2]
            out['end'][name] = pred[:, -1].numpy()                    # [N, 2]

    steps = obs[:, 1:] - obs[:, :-1]                                  # [N, 7, 2]
    out['speed'] = (steps.norm(dim=-1).mean(dim=-1) / DT).numpy()     # [N] m/s
    v = steps[:, -1]                                                  # [N, 2] 마지막 관측 변위
    d = fut[:, -1]                                                    # [N, 2] 마지막 관측점(원점) -> 마지막 미래점
    v_norm, d_norm = v.norm(dim=-1), d.norm(dim=-1)
    cos = (v * d).sum(-1) / (v_norm * d_norm).clamp_min(1e-9)
    turn = torch.rad2deg(torch.arccos(cos.clamp(-1, 1)))
    defined = (v_norm >= SLOW_STEP) & (d_norm >= MIN_FUTURE_DIST)
    out['turn'] = torch.where(defined, turn, torch.full_like(turn, float('nan'))).numpy()
    out['heading'] = torch.where((v_norm >= SLOW_STEP)[:, None], v / v_norm.clamp_min(1e-9)[:, None],
                                 torch.zeros_like(v)).numpy()
    out['n'] = len(ds)
    return out


def bin_masks(sample, kind):
    """kind: 'turn' | 'speed' -> [(이름, 마스크 [N])]"""
    if kind == 'turn':
        t = sample['turn']
        return [(name, np.isnan(t) if lo is None else (t >= lo) & (t < hi)) for name, lo, hi in TURN_BINS]
    s = sample['speed']
    return [(name, (s >= lo) & (s < hi)) for name, lo, hi in SPEED_BINS]


def weighted_mean(values, weights):
    return float((values * weights).sum() / weights.sum()) if weights.sum() > 0 else float('nan')


def groups(data):
    """'hotel' 과 'all scenes' 를 (샘플 dict 목록, 장면별 가중치) 로."""
    return {'hotel': [(data['hotel'], 1.0)],
            'all scenes': [(data[s], 1.0 / data[s]['n']) for s in SCENES]}


def aggregate_bins(data, names, kind):
    """-> {그룹: {bin: {'n': 샘플 수, 모델: ADE}}}  ('all scenes' 는 장면 동일 비중 가중 평균)"""
    result = {}
    for group, members in groups(data).items():
        result[group] = {}
        bins = [b[0] for b in (TURN_BINS if kind == 'turn' else SPEED_BINS)]
        for i, bin_name in enumerate(bins):
            masks = [bin_masks(sample, kind)[i][1] for sample, _ in members]
            weights = np.concatenate([m * w for m, (_, w) in zip(masks, members)])
            row = {'n': int(sum(m.sum() for m in masks))}
            for name in names:
                ade = np.concatenate([sample['err'][name].mean(-1) for sample, _ in members])
                row[name] = weighted_mean(ade, weights)
            result[group][bin_name] = row
    return result


def aggregate_steps(data, names):
    """-> {그룹: {모델: [12] 스텝별 평균 오차}}"""
    result = {}
    for group, members in groups(data).items():
        weights = np.concatenate([np.full(sample['n'], w) for sample, w in members])
        result[group] = {}
        for name in names:
            err = np.concatenate([sample['err'][name] for sample, _ in members])     # [ΣN, 12]
            result[group][name] = (err * weights[:, None]).sum(0) / weights.sum()
    return result


def plot_steps(steps, names, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    t = np.arange(1, 13)
    for ax, (group, curves) in zip(axes, steps.items()):
        for color, name in zip(COLORS, names):
            ax.plot(t, curves[name], '-o', color=color, lw=2, ms=4, label=label_of(name))
        ax.set_title(group, fontsize=11)
        ax.set_xticks(t)
        ax.set_xlabel('prediction step (× 0.4 s)', fontsize=10)
        ax.grid(True, color=GRID_COLOR)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel('mean displacement error (m)', fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(names), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_bins(bins, names, xlabel, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    width = 0.8 / len(names)
    for ax, (group, rows) in zip(axes, bins.items()):
        x = np.arange(len(rows))
        for j, (color, name) in enumerate(zip(COLORS, names)):
            values = [row[name] for row in rows.values()]
            ax.bar(x + (j - (len(names) - 1) / 2) * width, values, width * 0.92, color=color,
                   label=label_of(name))
        ax.set_xticks(x, [f'{b}\nn={row["n"]:,}' for b, row in rows.items()], fontsize=8)
        ax.set_title(group, fontsize=11)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.grid(True, axis='y', color=GRID_COLOR)
        ax.set_axisbelow(True)
        ax.tick_params(axis='y', labelsize=8)
    axes[0].set_ylabel('ADE (m)', fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(names), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def print_bins(title, bins, names):
    print(f'\n### {title}  (ADE, m)')
    for group, rows in bins.items():
        print(f'\n{group}\n| bin | n | ' + ' | '.join(label_of(n) for n in names) + ' |')
        print('| :--- | --: |' + ' --: |' * len(names))
        for b, row in rows.items():
            print(f'| {b} | {row["n"]:,} | ' + ' | '.join(f'{row[n]:.3f}' for n in names) + ' |')


def print_gap_decomposition(data, names):
    """hotel 에서 (LSTM ADE - 등속 ADE) 를 묶음별 기여로 나눈다. 기여 합 = 전체 차이."""
    hotel = data['hotel']
    n = hotel['n']
    for kind, bins in (('speed', SPEED_BINS), ('turn', TURN_BINS)):
        print(f'\n### hotel: ADE 차이(모델 - 등속) 의 {kind} 묶음별 기여 (m, 합 = 전체 차이)')
        print('| bin | 샘플 비율 | ' + ' | '.join(label_of(m) for m in names[1:]) + ' |')
        print('| :--- | --: |' + ' --: |' * (len(names) - 1))
        masks = bin_masks(hotel, kind)
        for bin_name, mask in masks:
            cells = []
            for m in names[1:]:
                gap = hotel['err'][m].mean(-1) - hotel['err']['cv'].mean(-1)     # [N]
                cells.append(f'{gap[mask].sum() / n:+.3f}')
            print(f'| {bin_name} | {mask.mean():.0%} | ' + ' | '.join(cells) + ' |')
        total = [f'{(hotel["err"][m].mean(-1) - hotel["err"]["cv"].mean(-1)).mean():+.3f}' for m in names[1:]]
        print('| 합 | 100% | ' + ' | '.join(total) + ' |')


def print_along_cross(data, names):
    """마지막 스텝 오차를 마지막 관측 진행 방향 기준으로 나눈다 (움직이는 샘플만)."""
    print('\n### 마지막 스텝 오차의 진행 방향 성분 (관측 마지막 변위 >= 0.08 m 인 샘플)')
    print('along 평균: + 면 예측이 실제보다 앞으로 더 감 / |along|, |cross|: 절댓값 평균 (m)')
    print('| 그룹 | n | 모델 | along 평균 | abs(along) | abs(cross) |')
    print('| :--- | --: | :--- | --: | --: | --: |')
    for group, members in groups(data).items():
        for name in names:
            along, cross, weights, count = [], [], [], 0
            for sample, w in members:
                h = sample['heading']                                    # [N, 2]
                moving = np.linalg.norm(h, axis=-1) > 0
                e = sample['final'][name][moving]                         # [M, 2]
                hm = h[moving]
                along.append((e * hm).sum(-1))
                cross.append(e[:, 0] * hm[:, 1] - e[:, 1] * hm[:, 0])
                weights.append(np.full(moving.sum(), w))
                count += int(moving.sum())
            along, cross, weights = map(np.concatenate, (along, cross, weights))
            print(f'| {group} | {count:,} | {label_of(name)} | {weighted_mean(along, weights):+.3f} | '
                  f'{weighted_mean(np.abs(along), weights):.3f} | {weighted_mean(np.abs(cross), weights):.3f} |')


def print_distance_response(data, names):
    """예측 이동 거리 ~ 등속 이동 거리 회귀 기울기 (움직이는 샘플, 장면 동일 비중 가중)."""
    print('\n### 4.8 s 이동 거리: 평균과, 등속 이동 거리(12 x 마지막 관측 변위)에 대한 회귀 기울기')
    print('| 그룹 | n | 대상 | 평균 이동 거리 (m) | 기울기 |')
    print('| :--- | --: | :--- | --: | --: |')
    for group, members in groups(data).items():
        cv_dist, weights, count = [], [], 0
        dists = {name: [] for name in ['gt'] + names}
        for sample, w in members:
            moving = np.linalg.norm(sample['heading'], axis=-1) > 0
            cv_dist.append(np.linalg.norm(sample['end']['cv'][moving], axis=-1))
            for name in dists:
                dists[name].append(np.linalg.norm(sample['end'][name][moving], axis=-1))
            weights.append(np.full(moving.sum(), w))
            count += int(moving.sum())
        x, weights = np.concatenate(cv_dist), np.concatenate(weights)
        for name, values in dists.items():
            y = np.concatenate(values)
            slope = np.polyfit(x, y, 1, w=np.sqrt(weights))[0]          # 가중 최소제곱 직선의 기울기
            target = 'actual future' if name == 'gt' else label_of(name)
            print(f'| {group} | {count:,} | {target} | {weighted_mean(y, weights):.3f} | {slope:.2f} |')


def print_slow_response(data, names):
    """관측 평균 속력 < 0.2 m/s 샘플의 4.8 s 예측 이동 거리 (모델이 서 있는 사람을 움직이게 하는지)."""
    print('\n### 관측 평균 속력 < 0.2 m/s 샘플: 4.8 s 이동 거리 평균 (m)')
    print('| 그룹 | n | ' + ' | '.join(['actual future'] + [label_of(n) for n in names]) + ' |')
    print('| :--- | --: |' + ' --: |' * (len(names) + 1))
    for group, members in groups(data).items():
        cells, count = [], 0
        for name in ['gt'] + names:
            values, weights = [], []
            for sample, w in members:
                slow = sample['speed'] < 0.2
                values.append(np.linalg.norm(sample['end'][name][slow], axis=-1))
                weights.append(np.full(slow.sum(), w))
            cells.append(f'{weighted_mean(np.concatenate(values), np.concatenate(weights)):.3f}')
            count = sum(len(v) for v in values)
        print(f'| {group} | {count:,} | ' + ' | '.join(cells) + ' |')


def print_split_slopes(root):
    """실제 미래 이동 거리 ~ 등속 이동 거리 기울기를 분할별로 (가중치 없음)."""
    print('\n### 실제 4.8 s 이동 거리의 등속 이동 거리에 대한 기울기 (마지막 관측 변위 >= 0.08 m)')
    print('| 장면 | test 기울기 (n) | train 기울기 (n) |')
    print('| :--- | --: | --: |')
    for scene in SCENES:
        cells = []
        for split in ('test', 'train'):
            ds = load_split(root, scene, split, normalize=True)
            v = ds.obs[:, -1] - ds.obs[:, -2]                      # [N, 2]
            moving = v.norm(dim=-1) >= SLOW_STEP
            x = 12 * v[moving].norm(dim=-1).numpy()
            y = ds.fut[moving, -1].norm(dim=-1).numpy()
            cells.append(f'{np.polyfit(x, y, 1)[0]:.2f} ({int(moving.sum()):,})')
        print(f'| {scene} | ' + ' | '.join(cells) + ' |')


def print_scaled_cv(root, factors=(1.0, 0.9, 0.88, 0.85, 0.8)):
    """속도를 k 배로 줄인 등속 모델. 원점이 마지막 관측점이라 예측에 k 를 곱하면 된다."""
    print('\n### 속도를 k 배로 줄인 등속 모델의 test ADE (m)')
    print('| k | ' + ' | '.join(SCENES) + ' | AVG |')
    print('| --: |' + ' --: |' * (len(SCENES) + 1))
    tests = {scene: load_split(root, scene, 'test', normalize=True) for scene in SCENES}
    model = ConstantVelocity(pred_len=12)
    for k in factors:
        values = [float(displacement(model(ds.obs) * k, ds.fut).mean()) for ds in tests.values()]
        print(f'| {k} | ' + ' | '.join(f'{v:.3f}' for v in values) + f' | {sum(values) / len(values):.3f} |')


def save_csv(bins_by_kind, steps, names, folder, tag, data):
    with open(folder / f'{tag}_bins.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['kind', 'group', 'bin', 'n'] + [f'ade_{n}' for n in names])
        for kind, bins in bins_by_kind.items():
            for group, rows in bins.items():
                for b, row in rows.items():
                    writer.writerow([kind, group, b, row['n']] + [f'{row[n]:.4f}' for n in names])
            # 장면별 값도 남긴다
            for scene in SCENES:
                sample = data[scene]
                for b, mask in bin_masks(sample, kind):
                    writer.writerow([kind, scene, b, int(mask.sum())] +
                                    [f'{sample["err"][n].mean(-1)[mask].mean():.4f}' if mask.any() else ''
                                     for n in names])
    with open(folder / f'{tag}_steps.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['group', 'model'] + [f'step{k}' for k in range(1, 13)])
        for group, curves in steps.items():
            for name in names:
                writer.writerow([group, name] + [f'{v:.4f}' for v in curves[name]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/raw')
    parser.add_argument('--runs', nargs='+', default=['none_seed0', 'rotate_seed0'])
    parser.add_argument('--tag', default='main')
    parser.add_argument('--figures', default='docs/figures')
    parser.add_argument('--out', default='results/analysis')
    args = parser.parse_args()
    figures, out = Path(args.figures), Path(args.out)
    figures.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    names = ['cv'] + args.runs
    data = {scene: scene_samples(args.data, scene, args.runs) for scene in SCENES}

    steps = aggregate_steps(data, names)
    bins_by_kind = {kind: aggregate_bins(data, names, kind) for kind in ('turn', 'speed')}

    print('### 스텝별 평균 오차 (m): step 1, 4, 8, 12 / 전체 평균(= ADE)')
    for group, curves in steps.items():
        for name in names:
            c = curves[name]
            print(f'{group:<11}{label_of(name):<26}' + '  '.join(f'{c[k - 1]:.3f}' for k in (1, 4, 8, 12))
                  + f'  | ADE {c.mean():.3f}')
    print_bins('방향 전환 각도별', bins_by_kind['turn'], names)
    print_bins('관측 속력별', bins_by_kind['speed'], names)
    print_gap_decomposition(data, names)
    print_along_cross(data, names)
    print_distance_response(data, names)
    print_slow_response(data, names)
    print_split_slopes(args.data)
    print_scaled_cv(args.data)

    plot_steps(steps, names, figures / f'{args.tag}_step_error.png')
    plot_bins(bins_by_kind['turn'], names, 'heading change over the 4.8 s future',
              figures / f'{args.tag}_turn_ade.png')
    plot_bins(bins_by_kind['speed'], names, 'mean observed speed', figures / f'{args.tag}_speed_ade.png')
    save_csv(bins_by_kind, steps, names, out, args.tag, data)
    print(f'\nsaved {figures}/{args.tag}_*.png, {out}/{args.tag}_*.csv')


if __name__ == '__main__':
    main()
