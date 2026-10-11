"""여러 시드로 돌린 LSTM 결과를 모아 장면별 평균 ± 표준편차 표를 만든다.

입력 : results/lstm/<condition>_seed<seed>.csv  (scripts/train_lstm.py 출력. 시드 파일이 다 없는 조건은 건너뛴다)
       results/cv.csv                           (등속 모델, 비교용)
출력 : results/lstm/summary.csv
       화면에 README 용 마크다운 표 (ADE 표, FDE 표)

표준편차는 시드 간 표본 표준편차(n-1 로 나눔)다. 굵은 글씨는 장면마다 (반올림한) 평균이 가장 작은 칸이다.
AVG 행은 시드마다 5개 장면 평균을 먼저 구한 뒤, 그 값들의 평균 ± 표준편차다.

    .venv/bin/python scripts/summarize_lstm.py --seeds 0 1 2
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from trajpred.data import SCENES

# (run 이름 앞부분, 표에 쓸 이름)
CONDITIONS = [
    ('none', '증강 없음'),
    ('rotate', '회전 증강'),
    ('none_heading', 'heading'),
    ('none_residual', 'residual'),
    ('none_heading_residual', 'heading + residual'),
    ('rotate_heading', 'heading + 회전 증강'),
    ('rotate_heading_social_g4d5', '+ social pooling'),
    ('tf_none_heading', 'Transformer'),
    ('tf_none_heading_social', 'Transformer + 이웃 attention'),
    ('tf_none_heading_social_shuffle', 'Transformer + 섞은 이웃'),
    ('tf_none_heading_residual_social', 'Transformer + 이웃 attention + residual'),
]
ROWS = SCENES + ['AVG']


def read_run(path):
    """-> {scene: (ade, fde)}  (AVG 행 포함)"""
    with open(path) as f:
        return {r['scene']: (float(r['ade_m']), float(r['fde_m'])) for r in csv.DictReader(f)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default='results/lstm')
    parser.add_argument('--cv', default='results/cv.csv')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    args = parser.parse_args()
    folder = Path(args.dir)

    cv = read_run(args.cv)
    # stats[condition][scene] = (ade 평균, ade 표준편차, fde 평균, fde 표준편차)
    stats = {}
    for condition, _ in CONDITIONS:
        paths = [folder / f'{condition}_seed{seed}.csv' for seed in args.seeds]
        if not all(p.exists() for p in paths):
            print(f'건너뜀: {condition} (시드 파일이 다 있지 않다)')
            continue
        runs = [read_run(p) for p in paths]
        stats[condition] = {}
        for scene in ROWS:
            values = np.array([run[scene] for run in runs])     # [시드 수, 2]  (ade, fde)
            mean, std = values.mean(axis=0), values.std(axis=0, ddof=1)
            stats[condition][scene] = (mean[0], std[0], mean[1], std[1])

    with open(folder / 'summary.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['scene', 'condition', 'n_seeds', 'ade_mean', 'ade_std', 'fde_mean', 'fde_std'])
        for scene in ROWS:
            for condition in stats:
                writer.writerow([scene, condition, len(args.seeds)] + [f'{v:.6f}' for v in stats[condition][scene]])

    labels = dict(CONDITIONS)
    for metric, col, cv_col in (('ADE', 0, 0), ('FDE', 2, 1)):
        print(f'\n{metric} (m), LSTM 은 시드 {args.seeds} 평균 ± 표준편차\n')
        print('| 장면 | 등속 | ' + ' | '.join(labels[c] for c in stats) + ' |')
        print('| :--- | :-: |' + ' :-: |' * len(stats))
        for scene in ROWS:
            means = [f'{cv[scene][cv_col]:.3f}'] + [f'{stats[c][scene][col]:.3f}' for c in stats]
            cells = [means[0]] + [f'{m} ± {stats[c][scene][col + 1]:.3f}' for m, c in zip(means[1:], stats)]
            best = min(means, key=float)                      # 반올림한 평균이 가장 작은 칸을 굵게
            cells = [f'**{cell}**' if m == best else cell for cell, m in zip(cells, means)]
            name = '**AVG**' if scene == 'AVG' else scene
            print(f'| {name} | ' + ' | '.join(cells) + ' |')
    print(f'\nsaved {folder / "summary.csv"}')


if __name__ == '__main__':
    main()
