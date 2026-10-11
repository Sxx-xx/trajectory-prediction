"""여러 시드로 돌린 LSTM 결과를 모아 장면별 평균 ± 표준편차 표를 만든다.

입력 : results/lstm/<augment>_seed<seed>.csv   (scripts/train_lstm.py 출력)
       results/cv.csv                           (등속 모델, 비교용)
출력 : results/lstm/summary.csv
       화면에 README 용 마크다운 표

표준편차는 시드 간 표본 표준편차(n-1 로 나눔)다.
AVG 행은 시드마다 5개 장면 평균을 먼저 구한 뒤, 그 값들의 평균 ± 표준편차다.

    .venv/bin/python scripts/summarize_lstm.py --seeds 0 1 2
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from trajpred.data import SCENES

AUGMENTS = ['none', 'rotate']
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
    # stats[augment][scene] = (ade 평균, ade 표준편차, fde 평균, fde 표준편차)
    stats = {}
    for augment in AUGMENTS:
        runs = [read_run(folder / f'{augment}_seed{seed}.csv') for seed in args.seeds]
        stats[augment] = {}
        for scene in ROWS:
            values = np.array([run[scene] for run in runs])     # [시드 수, 2]  (ade, fde)
            mean, std = values.mean(axis=0), values.std(axis=0, ddof=1)
            stats[augment][scene] = (mean[0], std[0], mean[1], std[1])

    with open(folder / 'summary.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['scene', 'augment', 'n_seeds', 'ade_mean', 'ade_std', 'fde_mean', 'fde_std'])
        for scene in ROWS:
            for augment in AUGMENTS:
                writer.writerow([scene, augment, len(args.seeds)] + [f'{v:.4f}' for v in stats[augment][scene]])

    print(f'seeds {args.seeds}  (ADE / FDE, m, 평균 ± 표준편차)\n')
    print('| 장면 | 등속 ADE / FDE | LSTM 증강 없음 ADE | FDE | LSTM 회전 증강 ADE | FDE |')
    print('| :--- | :-: | :-: | :-: | :-: | :-: |')
    for scene in ROWS:
        cells = [f'{cv[scene][0]:.3f} / {cv[scene][1]:.3f}']
        for augment in AUGMENTS:
            a_mean, a_std, f_mean, f_std = stats[augment][scene]
            cells += [f'{a_mean:.3f} ± {a_std:.3f}', f'{f_mean:.3f} ± {f_std:.3f}']
        print(f'| {scene} | ' + ' | '.join(cells) + ' |')
    print(f'\nsaved {folder / "summary.csv"}')


if __name__ == '__main__':
    main()
