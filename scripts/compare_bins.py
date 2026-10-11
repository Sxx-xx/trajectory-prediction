"""시드별 오차 분해 결과(analyze_errors.py 의 *_bins.csv)를 모아 묶음별 평균 ± 표준편차를 낸다.

analyze_errors.py 를 시드마다 다른 tag 로 돌려 둔 뒤 쓴다. 예:

    for s in 0 1 2; do
      .venv/bin/python scripts/analyze_errors.py --tag B_seed$s \
          --runs rotate_heading_seed$s rotate_heading_social_seed$s > results/analysis/B_seed${s}_report.md
    done
    .venv/bin/python scripts/compare_bins.py --tags B_seed0 B_seed1 B_seed2 \
        --conditions rotate_heading rotate_heading_social

출력: 화면에 마크다운 표 (hotel, all scenes 그룹만), results/analysis/<--out-tag>_bins_seeds.csv
표준편차는 시드 간 표본 표준편차(n-1)다. 등속 모델은 결정적이라 한 값이다.
"""

import argparse
import csv
from pathlib import Path

import numpy as np


def read_bins(path):
    """-> {(kind, group, bin): row dict}"""
    with open(path) as f:
        return {(r['kind'], r['group'], r['bin']): r for r in csv.DictReader(f)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default='results/analysis')
    parser.add_argument('--tags', nargs='+', required=True, help='시드 순서대로의 tag (예: B_seed0 B_seed1 B_seed2)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2], help='tags 와 같은 순서의 시드')
    parser.add_argument('--conditions', nargs='+', required=True, help='run 이름에서 _seed<s> 를 뺀 앞부분')
    parser.add_argument('--out-tag', default='B')
    args = parser.parse_args()
    if len(args.tags) != len(args.seeds):
        raise ValueError('tags 와 seeds 의 개수가 같아야 한다')

    folder = Path(args.dir)
    tables = [read_bins(folder / f'{tag}_bins.csv') for tag in args.tags]
    keys = [k for k in tables[0] if k[1] in ('hotel', 'all scenes')]

    rows = []
    for kind, group, bin_name in keys:
        first = tables[0][(kind, group, bin_name)]
        row = {'kind': kind, 'group': group, 'bin': bin_name, 'n': first['n'], 'cv': float(first['ade_cv'])}
        for cond in args.conditions:
            values = np.array([float(t[(kind, group, bin_name)][f'ade_{cond}_seed{s}'])
                               for t, s in zip(tables, args.seeds)])
            row[cond] = (values.mean(), values.std(ddof=1))
        rows.append(row)

    with open(folder / f'{args.out_tag}_bins_seeds.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['kind', 'group', 'bin', 'n', 'ade_cv']
                        + [f'{c}_{m}' for c in args.conditions for m in ('mean', 'std')])
        for r in rows:
            writer.writerow([r['kind'], r['group'], r['bin'], r['n'], f"{r['cv']:.4f}"]
                            + [f'{v:.4f}' for c in args.conditions for v in r[c]])

    print(f'seeds {args.seeds}, ADE (m) 평균 ± 표준편차')
    for kind in dict.fromkeys(r['kind'] for r in rows):
        for group in ('hotel', 'all scenes'):
            print(f'\n### {kind} / {group}')
            print('| bin | n | 등속 | ' + ' | '.join(args.conditions) + ' | 차이 (마지막 - 첫 조건) |')
            print('| :--- | --: | --: |' + ' --: |' * len(args.conditions) + ' --: |')
            for r in rows:
                if r['kind'] != kind or r['group'] != group:
                    continue
                cells = [f'{r[c][0]:.3f} ± {r[c][1]:.3f}' for c in args.conditions]
                diff = r[args.conditions[-1]][0] - r[args.conditions[0]][0]
                print(f"| {r['bin']} | {int(r['n']):,} | {r['cv']:.3f} | " + ' | '.join(cells) + f' | {diff:+.3f} |')
    print(f'\nsaved {folder / (args.out_tag + "_bins_seeds.csv")}')


if __name__ == '__main__':
    main()
