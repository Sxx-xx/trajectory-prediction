"""등속 모델을 5개 장면의 test 분할에서 평가한다.

출력
    화면       : 장면별 샘플 수, ADE, FDE 와 5개 장면 평균
    results/cv.csv          : 같은 표
    results/cv_examples.png : 장면마다 전형적인 예 1개와 FDE 가 가장 큰 예 1개

    .venv/bin/python scripts/eval_cv.py
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from trajpred.data import SCENES, load_split  # noqa: E402
from trajpred.metrics import ade, fde  # noqa: E402
from trajpred.models import ConstantVelocity  # noqa: E402

PAST_COLOR = '#8a8983'     # 과거 = 회색
FUTURE_COLOR = '#008300'   # 실제 미래 = 초록
PRED_COLOR = '#e0393e'     # 예측 = 빨강


def displacement_speed(obs, dt=0.4):
    """관측 구간 평균 속력. obs: [N, obs_len, 2] -> [N] (m/s)"""
    return torch.linalg.norm(obs[:, 1:] - obs[:, :-1], dim=-1).mean(dim=-1) / dt


def evaluate(root):
    model = ConstantVelocity(pred_len=12)
    rows, examples = [], {}
    for scene in SCENES:
        # 마지막 관측점 기준 상대 좌표. 등속 모델은 평행이동에 무관하므로 결과는 절대 좌표와 같다.
        ds = load_split(root, scene, 'test', normalize=True)
        with torch.no_grad():
            pred = model(ds.obs)                     # [N, 12, 2]
        ade_per = ade(pred, ds.fut)                  # [N]
        fde_per = fde(pred, ds.fut)                  # [N]
        rows.append({'scene': scene, 'samples': len(ds),
                     'ade': ade_per.mean().item(), 'fde': fde_per.mean().item()})

        # 그림용 예: 움직이는 보행자(관측 평균 속도 >= 0.5 m/s) 중 FDE 가 중앙값에 가장 가까운 것,
        # 그리고 장면 전체에서 FDE 가 가장 큰 것. 서 있는 사람은 좌표 떨림만 보여서 제외한다.
        speed = displacement_speed(ds.obs)                        # [N] m/s
        moving = torch.nonzero(speed >= 0.5).squeeze(1)           # [M]
        median = fde_per[moving].median()
        typical_idx = moving[(fde_per[moving] - median).abs().argmin()].item()
        worst_idx = fde_per.argmax().item()
        examples[scene] = [
            ('typical', ds.obs[typical_idx], ds.fut[typical_idx], pred[typical_idx], fde_per[typical_idx].item()),
            ('worst FDE', ds.obs[worst_idx], ds.fut[worst_idx], pred[worst_idx], fde_per[worst_idx].item()),
        ]
    return rows, examples


def print_table(rows, average):
    print(f"{'scene':<8}{'samples':>9}{'ADE (m)':>10}{'FDE (m)':>10}")
    for r in rows + [average]:
        samples = '' if r['samples'] is None else r['samples']
        print(f"{r['scene']:<8}{samples:>9}{r['ade']:>10.3f}{r['fde']:>10.3f}")


def save_csv(rows, average, path):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['scene', 'samples', 'ade_m', 'fde_m'])
        for r in rows + [average]:
            writer.writerow([r['scene'], '' if r['samples'] is None else r['samples'],
                             f"{r['ade']:.4f}", f"{r['fde']:.4f}"])


def plot_examples(examples, path):
    fig, axes = plt.subplots(2, len(SCENES), figsize=(3.2 * len(SCENES), 6.6))
    for col, scene in enumerate(SCENES):
        for row, (kind, obs, fut, pred, sample_fde) in enumerate(examples[scene]):
            ax = axes[row, col]
            # 관측 마지막 점과 미래 첫 점을 이어 그려 끊기지 않게 한다.
            ax.plot(obs[:, 0], obs[:, 1], '-o', color=PAST_COLOR, ms=3, lw=2)
            ax.plot(torch.cat([obs[-1:, 0], fut[:, 0]]), torch.cat([obs[-1:, 1], fut[:, 1]]),
                    '-o', color=FUTURE_COLOR, ms=3, lw=2)
            ax.plot(torch.cat([obs[-1:, 0], pred[:, 0]]), torch.cat([obs[-1:, 1], pred[:, 1]]),
                    '--o', color=PRED_COLOR, ms=3, lw=2)
            ax.set_aspect('equal', adjustable='datalim')
            ax.grid(True, color='#e4e3df')
            ax.set_title(f'{scene} · {kind}\nFDE {sample_fde:.2f} m', fontsize=10)
            ax.tick_params(labelsize=8)
    handles = [plt.Line2D([], [], color=c, lw=2, ls=ls, marker='o', ms=3)
               for c, ls in ((PAST_COLOR, '-'), (FUTURE_COLOR, '-'), (PRED_COLOR, '--'))]
    fig.legend(handles, ['past (8 steps, observed)', 'future (12 steps, ground truth)',
                         'constant velocity prediction'],
               loc='upper center', ncol=3, frameon=False)
    fig.supxlabel('x (m), relative to last observed point', fontsize=10)
    fig.supylabel('y (m)', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/raw')
    parser.add_argument('--out', default='results')
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows, examples = evaluate(args.data)
    # 논문 관례대로 평균은 5개 장면 값의 단순 평균이다 (샘플 수 가중 아님).
    average = {'scene': 'AVG', 'samples': None,
               'ade': sum(r['ade'] for r in rows) / len(rows),
               'fde': sum(r['fde'] for r in rows) / len(rows)}

    print_table(rows, average)
    save_csv(rows, average, out / 'cv.csv')
    plot_examples(examples, out / 'cv_examples.png')
    print(f'saved {out / "cv.csv"}, {out / "cv_examples.png"}')


if __name__ == '__main__':
    main()
