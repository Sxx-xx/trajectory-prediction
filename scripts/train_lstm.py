"""Vanilla LSTM encoder-decoder 를 leave-one-out 5개 분할에서 학습 · 평가한다.

장면 하나마다
    1. data/raw/<scene>/train 으로 학습, data/raw/<scene>/val 로 검증
       (둘 다 <scene> 을 뺀 나머지 4개 장면이다. test 장면은 학습에 전혀 쓰지 않는다)
    2. 에폭마다 val ADE 를 재서 가장 좋았던 모델을 저장한다 (early stopping)
       --augment rotate 이면 학습 배치마다 궤적을 무작위 각도로 회전한다 (val, test 는 그대로)
    3. 저장한 best 모델로 data/raw/<scene>/test 에서 ADE/FDE 를 잰다

출력 (<run> = <augment>_seed<seed>, 예: rotate_seed0)
    checkpoints/lstm_<run>_<scene>.pt   : best 모델 (state_dict 와 설정)
    results/lstm/<run>.csv              : 장면별 test ADE/FDE 와 평균
    results/lstm/<run>_history.csv      : 에폭별 train/val loss, val ADE
    results/lstm/<run>_curves.png       : 장면별 학습 곡선
    화면                                : 등속 모델(results/cv.csv)과의 비교표

    .venv/bin/python scripts/train_lstm.py --augment none --seed 0
    .venv/bin/python scripts/train_lstm.py --augment rotate --seed 0
"""

import argparse
import csv
import time
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from trajpred.data import SCENES, load_split  # noqa: E402
from trajpred.metrics import ade, fde  # noqa: E402
from trajpred.models import LSTMEncoderDecoder  # noqa: E402

TRAIN_COLOR = '#2a78d6'    # train loss = 파랑, 실선
VAL_COLOR = '#eb6834'      # val loss = 주황, 점선
GRID_COLOR = '#e4e3df'


def random_rotate(obs, fut, generator):
    """샘플마다 0~2π 의 무작위 각도로 관측 · 미래 궤적을 함께 회전한다 (학습 데이터 증강).

    좌표가 마지막 관측점 기준 상대 좌표이므로 그 점을 중심으로 돈다.
    걷는 방향만 바뀌고 속도 · 곡률 같은 움직임의 모양은 그대로다.
    obs: [B, 8, 2], fut: [B, 12, 2] -> 같은 shape
    """
    theta = torch.rand(obs.shape[0], generator=generator) * 2 * torch.pi    # [B]
    cos, sin = torch.cos(theta), torch.sin(theta)
    rot = torch.stack([torch.stack([cos, -sin], -1),
                       torch.stack([sin, cos], -1)], -2).to(obs.device)    # [B, 2, 2]
    # 점 p (행 벡터) 를 R p 로 돌린다: p @ R^T
    return obs @ rot.transpose(1, 2), fut @ rot.transpose(1, 2)


def batches(n, batch_size, shuffle, generator=None):
    """0..n-1 인덱스를 batch_size 씩 자른다."""
    order = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
    return order.split(batch_size)


@torch.no_grad()
def predict(model, obs, batch_size=4096):
    """obs: [N, 8, 2] -> [N, 12, 2]. 메모리를 아끼려고 나눠서 돌린다."""
    model.eval()
    return torch.cat([model(obs[idx]) for idx in batches(len(obs), batch_size, shuffle=False)])


def train_scene(scene, args, device):
    """장면 하나를 test 로 빼고 학습한다. 반환: (best 모델, 에폭별 기록)"""
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)

    # 마지막 관측점 기준 상대 좌표. 데이터 전체가 GPU 메모리에 올라간다 (수만 x 20 x 2 float).
    train = load_split(args.data, scene, 'train', normalize=True)
    val = load_split(args.data, scene, 'val', normalize=True)
    train_obs, train_fut = train.obs.to(device), train.fut.to(device)   # [N, 8, 2], [N, 12, 2]
    val_obs, val_fut = val.obs.to(device), val.fut.to(device)           # [M, 8, 2], [M, 12, 2]
    print(f'[{scene}] train {len(train)}  val {len(val)}')

    model = LSTMEncoderDecoder(pred_len=12, embed_dim=args.embed_dim, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()   # 예측 위치와 실제 위치의 좌표별 제곱 오차 평균 (m^2)

    history, best_val_ade, best_epoch, best_state = [], float('inf'), 0, None
    for epoch in range(1, args.max_epochs + 1):
        start = time.time()
        model.train()
        loss_sum = 0.0
        for idx in batches(len(train_obs), args.batch_size, shuffle=True, generator=generator):
            idx = idx.to(device)
            obs, fut = train_obs[idx], train_fut[idx]       # [B, 8, 2], [B, 12, 2]
            if args.augment == 'rotate':
                obs, fut = random_rotate(obs, fut, generator)
            pred = model(obs)                               # [B, 12, 2]
            loss = loss_fn(pred, fut)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # 순환 신경망의 기울기 폭주 방지
            optimizer.step()
            loss_sum += loss.item() * len(idx)
        train_loss = loss_sum / len(train_obs)

        val_pred = predict(model, val_obs)                  # [M, 12, 2]
        val_loss = loss_fn(val_pred, val_fut).item()
        val_ade = ade(val_pred, val_fut).mean().item()
        history.append({'scene': scene, 'epoch': epoch, 'train_loss': train_loss,
                        'val_loss': val_loss, 'val_ade': val_ade})

        # early stopping: val ADE 가 patience 에폭 동안 좋아지지 않으면 멈춘다
        improved = val_ade < best_val_ade
        if improved:
            best_val_ade, best_epoch = val_ade, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f'[{scene}] epoch {epoch:3d}  train {train_loss:.4f}  val {val_loss:.4f}  '
              f'val ADE {val_ade:.4f}{"  *" if improved else ""}  ({time.time() - start:.1f}s)')
        if epoch - best_epoch >= args.patience:
            print(f'[{scene}] early stop: best epoch {best_epoch}, val ADE {best_val_ade:.4f}')
            break

    model.load_state_dict(best_state)
    return model, history, best_epoch


def load_cv(path):
    """등속 모델 결과 results/cv.csv -> {scene: (ade, fde)}"""
    with open(path) as f:
        return {r['scene']: (float(r['ade_m']), float(r['fde_m'])) for r in csv.DictReader(f)}


def plot_curves(history, best_epochs, path, run):
    fig, axes = plt.subplots(1, len(SCENES), figsize=(3.2 * len(SCENES), 3.4), sharey=True)
    for ax, scene in zip(axes, SCENES):
        rows = [h for h in history if h['scene'] == scene]
        epochs = [h['epoch'] for h in rows]
        ax.plot(epochs, [h['train_loss'] for h in rows], '-', color=TRAIN_COLOR, lw=2)
        ax.plot(epochs, [h['val_loss'] for h in rows], '--', color=VAL_COLOR, lw=2)
        ax.axvline(best_epochs[scene], color='#8a8983', lw=1, ls=':')
        ax.set_yscale('log')
        ax.set_title(f'test = {scene}\nbest epoch {best_epochs[scene]}', fontsize=10)
        ax.grid(True, color=GRID_COLOR)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel('MSE loss (m², log scale)', fontsize=10)
    handles = [plt.Line2D([], [], color=TRAIN_COLOR, lw=2, ls='-'),
               plt.Line2D([], [], color=VAL_COLOR, lw=2, ls='--'),
               plt.Line2D([], [], color='#8a8983', lw=1, ls=':')]
    fig.legend(handles, ['train loss', 'val loss', 'best epoch (lowest val ADE)'],
               loc='upper center', ncol=3, frameon=False)
    fig.supxlabel(f'epoch    (run: {run})', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/raw')
    parser.add_argument('--out', default='results/lstm')
    parser.add_argument('--cv', default='results/cv.csv', help='비교용 등속 모델 결과')
    parser.add_argument('--ckpt', default='checkpoints')
    parser.add_argument('--scenes', nargs='+', default=SCENES)
    parser.add_argument('--embed-dim', type=int, default=64)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max-epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--augment', choices=['none', 'rotate'], default='none')
    args = parser.parse_args()
    run = f'{args.augment}_seed{args.seed}'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device, ' run:', run)
    out, ckpt = Path(args.out), Path(args.ckpt)
    out.mkdir(parents=True, exist_ok=True)
    ckpt.mkdir(parents=True, exist_ok=True)

    rows, history, best_epochs = [], [], {}
    for scene in args.scenes:
        model, scene_history, best_epoch = train_scene(scene, args, device)
        history += scene_history
        best_epochs[scene] = best_epoch
        torch.save({'state_dict': model.state_dict(), 'args': vars(args), 'best_epoch': best_epoch},
                   ckpt / f'lstm_{run}_{scene}.pt')

        test = load_split(args.data, scene, 'test', normalize=True)
        pred = predict(model, test.obs.to(device)).cpu()      # [N, 12, 2]
        rows.append({'scene': scene, 'samples': len(test), 'best_epoch': best_epoch,
                     'ade': ade(pred, test.fut).mean().item(), 'fde': fde(pred, test.fut).mean().item()})
        print(f'[{scene}] test ADE {rows[-1]["ade"]:.3f}  FDE {rows[-1]["fde"]:.3f}')

    with open(out / f'{run}_history.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['scene', 'epoch', 'train_loss', 'val_loss', 'val_ade'])
        writer.writeheader()
        writer.writerows(history)

    if len(rows) < len(SCENES):
        return   # 일부 장면만 돌린 경우 (디버그용): 평균 · 그림은 만들지 않는다

    # 논문 관례대로 평균은 5개 장면 값의 단순 평균이다.
    average = {'scene': 'AVG', 'samples': '', 'best_epoch': '',
               'ade': sum(r['ade'] for r in rows) / len(rows),
               'fde': sum(r['fde'] for r in rows) / len(rows)}
    with open(out / f'{run}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['scene', 'samples', 'best_epoch', 'ade_m', 'fde_m'])
        for r in rows + [average]:
            writer.writerow([r['scene'], r['samples'], r['best_epoch'], f"{r['ade']:.4f}", f"{r['fde']:.4f}"])
    plot_curves(history, best_epochs, out / f'{run}_curves.png', run)

    cv = load_cv(args.cv)
    print(f"\n{'scene':<8}{'CV ADE':>9}{'CV FDE':>9}{'LSTM ADE':>10}{'LSTM FDE':>10}")
    for r in rows + [average]:
        cv_ade, cv_fde = cv[r['scene']]
        print(f"{r['scene']:<8}{cv_ade:>9.3f}{cv_fde:>9.3f}{r['ade']:>10.3f}{r['fde']:>10.3f}")
    print(f'saved {out}/{run}.csv, {run}_history.csv, {run}_curves.png, {ckpt}/lstm_{run}_*.pt')


if __name__ == '__main__':
    main()
