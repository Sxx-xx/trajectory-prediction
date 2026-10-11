"""D단계: K개의 미래를 내는 Transformer 를 leave-one-out 5개 분할에서 학습 · 평가한다.

모델은 C단계 Transformer (heading frame) 에 num_modes=K 를 준 것이다 (src/trajpred/models/transformer.py).
손실 = winner-takes-all (Social GAN 의 variety loss 와 같은 꼴)
    샘플마다 K개 예측 각각의 MSE [B, K] 를 재고, 가장 작은 모드(winner) 의 MSE 만 줄인다.
    + 모드 점수: winner 를 정답으로 하는 cross-entropy. 점수 MLP 는 h.detach() 를 보므로
      이 항은 궤적 쪽 가중치를 바꾸지 않는다.
early stopping 은 val minADE_K 로 한다.

평가 (test)
    minADE_k / minFDE_k, k = 1..K. k < K 이면 점수가 높은 k개만 쓴다 (k = 1 = 가장 그럴듯한 하나).

출력 (<run> = mm<K>_tf_heading[_residual][_social][_<tag>]_seed<seed>)
    checkpoints/mm_<run>_<scene>.pt
    results/multimodal/<run>.csv          : 장면별 k = 1, 5, K 의 minADE / minFDE 와 평균
    results/multimodal/<run>_kcurve.csv   : 장면별 k = 1..K 전체
    results/multimodal/<run>_history.csv, <run>_curves.png

    .venv/bin/python scripts/train_multimodal.py --modes 20 --social --residual --seed 0
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_lstm import batches, model_inputs, plot_curves, random_rotate, split_tensors  # noqa: E402

from trajpred.data import SCENES  # noqa: E402
from trajpred.metrics import min_ade, min_fde, top_k  # noqa: E402
from trajpred.models import TransformerPredictor  # noqa: E402

REPORT_K = (1, 5, 20)


def wta_loss(preds, logits, fut):
    """preds: [B, K, 12, 2], logits: [B, K], fut: [B, 12, 2] -> (회귀 손실, 점수 손실)"""
    per_mode = ((preds - fut[:, None]) ** 2).mean(dim=(2, 3))     # [B, K] 모드별 MSE (m^2)
    winner = per_mode.argmin(dim=1)                               # [B]
    reg = per_mode.gather(1, winner[:, None]).mean()              # winner 의 MSE 만
    return reg, F.cross_entropy(logits, winner)


@torch.no_grad()
def predict(model, data, batch_size=2048):
    """-> (preds [N, K, 12, 2], logits [N, K])"""
    model.eval()
    outs = [model(*model_inputs(data, idx)) for idx in batches(len(data['obs']), batch_size, shuffle=False)]
    return torch.cat([o[0] for o in outs]), torch.cat([o[1] for o in outs])


def train_scene(scene, args, device):
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    train = split_tensors(args, scene, 'train', device)
    val = split_tensors(args, scene, 'val', device)
    print(f'[{scene}] train {len(train["obs"])}  val {len(val["obs"])}')

    model = TransformerPredictor(pred_len=12, d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
                                 ff_dim=args.ff_dim, dropout=args.dropout, frame=args.frame, social=args.social,
                                 residual=args.residual, num_modes=args.modes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history, best_val, best_epoch, best_state = [], float('inf'), 0, None
    for epoch in range(1, args.max_epochs + 1):
        start = time.time()
        model.train()
        loss_sum = 0.0
        for idx in batches(len(train['obs']), args.batch_size, shuffle=True, generator=generator):
            idx = idx.to(device)
            obs, nbr, nbr_mask = model_inputs(train, idx)
            fut = train['fut'][idx]
            if args.augment == 'rotate':
                obs, fut, nbr = random_rotate(obs, fut, generator, nbr)
            preds, logits = model(obs, nbr, nbr_mask)             # [B, K, 12, 2], [B, K]
            reg, cls = wta_loss(preds, logits, fut)
            optimizer.zero_grad()
            (reg + cls).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += reg.item() * len(idx)
        train_loss = loss_sum / len(train['obs'])

        val_preds, val_logits = predict(model, val)
        val_loss = wta_loss(val_preds, val_logits, val['fut'])[0].item()
        val_ade = min_ade(val_preds, val['fut']).mean().item()     # minADE_K
        val_ade1 = min_ade(top_k(val_preds, val_logits, 1), val['fut']).mean().item()
        history.append({'scene': scene, 'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
                        'val_ade': val_ade, 'val_ade_top1': val_ade1})
        improved = val_ade < best_val
        if improved:
            best_val, best_epoch = val_ade, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f'[{scene}] epoch {epoch:3d}  train {train_loss:.4f}  val {val_loss:.4f}  '
              f'val minADE{args.modes} {val_ade:.4f}  top1 {val_ade1:.4f}{"  *" if improved else ""}  '
              f'({time.time() - start:.1f}s)')
        if epoch - best_epoch >= args.patience:
            print(f'[{scene}] early stop: best epoch {best_epoch}, val minADE {best_val:.4f}')
            break

    model.load_state_dict(best_state)
    return model, history, best_epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/raw')
    parser.add_argument('--out', default='results/multimodal')
    parser.add_argument('--ckpt', default='checkpoints')
    parser.add_argument('--scenes', nargs='+', default=SCENES)
    parser.add_argument('--modes', type=int, default=20, help='예측 개수 K')
    parser.add_argument('--d-model', type=int, default=64)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--n-layers', type=int, default=2)
    parser.add_argument('--ff-dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max-epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--augment', choices=['none', 'rotate'], default='none')
    parser.add_argument('--frame', choices=['world', 'heading'], default='heading')
    parser.add_argument('--residual', action='store_true')
    parser.add_argument('--social', action='store_true', help='마지막 관측 프레임 이웃에 cross-attention')
    parser.add_argument('--max-neighbors', type=int, default=32)
    parser.add_argument('--neighbor-control', choices=['real', 'shuffle'], default='real')
    parser.add_argument('--tag', default='')
    args = parser.parse_args()
    args.model = 'transformer'       # load_lstm 이 TransformerPredictor 를 만들도록
    run = (f'mm{args.modes}_tf' + ('_rotate' if args.augment == 'rotate' else '')
           + ('_heading' if args.frame == 'heading' else '') + ('_residual' if args.residual else '')
           + ('_social' if args.social else '') + (f'_{args.tag}' if args.tag else '') + f'_seed{args.seed}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device, ' run:', run)
    out, ckpt = Path(args.out), Path(args.ckpt)
    out.mkdir(parents=True, exist_ok=True)
    ckpt.mkdir(parents=True, exist_ok=True)
    ks = [k for k in REPORT_K if k <= args.modes]

    rows, curve, history, best_epochs = [], [], [], {}
    for scene in args.scenes:
        model, scene_history, best_epoch = train_scene(scene, args, device)
        history += scene_history
        best_epochs[scene] = best_epoch
        torch.save({'state_dict': model.state_dict(), 'args': vars(args), 'best_epoch': best_epoch},
                   ckpt / f'mm_{run}_{scene}.pt')

        test = split_tensors(args, scene, 'test', device)
        preds, logits = predict(model, test)
        fut = test['fut']
        row = {'scene': scene, 'samples': len(fut), 'best_epoch': best_epoch}
        for k in range(1, args.modes + 1):
            kept = top_k(preds, logits, k)                                 # [N, k, 12, 2]
            a, f = min_ade(kept, fut).mean().item(), min_fde(kept, fut).mean().item()
            curve.append({'scene': scene, 'k': k, 'min_ade': a, 'min_fde': f})
            if k in ks:
                row[f'ade_k{k}'], row[f'fde_k{k}'] = a, f
        rows.append(row)
        print(f'[{scene}] test ' + '  '.join(f'k={k} {row[f"ade_k{k}"]:.3f}/{row[f"fde_k{k}"]:.3f}' for k in ks))

    metric_cols = [f'{m}_k{k}' for k in ks for m in ('ade', 'fde')]
    avg = {'scene': 'AVG', 'samples': sum(r['samples'] for r in rows), 'best_epoch': ''}
    avg.update({c: sum(r[c] for r in rows) / len(rows) for c in metric_cols})
    with open(out / f'{run}.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['scene', 'samples', 'best_epoch'] + metric_cols)
        writer.writeheader()
        for r in rows + [avg]:
            writer.writerow({k: (f'{v:.6f}' if isinstance(v, float) else v) for k, v in r.items()})
    with open(out / f'{run}_kcurve.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['scene', 'k', 'min_ade', 'min_fde'])
        writer.writeheader()
        writer.writerows({k: (f'{v:.6f}' if isinstance(v, float) else v) for k, v in r.items()} for r in curve)
    with open(out / f'{run}_history.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    plot_curves(history, best_epochs, out / f'{run}_curves.png', run)
    print('AVG ' + '  '.join(f'k={k} {avg[f"ade_k{k}"]:.3f}/{avg[f"fde_k{k}"]:.3f}' for k in ks))


if __name__ == '__main__':
    main()
