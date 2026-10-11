"""ETH/UCY 궤적 파일을 읽어 '과거 8스텝 → 미래 12스텝' 학습 샘플로 자른다.

파일 형식 (한 줄 = 한 보행자의 한 프레임, 탭 구분, 좌표 단위는 미터)
    frame_id  ped_id  x  y
프레임 간격은 0.4 s (2.5 Hz). 8 스텝 = 3.2 s 관측, 12 스텝 = 4.8 s 예측.

윈도우 만드는 규칙은 Social-GAN 데이터 로더(sgan/data/trajectories.py)와 같다.
그래야 논문들과 같은 샘플 집합으로 ADE/FDE 를 비교할 수 있다.
  - 연속한 20 프레임(8 + 12)을 한 칸씩 밀며 본다
  - 그 20 프레임 전부에 등장한 보행자만 샘플이 된다
  - 그런 보행자가 2명 이상인 윈도우만 쓴다 (Social-GAN 의 min_ped=1, '>' 비교)

leave-one-out 분할은 데이터 폴더 구조가 이미 담고 있다.
    data/raw/<scene>/train, val : 나머지 4개 장면의 학습/검증 구간
    data/raw/<scene>/test       : <scene> 장면 전체 (평가용)
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

SCENES = ['eth', 'hotel', 'univ', 'zara1', 'zara2']
OBS_LEN = 8
PRED_LEN = 12


def read_trajectory_file(path):
    """파일 하나를 읽는다.

    반환: np.ndarray [행 수, 4]  (frame_id, ped_id, x, y)
    """
    return np.loadtxt(path, delimiter='\t', dtype=np.float64, ndmin=2)


@dataclass
class WindowMeta:
    """샘플 하나가 어디서 왔는지. 같은 장면 · 같은 윈도우의 이웃을 다시 묶을 때 쓴다."""
    source_file: str
    window_id: int       # 파일 안에서 몇 번째 윈도우인지
    ped_id: int
    start_frame: int     # 관측 첫 프레임 번호


def extract_neighbors(rows, metas, obs_len=OBS_LEN, max_neighbors=16):
    """샘플마다 관측 프레임별로 같은 프레임의 다른 보행자(이웃)를 모은다.

    이웃은 그 프레임에 있는 모든 다른 보행자다. 20 프레임을 다 채운 보행자만이 아니다.
    관측 프레임만 쓰므로 미래 정보는 들어가지 않는다. 가까운 순서로 max_neighbors 명까지 남긴다.

    rows : [행 수, 4]  (frame_id, ped_id, x, y)
    metas: extract_windows 가 낸 (window_id, ped_id, start_frame) 목록 (이 순서대로 결과를 낸다)
    반환:
        nbr  : np.ndarray [N, obs_len, max_neighbors, 4]  (상대 위치 x, y, 속도 x, y). 빈 칸은 0
               상대 위치 = 이웃 위치 - 대상 위치 (같은 프레임), 속도 = 한 스텝 변위 (m / 0.4 s)
               이웃이 직전 프레임에 없으면 속도를 알 수 없어 0 으로 둔다
        mask : np.ndarray [N, obs_len, max_neighbors] bool  (실제 이웃이면 True)
    """
    n = len(metas)
    nbr = np.zeros((n, obs_len, max_neighbors, 4), dtype=np.float32)
    mask = np.zeros((n, obs_len, max_neighbors), dtype=bool)
    if n == 0:
        return nbr, mask

    frames = np.unique(rows[:, 0])
    # 프레임마다 (ped_id 오름차순, 위치, 속도) 를 한 번만 만든다
    ids, xys, vels = [], [], []
    for i, f in enumerate(frames):
        cur = rows[rows[:, 0] == f]
        cur = cur[np.argsort(cur[:, 1])]
        xy = np.around(cur[:, 2:4], decimals=4)                       # [P, 2] 데이터셋과 같은 반올림
        vel = np.zeros_like(xy)
        if i > 0:
            prev_ids, prev_xy = ids[-1], xys[-1]
            pos = np.searchsorted(prev_ids, cur[:, 1])
            pos = np.clip(pos, 0, max(len(prev_ids) - 1, 0))
            found = (len(prev_ids) > 0) & (prev_ids[pos] == cur[:, 1]) if len(prev_ids) else np.zeros(len(cur), bool)
            vel[found] = xy[found] - prev_xy[pos[found]]
        ids.append(cur[:, 1])
        xys.append(xy)
        vels.append(vel)

    # 같은 윈도우의 샘플끼리 묶어 프레임마다 한 번에 계산한다
    window_ids = np.array([m[0] for m in metas])
    ped_ids = np.array([m[1] for m in metas])
    for w in np.unique(window_ids):
        sample_idx = np.nonzero(window_ids == w)[0]                    # 이 윈도우의 샘플들 [S]
        for t in range(obs_len):
            fi = w + t
            target_pos = np.searchsorted(ids[fi], ped_ids[sample_idx])  # [S] 대상 보행자 위치
            rel = xys[fi][None, :, :] - xys[fi][target_pos][:, None, :]  # [S, P, 2]
            dist = np.linalg.norm(rel, axis=-1)                         # [S, P]
            dist[np.arange(len(sample_idx)), target_pos] = np.inf       # 자기 자신 제외
            order = np.argsort(dist, axis=1)[:, :max_neighbors]         # [S, <=K] 가까운 순
            k = order.shape[1]
            d_sorted = np.take_along_axis(dist, order, axis=1)          # [S, k]
            valid = np.isfinite(d_sorted)
            nbr[sample_idx, t, :k, 0:2] = np.take_along_axis(rel, order[..., None], axis=1)
            nbr[sample_idx, t, :k, 2:4] = vels[fi][order]
            mask[sample_idx, t, :k] = valid
    nbr[~mask] = 0
    return nbr, mask


def extract_windows(rows, obs_len=OBS_LEN, pred_len=PRED_LEN, min_peds=2):
    """한 파일의 행들에서 (보행자, 윈도우) 샘플을 뽑는다.

    rows: [행 수, 4]
    반환:
        trajs   : np.ndarray [N, obs_len + pred_len, 2]  절대 좌표 (m)
        metas   : (window_id, ped_id, start_frame) 튜플 N 개
    """
    seq_len = obs_len + pred_len
    frames = np.unique(rows[:, 0])                       # [F] 오름차순
    rows_by_frame = [rows[rows[:, 0] == f] for f in frames]

    trajs, metas = [], []
    for window_id in range(len(frames) - seq_len + 1):
        window_rows = np.concatenate(rows_by_frame[window_id:window_id + seq_len])
        window_trajs, window_metas = [], []
        for ped_id in np.unique(window_rows[:, 1]):
            ped_rows = window_rows[window_rows[:, 1] == ped_id]   # [<= seq_len, 4]
            if len(ped_rows) != seq_len:
                continue                                          # 20 프레임 중 빠진 프레임이 있다
            window_trajs.append(np.around(ped_rows[:, 2:4], decimals=4))   # [seq_len, 2] Social-GAN 과 같은 반올림
            window_metas.append((window_id, int(ped_id), int(frames[window_id])))
        if len(window_trajs) >= min_peds:
            trajs.extend(window_trajs)
            metas.extend(window_metas)

    if not trajs:
        return np.zeros((0, seq_len, 2)), []
    return np.stack(trajs), metas                                 # [N, seq_len, 2]


class TrajectoryDataset(Dataset):
    """폴더 안의 모든 .txt 파일에서 뽑은 샘플을 모은 데이터셋.

    normalize=True 이면 각 샘플을 마지막 관측점 기준 상대 좌표로 바꾼다.
    (관측 마지막 점이 원점이 된다. 원래 위치는 origin 에 남는다.)

    __getitem__ 반환 (모두 float32 텐서)
        obs    : [obs_len, 2]   과거 궤적
        fut    : [pred_len, 2]  실제 미래 궤적
        origin : [2]            마지막 관측점의 절대 좌표 (normalize 와 무관하게 항상 절대 좌표)

    max_neighbors > 0 이면 이웃 정보도 만든다 (extract_neighbors 참고). 속성으로만 둔다.
        nbr      : [N, obs_len, max_neighbors, 4]  상대 위치 (2), 속도 (2). 평행이동과 무관하다
        nbr_mask : [N, obs_len, max_neighbors]     bool
    """

    def __init__(self, folder, obs_len=OBS_LEN, pred_len=PRED_LEN, normalize=False, min_peds=2,
                 max_neighbors=0):
        self.folder = Path(folder)
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.normalize = normalize

        paths = sorted(self.folder.glob('*.txt'))
        if not paths:
            raise FileNotFoundError(f'{self.folder} 에 .txt 가 없다. scripts/download_data.sh 를 먼저 실행할 것')

        all_trajs, all_nbr, all_mask, self.metas = [], [], [], []
        for path in paths:
            rows = read_trajectory_file(path)
            trajs, metas = extract_windows(rows, obs_len, pred_len, min_peds)
            all_trajs.append(trajs)
            self.metas.extend(WindowMeta(path.name, *m) for m in metas)
            if max_neighbors > 0:
                nbr, mask = extract_neighbors(rows, metas, obs_len, max_neighbors)
                all_nbr.append(nbr)
                all_mask.append(mask)

        trajs = torch.from_numpy(np.concatenate(all_trajs)).float()   # [N, obs_len + pred_len, 2]
        self.obs = trajs[:, :obs_len]                                # [N, obs_len, 2]
        self.fut = trajs[:, obs_len:]                                # [N, pred_len, 2]
        self.origin = self.obs[:, -1].clone()                        # [N, 2]
        if normalize:
            self.obs = self.obs - self.origin[:, None, :]
            self.fut = self.fut - self.origin[:, None, :]
        if max_neighbors > 0:
            self.nbr = torch.from_numpy(np.concatenate(all_nbr))           # [N, obs_len, K, 4]
            self.nbr_mask = torch.from_numpy(np.concatenate(all_mask))     # [N, obs_len, K]

    def __len__(self):
        return self.obs.shape[0]

    def __getitem__(self, index):
        return self.obs[index], self.fut[index], self.origin[index]


def load_split(root, scene, split, **kwargs):
    """leave-one-out 분할 하나를 연다.

    scene: SCENES 중 하나 (평가에서 빼 둘 장면)
    split: 'train' | 'val' | 'test'
        train, val -> 나머지 4개 장면
        test       -> scene 장면
    """
    if scene not in SCENES:
        raise ValueError(f'scene 은 {SCENES} 중 하나: {scene}')
    if split not in ('train', 'val', 'test'):
        raise ValueError(f"split 은 'train' | 'val' | 'test': {split}")
    return TrajectoryDataset(Path(root) / scene / split, **kwargs)
