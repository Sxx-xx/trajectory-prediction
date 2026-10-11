"""궤적 예측 평가 지표. 좌표가 미터 단위이므로 결과도 미터다.

ADE (Average Displacement Error): 예측 구간 모든 스텝에서 예측점과 실제점 사이
                                   유클리드 거리의 평균
FDE (Final Displacement Error)  : 마지막 스텝에서의 거리

장면 하나의 값은 그 장면 모든 샘플(보행자 × 윈도우)의 평균이다.
"""

import torch


def displacement(pred, gt):
    """스텝마다의 거리.

    pred, gt: [N, T, 2]
    반환    : [N, T]
    """
    return torch.linalg.norm(pred - gt, dim=-1)


def ade(pred, gt):
    """샘플별 ADE. pred, gt: [N, T, 2] -> [N]"""
    return displacement(pred, gt).mean(dim=-1)


def fde(pred, gt):
    """샘플별 FDE. pred, gt: [N, T, 2] -> [N]"""
    return displacement(pred, gt)[:, -1]
