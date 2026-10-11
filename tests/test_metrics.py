"""평가 지표 테스트. 표준 라이브러리 unittest 로 돌린다.

    .venv/bin/python -m unittest discover tests
"""

import unittest

import torch

from trajpred.metrics import ade, fde, min_ade, min_fde, top_k


class TestSingle(unittest.TestCase):
    def test_ade_fde(self):
        gt = torch.zeros(1, 3, 2)
        pred = torch.tensor([[[3.0, 4.0], [0.0, 0.0], [0.0, 1.0]]])   # 거리 5, 0, 1
        self.assertAlmostEqual(ade(pred, gt).item(), 2.0)
        self.assertAlmostEqual(fde(pred, gt).item(), 1.0)


class TestBestOfK(unittest.TestCase):
    def setUp(self):
        self.gt = torch.zeros(1, 2, 2)                       # [N=1, T=2, 2]
        # 모드 0: 거리 (0, 4) -> ADE 2, FDE 4
        # 모드 1: 거리 (3, 3) -> ADE 3, FDE 3
        self.preds = torch.tensor([[[[0.0, 0.0], [4.0, 0.0]],
                                    [[0.0, 3.0], [0.0, 3.0]]]])  # [1, K=2, 2, 2]

    def test_min_ade_and_fde_chosen_separately(self):
        self.assertAlmostEqual(min_ade(self.preds, self.gt).item(), 2.0)   # 모드 0
        self.assertAlmostEqual(min_fde(self.preds, self.gt).item(), 3.0)   # 모드 1

    def test_k1_equals_single(self):
        pred = torch.randn(5, 12, 2)
        gt = torch.randn(5, 12, 2)
        torch.testing.assert_close(min_ade(pred[:, None], gt), ade(pred, gt))
        torch.testing.assert_close(min_fde(pred[:, None], gt), fde(pred, gt))

    def test_more_modes_never_worse(self):
        torch.manual_seed(0)
        preds, gt = torch.randn(50, 20, 12, 2), torch.randn(50, 12, 2)
        values = [min_ade(preds[:, :k], gt).mean().item() for k in (1, 5, 20)]
        self.assertTrue(values[0] >= values[1] >= values[2])

    def test_top_k(self):
        preds = torch.arange(3.0)[None, :, None, None].expand(2, 3, 4, 2)   # 모드 k 의 값 = k
        scores = torch.tensor([[0.1, 0.9, 0.5], [0.8, 0.0, 0.3]])
        kept = top_k(preds, scores, 2)                                       # [2, 2, 4, 2]
        self.assertEqual(kept[0, :, 0, 0].tolist(), [1.0, 2.0])
        self.assertEqual(kept[1, :, 0, 0].tolist(), [0.0, 2.0])


if __name__ == '__main__':
    unittest.main()
