"""PyTorch 가 GPU(CUDA)를 쓸 수 있는지 확인한다.

    .venv/bin/python scripts/check_gpu.py
"""

import torch

print('torch        :', torch.__version__)
print('built w/ CUDA:', torch.version.cuda)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU          :', torch.cuda.get_device_name(0))
    x = torch.randn(1024, 1024, device='cuda')                 # [1024, 1024]
    print('matmul check :', float((x @ x).sum().cpu()) == float((x @ x).sum().cpu()))
