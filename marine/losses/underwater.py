"""수중 전용 손실 항 — gray-world 색항상성(color constancy).

LUNA2 ``CombinedRestorationLoss.add_term(name, fn, weight)`` 으로 등록되는 추가
손실. 시그니처는 ``fn(pred, target, **ctx) -> scalar``. 수중 영상의 청/녹 색캐스트를
억제하기 위해 **출력의 채널별 평균을 균등화**하는 gray-world prior 를 건다(참조 불요).

주의: gray-world 가정(전체 평균색=회색)은 강한 prior 이므로 **작은 가중치**(0.05~0.2)로
복원 손실과 병행한다. 과도하면 본래 색이 있는 장면을 탈색시킬 수 있다.
"""
from __future__ import annotations

import torch


def gray_world_loss(pred: torch.Tensor, target: torch.Tensor = None, **ctx) -> torch.Tensor:
    """채널 평균 불균형 페널티 (gray-world). 입력 ``[-1,1]`` (B,3,H,W).

    L = mean_B( Σ_c (mean_c - mean_gray)^2 ),  mean_c = 채널 c 의 공간평균([0,1]).
    target 은 사용하지 않는다(참조 불요 prior).
    """
    x = (pred + 1.0) * 0.5                      # [-1,1] → [0,1]
    ch_mean = x.mean(dim=(2, 3))               # (B, 3)
    gray = ch_mean.mean(dim=1, keepdim=True)   # (B, 1)
    return ((ch_mean - gray) ** 2).mean()
