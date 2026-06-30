"""GuidanceNet 변형 — M3.2 ablation (수중 guidance anchor 비교).

LUNA2 ``GuidanceNet`` 은 bilateral grid 의 z(luma)축 좌표를 **Rec.601 휘도(밝기)** 에
anchor 한다. 저조도에선 밝기가 열화 축이라 타당하지만, 수중 열화의 주축은 밝기가
아니라 **파장별 감쇠(적색 손실)** 다. 따라서 anchor 후보를 바꿔 비교한다.

* ``MaxRGBGuidanceNet`` : anchor = max_c(RGB). 적색 감쇠가 큰 수중에서 화소의
  "남아있는 최대 채널"을 좌표로 써, 감쇠/투과 정도에 따라 z 를 분포시킨다.

구조(conv1/conv2/conv3, zero-init conv3)는 LUNA2 GuidanceNet 과 **동일 키**로 맞춰
warm-start 체크포인트의 guidance 가중치를 그대로 로드할 수 있게 한다(anchor 만 다름).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MaxRGBGuidanceNet(nn.Module):
    """풀해상도 입력 → guidance map ∈ [0,1], anchor = max over RGB.

    ``g = maxRGB + 0.5·tanh(conv3∘conv2∘conv1(x))`` → clamp[0,1].
    conv3 zero-init → 초기 guidance ≈ maxRGB (z 전구간 prior).
    """

    def __init__(self, c_hidden: int = 16) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, c_hidden, 3, padding=1)
        self.conv2 = nn.Conv2d(c_hidden, c_hidden, 1)
        self.conv3 = nn.Conv2d(c_hidden, 1, 1)
        self.act = nn.ReLU(inplace=True)
        nn.init.zeros_(self.conv3.weight)
        nn.init.zeros_(self.conv3.bias)

    def forward(self, x_full: torch.Tensor) -> torch.Tensor:
        x01 = (x_full + 1.0) * 0.5                       # [-1,1] → [0,1]
        anchor = x01.amax(dim=1, keepdim=True)          # (B,1,H,W) max over RGB
        r = self.act(self.conv1(x_full))
        r = self.act(self.conv2(r))
        r = self.conv3(r)
        g = anchor + 0.5 * torch.tanh(r)
        return g.clamp(0.0, 1.0)


def swap_guidance(model, variant: str, c_hidden: int = 16):
    """model.guidance_net 을 변형으로 교체. variant: 'maxrgb' | 'luma'(원본 유지)."""
    if variant in (None, "", "luma", "default"):
        return model
    if variant == "maxrgb":
        device = next(model.parameters()).device
        model.guidance_net = MaxRGBGuidanceNet(c_hidden=c_hidden).to(device)
        return model
    raise ValueError(f"알 수 없는 guidance_variant: {variant} (luma|maxrgb)")
