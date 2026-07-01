"""수중 향상 후처리 — 화이트밸런스 + CLAHE 대비 + 채도 보정.

네트워크(BilateralLowLightNet) 출력이 색은 잡아도 **대비/선명도가 약해 muddy**해
보이는 문제를 보완하는 고전적 후처리. 수중 영상 향상에서 널리 쓰이는 조합:

1. **Gray-world 화이트밸런스**(선택) — 잔여 색캐스트 중화.
2. **CLAHE** (LAB L채널) — 국소 대비 향상(haze/veil 완화). 전역 히스토그램보다 안전.
3. **채도 보정** — 색 생동감.

``strength`` 로 원본↔후처리 블렌드. 전부 RGB uint8 (H,W,3) 입출력.
"""
from __future__ import annotations

import cv2
import numpy as np


def gray_world_wb(
    rgb: np.ndarray,
    clip: tuple = (0.6, 1.6),
    strength: float = 0.6,
) -> np.ndarray:
    """제한(clamp)·부분블렌드형 gray-world WB (과보정 방지).

    캐스트가 강한 입력(예: 진한 초록)에서 원본 gray-world 는 반대색(파랑/보라)으로
    과보정된다. 채널 게인을 ``clip`` 범위로 제한하고 ``strength`` 만큼만 적용한다.
    """
    a = rgb.astype(np.float32)
    means = a.reshape(-1, 3).mean(0)
    gray = float(means.mean())
    scale = np.clip(gray / np.clip(means, 1e-3, None), clip[0], clip[1])
    wb = np.clip(a * scale, 0, 255)
    out = a * (1.0 - strength) + wb * strength
    return np.clip(out, 0, 255).astype(np.uint8)


def clahe_contrast(rgb: np.ndarray, clip: float = 2.0, tile: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)


def boost_saturation(rgb: np.ndarray, gain: float = 1.15) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * gain, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def postprocess(
    rgb: np.ndarray,
    wb: bool = False,
    clahe_clip: float = 2.0,
    sat_gain: float = 1.15,
    strength: float = 1.0,
) -> np.ndarray:
    """WB(선택)→CLAHE→채도. strength<1 이면 원본과 블렌드."""
    out = rgb
    if wb:
        out = gray_world_wb(out)
    out = clahe_contrast(out, clip=clahe_clip)
    if sat_gain and abs(sat_gain - 1.0) > 1e-3:
        out = boost_saturation(out, gain=sat_gain)
    if strength < 0.999:
        out = (rgb.astype(np.float32) * (1 - strength)
               + out.astype(np.float32) * strength).clip(0, 255).astype(np.uint8)
    return out
