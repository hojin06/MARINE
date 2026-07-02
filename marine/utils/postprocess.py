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
    clip: tuple = (0.5, 1.8),
    base_strength: float = 0.5,
    max_strength: float = 0.85,
    adaptive: bool = True,
) -> np.ndarray:
    """캐스트-비례 적응형 gray-world WB (과보정 방지 + 강캐스트 더 교정).

    채널 게인을 ``clip`` 으로 제한하고, 적용 강도를 **캐스트 강도에 비례**시킨다:
    거의 중성이면 약하게, 진한 초록/노랑이면 강하게(단 ``max_strength`` 로 상한).
    게인 clamp + 강도 상한의 이중 안전장치로 파랑/보라 아티팩트를 막는다.
    """
    a = np.asarray(rgb).astype(np.float32)
    means = a.reshape(-1, 3).mean(0)
    gray = float(means.mean())
    if adaptive:
        cast = float(means.std() / (gray + 1e-6))          # 색 불균형(0=중성)
        strength = float(np.clip(base_strength + 1.5 * cast, base_strength, max_strength))
    else:
        strength = base_strength
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
