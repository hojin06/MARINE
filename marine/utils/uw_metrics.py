"""수중 화질 no-reference 지표 — UIQM, UCIQE.

참조 없는(no-reference) 수중 영상 화질 평가 표준 지표 2종.

* **UIQM** (Underwater Image Quality Measure; Panetta et al., IEEE JOE 2016)
  = c1·UICM + c2·UISM + c3·UIConM
    - UICM  : colorfulness (RG/YB 비대칭 절단평균·분산)
    - UISM  : sharpness (채널별 Sobel edge map 의 EME)
    - UIConM: contrast (intensity 의 logAMEE)
  계수는 원논문값 c=(0.0282, 0.2953, 3.5753). FUnIE-GAN(uqim_utils) 구현을 이식.

* **UCIQE** (Underwater Color Image Quality Evaluation; Yang & Sowmya, TIP 2015)
  = c1·σ_chroma + c2·contrast_L + c3·mean_saturation  (CIELab)
  계수 c=(0.4680, 0.2745, 0.2576). Lab 변환은 cv2 사용(skimage 미설치).

입력 규약
---------
* ``getUIQM(rgb)`` / ``getUCIQE(rgb)`` : RGB ``uint8`` (H,W,3) 또는 float[0,255].
* ``uiqm_uciqe_from_tensor(t)`` : ``[-1,1]`` (3,H,W) 또는 (1,3,H,W) 텐서 → (uiqm, uciqe).
값이 클수록 좋음(둘 다).
"""
from __future__ import annotations

import math
from typing import Tuple

import cv2
import numpy as np
from scipy import ndimage


# ===========================================================================
# UIQM
# ===========================================================================
def _mu_a(x: np.ndarray, alpha_L: float = 0.1, alpha_R: float = 0.1) -> float:
    """비대칭 절단평균(asymmetric alpha-trimmed mean)."""
    x = np.sort(x.ravel())
    K = x.size
    TaL = int(math.ceil(alpha_L * K))
    TaR = int(math.floor(alpha_R * K))
    denom = max(K - TaL - TaR, 1)
    return float(np.sum(x[TaL:K - TaR]) / denom)


def _s_a(x: np.ndarray, mu: float) -> float:
    return float(np.mean((x.ravel() - mu) ** 2))


def _uicm(rgb: np.ndarray) -> float:
    R = rgb[..., 0].astype(np.float64)
    G = rgb[..., 1].astype(np.float64)
    B = rgb[..., 2].astype(np.float64)
    RG = R - G
    YB = (R + G) / 2.0 - B
    mu_rg, mu_yb = _mu_a(RG), _mu_a(YB)
    s_rg, s_yb = _s_a(RG, mu_rg), _s_a(YB, mu_yb)
    l = math.sqrt(mu_rg ** 2 + mu_yb ** 2)
    r = math.sqrt(s_rg + s_yb)
    return -0.0268 * l + 0.1586 * r


def _eme(ch: np.ndarray, window_size: int = 10) -> float:
    """Enhancement Measure Estimation (블록별 log(max/min) 합)."""
    k1 = ch.shape[1] // window_size
    k2 = ch.shape[0] // window_size
    if k1 == 0 or k2 == 0:
        return 0.0
    ch = ch[:k2 * window_size, :k1 * window_size]
    w = 2.0 / (k1 * k2)
    val = 0.0
    for l in range(k2):
        for k in range(k1):
            blk = ch[l * window_size:(l + 1) * window_size,
                     k * window_size:(k + 1) * window_size]
            mx = float(blk.max())
            mn = float(blk.min())
            if mn > 0 and mx > 0:
                val += math.log(mx / mn)
    return w * val


def _sobel(ch: np.ndarray) -> np.ndarray:
    dx = ndimage.sobel(ch, axis=0, mode="reflect")
    dy = ndimage.sobel(ch, axis=1, mode="reflect")
    return np.hypot(dx, dy)


def _uism(rgb: np.ndarray) -> float:
    R = rgb[..., 0].astype(np.float64)
    G = rgb[..., 1].astype(np.float64)
    B = rgb[..., 2].astype(np.float64)
    # 각 채널 edge map = sobel ⊙ 원채널 (grayscale edge)
    Rem = _sobel(R) * R
    Gem = _sobel(G) * G
    Bem = _sobel(B) * B
    r_eme, g_eme, b_eme = _eme(Rem), _eme(Gem), _eme(Bem)
    lam_r, lam_g, lam_b = 0.299, 0.587, 0.114
    return lam_r * r_eme + lam_g * g_eme + lam_b * b_eme


def _uiconm(rgb: np.ndarray, window_size: int = 10) -> float:
    """intensity logAMEE 기반 contrast."""
    x = rgb.astype(np.float64).sum(axis=2) / 3.0  # intensity
    k1 = x.shape[1] // window_size
    k2 = x.shape[0] // window_size
    if k1 == 0 or k2 == 0:
        return 0.0
    x = x[:k2 * window_size, :k1 * window_size]
    w = -1.0 / (k1 * k2)
    val = 0.0
    for l in range(k2):
        for k in range(k1):
            blk = x[l * window_size:(l + 1) * window_size,
                    k * window_size:(k + 1) * window_size]
            mx = float(blk.max())
            mn = float(blk.min())
            top = mx - mn
            bot = mx + mn
            if bot > 0 and top > 0:
                ratio = top / bot
                val += ratio * math.log(ratio)
    return w * val


def getUIQM(rgb: np.ndarray) -> float:
    """UIQM (클수록 좋음). rgb: (H,W,3) uint8 또는 float[0,255]."""
    rgb = np.asarray(rgb).astype(np.float64)
    uicm = _uicm(rgb)
    uism = _uism(rgb)
    uiconm = _uiconm(rgb, 10)
    return float(0.0282 * uicm + 0.2953 * uism + 3.5753 * uiconm)


# ===========================================================================
# UCIQE
# ===========================================================================
def getUCIQE(rgb: np.ndarray) -> float:
    """UCIQE (클수록 좋음). rgb: (H,W,3) uint8 또는 float[0,255]."""
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    # cv2 Lab(8bit): L∈[0,255], a,b∈[0,255](+128 offset) → 표준 범위로 환산
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab).astype(np.float64)
    L = lab[..., 0] * 100.0 / 255.0
    a = lab[..., 1] - 128.0
    b = lab[..., 2] - 128.0

    chroma = np.sqrt(a ** 2 + b ** 2)
    sigma_c = float(np.std(chroma))

    # 휘도 대비 = 상위1% 평균 − 하위1% 평균
    Lf = np.sort(L.ravel())
    n = Lf.size
    k = max(int(round(0.01 * n)), 1)
    con_l = float(Lf[-k:].mean() - Lf[:k].mean())

    # 평균 채도 = chroma / sqrt(chroma^2 + L^2)
    denom = np.sqrt(chroma ** 2 + L ** 2)
    sat = np.divide(chroma, denom, out=np.zeros_like(chroma), where=denom > 0)
    mu_s = float(np.mean(sat))

    return float(0.4680 * sigma_c + 0.2745 * con_l + 0.2576 * mu_s)


# ===========================================================================
# 텐서 편의 래퍼
# ===========================================================================
def _tensor_to_rgb_uint8(t) -> np.ndarray:
    """[-1,1] (3,H,W) 또는 (1,3,H,W) 텐서 → (H,W,3) uint8 RGB."""
    import torch  # 지연 import
    if isinstance(t, torch.Tensor):
        if t.dim() == 4:
            t = t[0]
        arr = ((t.detach().float().clamp(-1, 1) + 1.0) * 0.5
               ).permute(1, 2, 0).cpu().numpy()
    else:
        arr = np.asarray(t)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.shape[0] == 3:
            arr = np.transpose(arr, (1, 2, 0))
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def uiqm_uciqe_from_tensor(t) -> Tuple[float, float]:
    """[-1,1] 텐서 → (UIQM, UCIQE)."""
    rgb = _tensor_to_rgb_uint8(t)
    return getUIQM(rgb), getUCIQE(rgb)
