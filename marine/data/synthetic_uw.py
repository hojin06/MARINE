"""합성 수중 열화 — clean 이미지에 물리 기반 색캐스트/산란을 입혀 (deg, clean) 페어 생성.

목적(참조 상한 돌파 + 도메인 갭 해소)
------------------------------------
UIEB/EUVP 는 대부분 파란 바다라 강한 **초록/노랑 담수** 캐스트를 네트워크가 못 배운다.
깨끗한(good) 이미지에 **랜덤 강도의 수중 열화**를 인위로 입혀 입력을 만들고, 원본 clean 을
타겟으로 학습하면: (1) 타겟이 중성이라 완전 중성화를 배우고, (2) 초록/노랑 강캐스트를
명시적으로 커버한다.

열화 모델 (단순 수중 영상 형성식)
--------------------------------
    I_c = J_c · t_c + B_c · (1 − t_c)
  J = clean, t_c = 채널별 투과율(∈[0,1], 낮을수록 감쇠 큼), B = 수중 veil(물색).
초록/노랑 물: B 가 녹황색, 적색 투과율 t_R 이 낮음(적색 먼저 소실).
추가로 대비저하·경미한 blur/noise 로 haze 를 모사.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import List, Sequence

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def synth_underwater(J: torch.Tensor) -> torch.Tensor:
    """clean (3,H,W)∈[0,1] → 합성 열화 (3,H,W)∈[0,1].

    랜덤 물색/투과율/강도로 다양한 캐스트(초록·노랑·청록·파랑)를 생성한다.
    """
    dev = J.device
    # --- veil(물색) B: 초록-노랑 위주 + 가끔 파랑 ---
    if random.random() < 0.75:  # 초록/노랑/청록 (담수·탁수)
        r = random.uniform(0.05, 0.40)
        g = random.uniform(0.45, 0.95)
        b = random.uniform(0.15, 0.55)
    else:                       # 파랑 (바다)
        r = random.uniform(0.05, 0.30)
        g = random.uniform(0.35, 0.70)
        b = random.uniform(0.55, 0.95)
    B = torch.tensor([r, g, b], device=dev).view(3, 1, 1)

    # --- 투과율 t_c: 적색이 가장 낮게(먼저 소실). 강도 s 로 전체 스케일 ---
    s = random.uniform(0.35, 0.85)          # 열화 강도(클수록 심함)
    t_r = 1.0 - s * random.uniform(0.75, 1.0)
    t_g = 1.0 - s * random.uniform(0.30, 0.65)
    t_b = 1.0 - s * random.uniform(0.45, 0.85)
    t = torch.tensor([t_r, t_g, t_b], device=dev).clamp(0.05, 1.0).view(3, 1, 1)

    I = J * t + B * (1.0 - t)

    # --- 대비 저하(veil 성분이 이미 대비 낮춤) + 경미한 blur/noise ---
    if random.random() < 0.5:
        k = random.choice([3, 5])
        I = _gauss_blur(I, k)
    if random.random() < 0.5:
        I = I + torch.randn_like(I) * random.uniform(0.005, 0.02)

    return I.clamp(0.0, 1.0)


def _gauss_blur(x: torch.Tensor, k: int) -> torch.Tensor:
    sigma = 0.3 * ((k - 1) * 0.5 - 1) + 0.8
    coords = torch.arange(k, dtype=x.dtype, device=x.device) - (k - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2)); g = g / g.sum()
    kernel = (g[:, None] * g[None, :]).expand(3, 1, k, k)
    return F.conv2d(x.unsqueeze(0), kernel, padding=k // 2, groups=3).squeeze(0)


class SyntheticUWDataset(Dataset):
    """clean 이미지 폴더들 → (합성열화, clean) 페어 [-1,1].

    clean 소스는 EUVP trainB(good) / UIEB reference 등 상대적으로 깨끗한 영상.
    """

    def __init__(
        self,
        clean_dirs: Sequence[Path],
        image_size: int = 256,
        augment: bool = True,
        limit: int = 0,
        name: str = "SynthUW",
    ) -> None:
        self.image_size = image_size
        self.augment = augment
        self.name = name
        files: List[Path] = []
        for d in clean_dirs:
            d = Path(d)
            if d.is_dir():
                files += [p for p in d.iterdir()
                          if p.suffix.lower() in _IMG_EXTS and "__MACOSX" not in str(p)]
        files = sorted(files)
        if limit and len(files) > limit:
            idx = torch.linspace(0, len(files) - 1, limit).round().long().tolist()
            files = [files[i] for i in idx]
        self.files = files
        if not self.files:
            raise RuntimeError(f"{name}: clean 이미지 0개 ({[str(d) for d in clean_dirs]})")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        img = Image.open(self.files[idx]).convert("RGB")
        s = self.image_size
        if self.augment:
            # random resized crop + flip (기하만; 광학은 synth 가 담당)
            W, H = img.size
            scale = random.uniform(0.7, 1.0)
            ch, cw = max(int(H * scale), 16), max(int(W * scale), 16)
            top = random.randint(0, max(H - ch, 0)); left = random.randint(0, max(W - cw, 0))
            img = TF.resized_crop(img, top, left, ch, cw, [s, s],
                                  interpolation=InterpolationMode.BILINEAR)
            if random.random() < 0.5:
                img = TF.hflip(img)
        else:
            img = TF.resize(img, [s, s], interpolation=InterpolationMode.BILINEAR)
        clean = TF.to_tensor(img)                 # [0,1]
        deg = synth_underwater(clean)             # [0,1]
        return deg * 2.0 - 1.0, clean * 2.0 - 1.0
