"""MARINE 수중 페어 데이터셋 어댑터.

LUNA2 의 ``PairedImageDataset`` / ``PairedAugment`` 를 **그대로 재사용**(import)하여
수중 (degraded, reference) 페어를 만든다. low=degraded, high=reference 규약은
LUNA2 와 동일하며 ``[-1, 1]`` 텐서 페어를 반환한다.

핵심 차이 (수중 도메인)
-----------------------
* ``UnderwaterAugment`` : LUNA2 ``PairedAugment`` 를 상속하되 저조도 전용
  광학증강(gamma 암화 / 밝기 down / 노이즈)을 **비활성**(p=0)한다. 수중 입력을
  인위적으로 더 어둡게 만들면 안 되기 때문. 기하변환(flip/rotate/crop/perspective)은
  페어 동기화된 채 그대로 유지한다.

데이터 소스
-----------
* UIEB : ``raw-890/<name>.png`` ↔ ``reference-890/<name>.png`` (동일 파일명).
         split 은 configs/uieb_split.csv (stem,split) 로 필터.
* EUVP : ``Paired/underwater_{imagenet,dark}/{trainA,trainB}`` (동일 파일명 페어).
"""
from __future__ import annotations

import csv
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

import torch

# --- LUNA2 src 를 import 경로에 추가 (marine 패키지 __init__ 이 이미 했더라도 안전망) ---
_LUNA2_ROOT = Path(__file__).resolve().parents[3] / "LUNA2"
if str(_LUNA2_ROOT) not in sys.path:
    sys.path.insert(0, str(_LUNA2_ROOT))

from torch.utils.data import ConcatDataset, Dataset  # noqa: E402

from src.data.lowlight_dataset import PairedImageDataset, PairedAugment  # noqa: E402


# ===========================================================================
# 수중 증강 — 저조도 광학증강 비활성, 기하변환 유지
# ===========================================================================
class UnderwaterAugment(PairedAugment):
    """수중용 페어 증강.

    LUNA2 ``PairedAugment`` 와 동일하되 ``_photometric_low_only`` 의 세 항목
    (gamma/brightness/noise)을 확률 0 으로 꺼서 입력을 인위적으로 어둡게/노이즈화
    하지 않는다. 기하변환(crop/flip/rotate/perspective)은 그대로.
    ``training=False`` 면 resize-only (결정론적 평가).
    """

    def __init__(
        self,
        image_size: int = 256,
        training: bool = True,
        full_resize: bool = False,
        p_colorcast: float = 0.0,
        cast_range: tuple = (0.6, 1.3),
        **kwargs,
    ) -> None:
        # 저조도 전용 광학증강 비활성 (수중 도메인 핵심 차이)
        kwargs.update(p_gamma=0.0, p_brightness=0.0, p_noise=0.0)
        super().__init__(
            image_size=image_size, training=training, full_resize=full_resize, **kwargs
        )
        # 수중 색캐스트 augmentation (low 입력에만; target 불변 → 색항상성 일반화)
        self.p_colorcast = p_colorcast
        self.cast_range = cast_range

    def _photometric_low_only(self, low_t: torch.Tensor) -> torch.Tensor:
        """저조도 광학증강 대신 수중 색캐스트(채널별 랜덤 게인)를 low 에만 적용."""
        if self.p_colorcast > 0 and random.random() < self.p_colorcast:
            gains = torch.tensor(
                [random.uniform(*self.cast_range) for _ in range(3)],
                dtype=low_t.dtype,
            ).view(3, 1, 1)
            low_t = (low_t * gains).clamp(0.0, 1.0)
        return low_t


# ===========================================================================
# UIEB
# ===========================================================================
def _read_split(csv_path: Path | str, split: str) -> Set[str]:
    """uieb_split.csv (열: stem,split) 에서 해당 split 의 stem 집합."""
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"UIEB split CSV 없음: {csv_path} "
                                f"(experiments/make_uieb_split.py 먼저 실행)")
    stems: Set[str] = set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] == split:
                stems.add(row["stem"])
    return stems


class UIEBDataset(PairedImageDataset):
    """UIEB (raw-890 ↔ reference-890) 페어. split 은 CSV(stem,split)로 필터."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        split_csv: str | Path,
        image_size: int = 256,
        augment: bool = True,
        full_resize: bool = False,
        transform=None,
        colorcast: float = 0.0,
    ) -> None:
        root = Path(root)
        low_dir = root / "raw-890"
        high_dir = root / "reference-890"
        tf = transform or UnderwaterAugment(
            image_size=image_size, training=augment, full_resize=full_resize,
            p_colorcast=colorcast,
        )
        super().__init__(
            low_dir=low_dir, high_dir=high_dir, image_size=image_size,
            augment=augment, full_resize=full_resize, transform=tf,
            name=f"UIEB[{split}]",
        )
        keep = _read_split(split_csv, split)
        self.pairs = [(lo, hi) for (lo, hi) in self.pairs if lo.stem in keep]
        if not self.pairs:
            raise RuntimeError(
                f"UIEB[{split}] 페어 0개. split CSV({split_csv})와 파일명 정합 확인."
            )


# ===========================================================================
# EUVP
# ===========================================================================
EUVP_SUBSETS = ("underwater_imagenet", "underwater_dark")  # underwater_scenes 미확보


class EUVPPairedDataset(PairedImageDataset):
    """EUVP 단일 paired 서브셋 (trainA=degraded ↔ trainB=reference)."""

    def __init__(
        self,
        subset_dir: str | Path,
        image_size: int = 256,
        augment: bool = True,
        full_resize: bool = False,
        transform=None,
        name: str = "EUVP",
        colorcast: float = 0.0,
    ) -> None:
        subset_dir = Path(subset_dir)
        tf = transform or UnderwaterAugment(
            image_size=image_size, training=augment, full_resize=full_resize,
            p_colorcast=colorcast,
        )
        super().__init__(
            low_dir=subset_dir / "trainA", high_dir=subset_dir / "trainB",
            image_size=image_size, augment=augment, full_resize=full_resize,
            transform=tf, name=name,
        )


# ===========================================================================
# 빌더 — Stage A train / eval 데이터셋
# ===========================================================================
def build_marine_train(paths: Dict[str, str], image_size: int = 256,
                       colorcast: float = 0.0) -> Dataset:
    """Stage A 학습셋 = EUVP(imagenet+dark) + UIEB train split (ConcatDataset).

    colorcast>0 이면 low 입력에 채널별 랜덤 색캐스트 augmentation(도메인 일반화).
    """
    dsets: List[Dataset] = []
    if colorcast > 0:
        print(f"  [aug] color-cast p={colorcast}")

    euvp_root = Path(paths["euvp"])
    for sd in EUVP_SUBSETS:
        d = euvp_root / sd
        if (d / "trainA").is_dir() and (d / "trainB").is_dir():
            ds = EUVPPairedDataset(d, image_size=image_size, augment=True,
                                   name=f"EUVP-{sd}", colorcast=colorcast)
            dsets.append(ds)
            print(f"  [train] EUVP/{sd}: {len(ds)} pairs")
        else:
            print(f"  [skip]  EUVP/{sd}: trainA/trainB 없음 ({d})")

    try:
        u = UIEBDataset(paths["uieb"], "train", paths["uieb_split_csv"],
                        image_size=image_size, augment=True, colorcast=colorcast)
        dsets.append(u)
        print(f"  [train] UIEB[train]: {len(u)} pairs")
    except (FileNotFoundError, RuntimeError) as e:
        print(f"  [skip]  UIEB[train]: {e}")

    if not dsets:
        raise RuntimeError("MARINE 학습 데이터셋이 0개입니다. paths_marine.yaml 확인.")
    total = sum(len(d) for d in dsets)
    print(f"  [train] 합계: {total} pairs ({len(dsets)} sources)")
    return ConcatDataset(dsets) if len(dsets) > 1 else dsets[0]


def build_marine_eval(paths: Dict[str, str], eval_size: int = 256) -> Dataset:
    """Stage A 검증셋 = UIEB test split (paired, resize-only)."""
    ds = UIEBDataset(paths["uieb"], "test", paths["uieb_split_csv"],
                     image_size=eval_size, augment=False)
    print(f"  [eval]  UIEB[test]: {len(ds)} pairs (resize {eval_size}, augment off)")
    return ds
