"""M4.3b 준비: DUO train/test 를 MARINE 으로 향상 → 향상도메인 검출 학습용 데이터.

images/{train,test}_marine 생성 + labels 복사 + configs/duo_marine.yaml 작성.
(test_marine 은 M4.3a 에서 생성됐을 수 있어 있으면 재사용.)

사용: python experiments/prep_duo_marine.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_MARINE_ROOT = _HERE.parents[1]
_LUNA2_ROOT = _MARINE_ROOT.parent / "LUNA2"
for p in (str(_LUNA2_ROOT), str(_MARINE_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore
    except Exception:
        pass

import numpy as np
import torch
import torchvision.transforms.functional as TF
import yaml
from PIL import Image

from src.models.bilateral_grid import build_from_config


@torch.no_grad()
def enhance_split(model, src: Path, dst: Path, device: str, max_side: int) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = [p for p in src.iterdir() if p.suffix.lower() in exts]
    done = 0
    for fp in files:
        outp = dst / f"{fp.stem}.jpg"
        if outp.exists():
            done += 1
            continue
        img = Image.open(fp).convert("RGB")
        w, h = img.size
        im = img.resize((int(w * max_side / max(w, h)), int(h * max_side / max(w, h)))) \
            if (max_side > 0 and max(w, h) > max_side) else img
        t = TF.to_tensor(im).unsqueeze(0).to(device) * 2 - 1
        o = model(t).clamp(-1, 1)
        arr = (((o[0] + 1) * 0.5).permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(arr).resize((w, h)).save(outp, quality=95)
        done += 1
        if done % 500 == 0:
            print(f"  {dst.name}: {done}/{len(files)}")
    return len(files)


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    duo = yaml.safe_load(open(_MARINE_ROOT / "configs" / "duo.yaml", encoding="utf-8"))
    duo_root = Path(duo["path"])

    ck = torch.load(_MARINE_ROOT / "runs" / "marine_best.pth", map_location=device, weights_only=False)
    m = build_from_config({"model": ck["model_cfg"]}).to(device)
    m.load_state_dict(ck["model"]); m.eval()

    for split in ("train", "test"):
        n = enhance_split(m, duo_root / "images" / split,
                          duo_root / "images" / f"{split}_marine", device, 1280)
        lsrc = duo_root / "labels" / split
        ldst = duo_root / "labels" / f"{split}_marine"
        if not ldst.exists():
            shutil.copytree(lsrc, ldst)
        print(f"{split}: {n}장 향상 + 라벨 복사")

    dd = dict(duo); dd["train"] = "images/train_marine"; dd["val"] = "images/test_marine"
    outp = _MARINE_ROOT / "configs" / "duo_marine.yaml"
    yaml.safe_dump(dd, open(outp, "w", encoding="utf-8"), allow_unicode=True, sort_keys=False)
    print(f"duo_marine.yaml → {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
