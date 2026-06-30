"""MARINE no-reference 수중 화질 평가 — UIQM / UCIQE (입력 vs 향상).

참조(GT)가 없는 실제 수중 영상에서 향상 효과를 측정한다. 각 데이터셋에 대해
**입력**과 **MARINE 향상 출력**의 UIQM/UCIQE 평균을 비교한다(클수록 좋음).

대상(기본): RUIE-UIQS, RUIE-UCCS, UIEB-challenging.

사용:
    python experiments/eval_noref_uw.py \
        --ckpt runs/marineA_warmstart/checkpoints/best_psnr.pth
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

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
from torchvision.transforms import InterpolationMode

from src.models.bilateral_grid import build_from_config
from marine.utils.uw_metrics import getUIQM, getUCIQE

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def load_marine_paths(path: Path) -> Dict[str, str]:
    cfg = yaml.safe_load(open(path, encoding="utf-8"))
    root = Path(cfg["marine_root"]).resolve()
    return {k: str((root / v).resolve()) for k, v in cfg["datasets"].items()}


def list_images(root: Path, limit: int = 0) -> List[Path]:
    files = sorted(p for p in root.rglob("*")
                   if p.is_file() and p.suffix.lower() in _IMG_EXTS
                   and "__MACOSX" not in str(p))
    if limit > 0 and len(files) > limit:
        # 균등 샘플링 (앞쪽 편향 방지)
        idx = np.linspace(0, len(files) - 1, limit).round().astype(int)
        files = [files[i] for i in idx]
    return files


@torch.no_grad()
def enhance(model, img: Image.Image, device: str, max_side: int) -> np.ndarray:
    """PIL → 모델 향상 → uint8 RGB (H,W,3). max_side 초과 시 비율 유지 축소."""
    w, h = img.size
    if max_side > 0 and max(w, h) > max_side:
        s = max_side / max(w, h)
        img = TF.resize(img, [int(round(h * s)), int(round(w * s))],
                        interpolation=InterpolationMode.BILINEAR)
    t = TF.to_tensor(img).unsqueeze(0).to(device) * 2.0 - 1.0
    out = model(t).clamp(-1, 1)
    arr = ((out[0] + 1.0) * 0.5).permute(1, 2, 0).cpu().numpy()
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--paths", default="configs/paths_marine.yaml")
    ap.add_argument("--limit", type=int, default=150, help="데이터셋당 평가 장수(0=전체)")
    ap.add_argument("--max_side", type=int, default=1280, help="긴 변 상한(VRAM)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    paths = load_marine_paths(_MARINE_ROOT / args.paths)

    # 평가 대상 (이름 → 디렉토리)
    targets = {
        "RUIE-UIQS":        Path(paths["ruie"]) / "UIQS",
        "RUIE-UCCS":        Path(paths["ruie"]) / "UCCS",
        "UIEB-challenging": Path(paths["uieb"]) / "challenging-60",
    }

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = {"model": ckpt.get("model_cfg", ckpt.get("config", {}).get("model", {}))}
    model = build_from_config(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"ckpt: {args.ckpt} (epoch {ckpt.get('epoch')})  device={device}  "
          f"limit={args.limit} max_side={args.max_side}")
    print("=" * 86)
    print(f"{'dataset':18} {'n':>4} | {'UIQM_in':>8} {'UIQM_out':>9} {'ΔUIQM':>7} | "
          f"{'UCIQE_in':>9} {'UCIQE_out':>10} {'ΔUCIQE':>8}")
    print("-" * 86)

    for name, root in targets.items():
        if not root.is_dir():
            print(f"{name:18}  [skip] 디렉토리 없음: {root}")
            continue
        files = list_images(root, args.limit)
        ui_in = ui_out = uc_in = uc_out = 0.0
        n = 0
        for fp in files:
            try:
                img = Image.open(fp).convert("RGB")
            except Exception:
                continue
            inp = np.array(img)
            out = enhance(model, img, device, args.max_side)
            ui_in += getUIQM(inp);  ui_out += getUIQM(out)
            uc_in += getUCIQE(inp); uc_out += getUCIQE(out)
            n += 1
        if n == 0:
            print(f"{name:18}  [skip] 이미지 0개")
            continue
        ui_in, ui_out = ui_in / n, ui_out / n
        uc_in, uc_out = uc_in / n, uc_out / n
        print(f"{name:18} {n:4d} | {ui_in:8.3f} {ui_out:9.3f} {ui_out-ui_in:+7.3f} | "
              f"{uc_in:9.3f} {uc_out:10.3f} {uc_out-uc_in:+8.3f}")
    print("=" * 86)
    print("(UIQM/UCIQE 모두 클수록 좋음. Δ가 양수면 MARINE 향상이 화질을 개선.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
