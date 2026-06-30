"""MARINE paired 복원 평가 — UIEB test (네이티브 해상도) PSNR/SSIM + UIQM/UCIQE.

학습 중 검증은 256 리사이즈였지만, 여기서는 **원본 해상도**로 입력↔참조,
출력↔참조를 비교한다(BilateralLowLightNet 은 네이티브 해상도 보존).

사용:
    python experiments/eval_restoration_uw.py \
        --ckpt runs/marineA_warmstart/checkpoints/best_psnr.pth
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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
from src.utils.metrics import psnr_metric, ssim_metric
from marine.utils.uw_metrics import getUIQM, getUCIQE

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def load_marine_paths(path: Path) -> Dict[str, str]:
    cfg = yaml.safe_load(open(path, encoding="utf-8"))
    root = Path(cfg["marine_root"]).resolve()
    out = {k: str((root / v).resolve()) for k, v in cfg["datasets"].items()}
    out["uieb_split_csv"] = str((root / cfg["uieb_split_csv"]).resolve())
    return out


def uieb_test_pairs(paths: Dict[str, str], split: str = "test") -> List[Tuple[Path, Path]]:
    raw = Path(paths["uieb"]) / "raw-890"
    ref = Path(paths["uieb"]) / "reference-890"
    stems = set()
    with open(paths["uieb_split_csv"], encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] == split:
                stems.add(row["stem"])
    ref_idx = {p.stem: p for p in ref.iterdir() if p.suffix.lower() in _IMG_EXTS}
    pairs = []
    for rp in sorted(raw.iterdir()):
        if rp.suffix.lower() in _IMG_EXTS and rp.stem in stems and rp.stem in ref_idx:
            pairs.append((rp, ref_idx[rp.stem]))
    return pairs


def _to_t(img: Image.Image, device: str) -> torch.Tensor:
    return TF.to_tensor(img).unsqueeze(0).to(device) * 2.0 - 1.0


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--paths", default="configs/paths_marine.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max_side", type=int, default=1280)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    paths = load_marine_paths(_MARINE_ROOT / args.paths)
    pairs = uieb_test_pairs(paths, args.split)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = {"model": ckpt.get("model_cfg", ckpt.get("config", {}).get("model", {}))}
    model = build_from_config(cfg).to(device)
    _gv = cfg["model"].get("guidance_variant", "luma") if isinstance(cfg["model"], dict) else "luma"
    if _gv and _gv != "luma":
        from marine.models.guidance_variants import swap_guidance
        swap_guidance(model, _gv, c_hidden=cfg["model"].get("guidance_channels", 16))
    model.load_state_dict(ckpt["model"])
    model.eval()

    acc = {"psnr_in": 0.0, "ssim_in": 0.0, "psnr_out": 0.0, "ssim_out": 0.0,
           "uiqm_in": 0.0, "uiqm_out": 0.0, "uciqe_in": 0.0, "uciqe_out": 0.0}
    n = 0
    for rp, fp in pairs:
        raw = Image.open(rp).convert("RGB")
        ref = Image.open(fp).convert("RGB")
        if raw.size != ref.size:
            ref = TF.resize(ref, [raw.size[1], raw.size[0]],
                            interpolation=InterpolationMode.BILINEAR)
        if args.max_side > 0 and max(raw.size) > args.max_side:
            s = args.max_side / max(raw.size)
            sz = [int(round(raw.size[1] * s)), int(round(raw.size[0] * s))]
            raw = TF.resize(raw, sz, interpolation=InterpolationMode.BILINEAR)
            ref = TF.resize(ref, sz, interpolation=InterpolationMode.BILINEAR)
        x = _to_t(raw, device)
        y = _to_t(ref, device)
        out = model(x).clamp(-1, 1)
        acc["psnr_in"] += psnr_metric(x, y);   acc["ssim_in"] += ssim_metric(x, y)
        acc["psnr_out"] += psnr_metric(out, y); acc["ssim_out"] += ssim_metric(out, y)
        ai = np.array(raw)
        ao = np.clip(((out[0] + 1) * 0.5).permute(1, 2, 0).cpu().numpy() * 255, 0, 255).astype(np.uint8)
        acc["uiqm_in"] += getUIQM(ai);   acc["uiqm_out"] += getUIQM(ao)
        acc["uciqe_in"] += getUCIQE(ai); acc["uciqe_out"] += getUCIQE(ao)
        n += 1

    if n == 0:
        print("페어 0개 — split/경로 확인")
        return 1
    for k in acc:
        acc[k] /= n
    print(f"ckpt: {args.ckpt} (epoch {ckpt.get('epoch')})  UIEB[{args.split}] n={n} "
          f"(네이티브, max_side={args.max_side})")
    print("=" * 70)
    print(f"{'':10} {'PSNR':>8} {'SSIM':>8} {'UIQM':>8} {'UCIQE':>8}")
    print(f"{'입력':10} {acc['psnr_in']:8.3f} {acc['ssim_in']:8.4f} "
          f"{acc['uiqm_in']:8.3f} {acc['uciqe_in']:8.3f}")
    print(f"{'MARINE':10} {acc['psnr_out']:8.3f} {acc['ssim_out']:8.4f} "
          f"{acc['uiqm_out']:8.3f} {acc['uciqe_out']:8.3f}")
    print(f"{'Δ':10} {acc['psnr_out']-acc['psnr_in']:+8.3f} "
          f"{acc['ssim_out']-acc['ssim_in']:+8.4f} "
          f"{acc['uiqm_out']-acc['uiqm_in']:+8.3f} "
          f"{acc['uciqe_out']-acc['uciqe_in']:+8.3f}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
