"""MARINE TURBID 탁도단계별 robustness 평가.

TURBID(Milk/Chlorophyll/DeepBlue/TURBID3D)는 탁도를 단계적으로 높인 시퀀스다.
파일명 자연순서를 탁도 proxy 로 보고, 각 단계에서 입력 vs MARINE 향상의
UIQM/UCIQE 를 측정한다. 탁도가 올라가도 향상(Δ>0)이 유지되는지 본다.

출력: 콘솔 per-subset 요약 + 시퀀스 분위(저/중/고 탁도) Δ + per-image CSV(플롯용).

사용:
    python experiments/eval_turbid_curve.py \
        --ckpt runs/marineA_warmstart/checkpoints/best_psnr.pth \
        --out runs/marineA_warmstart/turbid_curve.csv
"""
from __future__ import annotations

import argparse
import csv
import re
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
SUBSETS = ["Milk", "Chlorophyll", "DeepBlue", "TURBID3D"]


def _natkey(p: Path):
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", p.name)]


def list_seq(root: Path) -> List[Path]:
    files = [p for p in root.rglob("*")
             if p.is_file() and p.suffix.lower() in _IMG_EXTS and "__MACOSX" not in str(p)]
    return sorted(files, key=_natkey)


@torch.no_grad()
def enhance(model, img: Image.Image, device: str, max_side: int) -> np.ndarray:
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
    ap.add_argument("--out", default=None, help="per-image CSV 저장 경로")
    ap.add_argument("--max_side", type=int, default=1024)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg_paths = yaml.safe_load(open(_MARINE_ROOT / args.paths, encoding="utf-8"))
    troot = Path(cfg_paths["marine_root"]) / cfg_paths["datasets"]["turbid"]

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = {"model": ckpt.get("model_cfg", ckpt.get("config", {}).get("model", {}))}
    model = build_from_config(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    rows = []
    print(f"ckpt: {args.ckpt} (epoch {ckpt.get('epoch')})  device={device}")
    print("=" * 92)
    print(f"{'subset':12} {'n':>3} | {'ΔUIQM(전체)':>11} | "
          f"{'ΔUIQM 저탁도':>12} {'중':>7} {'고':>7} | {'ΔUCIQE 저':>10} {'중':>7} {'고':>7}")
    print("-" * 92)
    for sub in SUBSETS:
        root = troot / sub
        if not root.is_dir():
            print(f"{sub:12}  [skip] 없음")
            continue
        files = list_seq(root)
        if not files:
            print(f"{sub:12}  [skip] 이미지 0")
            continue
        dU, dC = [], []
        for i, fp in enumerate(files):
            try:
                img = Image.open(fp).convert("RGB")
            except Exception:
                continue
            inp = np.array(img)
            out = enhance(model, img, device, args.max_side)
            ui_i, ui_o = getUIQM(inp), getUIQM(out)
            uc_i, uc_o = getUCIQE(inp), getUCIQE(out)
            dU.append(ui_o - ui_i); dC.append(uc_o - uc_i)
            rows.append([sub, i, f"{ui_i:.4f}", f"{ui_o:.4f}", f"{uc_i:.4f}", f"{uc_o:.4f}"])
        dU, dC = np.array(dU), np.array(dC)
        n = len(dU)
        # 시퀀스 3분할(저/중/고 탁도)
        t = max(n // 3, 1)
        duL, duM, duH = dU[:t].mean(), dU[t:2 * t].mean(), dU[2 * t:].mean()
        dcL, dcM, dcH = dC[:t].mean(), dC[t:2 * t].mean(), dC[2 * t:].mean()
        print(f"{sub:12} {n:3d} | {dU.mean():+11.3f} | "
              f"{duL:+12.3f} {duM:+7.3f} {duH:+7.3f} | {dcL:+10.3f} {dcM:+7.3f} {dcH:+7.3f}")
    print("=" * 92)
    print("(저/중/고 = 시퀀스 앞/중/뒤 1/3 = 탁도 낮음/중간/높음. Δ>0 유지면 탁도 robust.)")

    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["subset", "seq_idx", "uiqm_in", "uiqm_out", "uciqe_in", "uciqe_out"])
            w.writerows(rows)
        print(f"per-image CSV → {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
