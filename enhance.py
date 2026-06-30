"""MARINE 추론 CLI — 수중 영상 향상 (바로 사용 가능한 프로토타입 진입점).

학습된 MARINE(BilateralLowLightNet) 가중치로 단일 이미지 또는 폴더를 향상해 저장한다.
네이티브 해상도 보존(필요 시 --max_side 로 VRAM 제한). 입출력 [-1,1] 내부 처리.

기본 체크포인트 탐색 순서
-------------------------
1) --ckpt 인자
2) runs/marine_best.pth            (최고 모델 안정 경로; 학습 파이프라인이 갱신)
3) runs/marineA_warmstart/checkpoints/best_psnr.pth   (Stage A 베이스라인)

사용 예
-------
    python enhance.py --input sample.jpg --output out/
    python enhance.py --input some_folder/ --output out/ --metrics
    python enhance.py --input img.png --output out/ --ckpt runs/마이체크포인트.pth
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

_HERE = Path(__file__).resolve()
_MARINE_ROOT = _HERE.parent
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
from PIL import Image
from torchvision.transforms import InterpolationMode

from src.models.bilateral_grid import build_from_config

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _default_ckpt() -> Optional[Path]:
    for cand in (_MARINE_ROOT / "runs" / "marine_best.pth",
                 _MARINE_ROOT / "runs" / "marineA_warmstart" / "checkpoints" / "best_psnr.pth"):
        if cand.is_file():
            return cand
    return None


def load_model(ckpt_path: Path, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = {"model": ckpt.get("model_cfg") or ckpt.get("config", {}).get("model", {})}
    model = build_from_config(cfg).to(device)
    _gv = cfg["model"].get("guidance_variant", "luma") if isinstance(cfg["model"], dict) else "luma"
    if _gv and _gv != "luma":
        from marine.models.guidance_variants import swap_guidance
        swap_guidance(model, _gv, c_hidden=cfg["model"].get("guidance_channels", 16))
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(sd)
    model.eval()
    return model, ckpt.get("epoch")


def gather_inputs(inp: Path) -> List[Path]:
    if inp.is_file():
        return [inp]
    if inp.is_dir():
        return sorted(p for p in inp.rglob("*")
                      if p.is_file() and p.suffix.lower() in _IMG_EXTS
                      and "__MACOSX" not in str(p))
    return []


@torch.no_grad()
def enhance_image(model, img: Image.Image, device: str, max_side: int) -> np.ndarray:
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
    ap = argparse.ArgumentParser(description="MARINE 수중 영상 향상")
    ap.add_argument("--input", required=True, help="이미지 파일 또는 폴더")
    ap.add_argument("--output", default="marine_out", help="출력 폴더")
    ap.add_argument("--ckpt", default=None, help="체크포인트(.pth). 생략 시 자동 탐색")
    ap.add_argument("--max_side", type=int, default=1536, help="긴 변 상한(VRAM). 0=무제한")
    ap.add_argument("--device", default=None)
    ap.add_argument("--metrics", action="store_true", help="입력/출력 UIQM·UCIQE 출력")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.ckpt) if args.ckpt else _default_ckpt()
    if ckpt_path is None or not ckpt_path.is_file():
        print("[error] 체크포인트를 찾을 수 없습니다. --ckpt 로 지정하거나 학습을 먼저 수행하세요.")
        return 1

    inputs = gather_inputs(Path(args.input))
    if not inputs:
        print(f"[error] 입력 이미지가 없습니다: {args.input}")
        return 1

    model, epoch = load_model(ckpt_path, device)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_fn = None
    if args.metrics:
        from marine.utils.uw_metrics import getUIQM, getUCIQE
        metric_fn = (getUIQM, getUCIQE)

    print(f"MARINE enhance | ckpt={ckpt_path.name} (epoch {epoch}) | device={device} "
          f"| {len(inputs)} 장 | max_side={args.max_side}")
    print("-" * 70)
    t0 = time.time()
    for i, fp in enumerate(inputs, 1):
        try:
            img = Image.open(fp).convert("RGB")
        except Exception as e:
            print(f"  [skip] {fp.name}: {e}")
            continue
        out = enhance_image(model, img, device, args.max_side)
        out_path = out_dir / f"{fp.stem}_marine.png"
        Image.fromarray(out).save(out_path)
        line = f"  [{i}/{len(inputs)}] {fp.name} → {out_path.name}"
        if metric_fn is not None:
            gU, gC = metric_fn
            inp = np.array(img)
            line += (f"  | UIQM {gU(inp):.2f}→{gU(out):.2f}  "
                     f"UCIQE {gC(inp):.1f}→{gC(out):.1f}")
        print(line)
    dt = time.time() - t0
    print("-" * 70)
    print(f"완료: {len(inputs)}장 → {out_dir}  ({dt:.1f}s, {dt/max(len(inputs),1)*1000:.0f} ms/장)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
