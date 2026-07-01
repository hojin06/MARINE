"""M4.3a: MARINE 향상이 수중 검출(DUO)에 도움되는가 — frozen YOLO 평가.

프로토콜: raw DUO 로 학습한 YOLO(frozen)를 그대로 두고,
  (1) raw DUO test  → mAP
  (2) MARINE-향상 DUO test → mAP
를 비교한다. Δ>0 이면 향상이 고정 검출기 성능을 높임(전처리 유효).

MARINE 향상은 네트워크 출력(디헤이즈/색보정)만 사용(후처리 CLAHE/WB 제외 옵션).

사용: python experiments/eval_detection_uw.py
"""
from __future__ import annotations

import argparse
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
from marine.utils.postprocess import postprocess


def enhance_dir(model, src: Path, dst: Path, device: str, max_side: int, post: bool) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = [p for p in src.iterdir() if p.suffix.lower() in exts]
    for i, fp in enumerate(files):
        img = Image.open(fp).convert("RGB")
        w, h = img.size
        im = img
        if max_side > 0 and max(w, h) > max_side:
            s = max_side / max(w, h)
            im = img.resize((int(w * s), int(h * s)))
        t = TF.to_tensor(im).unsqueeze(0).to(device) * 2 - 1
        with torch.no_grad():
            o = model(t).clamp(-1, 1)
        arr = (((o[0] + 1) * 0.5).permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        if post:
            arr = postprocess(arr, wb=True, sat_gain=1.15)
        # 원본 해상도로 복원 저장(라벨 좌표 정합 유지)
        Image.fromarray(arr).resize((w, h)).save(dst / f"{fp.stem}.jpg", quality=95)
    return len(files)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(_MARINE_ROOT / "runs/duo_detector/duo_yolov8n/weights/best.pt"))
    ap.add_argument("--marine", default=str(_MARINE_ROOT / "runs/marine_best.pth"))
    ap.add_argument("--duo_yaml", default=str(_MARINE_ROOT / "configs/duo.yaml"))
    ap.add_argument("--max_side", type=int, default=1280)
    ap.add_argument("--post", action="store_true", help="후처리(CLAHE/WB)까지 적용")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from ultralytics import YOLO

    duo = yaml.safe_load(open(args.duo_yaml, encoding="utf-8"))
    duo_root = Path(duo["path"])
    test_img = duo_root / "images" / "test"

    yolo = YOLO(args.weights)
    print("=" * 60)
    r_raw = yolo.val(data=args.duo_yaml, split="val", verbose=False, plots=False)
    print(f"[RAW]      mAP50={r_raw.box.map50:.4f}  mAP50-95={r_raw.box.map:.4f}")

    # MARINE 향상본 생성
    ck = torch.load(args.marine, map_location=device, weights_only=False)
    m = build_from_config({"model": ck["model_cfg"]}).to(device)
    m.load_state_dict(ck["model"]); m.eval()
    tag = "test_marine_post" if args.post else "test_marine"
    enh_img = duo_root / "images" / tag
    n = enhance_dir(m, test_img, enh_img, device, args.max_side, args.post)
    # 라벨 복사(images→labels 규칙)
    lbl_src = duo_root / "labels" / "test"
    lbl_dst = duo_root / "labels" / tag
    if lbl_dst.exists():
        shutil.rmtree(lbl_dst)
    shutil.copytree(lbl_src, lbl_dst)
    # enhanced yaml
    enh_yaml = _MARINE_ROOT / "configs" / f"duo_{tag}.yaml"
    dd = dict(duo); dd["val"] = f"images/{tag}"
    yaml.safe_dump(dd, open(enh_yaml, "w", encoding="utf-8"), allow_unicode=True, sort_keys=False)
    print(f"  향상 {n}장 → {enh_img.name}, 라벨 복사 완료")

    r_enh = yolo.val(data=str(enh_yaml), split="val", verbose=False, plots=False)
    print(f"[MARINE{'+post' if args.post else ''}] mAP50={r_enh.box.map50:.4f}  mAP50-95={r_enh.box.map:.4f}")
    print("-" * 60)
    print(f"Δ mAP50    = {r_enh.box.map50 - r_raw.box.map50:+.4f}")
    print(f"Δ mAP50-95 = {r_enh.box.map - r_raw.box.map:+.4f}")
    print("(Δ>0 이면 MARINE 향상이 고정 검출기의 검출 성능을 높임)")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
