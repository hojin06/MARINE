"""DUO 검출기 학습 (M4.2) — YOLOv8n on DUO, 이후 freeze 하여 detection-aware 에 사용.

ultralytics YOLO 로 DUO(4클래스)를 학습한다. ultralytics 학습 루프에는 MARINE 의
--gpu_util throttle 이 안 통하므로, per-batch sleep 콜백으로 평균 GPU 가동률을 제한한다.

사용:
    python experiments/train_duo_detector.py --epochs 100 --gpu_util 0.8
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_MARINE_ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore
    except Exception:
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(_MARINE_ROOT / "configs" / "duo.yaml"))
    ap.add_argument("--model", default="yolov8n.pt", help="사전학습 가중치(자동 다운로드)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--gpu_util", type=float, default=0.8, help="목표 GPU 가동률(per-batch sleep)")
    ap.add_argument("--name", default="duo_yolov8n")
    args = ap.parse_args()

    from ultralytics import YOLO

    throttle = (1.0 / max(args.gpu_util, 1e-3) - 1.0) if args.gpu_util < 0.999 else 0.0
    _st = {"t": None}

    def _on_batch_start(trainer):
        _st["t"] = time.time()

    def _on_batch_end(trainer):
        if throttle > 0 and _st["t"] is not None:
            time.sleep((time.time() - _st["t"]) * throttle)

    model = YOLO(args.model)
    if throttle > 0:
        model.add_callback("on_train_batch_start", _on_batch_start)
        model.add_callback("on_train_batch_end", _on_batch_end)
        print(f"[throttle] gpu_util≈{args.gpu_util} (per-batch sleep {throttle:.2f}×batch_time)")

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=0,
        project=str(_MARINE_ROOT / "runs" / "duo_detector"),
        name=args.name,
        exist_ok=True,
        verbose=True,
    )
    print("DUO 검출기 학습 완료. best.pt:",
          _MARINE_ROOT / "runs" / "duo_detector" / args.name / "weights" / "best.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
