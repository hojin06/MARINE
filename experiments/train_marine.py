"""MARINE Stage A 학습 — 수중 지도복원 (LUNA2 BilateralLowLightNet warm-start).

개요
----
* 모델/손실/평가/메트릭은 LUNA2 ``src`` 에서 **import 재사용**(코드 복사 금지).
* 데이터/증강만 MARINE ``marine.data`` 의 수중 어댑터로 교체.
* warm-start: LUNA2 Phase1 best 체크포인트(guidefix)의 model 가중치를 로드 후 fine-tune.
* 손실: L1 + VGG(relu3_3) + SSIM (LUNA2 CombinedRestorationLoss). gray-world 는 M2.
* 평가: UIEB test split PSNR/SSIM. best/last 체크포인트 저장.

사용
----
    python experiments/train_marine.py --config configs/marine_stageA.yaml
    # 스모크(소량·few iters):
    python experiments/train_marine.py --config configs/marine_stageA.yaml \
        --smoke --exp_name marineA_smoke
"""
from __future__ import annotations

import argparse
import csv as _csv
import random
import sys
import time
from pathlib import Path
from typing import Dict

# --- 경로: MARINE 루트 + LUNA2 루트를 import 경로에 등록 ---
_HERE = Path(__file__).resolve()
_MARINE_ROOT = _HERE.parents[1]
_LUNA2_ROOT = _MARINE_ROOT.parent / "LUNA2"
for p in (str(_LUNA2_ROOT), str(_MARINE_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

# --- LUNA2 재사용 ---
from src.models.bilateral_grid import build_from_config
from src.losses.restoration import build_restoration_loss
from src.utils.metrics import evaluate
# --- MARINE 자체 ---
from marine.data.underwater_dataset import build_marine_train, build_marine_eval
from marine.losses.underwater import gray_world_loss

HRULE = "=" * 78


# ===========================================================================
# 설정 / 경로
# ===========================================================================
def load_config(path: Path | str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_marine_paths(path: Path | str) -> Dict[str, str]:
    cfg = yaml.safe_load(open(path, encoding="utf-8"))
    root = Path(cfg["marine_root"]).resolve()
    out: Dict[str, str] = {}
    for k, v in cfg["datasets"].items():
        out[k] = str((root / v).resolve())
    out["uieb_split_csv"] = str((root / cfg["uieb_split_csv"]).resolve())
    out["warmstart"] = str(Path(cfg["warmstart"]["luna2_p1"]).resolve())
    out["runs"] = str((root / cfg["outputs"]["runs"]).resolve())
    return out


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ===========================================================================
# 체크포인트
# ===========================================================================
def save_ckpt(path: Path, model, optimizer, scheduler, scaler, cfg,
              epoch, global_step, best_psnr, best_ssim) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "model_cfg": cfg["model"],
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch, "global_step": global_step,
        "best_psnr": best_psnr, "best_ssim": best_ssim, "config": cfg,
    }, path)


def load_warmstart(model, ckpt_path: Path, device: str) -> None:
    """LUNA2 Phase1 체크포인트의 model 가중치를 로드 (구조 동일 가정)."""
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  warm-start ← {ckpt_path}")
    if missing:
        print(f"    [warn] missing keys: {len(missing)} (예: {missing[:3]})")
    if unexpected:
        print(f"    [warn] unexpected keys: {len(unexpected)} (예: {unexpected[:3]})")
    if not missing and not unexpected:
        print("    가중치 완전 일치 (구조 동일).")


def load_resume(path: Path, model, optimizer, scheduler, scaler, device) -> dict:
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    if state.get("optimizer"):
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler"):
        scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and state.get("scaler"):
        scaler.load_state_dict(state["scaler"])
    return state


@torch.no_grad()
def save_samples(model, eval_loader, out_path: Path, n: int, device: str) -> None:
    try:
        from torchvision.utils import make_grid, save_image
    except Exception:
        return
    model.eval()
    rows, count = [], 0
    for low, high in eval_loader:
        low, high = low.to(device), high.to(device)
        enh = model(low).clamp(-1.0, 1.0)
        for b in range(low.size(0)):
            rows.append(torch.stack([low[b], enh[b], high[b]], dim=0))
            count += 1
            if count >= n:
                break
        if count >= n:
            break
    if not rows:
        return
    grid_in = (torch.cat(rows, dim=0) + 1.0) * 0.5
    grid = make_grid(grid_in, nrow=3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, str(out_path))


# ===========================================================================
# CLI
# ===========================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MARINE Stage A 학습 (수중 지도복원)")
    p.add_argument("--config", type=str, default="configs/marine_stageA.yaml")
    p.add_argument("--paths", type=str, default="configs/paths_marine.yaml")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--exp_name", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--no_warmstart", action="store_true")
    p.add_argument("--warmstart_ckpt", type=str, default=None,
                   help="warm-start 소스 체크포인트 override(기본: paths_marine.warmstart)")
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--gpu_util", type=float, default=1.0,
                   help="목표 GPU 평균 가동률(0~1). 예: 0.8 → 매 iter 약 25%% 휴식으로 "
                        "다른 작업용 여유 확보(학습은 그만큼 느려짐). 1.0=제한없음")
    p.add_argument("--grayworld", type=float, default=None,
                   help="gray-world 색항상성 손실 가중치(None=config의 loss.lambda_grayworld 사용)")
    p.add_argument("--guidance", type=str, default=None, choices=[None, "luma", "maxrgb"],
                   help="guidance anchor 변형(M3.2 ablation). None=config/luma")
    p.add_argument("--smoke", action="store_true",
                   help="소량 subset + 50 iter 로 빠른 파이프라인 점검")
    p.add_argument("--max_train_samples", type=int, default=0)
    p.add_argument("--max_eval_samples", type=int, default=0)
    p.add_argument("--max_iters", type=int, default=0)
    return p.parse_args()


def apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    if args.exp_name:
        cfg["experiment_name"] = args.exp_name
    if args.epochs is not None:
        cfg["training"]["num_epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["training"]["lr"] = args.lr
    if args.num_workers is not None:
        cfg["data"]["num_workers"] = args.num_workers
    if args.no_warmstart:
        cfg["training"]["warmstart"] = False
    if args.no_amp:
        cfg["training"]["amp"] = False
    if args.smoke:
        if not args.exp_name:
            cfg["experiment_name"] = cfg["experiment_name"] + "_smoke"
        if args.max_train_samples == 0:
            args.max_train_samples = 64
        if args.max_iters == 0:
            args.max_iters = 50
        cfg["data"]["num_workers"] = 0
    return cfg


# ===========================================================================
# 학습
# ===========================================================================
def train(cfg: dict, paths: Dict[str, str], args: argparse.Namespace) -> None:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.get("seed", 42))

    tr, dc, lg = cfg["training"], cfg["data"], cfg["logging"]
    exp = cfg["experiment_name"]
    run_dir = Path(paths["runs"]) / exp
    ckpt_dir, log_dir, sample_dir = run_dir / "checkpoints", run_dir / "logs", run_dir / "samples"
    for d in (ckpt_dir, log_dir, sample_dir):
        d.mkdir(parents=True, exist_ok=True)

    img_size = dc.get("image_size", 256)
    eval_size = dc.get("eval_size", 256)
    batch_size = tr.get("batch_size", 16)
    num_workers = dc.get("num_workers", 4)
    num_epochs = tr.get("num_epochs", 30)
    use_amp = tr.get("amp", True) and device.startswith("cuda")
    grad_clip = float(tr.get("grad_clip", 0.0))
    # GPU 가동률 throttle: 목표 util u 이면 매 iter 계산시간의 (1/u - 1) 만큼 sleep
    gpu_util = float(getattr(args, "gpu_util", 1.0) or 1.0)
    throttle = (1.0 / max(gpu_util, 1e-3) - 1.0) if gpu_util < 0.999 else 0.0

    print(HRULE)
    print(f" MARINE Stage A 학습 — {exp}")
    print(HRULE)
    print(f"  device={device}  AMP={use_amp}  img_size={img_size}  batch/epoch={batch_size}/{num_epochs}")
    print(f"  run_dir={run_dir}")
    print(HRULE)

    # --- 데이터 ---
    train_set = build_marine_train(paths, image_size=img_size,
                                   colorcast=float(dc.get("colorcast", 0.0)),
                                   synth_limit=int(dc.get("synth_limit", 0)))
    eval_set = build_marine_eval(paths, eval_size=eval_size)
    if args.max_train_samples > 0:
        idx = list(range(min(args.max_train_samples, len(train_set))))
        train_set = Subset(train_set, idx)
        print(f"  [smoke] train subset → {len(train_set)}")
    if args.max_eval_samples > 0:
        idx = list(range(min(args.max_eval_samples, len(eval_set))))
        eval_set = Subset(eval_set, idx)
        print(f"  [smoke] eval subset → {len(eval_set)}")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=device.startswith("cuda"),
                              drop_last=True)
    eval_loader = DataLoader(eval_set, batch_size=1, shuffle=False,
                             num_workers=min(num_workers, 2), pin_memory=device.startswith("cuda"))
    print(f"  train iters/epoch: {len(train_loader)}")
    print(HRULE)

    # --- 모델 / warm-start ---
    model = build_from_config(cfg).to(device)
    # guidance ablation(M3.2): warm-start 전에 교체해야 guidance 가중치가 로드됨
    gvar = args.guidance if args.guidance is not None else cfg["model"].get("guidance_variant", "luma")
    if gvar and gvar != "luma":
        from marine.models.guidance_variants import swap_guidance
        swap_guidance(model, gvar, c_hidden=cfg["model"].get("guidance_channels", 16))
        cfg["model"]["guidance_variant"] = gvar
        print(f"  guidance variant: {gvar}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model params: {n_params:,} ({n_params/1e3:.1f} K)")
    if tr.get("warmstart", True) and not args.resume:
        ws = Path(args.warmstart_ckpt) if args.warmstart_ckpt else Path(paths["warmstart"])
        if ws.is_file():
            load_warmstart(model, ws, device)
        else:
            print(f"  [warn] warm-start 체크포인트 없음: {ws} — 무작위 초기화로 진행")
    else:
        print("  warm-start: off")

    criterion = build_restoration_loss(cfg).to(device)
    # gray-world 색항상성 손실(선택): CLI > config
    gw_w = args.grayworld if args.grayworld is not None else float(cfg["loss"].get("lambda_grayworld", 0.0))
    use_gw = gw_w and gw_w > 0
    if use_gw:
        criterion.add_term("grayworld", gray_world_loss, weight=gw_w)
        cfg["loss"]["lambda_grayworld"] = gw_w
        print(f"  loss: + gray-world (λ={gw_w})")

    lr = float(tr.get("lr", 1e-4))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 betas=tuple(tr.get("betas", [0.9, 0.999])),
                                 weight_decay=float(tr.get("weight_decay", 0.0)))
    scheduler = None
    if tr.get("scheduler", "cosine") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=lr * float(tr.get("eta_min_ratio", 0.01)))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # --- resume ---
    start_epoch, global_step, best_psnr, best_ssim = 0, 0, -1.0, -1.0
    resume_path = Path(args.resume) if args.resume else (
        ckpt_dir / "last.pth" if (ckpt_dir / "last.pth").is_file() else None)
    if resume_path and resume_path.is_file():
        st = load_resume(resume_path, model, optimizer, scheduler, scaler, device)
        start_epoch = st.get("epoch", 0) + 1
        global_step = st.get("global_step", 0)
        best_psnr = st.get("best_psnr", -1.0)
        best_ssim = st.get("best_ssim", -1.0)
        print(f"  resumed ← {resume_path} (epoch {start_epoch}, step {global_step})")

    with open(run_dir / "config.used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    csv_path = log_dir / "train_log.csv"
    csv_new = not csv_path.is_file()
    csv_file = open(csv_path, "a", newline="", encoding="utf-8")
    cw = _csv.writer(csv_file)
    if csv_new:
        cols = ["epoch", "global_step", "lr", "train_total", "train_l1",
                "train_vgg", "train_ssim"]
        if use_gw:
            cols.append("train_grayworld")
        cols += ["val_psnr", "val_ssim"]
        cw.writerow(cols)

    # --- 학습 루프 ---
    stop = False
    epoch = start_epoch
    for epoch in range(start_epoch, num_epochs):
        model.train()
        sums = {"total": 0.0, "l1": 0.0, "vgg": 0.0, "ssim": 0.0}
        if use_gw:
            sums["grayworld"] = 0.0
        nb, t0 = 0, time.time()
        for low, high in train_loader:
            _it0 = time.time()
            low = low.to(device, non_blocking=True)
            high = high.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            # 모델 forward 만 autocast(fp16). 손실(SSIM/VGG)은 fp32 (LUNA2 와 동일 가드).
            with torch.cuda.amp.autocast(enabled=use_amp):
                enhanced = model(low)
            losses = criterion(enhanced.float(), high.float())
            loss = losses["total"]
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            # GPU 가동률 throttle (다른 작업용 여유 확보)
            if throttle > 0:
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                time.sleep((time.time() - _it0) * throttle)

            global_step += 1
            nb += 1
            for k in sums:
                sums[k] += float(losses[k])
            if global_step % 20 == 0 or (args.max_iters and global_step <= 5):
                print(f"  e{epoch} step {global_step}  total={float(loss):.4f} "
                      f"l1={float(losses['l1']):.4f} vgg={float(losses['vgg']):.4f} "
                      f"ssim={float(losses['ssim']):.4f}")
            if args.max_iters and global_step >= args.max_iters:
                print(f"  [smoke] max_iters={args.max_iters} 도달 — 중단")
                stop = True
                break

        denom = max(nb, 1)
        avg = {k: v / denom for k, v in sums.items()}
        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"  [epoch {epoch}] avg total={avg['total']:.4f} l1={avg['l1']:.4f} "
              f"vgg={avg['vgg']:.4f} ssim={avg['ssim']:.4f} lr={cur_lr:.2e} "
              f"({time.time()-t0:.1f}s)")
        if scheduler is not None:
            scheduler.step()

        val_psnr = val_ssim = float("nan")
        if ((epoch + 1) % lg.get("eval_every", 1) == 0) or stop:
            m = evaluate(model, eval_loader, device=device)
            val_psnr, val_ssim = m["psnr"], m["ssim"]
            print(f"  [val {epoch}] PSNR={val_psnr:.4f} dB  SSIM={val_ssim:.4f}  (n={m['n']})")
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                save_ckpt(ckpt_dir / "best_psnr.pth", model, optimizer, scheduler,
                          scaler, cfg, epoch, global_step, best_psnr, best_ssim)
                print("           ↳ best PSNR → best_psnr.pth")
            if val_ssim > best_ssim:
                best_ssim = val_ssim
                save_ckpt(ckpt_dir / "best_ssim.pth", model, optimizer, scheduler,
                          scaler, cfg, epoch, global_step, best_psnr, best_ssim)
                print("           ↳ best SSIM → best_ssim.pth")

        row = [epoch, global_step, f"{cur_lr:.6e}", f"{avg['total']:.6f}",
               f"{avg['l1']:.6f}", f"{avg['vgg']:.6f}", f"{avg['ssim']:.6f}"]
        if use_gw:
            row.append(f"{avg['grayworld']:.6f}")
        row += [f"{val_psnr:.6f}", f"{val_ssim:.6f}"]
        cw.writerow(row)
        csv_file.flush()

        if ((epoch + 1) % lg.get("sample_every", 5) == 0) or stop:
            save_samples(model, eval_loader, sample_dir / f"epoch_{epoch:03d}.png",
                         n=lg.get("num_samples", 4), device=device)
        if ((epoch + 1) % lg.get("save_every", 5) == 0) or stop or (epoch + 1 == num_epochs):
            save_ckpt(ckpt_dir / "last.pth", model, optimizer, scheduler, scaler,
                      cfg, epoch, global_step, best_psnr, best_ssim)
        if stop:
            break

    save_ckpt(ckpt_dir / "last.pth", model, optimizer, scheduler, scaler,
              cfg, min(epoch, num_epochs - 1), global_step, best_psnr, best_ssim)
    csv_file.close()
    print(HRULE)
    print(f"  학습 종료. best PSNR={best_psnr:.4f}  best SSIM={best_ssim:.4f}")
    print(f"  산출물: {run_dir}")
    print(HRULE)


def main() -> int:
    args = parse_args()
    cfg = load_config(_MARINE_ROOT / args.config if not Path(args.config).is_absolute() else args.config)
    paths = load_marine_paths(_MARINE_ROOT / args.paths if not Path(args.paths).is_absolute() else args.paths)
    cfg = apply_overrides(cfg, args)
    train(cfg, paths, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
