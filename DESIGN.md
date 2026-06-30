# MARINE — 설계 문서 (구조분석 + 재학습 파이프라인)

**MARINE** = *Modified version of lunA Reconstructed In order to Navigate sea Environment*
목표: LUNA2(저조도 전처리기)를 **탁도 높은 수중환경**용 전처리기로 **재학습(도메인 전이)**.

> 방침: 모델 **아키텍처는 재사용**, 바뀌는 것은 **데이터 도메인 / 증강 / 손실 prior / 평가지표**.
> 코드 베이스는 `../LUNA2`를 부모로 두고 MARINE은 그 위에 **얇은 어댑터 레이어**만 추가한다.

---

## 1. LUNA2 구조 분석 (상속 자산)

### 1.1 핵심 모델 — `BilateralLowLightNet` (HDRNet 계열)
`LUNA2/src/models/bilateral_grid.py`

```
입력 (B,3,H,W) ∈ [-1,1]   ── 네이티브 해상도 보존 ──▶  출력 (B,3,H,W)
  ├─ [256 다운샘플] → CoefficientNet → bilateral grid (B,12,depth,gh,gw)
  ├─ GuidanceNet(풀해상도) → guidance map (B,1,H,W) ∈ [0,1]   ← luma anchored
  ├─ slice_grid (5D grid_sample, trilinear) → per-pixel affine (B,12,H,W)
  ├─ apply_affine (identity-prior) → affine_out (B,3,H,W)
  └─ refine(concat[x, affine_out]) → +residual(zero-init) → out
```

**도메인 전이에 유리한 점**
- 출력 = per-pixel **3×4 affine** (RGB 채널 혼합 + bias). 이는 곧 **per-pixel white-balance/색보정**이라
  수중의 파장별 감쇠(적색 손실)·색캐스트 보정에 **구조적으로 적합**하다.
- bilateral grid 의 공간변화 affine → backscatter(산란 veil)의 공간 불균일도 일부 대응.
- **identity prior + residual zero-init** → 학습 시작점이 passthrough라 warm-start 재학습이 안정적.
- 네이티브 해상도 보존 → 다운스트림 검출 입력 해상도 손실 없음(LUNA2의 핵심 가설 그대로 승계).

**도메인 전이에서 의심스러운 점 (MARINE에서 검토)**
- `GuidanceNet`이 guidance를 **Rec.601 luma(밝기)** 에 anchor 한다(`bilateral_grid.py:215`).
  저조도에선 "밝기"가 열화 축이라 타당하지만, **수중 열화의 주축은 밝기가 아니라 색(파장별 감쇠)+haze**다.
  → v1은 luma-anchor 유지(안정적), **ablation으로 transmission/색 기반 guidance**(max-RGB 또는 학습형 감쇠맵) 비교.

### 1.2 데이터 — `LUNA2/src/data/lowlight_dataset.py`
- 모든 dataset이 `(low, high) ∈ [-1,1]` 페어 반환. `PairedImageDataset`이 base, **stem(파일명) 매칭**(`_build_pairs`).
- `DATASET_REGISTRY`로 키→클래스 매핑, `build_dataset_by_name(...)`.
- **`PairedAugment`** = 기하변환(flip/rotate/crop/perspective, 페어 동기화) + **low 전용 광학증강**
  (`_photometric_low_only`: gamma 어둡게/밝기 down/노이즈) — **저조도 전용. 수중엔 부적합(인위적 암화 금지).**

### 1.3 손실 — `LUNA2/src/losses/restoration.py`
- `CombinedRestorationLoss` = λ_L1·L1 + λ_VGG·VGG(relu3_3) + λ_SSIM·SSIM.
- **`add_term(name, fn, weight)`** 로 손실 항 **모듈 확장 가능**(Phase2 detection-aware가 이걸로 붙음).

### 1.4 학습 — `LUNA2/experiments/train.py`
- config(`configs/bilateral_base.yaml`) + 경로주입(`configs/paths.yaml`) 구동.
- Adam + cosine + AMP(모델 forward만 fp16, **손실은 fp32**), grad_clip, resume, best_psnr/ssim 체크포인트.
- 체크포인트 포맷: `{model, model_cfg, optimizer, scheduler, scaler, epoch, best_psnr, config}`.
- **Phase 구조**(승계): P1 지도복원 → P2/P3 **detection-aware joint**(frozen YOLO, ExDark mAP 타깃).

> 주의(메모리): LUNA2 P2 `config.yaml`의 λ 표기는 실제와 다를 수 있음 → **체크포인트 메타로 검증**.

---

## 2. 도메인 차이: 저조도 → 수중 탁도

| 축 | LUNA2 (저조도) | MARINE (수중 탁도) |
|---|---|---|
| 주 열화 | 저휘도, 노이즈 | **파장별 감쇠(적색↓)·색캐스트(청/녹)·backscatter haze·대비저하** |
| 페어 GT | LOL 정상광 | UIEB/EUVP **참조영상(상대적 GT, 완벽X)**, TURBID 탱크 clear |
| 증강 | 인위적 암화 OK | **암화 금지**, 대신 수중 열화모사/색지터 |
| 핵심 prior | 밝기 복원 | **색항상성(white balance)** + 대비/디헤이즈 |
| 평가 | PSNR/SSIM + ExDark mAP | PSNR/SSIM(paired) + **UIQM/UCIQE(no-ref)** + 수중검출 mAP |

---

## 3. MARINE 아키텍처 결정

1. **모델 재사용**: `BilateralLowLightNet` 그대로 사용(`cw=32, grid=16, depth=8, low_res=256`).
   → 비교군 동일성·warm-start 호환. 파라미터/지연 동일.
2. **warm-start(권장)**: LUNA2 P1 best(`runs/bilateral_phase1_l1only_guidefix/.../last.pth`)에서 **가중치 로드 후 fine-tune**.
   identity-prior 구조라 안전. 대조군으로 from-scratch도 1회.
3. **(ablation) GuidanceNet 변형**: luma-anchor → (a) max-RGB anchor, (b) 입력 6ch(RGB+역적색) 등.
   v1 본선은 **원형 유지**, 변형은 부가실험.

---

## 4. 데이터 파이프라인 설계

### 4.1 신규 어댑터 (`MARINE/src/data/underwater_dataset.py`)
LUNA2의 `PairedImageDataset`를 **상속**해 (degraded, reference) 페어를 만든다. (low=degraded, high=reference)

| 데이터셋 | 페어 구성 | 수량 | 역할 |
|---|---|---|---|
| **UIEB** | `raw-890/<name>.png` ↔ `reference-890/<name>.png` (동일 파일명, **890쌍 정합 확인됨**) | 890 | **Stage A 보조** |
| └ challenging-60 | 참조 없음 | 60 | **no-ref 정성/UIQM 평가 전용** |
| **EUVP** | `Paired/underwater_{imagenet,dark}/trainA(deg)↔trainB(good)` (동일 파일명) | **11,678쌍 확보** | **Stage A 주력** |
| └ (누락) underwater_scenes | HF/공개 미러 없음 | 2,185 | 미확보(공식 GDrive 404) |
| **TURBID** | `Milk/Chlorophyll/DeepBlue` 각 시퀀스: **최저탁도 프레임=ref**, 나머지=degraded | 172 | **탁도단계별 robustness 평가**(부차 학습) |
| **RUIE-UIQS/UCCS** | 참조 없음(real unpaired) | 3630/300 | **no-ref 평가(UIQM/UCIQE)·정성** |
| **RUIE-UTTS** | 검출 라벨 지향 | 301 | **Stage B detection-aware + 검출 mAP** |

- 구현: `UIEBDataset(PairedImageDataset)` — low_dir/high_dir만 지정하면 base의 stem매칭 그대로 동작.
- `EUVPDataset` — 서브셋별 trainA/trainB 디렉토리 매핑.
- `TurbidSequenceDataset` — 시퀀스에서 ref 프레임 지정 + 나머지를 deg로 페어링.
- `DATASET_REGISTRY`에 `uieb / euvp / turbid` 키 추가. 경로는 `configs/paths.yaml`에 `marine` 블록 추가.

### 4.2 증강 (`MARINE`: `UnderwaterAugment`)
`PairedAugment`를 상속하되 **`_photometric_low_only`를 수중용으로 교체**:
- **유지**: 기하변환(flip/rotate/crop/perspective, 페어 동기화).
- **제거**: gamma 암화·밝기 down(수중 입력을 더 망가뜨림).
- **추가(선택)**: 약한 색채널 지터(청/녹 게인 ±), 경미한 blur/backscatter veil 모사(일반화용, low에만).

### 4.3 split (`configs/`)
- UIEB 890 → `marine_uieb_split.csv` (train/val/test = **780/55/55** 권장, seed 고정·재현용 CSV로 관리, LUNA2 `make_exdark_split.py` 패턴 차용).
- EUVP(11,678쌍)는 `trainA↔trainB` 그대로 사용, 서브셋별 소량을 val로 홀드아웃. TURBID는 탁도단계 stratified eval.

---

## 5. 손실 설계

기본 `CombinedRestorationLoss` 유지 + **수중 prior 항을 `add_term`으로 추가**:

1. **L1 + VGG(relu3_3) + SSIM** — 그대로(λ=1.0/0.5/1.0 시작).
2. **(+) 색항상성 손실** `gray_world`: 향상 출력의 채널 평균을 균등화(Σ‖mean_R−mean_G‖ 등).
   수중 색캐스트 억제의 강한 prior. λ 작게(0.05~0.2) 탐색.
3. **(선택) 적색채널 보존/복원항** 또는 saturation 항 — 과보정 방지.
4. 평가전용(미분불가, 손실엔 미포함): **UIQM, UCIQE**.

> UIEB/EUVP 참조는 "상대적 GT"라 L1/SSIM 상한이 참조품질에 묶임 → no-ref 지표(UIQM/UCIQE)를 **공동 모니터링**.

---

## 6. 학습 단계 (LUNA2 phase 구조 승계)

### Stage A — 지도 복원 (필수, 1차 산출물)
- 데이터: UIEB(+EUVP) paired, `UnderwaterAugment`.
- init: **LUNA2 P1 warm-start**. 손실: L1+VGG+SSIM(+gray_world).
- 스크립트: `MARINE/experiments/train_marine.py` = LUNA2 `train.py`에 `--warmstart` + 수중 레지스트리/증강/eval만 주입(거의 재사용).
- 산출물: `MARINE/runs/marineA_*/`.
- 마일스톤 게이트: UIEB test PSNR/SSIM ↑, UIQM/UCIQE가 입력 대비 ↑, 정성적으로 색캐스트 제거.

### Stage B — 검출 인지 joint (선택, 2차) — **검출 셋업 = DUO + frozen YOLO (결정됨)**
LUNA2 Stage2/3(frozen YOLOv8n + ExDark mAP) 방법론을 수중으로 그대로 옮긴다.
- **데이터셋: DUO (Detecting Underwater Objects)** — URPC2017–2020을 중복제거·정제한 **COCO 포맷** 수중 검출 벤치마크
  (~7,782장, 4클래스: holothurian·echinus·scallop·starfish). **실제 탁한 바다** 영상이라 MARINE 도메인과 일치.
- **검출기: DUO로 사전학습한 YOLO(예: YOLOv8n)를 freeze**. LUNA2가 ExDark에 frozen YOLO를 쓴 구조의 1:1 대응.
- 절차: `MARINE(enhancer)` 출력을 frozen YOLO에 통과 → 검출 손실을 `add_term("det_aware", fn)`로 결합.
  타깃 = **DUO test mAP@0.5**. (복원항과 joint, λ는 LUNA2 P3 패턴대로 탐색.)
- **RUIE-UTTS(보유, 301장)** 는 보조 **실해역 정성/검출 eval**로만 사용(소규모·test성).

> 선택 근거: DUO는 (1) 깨끗한 COCO 라벨, (2) 실제 탁수 도메인, (3) LUNA2 검출인지 방법론을
> 변경 없이 재사용 가능 — 세 조건을 모두 만족하는 유일한 후보. URPC 원본은 라벨 중복/노이즈, RUIE-UTTS는
> 학습용 라벨 부족으로 단독 부적합. (DUO 다운로드는 M4 진입 시 수행.)

---

## 7. 평가 프로토콜 (`MARINE/experiments/eval_*`)
1. **Paired**: UIEB test / EUVP test → PSNR, SSIM (LUNA2 `eval_restoration.py` 재사용).
2. **No-reference**: RUIE-UIQS/UCCS, UIEB-challenging → **UIQM, UCIQE**(신규 `src/utils/uw_metrics.py`).
3. **Robustness**: TURBID 탁도단계별 지표 곡선.
4. **Downstream(선택)**: 수중 검출 mAP(Stage B).

---

## 8. 폴더 구조 (`MARINE/`) — ✅ M0/M1 구현됨
> **중요**: MARINE·LUNA2 둘 다 `src/` 패키지라 충돌 → MARINE 코드 패키지는 **`marine/`** 로 둔다.
> LUNA2 `src` 는 sys.path 에 LUNA2 루트를 추가해 그대로 import(복사 금지). `marine/__init__.py` 가 경로 주입.
```
MARINE/
├─ DESIGN.md
├─ datasets/                       # UIEB, TURBID, RUIE_repo, EUVP(imagenet+dark 11,678쌍) / DUO는 M4때
├─ marine/                         # ← MARINE 자체 패키지 (src 충돌 회피)
│  ├─ __init__.py                  #   LUNA2 루트를 sys.path 주입
│  └─ data/underwater_dataset.py   # ✅ UIEB/EUVP 어댑터 + UnderwaterAugment(암화 비활성)
│     (예정) losses/underwater.py  # gray_world (M2) , utils/uw_metrics.py UIQM/UCIQE (M2)
├─ configs/
│  ├─ paths_marine.yaml            # ✅ 수중 데이터 경로 + LUNA2 warmstart 참조
│  ├─ marine_stageA.yaml           # ✅ Stage A 학습 config (warmstart, lr 1e-4, 30ep)
│  └─ uieb_split.csv               # ✅ UIEB 780/55/55 (seed42, make_uieb_split.py 생성)
├─ experiments/
│  ├─ train_marine.py              # ✅ Stage A 학습 (LUNA2 model/loss/metric import 재사용 + warmstart)
│  ├─ make_uieb_split.py           # ✅ split CSV 생성기
│  └─ (예정) eval_restoration_uw.py / eval_noref_uw.py  # M2
└─ runs/                           # 산출물
```
**재사용 확정**: `train_marine.py` 가 LUNA2 `src.models.bilateral_grid`(모델),
`src.losses.restoration`(손실), `src.utils.metrics`(PSNR/SSIM)를 **import** 하고,
데이터/증강만 `marine.data.underwater_dataset` 로 교체. warm-start 키 **완전 일치** 확인됨.

---

## 9. 마일스톤 / 작업 순서
1. **M0 환경** ✅완료: `paths_marine.yaml`+`uieb_split.csv`(780/55/55)+`underwater_dataset.py`(UIEB+EUVP) + 스모크 통과.
2. **M1 Stage A v1** ✅완료: warm-start + EUVP+UIEB 12,458쌍 지도복원 30ep(879K params, RTX4060 ~210s/ep).
   - **UIEB test(55) 베이스라인**: 무처리 입력 18.03dB/0.798 · 저조도 warm-start 시작점 15.27dB(입력보다 **나쁨**)
     → **MARINE Stage A 20.30dB / 0.860** (best=`runs/marineA_warmstart/checkpoints/best_psnr.pth`, ep27).
   - 의미: 저조도 모델이 무처리보다 나쁜 점이 **도메인 시프트(저조도≠수중)를 실증** → 재학습 정당화.
     입력 대비 **+2.3dB**, warm-start 시작점 대비 **+5.0dB**. 정성적으로 청록 색캐스트·haze 제거 확인(samples/epoch_029.png).
3. **M2 손실/평가 강화**: gray_world 손실 추가, UIQM/UCIQE eval, TURBID robustness 곡선.
4. **M3 ablation**: from-scratch vs warm-start, guidance(luma vs 색) 비교.
5. **M4(선택) Stage B**: **DUO** 다운로드 → frozen YOLO 학습 → detection-aware joint → DUO mAP.

---

## 10. 데이터/결정 현황
- **EUVP** ✅ 확보(공식 GDrive 404 → **HF 미러 조립**): `underwater_imagenet` 6,128쌍(iamvastava zip) +
  `underwater_dark` 5,550쌍(Ken1053 parquet→이미지 변환) = **11,678 train 쌍**. `underwater_scenes`(2,185)만 미확보(공개 미러 없음, 영향 경미).
- **Stage B** ✅ 결정: **DUO + frozen YOLO**(§6). 다운로드는 M4 진입 시.
- LUNA2 코드 참조: **import 재사용**(복사 금지)으로 확정.
- LUNA2 코드 참조 방식: import 재사용(권장) vs 일부 포크.
