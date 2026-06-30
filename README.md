# MARINE

**M**odified version of lun**A** **R**econstructed **I**n order to **N**avigate sea **E**nvironment

저조도 영상 전처리기 **LUNA2** 를 **탁도 높은 수중환경**용으로 **재학습(도메인 전이)** 한 프로젝트.
수중 영상의 파장별 감쇠(적색 손실)·청록 색캐스트·산란(haze)을 제거해, 다운스트림 인식(검출 등)에
유리한 네이티브 해상도 전처리 출력을 만든다.

> LUNA2(저조도) → MARINE(수중). 모델 **아키텍처는 재사용**하고, 데이터 도메인·증강·손실 prior·평가지표를 수중용으로 교체한다.

---

## 핵심 결과 (Stage A 베이스라인)

UIEB test(55쌍, 256) PSNR/SSIM:

| 모델 | PSNR (dB) | SSIM |
|---|---|---|
| 무처리 입력 (degraded) | 18.03 | 0.798 |
| LUNA2 저조도 가중치 (전이 전) | 15.27 | ~0.76 |
| **MARINE Stage A** | **20.30** | **0.860** |

> 저조도 가중치를 그대로 적용하면 무처리보다도 **나쁘다(15.3 < 18.0)** — 저조도≠수중 **도메인 시프트**의 직접 증거이며,
> 재학습(15.3 → 20.3dB)이 그 격차를 메운다. 정성적으로 청록 색캐스트·haze 제거가 뚜렷하다.

---

## 아키텍처

LUNA2 `BilateralLowLightNet` (HDRNet 계열) 을 그대로 사용 (**879K 파라미터**, 네이티브 해상도 보존):

```
입력 (B,3,H,W) ∈ [-1,1]
  ├─ [256 다운샘플] → CoefficientNet → bilateral grid (B,12,depth,gh,gw)
  ├─ GuidanceNet(풀해상도) → guidance map (B,1,H,W)
  ├─ slice(trilinear) → per-pixel 3×4 affine
  ├─ apply affine (identity-prior)
  └─ refine(residual) → 출력 (B,3,H,W)   # 입력과 동일 해상도
```

per-pixel affine = per-pixel 화이트밸런스/색보정 → 수중 색캐스트 보정에 구조적으로 적합.
LUNA2 Phase1 가중치로 **warm-start** 후 fine-tune (identity-prior 구조라 안정적).

---

## 데이터셋

| 데이터셋 | 용도 | 비고 |
|---|---|---|
| **UIEB** (890쌍) | paired 학습/평가 | raw-890 ↔ reference-890 |
| **EUVP** (11,678쌍) | paired 학습 주력 | underwater_imagenet + dark (HF 미러 조립) |
| **TURBID** (172) | 탁도단계 robustness 평가 | Milk/Chlorophyll/DeepBlue/TURBID3D |
| **RUIE** | no-ref / 검출 평가 | UIQS/UCCS/UTTS |
| **DUO** (예정) | Stage B 검출 | M4에서 취득 |

데이터는 **레포에 포함하지 않는다**(대용량 + UIEB/EUVP **재배포 금지** 라이선스).
취득 방법은 [`datasets/README.md`](datasets/README.md) 참조 (EUVP 공식 GDrive 사망 → HF 미러 조립법 포함).

---

## 설치

```bash
pip install -r requirements.txt
```

> **LUNA2 의존성**: MARINE 은 LUNA2 의 모델/손실/평가 코드를 **import 재사용**한다(코드 복사 금지 원칙).
> LUNA2 저장소를 MARINE 과 **형제 디렉토리**(`../LUNA2`)에 두어야 한다. `marine/__init__.py` 가 경로를 주입한다.
> ```
> <parent>/
>   ├─ LUNA2/      # 원본(모델/손실/메트릭 제공)
>   └─ MARINE/     # 본 저장소
> ```

---

## 사용법

```bash
# 1) UIEB split 생성 (재현용 CSV)
python experiments/make_uieb_split.py

# 2) Stage A 학습 (LUNA2 Phase1 warm-start)
python experiments/train_marine.py --config configs/marine_stageA.yaml
#   재시작: 같은 명령 재실행 시 runs/.../last.pth 자동 감지하여 이어서 학습
#   스모크: --smoke
#   warm-start 끄기: --no_warmstart
```

산출물: `runs/{experiment_name}/` (checkpoints / logs / samples).

---

## 저장소 구조

```
MARINE/
├─ README.md            # 본 문서
├─ DESIGN.md            # 구조분석 + 재학습 파이프라인 설계
├─ ROADMAP.md           # 작업 가이드라인(M2~M4) + 예상시간 + 자율실행 정책
├─ requirements.txt
├─ marine/              # MARINE 자체 패키지 (LUNA2 src 와 충돌 회피용 별도 이름)
│  ├─ __init__.py       #   LUNA2 루트를 sys.path 주입
│  ├─ data/underwater_dataset.py   # UIEB/EUVP 어댑터 + UnderwaterAugment(암화 비활성)
│  └─ utils/uw_metrics.py          # UIQM / UCIQE (no-reference 지표)
├─ configs/
│  ├─ paths_marine.yaml            # 데이터 경로 + LUNA2 warmstart 참조
│  ├─ marine_stageA.yaml           # Stage A 학습 config
│  └─ uieb_split.csv               # UIEB 780/55/55 (seed 42)
├─ experiments/
│  ├─ make_uieb_split.py
│  └─ train_marine.py              # Stage A 학습 (LUNA2 import 재사용 + warm-start)
└─ datasets/  (git 제외; README 만 포함)
```

---

## 진행 현황 / 로드맵

- ✅ **P0–P2**: 폴더·데이터 확보·설계
- ✅ **M0**: 환경/어댑터 (스모크 통과)
- ✅ **M1**: Stage A 학습 (UIEB test 20.30dB / 0.860)
- 🔜 **M2**: 손실·평가 강화 (UIQM/UCIQE, gray-world 손실, TURBID robustness) — *~3.5h*
- 🔜 **M3**: Ablation (warm-start vs scratch, guidance) — *~4h*
- 🔜 **M4**: Stage B 검출 인지 joint (DUO + frozen YOLO) — *~1–2일*

상세·예상시간·자율실행 정책은 [`ROADMAP.md`](ROADMAP.md), 설계 근거는 [`DESIGN.md`](DESIGN.md).

---

## 라이선스 / 인용

- 코드: (TBD)
- 데이터셋은 각 원저작권자 라이선스를 따른다(UIEB/EUVP 등 학술·비상업 한정, 재배포 금지).
- 기반: LUNA2, HDRNet(Gharbi et al., SIGGRAPH 2017), UIEB(Li et al., TIP 2019), EUVP(Islam et al., RA-L 2020).
