# MARINE — 실행 가이드라인 (ROADMAP)

M1(Stage A 베이스라인, UIEB test 20.30dB/0.860) 완료 이후의 작업 순서·예상시간·합격기준·자율실행 정책.
설계 근거는 [DESIGN.md](DESIGN.md). 시간은 **코딩(상호작용)** 과 **GPU(RTX4060, ~210s/epoch)** 로 분리.

---

## 자율 실행 정책 (Autonomous Execution Policy)
- **자동 진행(질문 없이)**: M2 전체, M3.1(scratch ablation). 기존 인프라 재사용·저위험.
- **각 마일스톤 게이트마다 보고**: 수치표 + 샘플/곡선 + 다음 단계 예고.
- **중단·확인 요청**: M3.2(guidance 변형 — import한 LUNA2 코드 수정 필요), M4(DUO 외부 다운로드 위험 + 검출 설계 大). M4.1 DUO 다운로드를 자동 시도하되 EUVP처럼 막히면 즉시 보고·대기.
- 모든 GPU 학습은 background, resume 지원(`save_every=5`, best는 갱신마다). 실패 시 1회 자동 재시도 후 보고.

---

## M2 — 손실·평가 강화  (벽시계 ~3.5h: 코딩 ~2.5h ∥ GPU ~2.3h)
수중 전용 평가지표와 색항상성 손실을 추가해 "참조 없는 실제 수중 화질"까지 측정·개선.

| # | 작업 | 산출물 | 코딩 | GPU |
|---|---|---|---|---|
| M2.1 | **UIQM·UCIQE** no-ref 지표 구현 | `marine/utils/uw_metrics.py` | 40m | – |
| M2.2 | no-ref 평가 스크립트 + 실행(RUIE-UIQS/UCCS, UIEB-challenging: 입력 vs MARINE) | `experiments/eval_noref_uw.py` | 30m | ~10m |
| M2.3 | paired **네이티브 해상도** 평가(UIEB test, EUVP) PSNR/SSIM | `experiments/eval_restoration_uw.py` | 25m | ~10m |
| M2.4 | **gray-world 색항상성 손실** + `add_term` 결합 | `marine/losses/underwater.py` | 30m | – |
| M2.5 | gray-world 포함 재학습(`marineA_grayworld`) + 베이스라인 비교 | run 디렉토리 | 10m | ~1.8h |
| M2.6 | **TURBID 탁도단계별 robustness** 곡선(UIQM/UCIQE vs 탁도) | `experiments/eval_turbid_curve.py` | 30m | ~5m |

**게이트(M2 합격기준)**: ① no-ref/paired eval가 입력 대비 UIQM·UCIQE·PSNR 모두 ↑ ② gray-world가 색캐스트 추가 억제(UIQM↑, 과보정 없음) ③ 네 데이터셋(UIEB/EUVP/RUIE/TURBID) 모두에서 수치 산출.

---

## M3 — Ablation  (벽시계 ~4h: 코딩 ~1h ∥ GPU ~3.6h)
| # | 작업 | 코딩 | GPU |
|---|---|---|---|
| M3.1 | **warm-start vs from-scratch** (`marineA_scratch`, --no_warmstart) | 5m | ~1.8h |
| M3.2 | **guidance ablation**: luma-anchor vs 색/max-RGB anchor (GuidanceNet 변형 서브클래스) | 1h | ~1.8h |

**게이트**: warm-start 우위(수렴속도·최종 수치) 정량화 / guidance 변형이 수중에서 luma 대비 동등 이상인지 판정. → 본선 설정 확정.

---

## M4 — Stage B 검출 인지 joint  (벽시계 ~1–2일: 외부 다운로드 위험 포함)
| # | 작업 | 코딩 | GPU |
|---|---|---|---|
| M4.1 | **DUO** 다운로드·정리(COCO→YOLO 변환) | 30m | – |
| M4.2 | DUO로 YOLO 학습 후 **freeze** | 30m | 1–3h |
| M4.3 | enhancer→frozen YOLO **detection-aware joint**(`add_term`), DUO mAP 타깃 | 3–4h | 수시간 |

**게이트**: 향상기 통과 시 DUO mAP@0.5가 무처리/복원전용 대비 ↑. (LUNA2 Stage3 방법론 대응.)
**위험**: DUO 공식 배포(GDrive/Baidu) 접근성. 막히면 EUVP식 미러 탐색 후 보고·대기.

---

## 총 예상 시간 요약
- **M2 완료**: ~3.5h (오늘 내)
- **M2+M3 완료**: ~+4h → 누적 ~1일(벽시계, GPU 학습 대기 포함)
- **M4까지**: +1–2일 (DUO 접근성에 좌우)

> 진행 중 우선순위·중단·하이퍼파라미터 변경은 언제든 한마디로 가능. 각 게이트에서 결과 보고 후 다음으로 넘어간다.
