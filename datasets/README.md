# MARINE Datasets

MARINE(Modified version of lunA Reconstructed In order to Navigate sea Environment)은
LUNA2(저조도 이미지 전처리기)를 **탁도 높은 수중환경**에 적용하기 위해
수중 데이터셋으로 재학습/파인튜닝하는 것을 목표로 한다. 아래는 다운로드/검증 결과 요약이다.

마지막 검증: 2026-06-25

## 상태 요약

| 데이터셋 | 상태 | 위치 | 규모(검증값) | 비고 |
|---|---|---|---|---|
| **UIEB** | ✅ 완료 | `UIEB/` | raw 890 + reference 890 + challenging 60 | Google Drive(gdown). 실제 RAR5(비번 보호) |
| **TURBID** | ✅ 완료 | `TURBID/` | 172장 (Milk 20 / Chlorophyll 44 / DeepBlue 22 / TURBID3D 86) | 직접 HTTP 다운로드 |
| **RUIE** | ✅ 완료 | `RUIE_repo/` | UIQS 3630 + UTTS 301 + UCCS 300 | GitHub git clone (repo에 커밋된 공개분) |
| **EUVP** | ✅ 부분완료 | `EUVP/Paired/` | imagenet 6,128쌍 + dark 5,550쌍 = **11,678 train 쌍** | 공식 GDrive 404 → **HF 미러 조립**. scenes(2,185)만 미확보 |
| **UTIEB** | ⏸ 보류 | — | 1086장 | Baidu 전용 → 자동 다운로드 불가, 사용자 결정으로 제외 |

## 상세

### UIEB (Underwater Image Enhancement Benchmark)
- 출처: https://li-chongyi.github.io/proj_benchmark.html (Google Drive)
- **주의**: 배포 파일은 확장자가 `.zip`이지만 실제로는 **RAR5 + 비밀번호** 아카이브다.
  추출 비밀번호(프로젝트 페이지의 숫자 코드)는 다음과 같다:
  - `raw-890.zip`  → pw `1234567`  → `raw-890/`  (890장)
  - `reference-890.zip` → pw `8901234` → `reference-890/` (890장)
  - `challenging-60.zip` → pw `5678901` → `challenging-60/` (60장)
- 추출: `7z x -p<PW> <file>.zip` (7-Zip 사용; bsdtar/unzip 불가)

### TURBID (Turbid Underwater Image, Duarte 2016)
- 출처: http://amandaduarte.com.br/turbid/ (직접 zip 링크, HTTP만 / HTTPS 인증서 불일치)
- 서브셋: Milk / Chlorophyll / DeepBlue / TURBID3D — 각 zip을 동명 폴더에 추출 완료.

### RUIE (Real-world Underwater Image Enhancement Benchmark)
- 출처: https://github.com/dlut-dimt/Realworld-Underwater-Image-Enhancement-RUIE-Benchmark
- `git clone --depth 1` 로 받음. 하위:
  - `UIQS/` (A~E 가시성 등급, 3630장)
  - `UTTS/` (pic_A~E, 301장)
  - `UCCS/` (blue/green/... 색조별, 300장)
- 폴더명은 잠금으로 `RUIE_repo` 유지(내부에 `.git` 포함).

### EUVP — HF 미러로 조립 확보 ✅
- 공식 GDrive 폴더 `1ZEql33CajGfHHzPe1vFxUFCMcP0YbZb3` 는 **HTTP 404**(제거/비공개, eval_data용이었음).
  → 단일 공개 미러가 없어 **HuggingFace 두 미러를 조립**:
  - `underwater_imagenet` : [iamvastava/underwater_image_enhancement](https://huggingface.co/datasets/iamvastava/underwater_image_enhancement)
    의 `underwater_imagenet (1).zip`(449MB) → `Paired/underwater_imagenet/{trainA,trainB,test}` (6,128쌍 + test 1,813).
  - `underwater_dark` : [Ken1053/EUVP](https://huggingface.co/datasets/Ken1053/EUVP) 의 parquet(input_image/edited_image)
    → 스크립트로 디코드하여 `Paired/underwater_dark/{trainA,trainB}` (5,550쌍). trainA=열화(input), trainB=향상(edited).
- 구조: `EUVP/Paired/underwater_{imagenet,dark}/{trainA,trainB}` — trainA↔trainB **동일 파일명** 페어.
- **미확보**: `underwater_scenes`(2,185쌍). 공개 미러 없음(공식 GDrive 외 출처 부재). 학습 영향 경미.
- 재현:
  ```bash
  # imagenet
  curl -L -o ui.zip "https://huggingface.co/datasets/iamvastava/underwater_image_enhancement/resolve/main/underwater_imagenet%20(1).zip"
  7z x ui.zip   # → underwater_imagenet/{trainA,trainB,test}
  # dark (parquet → images): pyarrow 로 input_image→trainA, edited_image→trainB 디코드
  curl -L -o dark.parquet "https://huggingface.co/datasets/Ken1053/EUVP/resolve/main/data/train-00000-of-00001.parquet"
  ```

### UTIEB — 보류
- 출처: https://github.com/Peng-Lin-Dmu/UTIEB → **Baidu 넷디스크 전용**
  (https://pan.baidu.com/s/1NRgxTfFtPNMmeQ6PGcYGtQ). 해외 IP 자동 다운로드 불가.
- 사용자 결정으로 현재 단계에서 제외. 필요시 저자(plin@dlmu.edu.cn) 문의 또는 수동 다운로드.

## 재현 명령(참고)

```bash
# UIEB (Google Drive → 실제 RAR5, 7-Zip로 비번 추출)
python -m gdown 12W_kkblc2Vryb9zHQ6BfGQ_NKUfXYk13 -O UIEB/raw-890.zip
python -m gdown 1cA-8CzajnVEL4feBRKdBxjEe6hwql6Z7 -O UIEB/reference-890.zip
python -m gdown 1Ew_r83nXzVk0hlkfuomWqsAIxuq6kaN4 -O UIEB/challenging-60.zip
# 7z x -p1234567 raw-890.zip ; 7z x -p8901234 reference-890.zip ; 7z x -p5678901 challenging-60.zip

# TURBID (직접 HTTP)
for f in Milk Chlorophyll DeepBlue TURBID3D; do
  curl -o TURBID/$f.zip http://amandaduarte.com.br/turbid/turbid/$f.zip; done

# RUIE (git)
git clone --depth 1 https://github.com/dlut-dimt/Realworld-Underwater-Image-Enhancement-RUIE-Benchmark.git RUIE_repo
```
