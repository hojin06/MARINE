"""MARINE 패키지 — LUNA2 코드를 import 재사용하기 위한 경로 설정 포함.

MARINE 과 LUNA2 는 둘 다 ``src/`` 패키지를 갖기 때문에 이름 충돌을 피하려고
MARINE 자체 코드는 ``marine/`` 패키지에 두고, LUNA2 의 ``src`` 는 sys.path 에
LUNA2 루트를 추가해 그대로 import 한다 (코드 복사 금지 원칙).
"""
from __future__ import annotations

import sys
from pathlib import Path

# marine/__init__.py → parents[1]=MARINE, parents[2]=LUNA_paperWORKS
_LUNA2_ROOT = Path(__file__).resolve().parents[2] / "LUNA2"
if _LUNA2_ROOT.is_dir() and str(_LUNA2_ROOT) not in sys.path:
    sys.path.insert(0, str(_LUNA2_ROOT))

LUNA2_ROOT = _LUNA2_ROOT
