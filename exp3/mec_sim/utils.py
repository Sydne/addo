from __future__ import annotations

import json
import math
import os
from typing import Dict


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, data: Dict) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def clipped_sqrt(x: float) -> float:
    return math.sqrt(max(0.0, x))
