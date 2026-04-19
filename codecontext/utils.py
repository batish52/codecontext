from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def is_binary_bytes(data: bytes) -> bool:
    if not data:
        return False
    if b"\x00" in data:
        return True
    text = bytes(range(32, 127)) + b"\n\r\t\b\f"
    non_text = sum(byte not in text for byte in data[:4096])
    return (non_text / max(1, min(len(data), 4096))) > 0.30


def read_text_safely(path: Path, max_bytes: int | None = None) -> str:
    data = path.read_bytes() if max_bytes is None else path.read_bytes()[:max_bytes]
    return data.decode("utf-8", errors="replace")


def approx_tokens(text: str) -> int:
    return math.ceil(len(text) / 4)


def json_dumps(data: object) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clamp_text(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."
