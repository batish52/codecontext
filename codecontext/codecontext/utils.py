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
    # Read only the bytes we need. The previous implementation did
    # `path.read_bytes()[:max_bytes]` which loaded the entire file into
    # memory before slicing — so passing max_bytes=1024 against a 10GB
    # file still allocated 10GB. Use an open+read(N) stream instead so
    # max_bytes is a real memory bound.
    with path.open("rb") as f:
        data = f.read() if max_bytes is None else f.read(max_bytes)
    return data.decode("utf-8", errors="replace")


_WORD_CHAR_RE = re.compile(r"[A-Za-z0-9]")
# Boundary rule: a boundary is anything that is not an ASCII letter or
# digit. That means `_`, `.`, `/`, `-`, whitespace and punctuation all
# count as boundaries. This is intentionally more liberal than Python's
# `\b` (which treats `_` as a word char) because our use sites are a mix
# of natural language queries and path/identifier strings where people
# expect "test" to match "test_latest.py" and "config" to match
# "src/config/models.py".
_BOUNDARY_RE_CACHE: dict[str, re.Pattern[str]] = {}


def _word_pattern(word: str) -> re.Pattern[str]:
    cached = _BOUNDARY_RE_CACHE.get(word)
    if cached is not None:
        return cached
    pat = re.compile(
        r"(?:^|(?<=[^A-Za-z0-9]))" + re.escape(word) + r"(?=[^A-Za-z0-9]|$)"
    )
    _BOUNDARY_RE_CACHE[word] = pat
    return pat


def contains_word(text: str, word: str) -> bool:
    """Word-boundary-aware substring check.

    `word in text` matches false positives like "because" for "cause",
    "dialogs" for "logs", "latest_utils.py" for "test". Use this helper
    whenever you want to ask "does the text contain this *word*" rather
    than "does the text contain this substring".

    Boundaries are defined by anything that is not an ASCII letter or
    digit — so `_`, `.`, `/`, `-`, whitespace, and punctuation all count
    as boundaries. This makes the helper usable for both natural-language
    queries and file-path/identifier matching.

    Matching is case-sensitive; callers that want case-insensitive
    behaviour should lowercase both sides before calling.
    """
    if not word or not text:
        return False
    if not _WORD_CHAR_RE.search(word):
        return word in text
    return bool(_word_pattern(word).search(text))


def any_word(text: str, words) -> bool:
    """Shorthand for `any(contains_word(text, w) for w in words)`."""
    return any(contains_word(text, w) for w in words)


def approx_tokens(text: str) -> int:
    return math.ceil(len(text) / 4)


def json_dumps(data: object) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clamp_text(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."
