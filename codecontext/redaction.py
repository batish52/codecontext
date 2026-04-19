from __future__ import annotations

import re
from pathlib import Path

from .config import SECRET_PATTERNS


SECRET_VALUE_RE = re.compile(
    r"(?im)\b([A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD|PASSWD)[A-Z0-9_]*)\b\s*[:=]\s*([\"']?)([^\n\"']{6,})(\2)"
)
PEM_RE = re.compile(r"-----BEGIN [A-Z ]+-----.*?-----END [A-Z ]+-----", re.DOTALL)


def looks_secret_path(path: Path) -> bool:
    lower = path.as_posix().lower()
    return any(part in lower for part in (".env", ".pem", ".key", "id_rsa", "secrets"))


def redact_text(text: str) -> str:
    text = PEM_RE.sub("[REDACTED_PEM_BLOCK]", text)
    text = SECRET_VALUE_RE.sub(lambda m: f"{m.group(1)}={m.group(2)}[REDACTED]{m.group(4)}", text)
    for marker in SECRET_PATTERNS:
        text = re.sub(fr"(?i){re.escape(marker)}", f"{marker[:2]}[REDACTED_MARKER]", text)
    return text
