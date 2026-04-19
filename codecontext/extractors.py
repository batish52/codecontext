from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

from .utils import clamp_text


@dataclass(slots=True)
class ExtractedContent:
    tiny_summary: str
    detailed_summary: str
    symbol_summary: str
    snippets: list[dict]
    symbols: list[dict]
    todos: list[str]


IMPORT_RE = re.compile(r"^(?:from\s+\S+\s+import\s+.+|import\s+.+)$", re.MULTILINE)
TODO_RE = re.compile(r".*(?:TODO|FIXME|HACK|BUG).*", re.IGNORECASE)
HEADING_RE = re.compile(r"^(#+)\s+(.+)$", re.MULTILINE)
BULLET_RE = re.compile(r"^\s*[-*+]\s+(.+)$", re.MULTILINE)
JSON_KEY_RE = re.compile(r'"([^"]+)"\s*:')


def chunk_by_lines(text: str, chunk_size: int = 80) -> list[dict]:
    lines = text.splitlines()
    chunks = []
    for start in range(0, len(lines), chunk_size):
        end = min(len(lines), start + chunk_size)
        body = "\n".join(lines[start:end])
        chunks.append({"start_line": start + 1, "end_line": end, "text": body})
    return chunks


def _walk_python_symbols(nodes, prefix: str = "") -> list[dict]:
    symbols: list[dict] = []
    for node in nodes:
        if isinstance(node, ast.ClassDef):
            full_name = f"{prefix}{node.name}" if not prefix else f"{prefix}.{node.name}"
            symbols.append(
                {
                    "name": full_name,
                    "type": node.__class__.__name__,
                    "line": getattr(node, "lineno", None),
                    "end_line": getattr(node, "end_lineno", getattr(node, "lineno", None)),
                    "doc": clamp_text(ast.get_docstring(node) or "", 180),
                }
            )
            symbols.extend(_walk_python_symbols(node.body, prefix=full_name))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            full_name = f"{prefix}{node.name}" if not prefix else f"{prefix}.{node.name}"
            symbols.append(
                {
                    "name": full_name,
                    "type": node.__class__.__name__,
                    "line": getattr(node, "lineno", None),
                    "end_line": getattr(node, "end_lineno", getattr(node, "lineno", None)),
                    "doc": clamp_text(ast.get_docstring(node) or "", 180),
                }
            )
    return symbols


def extract_python(text: str) -> tuple[list[dict], list[str], str]:
    imports = IMPORT_RE.findall(text)
    symbols: list[dict] = []
    try:
        tree = ast.parse(text)
        symbols = _walk_python_symbols(tree.body)
    except SyntaxError:
        pass
    return symbols, imports, ", ".join(f"{s['type']}:{s['name']}@{s['line']}" for s in symbols[:20])


def extract_markdown(text: str) -> tuple[list[str], list[str]]:
    headings = [m[1] for m in HEADING_RE.findall(text)]
    bullets = BULLET_RE.findall(text)
    return headings[:20], bullets[:20]


def extract_structured(text: str) -> list[str]:
    return JSON_KEY_RE.findall(text)[:40]


def summarize_text(path: Path, text: str) -> ExtractedContent:
    suffix = path.suffix.lower()
    todos = TODO_RE.findall(text)
    snippets = chunk_by_lines(text)
    symbols: list[dict] = []
    detail_bits: list[str] = []
    symbol_summary = ""

    if suffix == ".py":
        symbols, imports, symbol_summary = extract_python(text)
        if imports:
            detail_bits.append("imports: " + "; ".join(imports[:12]))
    elif suffix in {".md", ".rst"}:
        headings, bullets = extract_markdown(text)
        if headings:
            detail_bits.append("headings: " + " | ".join(headings))
        if bullets:
            detail_bits.append("bullets: " + " | ".join(bullets[:8]))
    elif suffix in {".json", ".yaml", ".yml", ".toml"}:
        keys = extract_structured(text)
        if keys:
            detail_bits.append("keys: " + ", ".join(keys))
    elif suffix in {".log", ".txt"}:
        warn_lines = [line for line in text.splitlines() if any(k in line.lower() for k in ("warn", "error", "exception", "fail"))]
        if warn_lines:
            detail_bits.append("log signals: " + " | ".join(warn_lines[:10]))

    if todos:
        detail_bits.append("todo/fixme: " + " | ".join(todos[:10]))

    first_nonempty = next((line.strip() for line in text.splitlines() if line.strip()), "")
    tiny = clamp_text(first_nonempty or f"{path.name} ({suffix or 'no extension'})", 160)
    detailed = clamp_text(" || ".join(detail_bits) or clamp_text(text[:800].replace("\n", " "), 500), 1200)
    symbol_summary = symbol_summary or (", ".join(s["name"] for s in symbols[:20]))

    return ExtractedContent(
        tiny_summary=tiny,
        detailed_summary=detailed,
        symbol_summary=clamp_text(symbol_summary, 1400),
        snippets=snippets,
        symbols=symbols,
        todos=todos[:20],
    )
