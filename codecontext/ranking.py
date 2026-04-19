from __future__ import annotations

CODE_HINTS = {
    "function", "class", "method", "symbol", "module", "import", "call", "patch",
    "edit", "bug", "fix", "refactor", "code", "logic", "retrieval", "embedding", "rerank",
    "context", "budget", "index", "search", "scan", "chunk", "file", "config",
}

DOC_HINTS = {"readme", "docs", "documentation", "explain", "overview", "architecture"}
ENTRYPOINT_FILES = {"main.py", "app.py", "manage.py", "run.py"}
MEMORY_PATH_MARKERS = {"memory/", "memory\\", "memory.md"}


def query_intent_terms(query: str) -> set[str]:
    return {part.lower() for part in query.replace("/", " ").replace("-", " ").split()}


def filetype_intent_boost(path: str, query: str) -> float:
    terms = query_intent_terms(query)
    lower = path.lower()
    score = 0.0
    if any(t in CODE_HINTS for t in terms):
        if lower.endswith((".py", ".ts", ".tsx", ".js", ".jsx", ".json", ".toml", ".yaml", ".yml", ".ps1")):
            score += 2.5
    if any(t in DOC_HINTS for t in terms):
        if lower.endswith((".md", ".rst")):
            score += 1.5
    if "test" in terms and ("test" in lower or "/tests/" in lower):
        score += 2.0
    if "config" in terms and any(k in lower for k in ("config", ".json", ".yaml", ".toml", ".env")):
        score += 2.0
    return score


def entrypoint_penalty(path: str, query: str) -> float:
    terms = query_intent_terms(query)
    lower = path.lower().replace('\\', '/')
    name = lower.rsplit('/', 1)[-1]
    if name not in ENTRYPOINT_FILES:
        return 0.0
    if any(t in terms for t in {"entrypoint", "startup", "boot", "main", "run", "launch"}):
        return 0.0
    return 1.0


def wants_exact_symbol_body(query: str) -> bool:
    terms = query_intent_terms(query)
    return bool({"exact", "body", "function", "class", "method", "defined", "definition", "helper"} & terms)


def memory_file_penalty(path: str, query: str) -> float:
    lower = path.lower().replace('\\', '/')
    if not any(marker in lower for marker in MEMORY_PATH_MARKERS):
        return 0.0
    terms = query_intent_terms(query)
    if any(t in terms for t in {"memory", "history", "log", "notes", "journal"}):
        return 0.0
    return 1.0
