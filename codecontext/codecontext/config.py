from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .llm_client import LLMClientConfig


DEFAULT_INCLUDE = ["**/*"]
DEFAULT_EXCLUDE = [
    ".git/**",
    ".venv/**",
    "venv/**",
    "node_modules/**",
    "dist/**",
    "build/**",
    "__pycache__/**",
    ".pytest_cache/**",
    ".mypy_cache/**",
    ".codecontext/**",
    "cache/**",
    "outputs/**",
    "logs/**",
    "tmp_*",
    "tmp_*.json",
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.webp",
    "*.pdf",
    "*.zip",
    "*.exe",
    "*.dll",
    "*.bin",
    "*.pyc",
]

IMPORTANT_FILE_NAMES = {
    "readme.md": 10,
    "package.json": 10,
    "pyproject.toml": 10,
    "requirements.txt": 9,
    "dockerfile": 8,
    ".env.example": 8,
    "codecontext.toml": 9,
    "main.py": 8,
    "app.py": 8,
    "manage.py": 8,
}

IMPORTANT_EXTENSIONS = {
    ".py": 6,
    ".ts": 6,
    ".tsx": 6,
    ".js": 5,
    ".jsx": 5,
    ".json": 4,
    ".yaml": 4,
    ".yml": 4,
    ".toml": 4,
    ".md": 4,
    ".sql": 4,
    ".sh": 3,
    ".ps1": 3,
}

SECRET_PATTERNS = [
    "api_key",
    "apikey",
    "secret",
    "token",
    "password",
    "passwd",
    "private_key",
    "authorization",
    "bearer ",
    "-----begin",
]


@dataclass(slots=True)
class AppConfig:
    root: Path
    data_dir: Path = field(init=False)
    db_path: Path = field(init=False)
    backup_dir: Path = field(init=False)
    manifest_path: Path = field(init=False)
    metrics_path: Path = field(init=False)
    include: list[str] = field(default_factory=lambda: list(DEFAULT_INCLUDE))
    exclude: list[str] = field(default_factory=lambda: list(DEFAULT_EXCLUDE))
    max_file_bytes: int = 512_000
    default_context_budget_tokens: int = 4000
    do_not_resend_window: int = 50
    watcher_poll_seconds: float = 2.0
    enable_embeddings: bool = True
    # Was `False` and silently ignored. Default is now `True` because
    # the previous code path ran semantic rerank unconditionally when
    # embeddings were available — flipping this to True preserves the
    # observed pre-fix behaviour. Users who explicitly want pure-BM25
    # ranking can now set this to False and have it actually take
    # effect.
    enable_reranker: bool = True
    max_chunk_candidates: int = 60
    max_chunks_per_file: int = 3
    min_combined_score: float = 3.5
    adjacent_chunk_merge_gap: int = 8
    max_merged_chunk_lines: int = 40
    symbol_context_radius: int = 12
    novelty_overlap_penalty: float = 3.0
    context_snippet_char_limit: int = 1200
    entrypoint_suppression_penalty: float = 8.0
    exact_symbol_body_bonus: float = 10.0
    index_schema_version: str = "2026-03-12-symbol-body-v2"
    max_escalation_cost_per_request: float = 0.08
    allow_external_for_explain_style: bool = True
    allow_external_for_product_workflow: bool = False
    prefer_local_for_plugin_runtime: bool = True
    prefer_local_for_phase_workflow: bool = True
    # Baseline bytes a user would have sent without CodeContext compaction.
    # Used to compute the "raw_bytes_avoided" / "estimated_saved_tokens"
    # marketing metrics. Previously hardcoded as 14000 deep inside
    # context_pack.py with no documentation. Override per project if the
    # typical uncompacted prompt size differs.
    naive_baseline_bytes: int = 14_000
    naive_baseline_tokens: int = 14_000
    llm_client: LLMClientConfig = field(default_factory=LLMClientConfig)

    def __post_init__(self) -> None:
        self.root = self.root.resolve()
        self.data_dir = self.root / ".codecontext" / "data"
        self.db_path = self.data_dir / "codecontext.db"
        self.backup_dir = self.data_dir / "backups"
        self.manifest_path = self.data_dir / "manifest.json"
        self.metrics_path = self.data_dir / "metrics.jsonl"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
