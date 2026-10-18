"""Project-wide constants and filesystem layout.

Everything under ``data/`` is derived from City of St. John's documents and is
gitignored. Paths can be redirected with the ``COUNCIL_RAG_DATA_DIR`` environment
variable (useful for tests and CI).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _default_root() -> Path:
    """Repository root when running from a source checkout, else the working directory."""
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "pyproject.toml").exists():
        return candidate
    return Path.cwd()


PROJECT_ROOT = _default_root()

# --- Source portal -----------------------------------------------------------------
ESCRIBE_BASE_URL = "https://pub-stjohns.escribemeetings.com"
CALENDAR_ENDPOINT = "/MeetingsCalendarView.aspx/GetCalendarMeetings"
USER_AGENT = (
    "stjohns-council-minutes-rag/0.1 (portfolio research; contact mostafa.wasib73@gmail.com)"
)
MEETING_NAME = "Regular Council Meeting"
DEFAULT_START_DATE = "2023-01-01"
MIN_REQUEST_INTERVAL_S = 1.1  # brief: at most one request per second

# --- Embedding model ---------------------------------------------------------------
# BAAI/bge-small-en-v1.5 via its ONNX port, the same weights transformers.js loads
# in the browser demo. The revision is pinned so vectors are reproducible.
EMBEDDING_REPO = "Xenova/bge-small-en-v1.5"
EMBEDDING_REVISION = os.environ.get(
    "COUNCIL_RAG_EMBEDDING_REVISION", "ea104dacec62c0de699686887e3f920caeb4f3e3"
)
EMBEDDING_ONNX_FILE = "onnx/model.onnx"
EMBEDDING_DIM = 384
EMBEDDING_MAX_TOKENS = 512
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# --- Generation (Claude) -----------------------------------------------------------
CLAUDE_MODEL = os.environ.get("COUNCIL_RAG_MODEL", "claude-opus-5-5")
CLAUDE_EFFORT = os.environ.get("COUNCIL_RAG_EFFORT", "medium")


@dataclass(frozen=True)
class DataPaths:
    """Locations of raw, parsed and indexed data for one data directory."""

    root: Path

    @property
    def raw_html(self) -> Path:
        return self.root / "raw" / "html"

    @property
    def raw_pdfs(self) -> Path:
        return self.root / "raw" / "pdfs"

    @property
    def calendar_cache(self) -> Path:
        return self.root / "raw" / "calendar"

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def parsed(self) -> Path:
        return self.root / "parsed"

    @property
    def index(self) -> Path:
        return self.root / "index"

    @property
    def logs(self) -> Path:
        return self.root / "logs"


def data_paths(root: str | os.PathLike[str] | None = None) -> DataPaths:
    """Resolve the data directory (argument, then env var, then ``<repo>/data``)."""
    if root is None:
        root = os.environ.get("COUNCIL_RAG_DATA_DIR", PROJECT_ROOT / "data")
    return DataPaths(Path(root))


def model_cache_dir() -> Path:
    """Where downloaded model files are cached (kept out of git)."""
    return Path(os.environ.get("COUNCIL_RAG_MODEL_CACHE", PROJECT_ROOT / ".cache" / "models"))
