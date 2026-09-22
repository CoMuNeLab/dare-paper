"""Repository and data path helpers."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def solve_path(path: str | Path) -> Path:
    """Resolve *path* against ``DATA_BASE_DIR`` or the repository root."""
    return Path(os.getenv("DATA_BASE_DIR", PROJECT_ROOT)).expanduser() / path


def output_path(timestamp: str, *parts: str | Path) -> Path:
    """Resolve an artifact path inside one timestamped output run."""
    return solve_path(Path("data") / "output" / timestamp).joinpath(*parts)
