from __future__ import annotations

import shutil
from pathlib import Path

from app.config import DEMO_MODE, DOCS_DIR, PROJECT_ROOT


def seed_demo_docs() -> bool:
    """Seed a tiny documentation set only when demo mode has no user docs."""
    if not DEMO_MODE or any(DOCS_DIR.iterdir()):
        return False
    source = PROJECT_ROOT / "demo_docs"
    if not source.is_dir():
        return False
    for directory in source.iterdir():
        if directory.is_dir():
            shutil.copytree(directory, DOCS_DIR / directory.name, dirs_exist_ok=True)
    return True
