import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def _base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT: Path = _base_dir()
load_dotenv(PROJECT_ROOT / ".env")

DOCS_DIR: Path = PROJECT_ROOT / "docs"
CHROMA_DB_DIR: Path = PROJECT_ROOT / "chroma_db"


def ensure_dirs() -> None:
    DOCS_DIR.mkdir(exist_ok=True)
    CHROMA_DB_DIR.mkdir(exist_ok=True)


EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "mistralai/mistral-7b-instruct")
LLM_TEMPERATURE: float = 0.4
LLM_MAX_TOKENS: int = 4096

CHUNK_SIZE: int = 800
CHUNK_OVERLAP: int = 150
RETRIEVAL_K: int = 8


def collection_name_for(language: str) -> str:
    """Normalise a language label into a valid ChromaDB collection name."""
    return language.strip().lower().replace(" ", "_")


APP_TITLE: str = "RAG Coding Assistant"
WINDOW_SIZE: str = "900x700"
FONT_FAMILY: str = "Consolas"
FONT_SIZE: int = 11
