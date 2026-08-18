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

DOCS_DIR: Path = Path(os.getenv("DOCS_DIR", PROJECT_ROOT / "docs"))
CHROMA_DB_DIR: Path = Path(os.getenv("CHROMA_DB_DIR", PROJECT_ROOT / "chroma_db"))
DATA_DIR: Path = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
OBSERVABILITY_DB_PATH: Path = Path(
    os.getenv("OBSERVABILITY_DB_PATH", DATA_DIR / "observability.db")
)


def ensure_dirs() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_BACKEND: str = os.getenv("EMBEDDING_BACKEND", "huggingface").lower()

OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "qwen/qwen3-next-80b-a3b-instruct:free")
LLM_TEMPERATURE: float = 0.4
LLM_MAX_TOKENS: int = 4096

CHUNK_SIZE: int = 800
CHUNK_OVERLAP: int = 150
RETRIEVAL_K: int = 8
RETRIEVAL_CANDIDATES: int = int(os.getenv("RETRIEVAL_CANDIDATES", "24"))
DENSE_WEIGHT: float = float(os.getenv("DENSE_WEIGHT", "0.65"))
SPARSE_WEIGHT: float = float(os.getenv("SPARSE_WEIGHT", "0.35"))
RRF_K: int = int(os.getenv("RRF_K", "60"))

RERANK_ENABLED: bool = os.getenv("RERANK_ENABLED", "true").lower() == "true"
RERANKER_MODEL_NAME: str = os.getenv(
    "RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
RERANK_TOP_K: int = int(os.getenv("RERANK_TOP_K", str(RETRIEVAL_K)))

# Override these for the selected OpenRouter model. Values are USD per million
# tokens; the default free model therefore records a zero-dollar cost.
LLM_INPUT_COST_PER_MILLION: float = float(
    os.getenv("LLM_INPUT_COST_PER_MILLION", "0")
)
LLM_OUTPUT_COST_PER_MILLION: float = float(
    os.getenv("LLM_OUTPUT_COST_PER_MILLION", "0")
)
DEMO_MODE: bool = os.getenv("DEMO_MODE", "false").lower() == "true"
DEMO_MIN_RELEVANCE: float = float(os.getenv("DEMO_MIN_RELEVANCE", "0.25"))


def collection_name_for(language: str) -> str:
    """Normalise a language label into a valid ChromaDB collection name."""
    name = language.strip().lower().replace(" ", "_")
    # ChromaDB requires names of 3+ characters; pad short ones (e.g. "go", "c")
    if len(name) < 3:
        name = f"{name}-docs"
    return name


APP_TITLE: str = "RAG Coding Assistant"
WINDOW_SIZE: str = "900x700"
FONT_FAMILY: str = "Consolas"
FONT_SIZE: int = 11
