import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Directory for language documents (one sub-directory per language)
DOCS_DIR: Path = PROJECT_ROOT / "docs"

# Where ChromaDB persists its collections between runs
CHROMA_DB_DIR: Path = PROJECT_ROOT / "chroma_db"

# Ensure these directories exist at import time
DOCS_DIR.mkdir(exist_ok=True)
CHROMA_DB_DIR.mkdir(exist_ok=True)

# Embedding Model
EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

# LLM
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "mistralai/mistral-7b-instruct")
LLM_TEMPERATURE: float = 0.4 # Lower temp = more factual
LLM_MAX_TOKENS: int = 4096 # Max tokens per answer

# Chunking
CHUNK_SIZE: int = 800 # character count per chunk
CHUNK_OVERLAP: int = 150 # characters shared between adjacent chunks
RETRIEVAL_K: int = 8 # chunks received per query

# ChromaDB
def collection_name_for(language: str) -> str:
    """Normalise a language label into a valid ChromaDB collection name."""
    return language.strip().lower().replace(" ", "_")

# GUI
APP_TITLE: str = "RAG Coding Assistant"
WINDOW_SIZE: str = "900x700"
FONT_FAMILY: str = "Consolas"
FONT_SIZE: int = 11