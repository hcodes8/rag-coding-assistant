"""Production ASGI entrypoint used by Docker and conventional web hosts."""

from app.config import ensure_dirs
from app.demo import seed_demo_docs
from app.rag_pipeline import RAGPipeline
from app.server import create_app
from app.vector_store import VectorStoreManager

ensure_dirs()
seed_demo_docs()
vector_store = VectorStoreManager()
pipeline = RAGPipeline(vector_store)
app = create_app(vector_store, pipeline)
