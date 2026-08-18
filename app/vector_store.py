from __future__ import annotations
import logging
from typing import Optional
import chromadb
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import (
    CHROMA_DB_DIR,
    EMBEDDING_BACKEND,
    EMBEDDING_MODEL_NAME,
    RETRIEVAL_K,
    collection_name_for,
)
from app.embeddings import HashEmbeddings

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    Manages ChromaDB collections for each programming language.

    1. Instantiate once at app startup.
    2. Calls ingest(language, documents) when a language's docs are not yet
       embedded or need refreshing.
    3. Calls get_retriever(language) to get a LangChain retriever for that
       language. 
    """

    def __init__(self) -> None:
        # HuggingFaceEmbeddings downloads the model weights on first use and
        # caches them in ~/.cache/huggingface.
        if EMBEDDING_BACKEND == "hash":
            logger.info("Using deterministic hash embeddings")
            self._embeddings = HashEmbeddings()
        else:
            logger.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
            self._embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )

        # Persistent ChromaDB client: no re ingestion between app sessions
        self._chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))

        # Cache of LangChain Chroma objects keyed by collection name.
        self._stores: dict[str, Chroma] = {}

    def collection_exists(self, language: str) -> bool:
        """Return True if this language has already been ingested."""
        name = collection_name_for(language)
        existing = [c.name for c in self._chroma_client.list_collections()]
        return name in existing

    def ingest(self, language: str, documents: list) -> None:
        """
        Embeds and stores documents for a language. If the collection already exists 
        it is deleted first to refresh with new documents

        Args:
            language: Language label (used to derive collection name).
            documents: List of LangChain Document objects (chunked).
        
        Raises:
            ValueError: If documents is empty.
        """
        if not documents:
            raise ValueError(
                f"Cannot ingest an empty document list for '{language}'. "
            )
        name = collection_name_for(language)

        # Delete collection if it exists
        if self.collection_exists(language):
            logger.info("Deleting existing collection '%s' before re-ingest", name)
            self._chroma_client.delete_collection(name)
        
        # Remove from cache
        self._stores.pop(name, None)
        logger.info("Ingesting %d chunks into collection '%s'", len(documents), name)

        # Chroma.from_documents creates the collection and adds all documents
        # in one call, handling batching internally.
        store = Chroma.from_documents(
            documents=documents,
            embedding=self._embeddings,
            collection_name=name,
            client=self._chroma_client,
        )

        self._stores[name] = store
        logger.info("Ingestion complete for '%s'", name)

    def get_retriever(self, language: str, k: int = RETRIEVAL_K):
        """
        Returns a LangChain retriever for the given language's vector store.

        Args:
            language: Must already be ingested.
            k: Number of chunks to retrieve.

        Returns:
            A LangChain BaseRetriever compatible with the RAG chain.

        Raises:
            RuntimeError: If the language has not been ingested yet.
        """
        name = collection_name_for(language)

        if not self.collection_exists(language):
            raise RuntimeError(
                f"Collection for '{language}' does not exist. "
                "Please ingest documents first."
            )

        # Build and cache the LangChain Chroma wrapper if not already cached
        if name not in self._stores:
            self._stores[name] = Chroma(
                collection_name=name,
                embedding_function=self._embeddings,
                client=self._chroma_client,
            )

        store = self._stores[name]
        return store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": k * 3},
        )

    def _get_store(self, language: str) -> Chroma:
        name = collection_name_for(language)
        if not self.collection_exists(language):
            raise RuntimeError(
                f"Collection for '{language}' does not exist. "
                "Please ingest documents first."
            )
        if name not in self._stores:
            self._stores[name] = Chroma(
                collection_name=name,
                embedding_function=self._embeddings,
                client=self._chroma_client,
            )
        return self._stores[name]

    def dense_search(self, language: str, query: str, k: int) -> list[tuple[Document, float]]:
        """Return dense candidates with normalized similarity scores."""
        store = self._get_store(language)
        results = store.similarity_search_with_score(query, k=k)
        return [
            (document, 1.0 / (1.0 + max(float(distance), 0.0)))
            for document, distance in results
        ]

    def all_documents(self, language: str) -> list[Document]:
        """Load stored chunks for sparse indexing without re-reading source files."""
        store = self._get_store(language)
        payload = store.get(include=["documents", "metadatas"])
        documents = payload.get("documents") or []
        metadatas = payload.get("metadatas") or []
        return [
            Document(page_content=text, metadata=metadata or {})
            for text, metadata in zip(documents, metadatas)
            if text
        ]

    def list_ingested_languages(self) -> list[str]:
        """Return language names (collection names) that are already embedded."""
        return [c.name for c in self._chroma_client.list_collections()]
