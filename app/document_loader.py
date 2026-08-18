from __future__ import annotations
import hashlib
import logging
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import CHUNK_OVERLAP, CHUNK_SIZE, DOCS_DIR

logger = logging.getLogger(__name__)

# Extend this set for more formats
SUPPORTED_EXTENSIONS: set[str] = {".txt", ".md", ".rst"}


def get_available_languages() -> List[str]:
    """
    Scan the docs/ directory and return a sorted list of language names.

    Each direct subdirectory of docs/ is treated as a language collection.
    Returns an empty list if docs/ has no subdirectories.
    """
    if not DOCS_DIR.exists():
        logger.warning("docs/ directory not found at %s", DOCS_DIR)
        return []

    # A language folder must be a directory
    languages = [d.name for d in sorted(DOCS_DIR.iterdir()) if d.is_dir()]
    logger.debug("Discovered language folders: %s", languages)
    return languages


def load_documents_for_language(language: str) -> List[Document]:
    """
    Load all supported text files from docs/<language>/ recursively.

    Each file becomes one or more LangChain Document objects after chunking.
    Metadata (source path, language) is attached to every Document so that
    the chatbot can cite where an answer came from.

    Args:
        language: Folder name under docs/ .

    Returns:
        A list of chunked Document objects ready for embedding.

    Raises:
        FileNotFoundError: If no matching language folder exists.
        ValueError: If the folder contains no supported files.
    """
    lang_dir = DOCS_DIR / language
    if not lang_dir.is_dir():
        raise FileNotFoundError(
            f"No documentation folder found for '{language}' at {lang_dir}"
        )

    file_paths: List[Path] = []
    for ext in SUPPORTED_EXTENSIONS:
        file_paths.extend(lang_dir.rglob(f"*{ext}"))

    if not file_paths:
        raise ValueError(
            f"No supported files ({SUPPORTED_EXTENSIONS}) found in {lang_dir}"
        )

    logger.info("Found %d file(s) for language '%s'", len(file_paths), language)

    # Read raw text
    raw_docs: List[Document] = []
    for fp in sorted(file_paths):
        try:
            text = fp.read_text(encoding="utf-8", errors="replace")
            raw_docs.append(
                Document(
                    page_content=text,
                    # Metadata is stored alongside each chunk in ChromaDB.
                    # 'source' lets us tell the user which doc an answer came from.
                    metadata={
                        "source": str(fp.relative_to(DOCS_DIR)),
                        "language": language,
                        "filename": fp.name,
                    },
                )
            )
        except OSError as exc:
            # Log and skip unreadable files rather than crashing the whole load
            logger.warning("Could not read %s: %s", fp, exc)

    if not raw_docs:
        raise ValueError(f"All files for '{language}' were unreadable.")

    # Chunk Documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )

    chunked_docs = splitter.split_documents(raw_docs)
    source_indexes: dict[str, int] = {}
    for doc in chunked_docs:
        source = str(doc.metadata.get("source", "unknown"))
        chunk_index = source_indexes.get(source, 0)
        source_indexes[source] = chunk_index + 1
        identity = (
            f"{source}:{doc.metadata.get('start_index', 0)}:{doc.page_content}"
        ).encode("utf-8")
        doc.metadata["chunk_index"] = chunk_index
        doc.metadata["chunk_id"] = hashlib.sha1(identity).hexdigest()[:16]
    logger.info(
        "Split %d raw doc(s) into %d chunk(s) for '%s'",
        len(raw_docs),
        len(chunked_docs),
        language,
    )
    return chunked_docs
