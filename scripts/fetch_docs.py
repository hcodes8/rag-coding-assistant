"""Download documentation sets into docs/<language>/ for the RAG assistant.

Usage:
    python scripts/fetch_docs.py --list
    python scripts/fetch_docs.py python rust
    python scripts/fetch_docs.py all --ingest

Only uses the standard library. Each source is a zip archive that is
downloaded to a temp file and selectively extracted (supported text formats
only), so image-heavy repos don't bloat the docs folder.
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DOCS_DIR = PROJECT_ROOT / "docs"
EXTENSIONS = (".txt", ".md", ".rst")
USER_AGENT = "rag-coding-assistant-docs-fetcher"


@dataclass
class Source:
    """A downloadable documentation set.

    url may be a callable for sources whose download link must be discovered
    at runtime (e.g. the versioned Python docs archive).
    """

    url: str | object
    subdir: str = ""  # only extract members under this path inside the archive
    note: str = ""


def _python_docs_url() -> str:
    """Find the current plain-text Python docs archive from the download page."""
    page = _get("https://docs.python.org/3/download.html").decode("utf-8", "replace")
    match = re.search(r'href="([^"]*python-[\d.]+-docs-text\.zip)"', page)
    if not match:
        raise RuntimeError("Could not find the plain-text archive link on docs.python.org")
    link = match.group(1)
    if link.startswith("http"):
        return link
    return f"https://docs.python.org/3/{link}"


SOURCES: dict[str, Source] = {
    "python": Source(
        url=_python_docs_url,
        note="official CPython docs, plain text",
    ),
    "javascript": Source(
        url="https://github.com/javascript-tutorial/en.javascript.info/archive/refs/heads/master.zip",
        note="The Modern JavaScript Tutorial (javascript.info)",
    ),
    "rust": Source(
        url="https://github.com/rust-lang/book/archive/refs/heads/main.zip",
        subdir="src/",
        note="The Rust Programming Language book",
    ),
    "go": Source(
        url="https://github.com/golang/website/archive/refs/heads/master.zip",
        subdir="_content/doc/",
        note="go.dev documentation pages",
    ),
}


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req) as resp:
        return resp.read()


def _download(url: str, dest) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    live = sys.stdout.isatty()  # animate in a terminal, stay quiet when piped
    with urllib.request.urlopen(req) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        read = 0
        while chunk := resp.read(1 << 16):
            dest.write(chunk)
            read += len(chunk)
            if live:
                suffix = f"/{total / 1e6:.1f}" if total else ""
                print(f"\r  downloading… {read / 1e6:.1f}{suffix} MB", end="")
    prefix = "\r" if live else ""
    print(f"{prefix}  downloaded {read / 1e6:.1f} MB" + " " * 10)


def fetch(language: str, force: bool = False) -> bool:
    source = SOURCES[language]
    target = DOCS_DIR / language

    if target.exists() and any(target.iterdir()) and not force:
        print(f"[{language}] docs/{language}/ already has files, skipping (use --force to refetch)")
        return True

    url = source.url() if callable(source.url) else source.url
    print(f"[{language}] {source.note}")

    with tempfile.TemporaryFile() as tmp:
        _download(url, tmp)
        tmp.seek(0)
        written = 0
        with zipfile.ZipFile(tmp) as zf:
            for member in zf.infolist():
                if member.is_dir():
                    continue
                # strip the archive's top-level folder (e.g. "book-main/")
                parts = member.filename.split("/", 1)
                relative = parts[1] if len(parts) == 2 else parts[0]
                if source.subdir and not relative.startswith(source.subdir):
                    continue
                if not relative.lower().endswith(EXTENSIONS):
                    continue
                out = target / relative[len(source.subdir):]
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(zf.read(member))
                written += 1

    if written == 0:
        print(f"[{language}] ERROR: archive contained no supported files")
        return False
    print(f"[{language}] wrote {written} file(s) to docs/{language}/")
    return True


def ingest(language: str) -> None:
    """Embed a language's docs now so the app UI is instant on first use."""
    from app.document_loader import load_documents_for_language
    from app.vector_store import VectorStoreManager

    print(f"[{language}] ingesting (chunk + embed; one-time, CPU-bound)…")
    manager = VectorStoreManager()
    docs = load_documents_for_language(language)
    manager.ingest(language, docs)
    print(f"[{language}] ingested {len(docs)} chunks")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("languages", nargs="*", help='language names, or "all"')
    parser.add_argument("--list", action="store_true", help="list available sources")
    parser.add_argument("--force", action="store_true", help="refetch even if docs exist")
    parser.add_argument("--ingest", action="store_true",
                        help="embed into the vector store immediately after download")
    args = parser.parse_args()

    if args.list or not args.languages:
        print("Available documentation sources:")
        for name, source in SOURCES.items():
            print(f"  {name:<12} {source.note}")
        return 0

    wanted = list(SOURCES) if "all" in args.languages else args.languages
    unknown = [lang for lang in wanted if lang not in SOURCES]
    if unknown:
        print(f"Unknown language(s): {', '.join(unknown)} (see --list)")
        return 1

    failed = []
    for lang in wanted:
        try:
            if not fetch(lang, force=args.force):
                failed.append(lang)
                continue
            if args.ingest:
                ingest(lang)
        except Exception as exc:
            print(f"[{lang}] ERROR: {exc}")
            failed.append(lang)

    if failed:
        print(f"\nFailed: {', '.join(failed)}")
        return 1
    print("\nDone. Launch the app with: python -m app.main")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
