import logging
import sys
import threading
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import uvicorn


class _NoSignalServer(uvicorn.Server):
    """Uvicorn server that skips signal-handler installation so it can run
    in a non-main thread (signal.signal only works on the main thread)."""

    def install_signal_handlers(self) -> None:  # type: ignore[override]
        return None


def _configure_logging() -> None:
    if getattr(sys, "frozen", False):
        log_dir = Path(sys.executable).parent
    else:
        log_dir = Path(__file__).resolve().parent.parent

    log_file = log_dir / "devdocs_chatbot.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
        force=True,
    )


def _wait_for_server(url: str, timeout: float = 20.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=0.5) as resp:
                if resp.status == 200:
                    return True
        except (URLError, OSError):
            pass
        time.sleep(0.1)
    return False


def main() -> None:
    _configure_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting RAG-Coding-Assistant")

    from app.config import ensure_dirs
    from app.gui import launch_gui
    from app.rag_pipeline import RAGPipeline
    from app.server import create_app
    from app.vector_store import VectorStoreManager

    ensure_dirs()

    HOST = "127.0.0.1"
    PORT = 5000

    logger.info("Loading embedding model…")
    vs_manager = VectorStoreManager()
    pipeline = RAGPipeline(vs_manager)

    fastapi_app = create_app(vs_manager, pipeline)
    config = uvicorn.Config(
        fastapi_app,
        host=HOST,
        port=PORT,
        log_level="warning",
    )
    server = _NoSignalServer(config)

    def _serve():
        try:
            server.run()
        except Exception:
            logger.exception("Server crashed")

    server_thread = threading.Thread(target=_serve, daemon=True)
    server_thread.start()

    if not _wait_for_server(f"http://{HOST}:{PORT}/"):
        logger.error("Server did not become ready in time")
        server.should_exit = True
        server_thread.join(timeout=5)
        return

    logger.info("Server ready; launching GUI at http://%s:%d/ui", HOST, PORT)

    def _on_closed() -> None:
        logger.info("GUI closed; stopping server")
        server.should_exit = True

    launch_gui(url=f"http://{HOST}:{PORT}/ui", on_closed=_on_closed)

    server_thread.join(timeout=5)
    logger.info("Application closed")


if __name__ == "__main__":
    main()
