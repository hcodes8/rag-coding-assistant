import logging
import sys
import threading
import time
import webbrowser
from pathlib import Path


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
    )


def main() -> None:
    _configure_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting RAG-Coding-Assistant (web mode)")

    from app.vector_store import VectorStoreManager
    from app.rag_pipeline import RAGPipeline
    from app.server import run_server
    from app.gui import launch_gui 

    HOST = "127.0.0.1"
    PORT = 5000

    logger.info("Loading embedding model…")
    vs_manager = VectorStoreManager()
    pipeline   = RAGPipeline(vs_manager)
    logger.info("Backend ready. Starting web server on http://%s:%d", HOST, PORT)

    # Open browser after a short delay so server is active
    server_thread = threading.Thread(
        target=run_server,
        args=(vs_manager, pipeline),
        kwargs={"host": HOST, "port": PORT},
        daemon=True,
    )
    server_thread.start()
    time.sleep(1.2)
    # Blocks here until the window is closed
    launch_gui(port=PORT)

    logger.info("Application closed")



if __name__ == "__main__":
    main()