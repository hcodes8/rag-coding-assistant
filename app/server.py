import asyncio
import json
import logging
import threading
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

WEB_DIR: Path = (Path(__file__).parent / "web").resolve()


def create_app(vs_manager, pipeline) -> FastAPI:
    app = FastAPI(title="RAG Coding Assistant")
    # One lock guards both activation and streaming so the active chain cannot
    # be swapped mid-answer. Fine for a single-user desktop app.
    pipeline_lock = asyncio.Lock()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    async def health():
        return {"status": "alive"}

    @app.get("/ui")
    async def serve_index():
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/ui/{filename:path}")
    async def serve_static(filename: str):
        candidate = (WEB_DIR / filename).resolve()
        try:
            candidate.relative_to(WEB_DIR)
        except ValueError:
            raise HTTPException(404, "not found")
        if not candidate.is_file():
            raise HTTPException(404, "not found")
        return FileResponse(candidate)

    @app.get("/api/languages")
    async def get_languages():
        from app.document_loader import get_available_languages
        return {"languages": get_available_languages()}

    @app.get("/api/status")
    async def get_status():
        return {
            "language": pipeline.current_language,
            "ready": pipeline.current_language is not None,
        }

    class ActivateRequest(BaseModel):
        language: str

    class AskRequest(BaseModel):
        question: str

    @app.post("/api/activate")
    async def activate_language(req: ActivateRequest):
        language = req.language.strip()
        if not language:
            raise HTTPException(400, "language required")

        async with pipeline_lock:
            if vs_manager.collection_exists(language):
                pipeline.set_language(language)
                return {"status": "ready", "language": language}

            loop = asyncio.get_running_loop()

            def _ingest():
                from app.document_loader import load_documents_for_language
                docs = load_documents_for_language(language)
                vs_manager.ingest(language, docs)
                pipeline.set_language(language)
                return len(docs)

            try:
                n = await loop.run_in_executor(None, _ingest)
                return {"status": "ingested", "language": language, "chunks": n}
            except Exception as exc:
                logger.exception("Activation failed for %s", language)
                raise HTTPException(500, str(exc))

    @app.post("/api/ask")
    async def ask(req: AskRequest):
        question = req.question.strip()
        if not question:
            raise HTTPException(400, "question required")
        if pipeline.current_language is None:
            raise HTTPException(400, "No language selected")

        async def token_stream() -> AsyncGenerator[str, None]:
            async with pipeline_lock:
                loop = asyncio.get_running_loop()
                queue: asyncio.Queue = asyncio.Queue()

                def _run():
                    try:
                        for token in pipeline.ask_stream(question):
                            loop.call_soon_threadsafe(queue.put_nowait, token)
                    except Exception as exc:
                        loop.call_soon_threadsafe(
                            queue.put_nowait, f"\n\nError: {exc}"
                        )
                    finally:
                        loop.call_soon_threadsafe(queue.put_nowait, None)

                threading.Thread(target=_run, daemon=True).start()
                while True:
                    token = await queue.get()
                    if token is None:
                        break
                    yield f"data: {json.dumps({'token': token})}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            token_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app