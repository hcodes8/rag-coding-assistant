import uvicorn
from datetime import timedelta, datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import threading
from typing import AsyncGenerator
from fastapi.responses import StreamingResponse
import json
from contextlib import asynccontextmanager
import os
import signal
from fastapi.responses import FileResponse
from pathlib import Path


def create_app(vs_manager, pipeline) -> FastAPI:
    last_ping = {"time": datetime.now()}
    TIMEOUT_SECONDS = 60

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async def _watchdog():
            await asyncio.sleep(10)  # grace period before watchdog activates
            while True:
                await asyncio.sleep(5)
                if datetime.now() - last_ping["time"] > timedelta(seconds=TIMEOUT_SECONDS):
                    os.kill(os.getpid(), signal.SIGTERM)
        asyncio.create_task(_watchdog())
        yield

    app = FastAPI(title="RAG Coding Assistant", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],       # file:// origin requires wildcard
        allow_credentials=False,   # credentials not compatible with wildcard
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    async def index():
        return {"status": "alive"}
    
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
    
    GUI_DIR = Path(__file__).parent # app folder
    @app.get("/ui/{filename}")
    async def serve_gui_file(filename: str):
        file = GUI_DIR / filename
        if not file.exists():
            raise HTTPException(404, "not found")
        return FileResponse(file)

    @app.get("/ui")
    async def serve_gui():
        return FileResponse(GUI_DIR / "index.html")
    
    class ActivateRequest(BaseModel):
        language: str

    class AskRequest(BaseModel):
        question: str

    # post endpoints
    @app.post("/api/ping")
    async def ping():
        last_ping["time"] = datetime.now()
        return {"ok": True}
    
    @app.post("/api/activate")
    async def activate_language(req: ActivateRequest):
        language = req.language
        if not language:
            raise HTTPException(400, "language required")

        if vs_manager.collection_exists(language):
            pipeline.set_language(language)
            return {"status": "ready", "language": language}
        
        loop = asyncio.get_event_loop()
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
            raise HTTPException(500, str(exc))
        
    @app.post("/api/ask")
    async def ask(req: AskRequest):
        if not req.question.strip():
            raise HTTPException(400, "question required")
        if pipeline.current_language is None:
            raise HTTPException(400, "No language selected")
        
        async def token_stream() -> AsyncGenerator[str, None]:
            loop = asyncio.get_event_loop()
            queue: asyncio.Queue = asyncio.Queue()

            def _run():
                try:
                    for token in pipeline.ask_stream(req.question):
                        loop.call_soon_threadsafe(queue.put_nowait, token)
                except Exception as exc:
                    loop.call_soon_threadsafe(queue.put_nowait, f"\n\nError: {exc}")
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, None)

            threading.Thread(target=_run, daemon=True).start()
            while True:
                token = await queue.get()
                if token is None:
                    break
                data = json.dumps({"token": token})
                yield f"data: {data}\n\n"

            yield "data: [DONE]\n\n"

        return StreamingResponse(
            token_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
            
    return app

def run_server(vs_manager, pipeline, host="127.0.0.1", port=5000):
    app = create_app(vs_manager, pipeline)
    uvicorn.run(app, host=host, port=port, log_level="warning")