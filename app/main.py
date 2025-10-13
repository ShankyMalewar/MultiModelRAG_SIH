# app/main.py  (SESSION-ENABLED)
import os
import uuid
import shutil
import asyncio
import logging
import pathlib
import importlib
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

logger = logging.getLogger("asklyne")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Multimodal RAG - Offline (session-enabled)", version="1.0")

DATA_VAULT = pathlib.Path("./data/vault")
DATA_VAULT.mkdir(parents=True, exist_ok=True)

# in-memory session registry (simple)
_sessions: Dict[str, Dict[str, Any]] = {}
_models_preloaded = False
_models_preload_task: Optional[asyncio.Task] = None
_ingest_status: Dict[str, Dict[str, Any]] = {}

_QA_ORCHESTRATOR = None


def _save_uploadfile(upload: UploadFile, folder: pathlib.Path) -> pathlib.Path:
    folder.mkdir(parents=True, exist_ok=True)
    dst = folder / f"{uuid.uuid4().hex}{pathlib.Path(upload.filename).suffix}"
    with open(dst, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    return dst


async def _process_ingest_background(file_path: str, task_id: str, session_id: Optional[str] = None):
    _ingest_status[task_id] = {"status": "processing", "msg": "Starting ingest."}
    try:
        ingest_mod = importlib.import_module("app.ingest.ingest")
        fn = getattr(ingest_mod, "run_ingest", None) or getattr(ingest_mod, "ingest_file", None)
        if not fn:
            msg = "No ingest function found in app.ingest.ingest"
            _ingest_status[task_id] = {"status": "error", "msg": msg}
            logger.error(msg)
            return

        logger.info("Background ingest (session=%s): running for %s", session_id, file_path)
        # call the ingest function with session_id if it accepts it
        def call():
            try:
                return fn(file_path) if session_id is None else fn(file_path, session_id=session_id)
            except TypeError:
                return fn(file_path)

        result = await run_in_threadpool(call)
        msg = "OK"
        if hasattr(result, "errors") and result.errors:
            msg = f"errors={result.errors}"
        _ingest_status[task_id] = {
            "status": "done",
            "msg": msg,
            "result": result.to_dict() if hasattr(result, "to_dict") else str(result),
        }
    except Exception as e:
        logger.exception("Background ingest failed for %s: %s", file_path, e)
        _ingest_status[task_id] = {"status": "error", "msg": str(e)}


# ---------- Session API ----------
@app.post("/session")
async def create_session():
    sid = uuid.uuid4().hex
    _sessions[sid] = {"created_at": asyncio.get_event_loop().time()}
    # ensure vault subfolder exists
    (DATA_VAULT / sid).mkdir(parents=True, exist_ok=True)
    return {"ok": True, "session_id": sid}


@app.post("/session/{session_id}/ingest")
async def session_ingest(
    session_id: str,
    uploads: Optional[List[UploadFile]] = File(None),
    paths: Optional[List[str]] = Form(None),
    background: bool = Form(False),
):
    """
    Accept multiple uploads (form field `uploads`) and server-local paths via `paths`.
    Each file is saved to data/vault/{session_id} and then ingested with session_id.
    """
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    saved_paths = []
    # handle uploaded files
    if uploads:
        for uf in uploads:
            dst = _save_uploadfile(uf, DATA_VAULT / session_id)
            saved_paths.append(str(dst.resolve()))

    # handle server-local paths (if any)
    if paths:
        for p in paths:
            pth = pathlib.Path(p)
            if not pth.exists():
                logger.warning("Provided path not found: %s", p)
                continue
            saved_paths.append(str(pth.resolve()))

    if not saved_paths:
        raise HTTPException(status_code=400, detail="No files provided")

    results = []
    for sp in saved_paths:
        task_id = uuid.uuid4().hex
        _ingest_status[task_id] = {"status": "queued", "msg": "Saved to vault"}
        if background:
            asyncio.create_task(_process_ingest_background(sp, task_id, session_id=session_id))
            results.append({"file": sp, "task_id": task_id, "status": "queued"})
        else:
            # debug-run synchronously to surf errors immediately
            try:
                ingest_mod = importlib.import_module("app.ingest.ingest")
                fn = getattr(ingest_mod, "ingest_file", None) or getattr(ingest_mod, "run_ingest", None)
                if fn is None:
                    raise RuntimeError("Ingest function not found")
                # call with session_id if function accepts it
                try:
                    res = await run_in_threadpool(lambda: fn(sp, session_id=session_id))
                except TypeError:
                    res = await run_in_threadpool(lambda: fn(sp))
                results.append({"file": sp, "task_id": task_id, "result": res.to_dict() if hasattr(res, "to_dict") else str(res)})
                _ingest_status[task_id] = {"status": "done", "msg": "completed"}
            except Exception as e:
                logger.exception("Ingest failed for %s: %s", sp, e)
                _ingest_status[task_id] = {"status": "error", "msg": str(e)}
                results.append({"file": sp, "task_id": task_id, "error": str(e)})

    return {"ok": True, "session_id": session_id, "results": results}


@app.post("/session/{session_id}/qa")
async def session_qa(session_id: str, query: str = Form(...), top_k: int = Form(8)):
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    global _QA_ORCHESTRATOR
    if _QA_ORCHESTRATOR is None:
        # lazy-init orchestrator
        from app.core.retriever import Retriever
        from app.core.context_builder import ContextBuilder
        from app.core.llm_client import LLMClient
        from app.core.qa_prompt import QAOrchestrator

        retriever = Retriever()
        ctx_builder = ContextBuilder()
        llm_client = LLMClient()
        _QA_ORCHESTRATOR = QAOrchestrator(retriever=retriever, context_builder=ctx_builder, llm_client=llm_client)

    # run QA with session_id so retrieval is scoped
    res = await _QA_ORCHESTRATOR.ask(question=query, top_k=top_k, session_id=session_id)
    return res


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """
    Delete the Qdrant vectors for this session and remove vault files.
    """
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    # delete qdrant points by session (best-effort)
    try:
        from app.services.qdrant_service import QdrantService
        q = QdrantService()
        q.delete_by_session(session_id)
    except Exception as e:
        logger.exception("Failed to delete qdrant points for session %s: %s", session_id, e)

    # remove vault folder
    try:
        folder = DATA_VAULT / session_id
        if folder.exists():
            shutil.rmtree(folder)
    except Exception as e:
        logger.exception("Failed to remove vault folder for session %s: %s", session_id, e)

    _sessions.pop(session_id, None)
    return {"ok": True, "session_id": session_id, "deleted": True}


# ---------- Misc / Backwards-compatible endpoints ----------
@app.post("/ingest")
async def ingest_file(upload: Optional[UploadFile] = File(None), path: Optional[str] = Form(None)):
    if upload is None and not path:
        raise HTTPException(status_code=400, detail="Provide a file upload or a path.")

    if upload:
        dst = _save_uploadfile(upload, DATA_VAULT)
        file_path = str(dst.resolve())
        logger.info("Saved upload to %s", file_path)
    else:
        p = pathlib.Path(path)
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"Path not found: {path}")
        file_path = str(p.resolve())
        logger.info("Using existing file path %s", file_path)

    task_id = uuid.uuid4().hex
    _ingest_status[task_id] = {"status": "queued", "msg": "Saved to vault"}

    try:
        await run_in_threadpool(lambda: importlib.import_module("app.ingest.ingest").run_ingest(file_path))
        _ingest_status[task_id] = {"status": "done", "msg": "completed (debug run)"}
    except Exception as e:
        logger.exception("Ingest run failed (debug mode): %s", e)
        _ingest_status[task_id] = {"status": "error", "msg": str(e)}
        return JSONResponse({"ok": False, "message": str(e)}, status_code=500)

    return {"ok": True, "task_id": task_id, "file_path": file_path, "message": "Ingest queued (debug-run)"}


@app.get("/ingest_status/{task_id}")
async def ingest_status(task_id: str):
    info = _ingest_status.get(task_id)
    if not info:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"task_id": task_id, "status": info}


# preload models on startup (non-blocking)
async def _preload_models():
    global _models_preloaded
    logger.info("Preload: starting model warmup.")
    try:
        try:
            from app.core.embedder import Embedder
            await run_in_threadpool(lambda: Embedder())
            logger.info("Preload: embedder ready")
        except Exception as e:
            logger.warning("Preload: embedder failed: %s", e)
        _models_preloaded = True
    except Exception:
        logger.exception("Preload error")

@app.on_event("startup")
async def startup_event():
    global _models_preload_task
    loop = asyncio.get_event_loop()
    _models_preload_task = loop.create_task(_preload_models())


@app.get("/health")
async def health():
    return {"ok": True, "preloaded": _models_preloaded, "sessions": list(_sessions.keys())}





# app/main.py (paste after `app = FastAPI()`)

import os
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI

# assume you already have `app = FastAPI()` above

# ---------- Configurable values ----------
# Comma-separated allowed origins (use env var to avoid code changes)
_frontend_origins = os.getenv(
    "FRONTEND_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000"
)
ALLOW_ALL_ORIGINS_DEV = os.getenv("ALLOW_ALL_ORIGINS_DEV", "false").lower() in ("1", "true")

if ALLOW_ALL_ORIGINS_DEV:
    allow_origins = ["*"]
else:
    # split and trim
    allow_origins = [o.strip() for o in _frontend_origins.split(",") if o.strip()]

# ---------- Add CORS middleware ----------
# Add middleware before mounting static files / routers so preflight requests are handled properly
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Serve static frontend ----------
# Compute static dir relative to this file (project-root/frontend/static)
BASE = Path(__file__).resolve().parent.parent  # adjust if your layout differs
STATIC_DIR = BASE / "frontend" / "static"

if not STATIC_DIR.exists():
    # optional: warn in logs to help devs
    import logging
    logging.getLogger("uvicorn").warning(f"Static directory not found: {STATIC_DIR} (frontend static may not be served)")

# Mount at root so visiting http://host:8000/ serves index.html
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="frontend")


# ---------- Optional helper endpoint for runtime config ----------
# Useful if you prefer the static page to read config instead of hardcoding API_BASE in the HTML.
# Returns empty string for same-origin (recommended), or you can set API_BASE env var to explicit URL.
from pydantic import BaseModel
from fastapi import APIRouter

router = APIRouter()

class ConfigResp(BaseModel):
    api_base: str

@router.get("/__config", response_model=ConfigResp)
async def get_config():
    """
    Returns frontend runtime config. If API is served same-origin, api_base is ''.
    Otherwise set API_BASE env var to e.g. 'http://localhost:8000'
    """
    api_base = os.getenv("API_BASE", "")  # set this in env if frontend needs an explicit host
    return {"api_base": api_base}

app.include_router(router, tags=["config"])
