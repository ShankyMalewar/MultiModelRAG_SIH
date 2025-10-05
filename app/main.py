# app/main.py
import os
import uuid
import shutil
import asyncio
import logging
import pathlib
import importlib
from typing import Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

# ========== Setup ==========
logger = logging.getLogger("asklyne")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Multimodal RAG - Offline", version="1.0")

DATA_VAULT = pathlib.Path("./data/vault")
DATA_VAULT.mkdir(parents=True, exist_ok=True)

# Global preload flags
_models_preloaded = False
_models_preload_task: Optional[asyncio.Task] = None
_ingest_status: Dict[str, Dict[str, Any]] = {}  # {task_id: {"status": "...", "msg": "..."}}

# Lazy QA orchestrator holder
_QA_ORCHESTRATOR = None


# ========== Utilities ==========
def _save_uploadfile(upload: UploadFile, folder: pathlib.Path) -> pathlib.Path:
    folder.mkdir(parents=True, exist_ok=True)
    dst = folder / f"{uuid.uuid4().hex}{pathlib.Path(upload.filename).suffix}"
    with open(dst, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    return dst


# ========== Background Ingest ==========
async def _process_ingest_background(file_path: str, task_id: str):
    """Runs the heavy ingest pipeline in a background thread."""
    _ingest_status[task_id] = {"status": "processing", "msg": "Starting ingest..."}
    try:
        ingest_mod = importlib.import_module("app.ingest.ingest")
        fn = getattr(ingest_mod, "run_ingest", None) or getattr(ingest_mod, "ingest_file", None)
        if not fn:
            msg = "No ingest function found in app.ingest.ingest"
            _ingest_status[task_id] = {"status": "error", "msg": msg}
            logger.error(msg)
            return

        logger.info("Background ingest: running for %s", file_path)
        # run_in_threadpool on a lambda so we can pass the file_path
        result = await run_in_threadpool(lambda: fn(file_path))
        logger.info("Background ingest: finished for %s", file_path)
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


@app.post("/ingest")
async def ingest_file(upload: Optional[UploadFile] = File(None), path: Optional[str] = Form(None)):
    """Accept a file or server-local path → enqueue background processing."""
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

    # Debug / interactive mode: run in threadpool synchronously to show logs & tracebacks immediately.
    # If you want background async enqueue behavior, replace the next line with:
    # asyncio.create_task(_process_ingest_background(file_path, task_id))
    try:
        # Use importlib.import_module to ensure we get the submodule object (not top-level app)
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


# ========== Preload Models ==========
async def _preload_models():
    global _models_preloaded
    logger.info("Preload: starting model warmup...")
    try:
        # Embedder preload
        try:
            from app.core.embedder import Embedder
            await run_in_threadpool(lambda: Embedder())
            logger.info("Preload: embedder ready")
        except Exception as e:
            logger.warning("Preload: embedder failed: %s", e)

        # ASR preload
        try:
            from app.extractors.asr_handler import ASRHandler
            await run_in_threadpool(lambda: ASRHandler())
            logger.info("Preload: ASR ready")
        except Exception as e:
            logger.warning("Preload: ASR failed: %s", e)

        # LLM ping
        try:
            from app.core.llm_client import LLMClient
            llm = LLMClient()
            if asyncio.iscoroutinefunction(llm.ping):
                await llm.ping()
            else:
                await run_in_threadpool(llm.ping)
            logger.info("Preload: LLM reachable")
        except Exception as e:
            logger.warning("Preload: LLM unreachable (%s)", e)

        _models_preloaded = True
        logger.info("Preload finished.")
    except Exception as e:
        logger.exception("Preload error: %s", e)


@app.on_event("startup")
async def startup_event():
    global _models_preload_task
    loop = asyncio.get_event_loop()
    _models_preload_task = loop.create_task(_preload_models())


# ========== Health / QA / Query / Answer ==========
@app.get("/health")
async def health():
    return {"ok": True, "preloaded": _models_preloaded, "vault_files": len(list(DATA_VAULT.glob('*')))}


@app.post("/query")
async def query(query: str = Form(...), top_k: int = Form(5)):
    try:
        from app.core.retriever import Retriever
        retriever = Retriever()
        hits = await run_in_threadpool(retriever.retrieve, query, top_k)
        return {"ok": True, "hits": hits}
    except Exception as e:
        logger.exception("Query failed: %s", e)
        return JSONResponse({"ok": False, "message": str(e)}, status_code=500)


# QA endpoint using QAOrchestrator (lazy init)
@app.post("/qa")
async def qa(query: str = Form(...), top_k: int = Form(8)):
    """
    Retrieve + generate answer using local QA orchestrator.

    POST form fields:
      - query: the user question (required)
      - top_k: number of candidate chunks to retrieve (optional)
    """
    global _QA_ORCHESTRATOR
    if not query:
        raise HTTPException(status_code=400, detail="Missing 'query' field")

    try:
        # Lazy build orchestrator (so startup remains fast & robust)
        if _QA_ORCHESTRATOR is None:
            from app.core.retriever import Retriever
            from app.core.context_builder import ContextBuilder
            from app.core.llm_client import LLMClient
            from app.core.qa_prompt import QAOrchestrator

            retriever = Retriever()
            ctx_builder = ContextBuilder()
            llm_client = LLMClient()
            _QA_ORCHESTRATOR = QAOrchestrator(retriever=retriever, context_builder=ctx_builder, llm_client=llm_client)

        # orchestrator.ask is async — call directly
        result = await _QA_ORCHESTRATOR.ask(query, top_k=top_k)
        return {"ok": True, "result": result}
    except Exception as e:
        logger.exception("QA failed: %s", e)
        return JSONResponse({"ok": False, "message": str(e)}, status_code=500)


# New: simple answer-only endpoint -> returns only the answer text (no metadata)
@app.post("/answer")
async def answer(query: str = Form(...), top_k: int = Form(8)):
    """
    Higher-level endpoint for user-facing answers. Returns minimal JSON:
      { "ok": True, "answer": "<text>" }
    This reuses the same QAOrchestrator but strips metadata from the response.
    """
    global _QA_ORCHESTRATOR
    if not query:
        raise HTTPException(status_code=400, detail="Missing 'query' field")

    try:
        # Lazy build orchestrator if necessary
        if _QA_ORCHESTRATOR is None:
            from app.core.retriever import Retriever
            from app.core.context_builder import ContextBuilder
            from app.core.llm_client import LLMClient
            from app.core.qa_prompt import QAOrchestrator

            retriever = Retriever()
            ctx_builder = ContextBuilder()
            llm_client = LLMClient()
            _QA_ORCHESTRATOR = QAOrchestrator(retriever=retriever, context_builder=ctx_builder, llm_client=llm_client)

        result = await _QA_ORCHESTRATOR.ask(query, top_k=top_k)

        # The orchestrator returns a dict like {"answer": "...", ...}
        # Be defensive: look for 'answer' or fallback to 'result["answer_text"]' or raw LLM text.
        answer_text = None
        if isinstance(result, dict):
            answer_text = result.get("answer") or result.get("answer_text") or result.get("query_text") or ""
            # If the orchestration put LLM output under 'raw_llm', try to extract
            if not answer_text:
                raw = result.get("raw_llm") or result.get("raw")
                if isinstance(raw, dict):
                    answer_text = raw.get("text") or raw.get("message", {}).get("content") if raw.get("message") else None
        if answer_text is None:
            answer_text = ""

        return {"ok": True, "answer": answer_text}
    except Exception as e:
        logger.exception("Answer endpoint failed: %s", e)
        return JSONResponse({"ok": False, "message": str(e)}, status_code=500)
