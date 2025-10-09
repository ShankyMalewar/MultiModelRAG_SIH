# app/core/qa_prompt.py
"""
QA orchestration and prompt templates for Asklyne Offline.

Responsibilities:
- Build grounded prompts from retrieved chunks
- Call local LLM via LLMClient
- Post-process LLM output: extract citations, list used chunk ids, return structured result

This version is session-aware: QAOrchestrator.ask accepts an optional session_id
which is forwarded to the retriever so retrieval is scoped to a single session.
"""

from typing import List, Dict, Any, Optional, Tuple
import re
import asyncio
import logging
from starlette.concurrency import run_in_threadpool

logger = logging.getLogger("asklyne.qa")

# Default system prompt: strict grounding
DEFAULT_SYSTEM_PROMPT = (
    "You are an assistant that must answer questions using only the information provided in the CONTEXT section.And can use some of your own brain too to structure the output in chat gpt style "
    "Do NOT invent facts. If the answer cannot be found in the CONTEXT, say exactly: \"I don't know\". "
    "Be friendly yet professional."
    "Keep answers in mid range between consise and elaborative. If asked for more details, provide a short summary first and then an optional expanded section."
)

# User prompt template: injected with question + context block
USER_PROMPT_TEMPLATE = (
    "QUESTION:\n{question}\n\n"
    "CONTEXT (use ONLY this information to answer):\n{context}\n\n"
    "INSTRUCTIONS:\n"
    "1) Answer the QUESTION using only the CONTEXT above.\n"
    "2) If you cannot answer fully, respond with exactly: \"I guess that's something which is not in the resources you provided!\".\n"
    "3) Keep answer length limited to ~200 words unless the user asks for more.\n"
)

CHUNK_FORMAT = "SOURCE: {filename}{page}{ts}{modality} | SCORE: {score}\nID: {id}\n{snippet}\n"

CITATION_RE = re.compile(
    r"\[source:\s*([^,\]\n]+)(?:,\s*page:\s*([0-9]+))?(?:,\s*ts:\s*([0-9:. -]+))?\]",
    flags=re.IGNORECASE,
)


def format_chunk_for_context(chunk: Dict[str, Any]) -> str:
    """
    Render a single chunk into a human-readable block inserted to the prompt context.
    Expects chunk has 'payload' or 'text' and some metadata fields (filename, page_num, ts_start).
    """
    payload = chunk.get("payload", {}) if isinstance(chunk, dict) else {}
    text = chunk.get("text", "") or payload.get("text", "") or payload.get("snippet", "")
    filename = payload.get("filename") or payload.get("source_path") or chunk.get("filename") or "unknown"
    page_num = payload.get("page_num") or chunk.get("page_num")
    ts_start = payload.get("ts_start") or chunk.get("ts_start")
    modality = payload.get("modality") or chunk.get("modality") or ""
    score = chunk.get("score", 0.0)

    page_str = f" page: {page_num}" if page_num is not None else ""
    ts_str = ""
    if ts_start is not None:
        ts_end = payload.get("ts_end") or chunk.get("ts_end")
        if ts_end is not None:
            try:
                ts_str = f", ts: {float(ts_start):.2f}-{float(ts_end):.2f}s"
            except Exception:
                ts_str = f", ts: {ts_start}-{ts_end}"
        else:
            try:
                ts_str = f", ts: {float(ts_start):.2f}s"
            except Exception:
                ts_str = f", ts: {ts_start}s"

    snippet = (text.strip()[:1200] + "...") if text and len(text) > 1200 else (text or "")

    return CHUNK_FORMAT.format(
        filename=filename,
        page=page_str,
        ts=ts_str,
        modality=f" modality: {modality}" if modality else "",
        score=f"{float(score):.4f}",
        id=chunk.get("id") or payload.get("id") or "unknown-id",
        snippet=snippet,
    )


def build_context_text(chunks: List[Dict[str, Any]], max_chunks: int = 8) -> str:
    """
    Convert top-ranked chunks into a bounded context string for the LLM prompt.
    Keeps up to max_chunks most relevant chunks (ordered by score desc).
    """
    if not chunks:
        return ""
    sorted_chunks = sorted(chunks, key=lambda c: -float(c.get("score", 0.0)))
    selected = sorted_chunks[:max_chunks]
    blocks = [format_chunk_for_context(c) for c in selected]
    return "\n---\n".join(blocks)


def extract_citations(text: str) -> List[Dict[str, Optional[str]]]:
    """
    Parse inline citations matching the pattern used in the prompt.
    Returns list of dicts: {"source": <filename>, "page": <page or None>, "ts": <ts or None>}
    """
    out = []
    for m in CITATION_RE.finditer(text):
        source = (m.group(1) or "").strip()
        page = (m.group(2) or None)
        ts = (m.group(3) or None)
        out.append({"source": source, "page": page, "ts": ts})
    return out


class QAOrchestrator:
    """
    High-level orchestrator that:
        - accepts a query
        - uses Retriever to get candidate chunks
        - builds a context using ContextBuilder and formatting helpers
        - calls LLMClient to generate an answer
        - post-processes to extract citations and used chunk ids

    Now accepts optional session_id in ask() to scope retrieval to a session.
    """

    def __init__(
        self,
        retriever: Any,
        context_builder: Any,
        llm_client: Any,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        user_template: str = USER_PROMPT_TEMPLATE,
        max_context_chunks: int = 8,
        llm_max_tokens: int = 512,
        llm_temperature: float = 0.0,
    ):
        self.retriever = retriever
        self.context_builder = context_builder
        self.llm = llm_client
        self.system_prompt = system_prompt
        self.user_template = user_template
        self.max_context_chunks = max_context_chunks
        self.llm_max_tokens = llm_max_tokens
        self.llm_temperature = llm_temperature

    def _assemble_messages(self, question: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Create message list for LLMClient.generate() (system + user).
        If context_builder provides `build()` that accepts chunks and returns text, prefer that,
        otherwise use our `build_context_text`.
        Also ensures the context isn't absurdly huge by truncating to reasonable size.
        """
        context_text = ""

        try:
            if hasattr(self.context_builder, "build"):
                maybe = self.context_builder.build(chunks)
                if isinstance(maybe, str) and maybe.strip():
                    context_text = maybe
        except Exception:
            logger.debug("context_builder.build failed; falling back to local formatter", exc_info=True)

        if not context_text:
            context_text = build_context_text(chunks, max_chunks=self.max_context_chunks)

        # Safety: limit prompt length (very important for local LLMs). Keep about ~3000-8000 characters.
        if len(context_text) > 2800:
            logger.debug("Truncating context from %d to 2800 chars", len(context_text))
            context_text = context_text[:2800] + "\n... [TRUNCATED]"

        user_prompt = self.user_template.format(question=question, context=context_text)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return messages

    async def _call_llm(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Call the LLM client asynchronously and return normalized response
        """
        # log the outgoing prompt (useful during debugging; may be large)
        try:
            logger.debug("LLM messages (system + user): %s", messages)
        except Exception:
            logger.debug("LLM messages not printable")

        resp = await self.llm.generate(messages=messages, max_tokens=self.llm_max_tokens, temperature=self.llm_temperature)
        try:
            logger.debug("LLM raw response: %s", getattr(resp, "get", lambda k, d=None: None)("raw", resp))
        except Exception:
            pass
        return resp

    async def ask(self, question: str, top_k: int = 5, filter_meta: Optional[Dict[str, Any]] = None, session_id: Optional[str] = None):
        """
        Retrieve candidate chunks, build a compact context, call the LLM and return a clean answer.
        session_id: optional string to restrict retrieval to a single session's uploaded chunks.
        Returns structure:
          {"ok": True, "query_text": question, "answer": <text>, "used_chunk_ids": [...], "citations": [...]}
        """
        try:
            # 1) retrieve candidates (runs blocking retriever in threadpool)
            # retriever.retrieve should return list of dicts: {"id": ..., "score": ..., "payload": {...}, "text": ...}
            candidates: List[Dict[str, Any]] = await run_in_threadpool(
                lambda: self.retriever.retrieve(question, top_k=top_k, filter_meta=filter_meta, session_id=session_id)
            )

            if not candidates:
                return {"ok": True, "query_text": question, "answer": "I don't know.", "used_chunk_ids": [], "citations": []}

            # Optional: dedupe by payload text or hash (avoid very similar chunks)
            seen_text = set()
            deduped = []
            for c in candidates:
                txt = (c.get("payload", {}).get("text") or c.get("text") or "").strip()
                key = txt[:800]  # use prefix to dedupe
                if not key or key in seen_text:
                    continue
                seen_text.add(key)
                deduped.append(c)
                if len(deduped) >= top_k:
                    break
            candidates = deduped

            # 2) Build context with your custom ContextBuilder (lazy import to avoid import-time cycles)
            try:
                from app.core.context_builder import ContextBuilder  # type: ignore
            except Exception:
                ContextBuilder = None

            if ContextBuilder and hasattr(ContextBuilder, "build_context_from_candidates"):
                cb = ContextBuilder(max_total_chars=3000, per_chunk_chars=800)
                try:
                    context = cb.build_context_from_candidates(
                        candidates, question, max_total_chars=3000, per_chunk_chars=800
                    )
                except Exception:
                    logger.debug("ContextBuilder.build_context_from_candidates failed; using simple build_context_text", exc_info=True)
                    context = build_context_text(candidates, max_chunks=self.max_context_chunks)
            else:
                # fallback
                context = build_context_text(candidates, max_chunks=self.max_context_chunks)

            # 3) Build messages using assembled helper (so formatting & truncation handled)
            messages = self._assemble_messages(question, candidates)
            logger.info("Retrieved %d candidates. First text preview: %s", len(candidates), candidates[0].get("text")[:300] if candidates else "NONE")


            # 4) Call LLM
            llm_out = await self._call_llm(messages)

            # Extract text from llm_out safely
            answer_text = None
            if isinstance(llm_out, dict):
                answer_text = llm_out.get("text")
                if not answer_text:
                    raw = llm_out.get("raw")
                    if isinstance(raw, dict):
                        # try common keys
                        answer_text = raw.get("message", {}).get("content") or raw.get("text") or raw.get("_raw_text")
                    else:
                        answer_text = str(raw)
            answer_text = (answer_text or "").strip()
            if not answer_text:
                answer_text = "I don't know."

            # 5) build citations list from used candidates (one per used chunk)
            citations = []
            for c in candidates:
                payload = c.get("payload", {})
                filename = payload.get("filename") or payload.get("source_path") or "uploaded_pdf"
                # page info may be nested in meta
                page = None
                meta = payload.get("meta")
                if isinstance(meta, dict):
                    page = meta.get("page") or meta.get("page_num")
                citations.append({"source": filename, "page": page})

            # Optionally extract inline citations that the LLM may have produced
            inline_citations = extract_citations(answer_text)

            return {
                "ok": True,
                "query_text": question,
                "answer": answer_text,
                "used_chunk_ids": [c.get("id") for c in candidates],
                "citations": citations,
                "inline_citations": inline_citations,
                "raw_llm": llm_out,
                "retrieved": candidates,
            }
        except Exception as e:
            # keep errors visible
            logger.exception("QA ask failed: %s", e)
            return {"ok": False, "message": str(e)}


# -------------------------
# Convenience sync wrapper
# -------------------------
def run_qa_sync(orchestrator: QAOrchestrator, question: str, top_k: int = 12, filter_meta: Optional[Dict[str, Any]] = None, session_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Synchronous wrapper to run the orchestrator.ask coroutine.
    Useful for quick CLI/script testing.
    """
    return asyncio.run(orchestrator.ask(question, top_k=top_k, filter_meta=filter_meta, session_id=session_id))


# -------------------------
# Quick test harness (manual)
# -------------------------
if __name__ == "__main__":
    """
    Basic manual run:
    - expects to be run from project root where app package is importable
    - this will create simple instances if available: Retriever, ContextBuilder, LLMClient
    """
    import os

    # Lazy imports so module can be used even if some pieces are missing
    try:
        from app.core.retriever import Retriever
    except Exception:
        Retriever = None

    try:
        from app.core.context_builder import ContextBuilder
    except Exception:
        ContextBuilder = None

    try:
        from app.core.llm_client import LLMClient
    except Exception:
        LLMClient = None

    # Build minimal components or raise helpful error
    if Retriever is None or ContextBuilder is None or LLMClient is None:
        logger.error("Cannot run quick test: missing Retriever/ContextBuilder/LLMClient implementations in your environment.")
        logger.error("Ensure app.core.retriever, app.core.context_builder and app.core.llm_client exist and are importable.")
        raise SystemExit(1)

    retriever = Retriever()
    context_builder = ContextBuilder()
    llm = LLMClient()

    orchestrator = QAOrchestrator(retriever=retriever, context_builder=context_builder, llm_client=llm)

    q = os.getenv("TEST_QUERY", "What is the main point of the first document?")
    # optional session_id from env for manual testing
    sid = os.getenv("TEST_SESSION_ID")
    print("Running QA for question:", q, "session_id:", sid)
    try:
        result = run_qa_sync(orchestrator, q, top_k=8, session_id=sid)
        print("=== ANSWER ===")
        print(result.get("answer"))
        print("\n=== CITATIONS ===")
        print(result.get("citations"))
        print("\n=== USED CHUNKS ===")
        print(result.get("used_chunk_ids"))
        print("\n=== TOP RETRIEVED ===")
        for r in result.get("retrieved", [])[:6]:
            print(f"- id={r.get('id')} score={r.get('score')} filename={r.get('payload',{}).get('filename')}")
    except Exception as e:
        logger.exception("QA test run failed: %s", e)
