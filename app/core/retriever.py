# app/core/retriever.py
"""
Robust Retriever for the project.

Key fixes included:
- Defensively normalizes embedder output (numpy / torch -> list of floats)
- Explicit session_id handling via filter_meta
- Clear, consistent logging and safe failures returning empty lists
- Uses QdrantService.search_by_vector (REST or client) and normalizes results
- Keeps optional Typesense + Reranker support but tolerates their absence

Drop this file into app/core/retriever.py (replace existing file).
"""
from typing import Any, Dict, List, Optional
import logging

from app.core.embedder import Embedder
try:
    from app.core.reranker import Reranker
    _has_reranker = True
except Exception:
    Reranker = None
    _has_reranker = False

try:
    from app.services.typesense_service import TypesenseService
    _has_typesense = True
except Exception:
    TypesenseService = None
    _has_typesense = False

from app.services.qdrant_service import QdrantService

LOG = logging.getLogger("asklyne.retriever")


class Retriever:
    def __init__(self, qdrant: Optional[QdrantService] = None, embedder: Optional[Embedder] = None, reranker: Optional[Any] = None):
        self.qdrant = qdrant or QdrantService()
        self.embedder = embedder or Embedder()
        self.reranker = reranker or (Reranker() if _has_reranker else None)
        self.typesense = TypesenseService() if _has_typesense else None

    def _ensure_list(self, vec: Any) -> Optional[List[float]]:
        """Convert numpy/torch arrays and other acceptable types into a python list of floats."""
        if vec is None:
            return None
        # prefer tolist when available
        if hasattr(vec, "tolist"):
            try:
                vec = vec.tolist()
            except Exception:
                pass
        if isinstance(vec, (list, tuple)):
            try:
                return [float(x) for x in vec]
            except Exception:
                LOG.exception("Failed converting embedding sequence to list of floats")
                return None
        LOG.warning("Embedding produced unsupported type: %s", type(vec))
        return None

    def _build_effective_filter(self, filter_meta: Optional[Dict[str, Any]], session_id: Optional[str]) -> Optional[Dict[str, Any]]:
        """Combine provided filter_meta and an optional session_id into the Qdrant filter shape.
        Keeps an existing bool-style filter if provided; otherwise builds one with 'must'.
        """
        if not filter_meta and not session_id:
            return None
        effective = (filter_meta or {}).copy()
        # normalize to top-level 'must' list when empty
        if not effective:
            effective = {"must": []}
        # ensure 'must' exists
        if "must" not in effective:
            effective = {"must": [effective]} if isinstance(effective, dict) else {"must": []}
        if session_id:
            must_list = effective.setdefault("must", [])
            # avoid duplicate session filters
            if not any(isinstance(m, dict) and m.get("key") == "session_id" for m in must_list):
                must_list.append({"key": "session_id", "match": {"value": session_id}})
        return effective

    def _normalize_qdrant_hits(self, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize Qdrant hit dicts to a consistent shape used by the rest of the app."""
        out: List[Dict[str, Any]] = []
        for h in hits or []:
            try:
                out.append({
                    "id": h.get("id"),
                    "score": float(h.get("score", 0.0)) if h.get("score") is not None else 0.0,
                    "payload": h.get("payload", {}),
                    "text": (h.get("payload") or {}).get("text", "")
                })
            except Exception:
                LOG.exception("Failed normalizing qdrant hit: %s", h)
        return out

    def retrieve(self, query: str, top_k: int = 10, filter_meta: Optional[Dict[str, Any]] = None, session_id: Optional[str] = None, with_payload: bool = True) -> List[Dict[str, Any]]:
        """Embed the query, call Qdrant (and optional Typesense), merge + optional rerank and return hits.

        Returns an empty list on failure (caller should handle empty results).
        """
        LOG.debug("Retriever.retrieve called: query=%s top_k=%s session_id=%s", query, top_k, session_id)

        # 1) Embed
        try:
            q_vec_raw = self.embedder.embed_text(query)
        except Exception as e:
            LOG.exception("Embedder failed: %s", e)
            return []

        q_vec = self._ensure_list(q_vec_raw)
        if q_vec is None:
            LOG.error("Embedding could not be converted to list - aborting retrieval")
            return []

        # 2) Build filter
        effective_filter = self._build_effective_filter(filter_meta, session_id)

        # 3) Qdrant semantic search
        try:
            qdrant_hits = self.qdrant.search_by_vector(q_vec, top_k=top_k, filter_meta=effective_filter, with_payload=with_payload)
        except Exception as e:
            LOG.exception("Qdrant search failed: %s", e)
            qdrant_hits = []

        semantic_candidates = self._normalize_qdrant_hits(qdrant_hits)

        # 4) Optional keyword search (Typesense)
        keyword_candidates: List[Dict[str, Any]] = []
        if self.typesense:
            try:
                keyword_candidates = self.typesense.search(query, top_k=top_k, filters=effective_filter) or []
            except Exception:
                LOG.exception("Typesense search failed")
                keyword_candidates = []

        # 5) Merge + dedupe
        merged = []
        seen_texts = set()
        for lst in (semantic_candidates, keyword_candidates):
            for item in lst:
                t = (item.get("text") or "").strip()
                if not t:
                    continue
                if t in seen_texts:
                    continue
                seen_texts.add(t)
                merged.append(item)

        # 6) Optional rerank
        try:
            if self.reranker is not None and getattr(self.reranker, "enabled", True):
                reranked = self.reranker.rerank(query, merged)
                if isinstance(reranked, list):
                    merged = reranked
                else:
                    LOG.warning("Reranker returned unexpected output; skipping rerank")
        except Exception:
            LOG.exception("Reranker failed; returning pre-reranked merged results")

        # 7) Final trim
        return merged[:top_k]
