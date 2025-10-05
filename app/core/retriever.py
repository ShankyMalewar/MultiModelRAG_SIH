# app/core/retriever.py
"""
Retriever orchestrates embedding, Qdrant search, optional Typesense keyword search,
merging, deduplication, and reranking with CrossEncoder.
"""

from typing import List, Dict, Any, Optional
from loguru import logger

from app.core.embedder import Embedder
from app.core.reranker import Reranker
from app.services.qdrant_service import QdrantService

try:
    from app.services.typesense_service import TypesenseService
    _has_typesense = True
except Exception:
    TypesenseService = None
    _has_typesense = False


class Retriever:
    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        reranker: Optional[Reranker] = None,
        qdrant: Optional[QdrantService] = None
    ):
        self.embedder = embedder or Embedder()
        self.reranker = reranker or Reranker()
        self.qdrant = qdrant or QdrantService()
        self.typesense = TypesenseService() if _has_typesense else None

    def _merge_and_dedupe(self, lists: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Merge and deduplicate retrieval lists."""
        seen = set()
        merged: List[Dict[str, Any]] = []
        for lst in lists:
            for item in lst:
                payload = item.get("payload", {})
                source = payload.get("source_path") or payload.get("filename") or ""
                page = payload.get("page_num")
                snippet = (payload.get("text") or payload.get("snippet") or "")[:200]
                key = f"{source}||{page}||{snippet}"
                if key in seen:
                    continue
                seen.add(key)
                merged.append(item)
        merged.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        return merged

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filter_meta: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Full retrieval pipeline: embed → Qdrant → (Typesense) → merge → rerank."""
        query_vec = self.embedder.embed_text(query)

        # --- Qdrant semantic search ---
        try:
            qdrant_hits = self.qdrant.search_by_vector(query_vec, top_k=top_k, filter_meta=filter_meta, with_payload=True)
        except Exception as e:
            logger.exception("Qdrant search failed: %s", e)
            qdrant_hits = []

        semantic_candidates = [
            {
                "id": h.get("id"),
                "score": h.get("score", 0.0),
                "payload": h.get("payload", {}),
                "text": h.get("payload", {}).get("text", ""),
            }
            for h in qdrant_hits
        ]

        # --- Typesense keyword search (optional) ---
        keyword_candidates = []
        if self.typesense:
            try:
                keyword_candidates = self.typesense.search(query, top_k=top_k, filters=filter_meta) or []
            except Exception as e:
                logger.exception("Typesense search failed: %s", e)

        # --- Merge results ---
        merged = self._merge_and_dedupe([semantic_candidates, keyword_candidates])

        # --- Rerank ---
        try:
            if self.reranker.enabled:
                reranked = self.reranker.rerank(query, merged)
                if not isinstance(reranked, list):
                    logger.warning("Reranker returned non-list output; using merged results.")
                    reranked = merged
            else:
                reranked = merged
        except Exception as e:
            logger.exception("Reranker failed: %s", e)
            reranked = merged

        # --- Final normalization ---
        unique = []
        seen = set()
        for c in reranked:
            t = c.get("text", "").strip()
            if t and t not in seen:
                seen.add(t)
                unique.append(c)

        reranked = unique[:6]  # keep top 6 unique chunks only
        return reranked
