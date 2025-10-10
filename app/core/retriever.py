from typing import Any, Dict, List, Optional, Set
import logging
from app.core.embedder import Embedder
from app.services.qdrant_service import QdrantService

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

LOG = logging.getLogger("asklyne.retriever")
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)


class Retriever:
    AUDIO_CLAUSES = [
        {"key": "modality", "match": {"value": "audio"}},
        {"key": "type", "match": {"value": "audio"}},
        {"key": "is_audio", "match": {"value": True}},
        {"key": "media_type", "match": {"value": "audio"}},
    ]

    AUDIO_KEYWORDS = ["audio", "sound", "listen", "hear", "speech", "transcript", "voice"]

    def __init__(self, qdrant: Optional[QdrantService] = None, embedder: Optional[Embedder] = None, reranker: Optional[Any] = None):
        self.qdrant = qdrant or QdrantService()
        self.embedder = embedder or Embedder()
        self.reranker = reranker or (Reranker() if _has_reranker else None)
        self.typesense = TypesenseService() if _has_typesense else None

    def _ensure_list(self, vec: Any) -> Optional[List[float]]:
        if vec is None:
            return None
        if hasattr(vec, "tolist"):
            vec = vec.tolist()
        if isinstance(vec, (list, tuple)):
            try:
                return [float(x) for x in vec]
            except Exception:
                return None
        return None

    def _normalize_hits(self, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for h in hits or []:
            out.append({
                "id": h.get("id"),
                "score": h.get("score", 0.0),
                "payload": h.get("payload", {}),
                "text": (h.get("payload") or {}).get("text", "")
            })
        return out

    def _build_filter(self, session_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if not session_id:
            return None
        return {"must": [{"key": "session_id", "match": {"value": session_id}}]}

    def _query_is_audio(self, query: str) -> bool:
        qlow = query.lower()
        return any(word in qlow for word in self.AUDIO_KEYWORDS)

    def retrieve(
        self,
        query: str,
        top_k: int = 8,
        filter_meta: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        with_payload: bool = True,
    ) -> List[Dict[str, Any]]:
        LOG.info("Retriever.retrieve: query='%s' session_id=%s", query, session_id)

        try:
            vec = self.embedder.embed_text(query)
            vec = self._ensure_list(vec)
        except Exception as e:
            LOG.exception("Retriever: embedding failed: %s", e)
            return []

        if vec is None:
            LOG.error("Retriever: invalid embedding vector")
            return []

        base_filter = self._build_filter(session_id) or {"must": []}
        if filter_meta:
            base_filter["must"].extend(filter_meta.get("must", []))

        def _dedupe_musts(musts: List[Dict[str, Any]]):
            seen = set()
            deduped = []
            for m in musts:
                key = m.get("key")
                val = str(m.get("match", {}).get("value"))
                ident = (key, val)
                if ident not in seen:
                    seen.add(ident)
                    deduped.append(m)
            return deduped

        is_audio_query = self._query_is_audio(query)

        if is_audio_query:
            LOG.info("Retriever: running audio-first search (query suggests audio)")
            audio_filter = {"must": [
                {"key": "session_id", "match": {"value": session_id}},
                {"key": "modality", "match": {"value": "audio"}},
            ]}
            audio_filter["must"].extend(base_filter.get("must", []))
            audio_filter["must"] = _dedupe_musts(audio_filter["must"])

            try:
                hits_audio = self.qdrant.search_by_vector(vec, top_k=top_k, filter_meta=audio_filter, with_payload=with_payload)
                audio_hits = self._normalize_hits(hits_audio)
                if audio_hits:
                    LOG.info("Retriever: returning %d audio hits", len(audio_hits))
                    return audio_hits[:top_k]
                else:
                    LOG.warning("Retriever: no audio hits — falling back to general search")
            except Exception as e:
                LOG.warning("Audio-filtered search failed: %s", e)

        try:
            merged_filter = {"must": _dedupe_musts(base_filter.get("must", []))}
            hits = self.qdrant.search_by_vector(vec, top_k=top_k, filter_meta=merged_filter, with_payload=with_payload)
        except Exception as e:
            LOG.exception("Retriever: general Qdrant search failed: %s", e)
            hits = []

        results = self._normalize_hits(hits)

        if self.reranker:
            try:
                results = self.reranker.rerank(query, results)
            except Exception:
                LOG.warning("Reranker failed, skipping rerank")

        return results[:top_k]
