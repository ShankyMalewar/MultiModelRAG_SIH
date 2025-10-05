from typing import List, Dict, Optional, Any
import os
import logging

logger = logging.getLogger(__name__)

class Reranker:
    """Cross-Encoder reranker (optional)."""

    def __init__(self, model_name: Optional[str] = None):
        self.enabled = os.getenv("RERANKER_ENABLED", "false").lower() in ("1", "true", "yes")
        self.model_name = model_name or os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self._model = None
        if not self.enabled:
            logger.debug("Reranker disabled (set RERANKER_ENABLED=1 to enable).")

    def _ensure_model(self) -> None:
        if not self.enabled or self._model:
            return
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        except Exception as e:
            logger.exception("Failed to initialize reranker model: %s", e)
            self._model = None

    def rerank(self, query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.enabled:
            return chunks
        self._ensure_model()
        if not self._model:
            return chunks

        pairs = [(query, c.get("text", "")) for c in chunks]
        if not pairs:
            return chunks

        try:
            scores = self._model.predict(pairs)
            for i, score in enumerate(scores):
                chunks[i]["score"] = float(score)
            return sorted(chunks, key=lambda x: -x.get("score", 0.0))
        except Exception as e:
            logger.exception("Reranker predict failed: %s", e)
            return chunks
