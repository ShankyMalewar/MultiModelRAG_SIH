# app/core/embedder.py
from typing import List, Optional, Dict, Any
import os
import logging

logger = logging.getLogger(__name__)

# cache of loaded sentence-transformers model instances
_loaded_models: Dict[str, Any] = {}

# Default models (you can override via EMBED_MODEL env var)
_DEFAULT_MODELS = {
    "text": os.getenv("EMBED_MODEL_TEXT", "BAAI/bge-large-en-v1.5"),
    "code": os.getenv("EMBED_MODEL_CODE", "microsoft/codebert-base"),
}


class Embedder:
    """
    Lazy-loading Embedder wrapper.

    - Does NOT import sentence_transformers at module import.
    - Instantiates model on first use via _ensure_model().
    """

    def __init__(self, mode: str = "text", model_name: Optional[str] = None):
        self.mode = (mode or "text").lower()
        if self.mode == "notes":
            self.mode = "text"

        self.model_name = model_name or os.getenv("EMBED_MODEL") or _DEFAULT_MODELS.get(self.mode)
        if not self.model_name:
            raise ValueError(
                "No embed model configured. Set EMBED_MODEL or EMBED_MODEL_TEXT/EMBED_MODEL_CODE in env."
            )

        # placeholder for the instance (populated lazily)
        self._model = None

    def _ensure_model(self):
        """Lazy-load and cache the underlying SentenceTransformer model."""
        global _loaded_models
        if self._model is not None:
            return

        if self.model_name in _loaded_models:
            self._model = _loaded_models[self.model_name]
            return

        # heavy import / instantiate happens here (only on demand)
        try:
            logger.info("[Embedder] Loading model %s for mode=%s", self.model_name, self.mode)
            from sentence_transformers import SentenceTransformer  # local import
        except Exception as e:
            logger.exception("Failed to import sentence_transformers: %s", e)
            raise

        model = SentenceTransformer(self.model_name)
        _loaded_models[self.model_name] = model
        self._model = model

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Primary batch embed API (used by doc_ingestor). This will lazily load the model.
        """
        self._ensure_model()
        arr = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return arr.tolist()

    def embed_chunks(self, chunks: List[str]) -> List[List[float]]:
        return self.embed_batch(chunks)

    def embed_text(self, text: str) -> List[float]:
        self._ensure_model()
        vec = self._model.encode([text], convert_to_numpy=True)[0]
        return vec.tolist()
