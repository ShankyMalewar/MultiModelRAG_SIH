# app/services/typesense_service.py
"""
Minimal TypesenseService stub for local dev.
This provides the same interface shape expected by Retriever:
- search(query, top_k=10, filters=None) -> list of candidate dicts similar to qdrant hits
For now it returns an empty list (no keyword results).
Replace with a real Typesense client if/when you run Typesense.
"""

from typing import List, Dict, Any, Optional
import os
import logging

logger = logging.getLogger(__name__)

class TypesenseService:
    def __init__(self, host: Optional[str] = None, api_key: Optional[str] = None):
        # Keep minimal: read from env if present
        self.host = host or os.getenv("TYPESENSE_HOST", "http://localhost:8108")
        self.api_key = api_key or os.getenv("TYPESENSE_API_KEY", "xyz")
        logger.info(f"TypesenseService initialized (stub). host={self.host}")

    def search(self, query: str, top_k: int = 10, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Stubbed search: return empty list. Real implementation should return:
        [
          {"id": "...", "score": 1.0, "payload": {"text": "...", "source_path": "..."}, "text": "..."}, ...
        ]
        """
        logger.debug(f"TypesenseService.search called (stub) query='{query}' top_k={top_k} filters={filters}")
        return []
