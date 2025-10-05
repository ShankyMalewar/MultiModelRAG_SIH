# app/core/schema.py
"""
Canonical schema dataclasses for Asklyne Offline.

Provides:
 - ChunkDoc: a single chunk (text + optional embedding + metadata)
 - Citation: simple citation metadata
 - RetrievalHit: a retrieved chunk + score
 - IngestResult: summary returned after an ingestion job
 - helpers: chunkdoc_to_payload, chunkdoc_to_point
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Tuple, Dict, Any
import uuid
import time
import json


def _now_ts() -> float:
    return time.time()


@dataclass
class ChunkDoc:
    id: str
    text: str
    embedding: Optional[List[float]] = None
    modality: str = "text"
    source_path: Optional[str] = None
    filename: Optional[str] = None
    page_num: Optional[int] = None
    bbox: Optional[Tuple[int, int, int, int]] = None
    ts_start: Optional[float] = None
    ts_end: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @classmethod
        
    def create(
        cls,
        text: str,
        modality: str = "text",
        source_path: Optional[str] = None,
        filename: Optional[str] = None,
        page_num: Optional[int] = None,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        ts_start: Optional[float] = None,
        ts_end: Optional[float] = None,
        meta: Optional[Dict[str, Any]] = None,
        id: Optional[str] = None,
    ) -> "ChunkDoc":
        """
        Factory helper to create a ChunkDoc with a generated id and default meta fields.
        Ensures id is a UUID string (with dashes), compatible with Qdrant.
        """
        import uuid as _uuid
        cid = id or str(_uuid.uuid4())  # use dashed UUID string (e.g. "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx")
        m = dict(meta or {})
        m.setdefault("created_at", _now_ts())
        return cls(
            id=cid,
            text=text,
            embedding=None,
            modality=modality,
            source_path=source_path,
            filename=filename,
            page_num=page_num,
            bbox=bbox,
            ts_start=ts_start,
            ts_end=ts_end,
            meta=m,
        )


    def to_payload(self) -> Dict[str, Any]:
        """
        Convert chunk into a JSON-serializable payload (for vector DB storage).
        Note: embedding is returned separately by helpers that create 'point' objects.
        """
        payload = {
            "text": self.text,
            "modality": self.modality,
            "source_path": self.source_path,
            "filename": self.filename,
            "page_num": self.page_num,
            "bbox": list(self.bbox) if self.bbox else None,
            "ts_start": self.ts_start,
            "ts_end": self.ts_end,
            "meta": self.meta or {},
        }
        # Remove keys with None values for compactness
        return {k: v for k, v in payload.items() if v is not None}

    def to_point(self) -> Tuple[str, List[float], Dict[str, Any]]:
        """
        Return a tuple suitable for Qdrant upsert: (id, vector, payload).
        Raises ValueError if embedding is missing.
        """
        if self.embedding is None:
            raise ValueError("ChunkDoc.embedding is None; cannot convert to point without vector.")
        return (self.id, list(self.embedding), self.to_payload())

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # ensure bbox serializable
        if d.get("bbox") and isinstance(d["bbox"], (list, tuple)):
            d["bbox"] = list(d["bbox"])
        return d

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"ChunkDoc(id={self.id}, len_text={len(self.text)}, modality={self.modality})"


@dataclass
class Citation:
    source_id: Optional[str] = None
    text: Optional[str] = None
    score: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"source_id": self.source_id, "text": self.text, "score": self.score, "meta": self.meta}


@dataclass
class RetrievalHit:
    chunk: ChunkDoc
    score: float

    def to_dict(self) -> Dict[str, Any]:
        return {"chunk": self.chunk.to_dict(), "score": float(self.score)}


@dataclass
class IngestResult:
    file_path: str
    file_name: str
    chunks_created: int = 0
    chunk_ids: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "file_name": self.file_name,
            "chunks_created": self.chunks_created,
            "chunk_ids": list(self.chunk_ids or []),
            "errors": list(self.errors or []),
        }

    def __repr__(self) -> str:
        return f"IngestResult(file_name={self.file_name}, chunks_created={self.chunks_created}, errors={self.errors})"


# Small helper: convert ChunkDoc to a payload dict (alias)
def chunkdoc_to_payload(chunk: ChunkDoc) -> Dict[str, Any]:
    return chunk.to_payload()


# Small helper: convert to the "point" tuple: (id, vector, payload)
def chunkdoc_to_point(chunk: ChunkDoc) -> Tuple[str, List[float], Dict[str, Any]]:
    return chunk.to_point()
