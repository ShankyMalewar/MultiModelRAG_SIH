# app/ingest/image_ingestor.py
"""
Image ingestor for Asklyne Offline (session-aware).
Reworked to produce explicit upsert payloads matching QdrantService.upsert_chunks:
    {"id": <optional>, "embedding": <list[float]>, "payload": {...}}
Ensures payload includes modality/media_type and session_id (if passed).
"""

from typing import List, Optional, Dict, Any
import os
import logging
from PIL import Image

from app.extractors.ocr_handler import OCRHandler
from app.core.chunker import Chunker
from app.core.embedder import Embedder
from app.services.qdrant_service import QdrantService
from app.core.schema import ChunkDoc, IngestResult

logger = logging.getLogger("asklyne.ingest.image")
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)


class ImageIngestor:
    def __init__(
        self,
        session_id: Optional[str] = None,
        embed_mode: str = "text",
        chunk_token_size: int = 512,
        chunk_overlap: int = 50,
        qdrant: Optional[QdrantService] = None,
        ocr: Optional[OCRHandler] = None,
    ):
        self.session_id = session_id
        self.embed_mode = embed_mode
        # local import style retained to avoid circulars
        from app.core.chunker import Chunker as _Chunker
        from app.core.embedder import Embedder as _Embedder

        self.chunker = _Chunker(max_tokens=chunk_token_size, overlap=chunk_overlap, file_type="text")
        self.embedder = _Embedder(mode=self.embed_mode)
        self.qdrant = qdrant or QdrantService()
        self.ocr = ocr or OCRHandler()

    def ingest_image(self, image_path: str, filename: Optional[str] = None) -> IngestResult:
        filename = filename or os.path.basename(image_path)
        chunks: List[ChunkDoc] = []
        errors: List[str] = []

        try:
            # 1) OCR blocks
            ocr_blocks = self._ocr_blocks_from_image(image_path)
            if not ocr_blocks:
                logger.debug("No OCR blocks; using fallback full-image OCR/caption.")
                full_text = self.ocr.extract_text_from_image(image_path)
                if full_text and full_text.strip():
                    ocr_blocks = [{"text": full_text.strip(), "bbox": None}]
                else:
                    ocr_blocks = [{"text": "", "bbox": None}]

            # 2) Create chunks from blocks
            for idx, block in enumerate(ocr_blocks):
                text = (block.get("text") or "").strip()
                bbox = block.get("bbox")
                if not text:
                    continue
                text_chunks = self.chunker.chunk_text(text) if hasattr(self.chunker, "chunk_text") else [text]
                for n, tc in enumerate(text_chunks):
                    meta: Dict[str, Any] = {
                        "ocr_block_index": idx,
                        "ocr_chunk_index": n,
                        "modality": "image",
                        "media_type": "image",
                        "is_image": True,
                    }
                    if self.session_id:
                        meta["session_id"] = self.session_id

                    chunk = ChunkDoc.create(
                        text=tc,
                        modality="image",
                        source_path=image_path,
                        filename=filename,
                        page_num=1,
                        bbox=bbox,
                        meta=meta,
                    )
                    chunks.append(chunk)

            # 3) Caption fallback for no chunks
            if not chunks:
                caption = self._caption_fallback(image_path)
                snippet = caption or f"[image: {filename}]"
                meta = {"caption_fallback": bool(caption), "modality": "image", "media_type": "image", "is_image": True}
                if self.session_id:
                    meta["session_id"] = self.session_id
                chunk = ChunkDoc.create(
                    text=snippet,
                    modality="image",
                    source_path=image_path,
                    filename=filename,
                    page_num=1,
                    bbox=None,
                    meta=meta,
                )
                chunks.append(chunk)

            # 4) Embed chunks
            texts = [c.text for c in chunks]
            if texts:
                embeddings = self.embedder.embed_chunks(texts)
                for c, emb in zip(chunks, embeddings):
                    # put embedding on attribute consistently
                    try:
                        c.embedding = emb
                    except Exception:
                        # fallback if chunk is dict-like
                        if isinstance(c, dict):
                            c["embedding"] = emb

            # 5) Build explicit upsert payloads expected by QdrantService.upsert_chunks
            upsert_items: List[Dict[str, Any]] = []
            for c in chunks:
                cid = getattr(c, "id", None) or (c.get("id") if isinstance(c, dict) else None)
                emb = getattr(c, "embedding", None) or (c.get("embedding") if isinstance(c, dict) else None)
                text = getattr(c, "text", None) or (c.get("text") if isinstance(c, dict) else "")
                filename_val = getattr(c, "filename", None) or filename
                source_path = getattr(c, "source_path", None) or image_path
                page_num = getattr(c, "page_num", None) or 1
                bbox_val = getattr(c, "bbox", None) or None
                meta_val = getattr(c, "meta", None) or (c.get("meta") if isinstance(c, dict) else {})

                payload = {
                    "text": text,
                    "filename": filename_val,
                    "source_path": source_path,
                    "page_num": page_num,
                    "bbox": bbox_val,
                    **(meta_val or {}),
                }

                # ensure modality/media_type/is_image exist on payload
                payload.setdefault("modality", "image")
                payload.setdefault("media_type", "image")
                payload.setdefault("is_image", True)
                if self.session_id:
                    payload.setdefault("session_id", self.session_id)

                # final item format: embedding + payload (this is what QdrantService expects)
                item = {
                    "id": cid,
                    "embedding": emb,
                    "payload": payload,
                }
                upsert_items.append(item)

            # 6) Upsert using QdrantService
            upserted_count, failed_ids = self.qdrant.upsert_chunks(upsert_items)

            logger.info("Image ingested: file=%s chunks=%d upserted=%s failed=%s",
                        filename, len(chunks), upserted_count, failed_ids)

            ingest_result = IngestResult(
                file_path=image_path,
                file_name=filename,
                chunks_created=len(chunks),
                chunk_ids=[getattr(c, "id", None) or (c.get("id") if isinstance(c, dict) else None) for c in chunks],
                errors=[str(e) for e in []] + failed_ids,
            )
            return ingest_result

        except Exception as e:
            logger.exception("Error ingesting image %s: %s", image_path, e)
            return IngestResult(file_path=image_path, file_name=filename, chunks_created=0, errors=[str(e)])

    # ----------------------------
    # Helpers
    # ----------------------------
    def _ocr_blocks_from_image(self, image_path: str) -> List[dict]:
        try:
            import pytesseract
            from pytesseract import Output
            img = Image.open(image_path).convert("RGB")
            data = pytesseract.image_to_data(img, output_type=Output.DICT)
            blocks: List[dict] = []
            n_boxes = len(data.get("level", []))
            current_block_idx = None
            current_texts = []
            current_bbox = None
            for i in range(n_boxes):
                text = (data.get("text", [""] * n_boxes)[i] or "").strip()
                if not text:
                    continue
                block_num = data.get("block_num", [0] * n_boxes)[i]
                left = data.get("left", [0] * n_boxes)[i]
                top = data.get("top", [0] * n_boxes)[i]
                width = data.get("width", [0] * n_boxes)[i]
                height = data.get("height", [0] * n_boxes)[i]
                bbox = (int(left), int(top), int(left + width), int(top + height))
                if current_block_idx is None:
                    current_block_idx = block_num
                    current_texts = [text]
                    current_bbox = bbox
                elif block_num != current_block_idx:
                    blocks.append({"text": " ".join(current_texts).strip(), "bbox": current_bbox})
                    current_block_idx = block_num
                    current_texts = [text]
                    current_bbox = bbox
                else:
                    current_texts.append(text)
                    if current_bbox:
                        x1 = min(current_bbox[0], bbox[0])
                        y1 = min(current_bbox[1], bbox[1])
                        x2 = max(current_bbox[2], bbox[2])
                        y2 = max(current_bbox[3], bbox[3])
                        current_bbox = (x1, y1, x2, y2)
            if current_block_idx is not None and current_texts:
                blocks.append({"text": " ".join(current_texts).strip(), "bbox": current_bbox})
            return blocks
        except Exception as e:
            logger.exception("OCR data extraction failed for %s: %s. Falling back to OCRHandler.", image_path, e)
            try:
                text = self.ocr.extract_text_from_image(image_path)
                return [{"text": text, "bbox": None}] if text else []
            except Exception:
                return []

    def _caption_fallback(self, image_path: str) -> Optional[str]:
        return None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Image path")
    parser.add_argument("--session", help="session_id", default=None)
    args = parser.parse_args()
    ing = ImageIngestor(session_id=args.session)
    print(ing.ingest_image(args.image))
