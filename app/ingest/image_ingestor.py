# app/ingestors/image_ingestor.py
"""
Image ingestor for Asklyne Offline.

Responsibilities:
 - Accept an image file (png/jpg/heic/...) path
 - Run OCR (OCRHandler) to extract text and bounding boxes
 - Create ChunkDoc objects per OCR block (or whole-image fallback)
 - Embed chunks via Embedder
 - Upsert chunks into Qdrant via QdrantService
 - Return an IngestResult summarizing the ingestion

Notes:
 - This module intentionally keeps things sync to match the rest of the codebase.
 - If you add an offline captioning model later, implement `caption_fallback` to produce
   a short summary for images with little OCR text.
"""

from typing import List, Tuple, Optional
import os
import logging
from PIL import Image

from app.extractors.ocr_handler import OCRHandler
from app.core.chunker import Chunker
from app.core.embedder import Embedder
from app.services.qdrant_service import QdrantService
from app.core.schema import ChunkDoc, IngestResult

logger = logging.getLogger(__name__)


class ImageIngestor:
    def __init__(
        self,
        embed_mode: str = "text",
        chunk_token_size: int = 512,
        chunk_overlap: int = 50,
        qdrant: Optional[QdrantService] = None,
        ocr: Optional[OCRHandler] = None,
    ):
        """
        Args:
            embed_mode: 'text'|'code' (passed to Embedder)
            chunk_token_size: approx token size per chunk
            chunk_overlap: overlap tokens
        """
        self.embed_mode = embed_mode
        # ensure a Chunker instance exists
        from app.core.chunker import Chunker  # local import to avoid circulars during top-level import
        self.chunker = Chunker(max_tokens=chunk_token_size, overlap=chunk_overlap, file_type="text")
        # embedder
        from app.core.embedder import Embedder
        self.embedder = Embedder(mode=self.embed_mode)
        self.qdrant = qdrant or QdrantService()
        self.ocr = ocr or OCRHandler()

    # ----------------------------
    # Public API
    # ----------------------------
    def ingest_image(self, image_path: str, filename: Optional[str] = None) -> IngestResult:
        """
        Ingest a single image file.

        Process:
         - Run OCR to extract text and bounding boxes (line/word level)
         - Group OCR text into reasonable chunks using Chunker
         - Create ChunkDoc objects (include bbox & page_num metadata where possible)
         - Embed chunk text via Embedder
         - Upsert to Qdrant

        Returns:
            IngestResult summarizing ingestion
        """
        filename = filename or os.path.basename(image_path)
        chunks: List[ChunkDoc] = []
        errors: List[str] = []

        try:
            # 1) OCR: we prefer detailed data (bbox, line-level)
            ocr_blocks = self._ocr_blocks_from_image(image_path)
            if not ocr_blocks:
                # fallback: try captioning or whole-image OCR text
                logger.debug("No OCR blocks extracted; using fallback full-image OCR/caption.")
                full_text = self.ocr.extract_text_from_image(image_path)
                if full_text and full_text.strip():
                    ocr_blocks = [{"text": full_text.strip(), "bbox": None}]
                else:
                    # last resort: create an empty chunk with a placeholder snippet
                    ocr_blocks = [{"text": "", "bbox": None}]

            # 2) For each block produce chunks (chunker works on long text)
            for idx, block in enumerate(ocr_blocks):
                text = (block.get("text") or "").strip()
                bbox = block.get("bbox")  # (x1,y1,x2,y2) or None

                if not text:
                    # skip completely empty text blocks unless you want captions
                    continue

                # chunk the text into smaller chunks (so we can embed/answer granularly)
                text_chunks = self.chunker.chunk_text(text) if isinstance(self.chunker, Chunker) else [text]
                for n, tc in enumerate(text_chunks):
                    chunk = ChunkDoc.create(
                        text=tc,
                        modality="image",
                        source_path=image_path,
                        filename=filename,
                        page_num=1,  # single-image ingest: use 1
                        bbox=bbox,
                        meta={
                            "ocr_block_index": idx,
                            "ocr_chunk_index": n,
                        },
                    )
                    chunks.append(chunk)

            # 3) If no chunks were created (sparse OCR), try caption fallback & add as chunk
            if not chunks:
                caption = self._caption_fallback(image_path)
                snippet = caption or f"[image: {filename}]"
                chunk = ChunkDoc.create(
                    text=snippet,
                    modality="image",
                    source_path=image_path,
                    filename=filename,
                    page_num=1,
                    bbox=None,
                    meta={"caption_fallback": bool(caption)},
                )
                chunks.append(chunk)

            # 4) Embed chunks in batch
            texts = [c.text for c in chunks]
            if texts:
                embeddings = self.embedder.embed_chunks(texts)
                # attach embeddings
                for c, emb in zip(chunks, embeddings):
                    c.embedding = emb

            # 5) Upsert to Qdrant
            upserted_count, failed_ids = self.qdrant.upsert_chunks(chunks)
            logger.info(f"Ingested image {filename}: upserted={upserted_count} failed={len(failed_ids)}")

            # 6) Build IngestResult
            ingest_result = IngestResult(
                file_path=image_path,
                file_name=filename,
                chunks_created=len(chunks),
                chunk_ids=[c.id for c in chunks],
                errors=[str(e) for e in errors] + failed_ids,
            )
            return ingest_result

        except Exception as e:
            logger.exception("Error ingesting image %s: %s", image_path, e)
            return IngestResult(file_path=image_path, file_name=filename, chunks_created=0, errors=[str(e)])

    # ----------------------------
    # Helpers
    # ----------------------------
    def _ocr_blocks_from_image(self, image_path: str) -> List[dict]:
        """
        Use OCRHandler to produce a list of blocks with text + bbox.
        Uses pytesseract's image_to_data-like interface (OCRHandler currently uses pytesseract).
        Returns list of: {"text": "...", "bbox": (x1,y1,x2,y2)}
        """
        # We'll use pytesseract directly here to get bbox details (image_to_data)
        try:
            import pytesseract
            from pytesseract import Output
            img = Image.open(image_path).convert("RGB")
            data = pytesseract.image_to_data(img, output_type=Output.DICT)
            blocks: List[dict] = []

            n_boxes = len(data.get("level", []))
            # group by block_num or by line_num depending on what we want
            # We'll group by block_num then by line_num fallback
            current_block_idx = None
            current_texts: List[str] = []
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
                    # flush previous
                    blocks.append({"text": " ".join(current_texts).strip(), "bbox": current_bbox})
                    # start new
                    current_block_idx = block_num
                    current_texts = [text]
                    current_bbox = bbox
                else:
                    current_texts.append(text)
                    # expand bbox
                    if current_bbox:
                        x1 = min(current_bbox[0], bbox[0])
                        y1 = min(current_bbox[1], bbox[1])
                        x2 = max(current_bbox[2], bbox[2])
                        y2 = max(current_bbox[3], bbox[3])
                        current_bbox = (x1, y1, x2, y2)

            # flush last block
            if current_block_idx is not None and current_texts:
                blocks.append({"text": " ".join(current_texts).strip(), "bbox": current_bbox})

            return blocks

        except Exception as e:
            logger.exception("Detailed OCR (image_to_data) failed for %s: %s. Falling back to simple OCR.", image_path, e)
            # fallback: use OCRHandler's simple extractor
            try:
                text = self.ocr.extract_text_from_image(image_path)
                return [{"text": text, "bbox": None}] if text else []
            except Exception as e2:
                logger.exception("Fallback OCR also failed for %s: %s", image_path, e2)
                return []

    def _caption_fallback(self, image_path: str) -> Optional[str]:
        """
        Optional caption fallback for images with little/no OCR text.
        This is a placeholder. If you later add a local caption model,
        implement the call here and return a short caption text.

        For now, this returns None (no caption).
        """
        # Example:
        #   # load caption model from local cache
        #   caption = my_caption_model.generate_caption(image_path)
        #   return caption
        return None


# ----------------------------
# Simple CLI test
# ----------------------------
if __name__ == "__main__":
    import argparse
    import asyncio
    from loguru import logger as lu_logger
    lu_logger.add(lambda msg: print(msg, end=""))

    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to image file to ingest")
    args = parser.parse_args()
    ingestor = ImageIngestor()
    result = ingestor.ingest_image(args.image)
    print("Ingest result:", result)
