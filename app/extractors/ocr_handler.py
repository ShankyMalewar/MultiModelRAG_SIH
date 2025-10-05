# app/extractors/ocr_handler.py
"""
OCR utilities for Asklyne Offline.

Provides OCRHandler class with:
 - extract_text_from_image(path)
 - image_to_data(path)
 - extract_text_from_pdf(path)
 - get_page_images_from_pdf(path)

Dependencies:
 - pillow
 - pytesseract + Tesseract binary installed
 - pdf2image + poppler (for PDF OCR)
"""

from typing import List, Optional, Dict, Any
import os
import logging

logger = logging.getLogger(__name__)

try:
    from PIL import Image
except ImportError:
    Image = None


class OCRHandler:
    def __init__(self, tesseract_cmd: Optional[str] = None, poppler_path: Optional[str] = None):
        self.tesseract_cmd = tesseract_cmd or os.environ.get("TESSERACT_CMD")
        self.poppler_path = poppler_path or os.environ.get("POPPLER_PATH")

        try:
            import pytesseract
            self._pytesseract = pytesseract
            if self.tesseract_cmd:
                self._pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
        except ImportError:
            self._pytesseract = None

        self._pdf2image = None

    # --- IMAGE OCR ---
    def extract_text_from_image(self, image_path: str, lang: Optional[str] = None) -> str:
        if Image is None:
            raise RuntimeError("Pillow not installed.")
        if not self._pytesseract:
            raise RuntimeError("pytesseract not installed.")

        img = Image.open(image_path).convert("RGB")
        try:
            return self._pytesseract.image_to_string(img, lang=lang).strip()
        except Exception as e:
            logger.exception("OCR failed: %s", e)
            return ""

    def image_to_data(self, image_path: str) -> Dict[str, Any]:
        if Image is None:
            raise RuntimeError("Pillow not installed.")
        if not self._pytesseract:
            raise RuntimeError("pytesseract not installed.")

        img = Image.open(image_path).convert("RGB")
        from pytesseract import Output
        try:
            return self._pytesseract.image_to_data(img, output_type=Output.DICT)
        except Exception as e:
            logger.exception("image_to_data failed: %s", e)
            return {"level": [], "text": []}

    # --- PDF OCR ---
    def _ensure_pdf2image(self):
        if self._pdf2image:
            return
        try:
            from pdf2image import convert_from_path
            self._pdf2image = convert_from_path
        except ImportError:
            raise RuntimeError("pdf2image not installed. Install it to OCR PDFs.")

    def get_page_images_from_pdf(self, pdf_path: str, dpi: int = 200) -> List["Image.Image"]:
        self._ensure_pdf2image()
        kwargs = {}
        if self.poppler_path:
            kwargs["poppler_path"] = self.poppler_path
        return self._pdf2image(pdf_path, dpi=dpi, **kwargs)

    def extract_text_from_pdf(self, pdf_path: str, lang: Optional[str] = None, dpi: int = 200) -> str:
        try:
            pages = self.get_page_images_from_pdf(pdf_path, dpi=dpi)
        except Exception as e:
            logger.exception("Failed to render PDF: %s", e)
            return ""

        texts = []
        for i, img in enumerate(pages, start=1):
            try:
                page_text = self._pytesseract.image_to_string(img, lang=lang).strip()
                texts.append(f"[page {i}]\n{page_text}")
            except Exception as e:
                logger.exception("OCR failed on page %d: %s", i, e)

        return "\n\n".join(texts)

    # --- Unified interface ---
    def extract_text(self, path: str, lang: Optional[str] = None) -> str:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf":
            return self.extract_text_from_pdf(path, lang=lang)
        return self.extract_text_from_image(path, lang=lang)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path to image or PDF")
    args = parser.parse_args()

    ocr = OCRHandler()
    text = ocr.extract_text(args.file)
    print(text[:1000])

# --- Legacy compatibility helpers ---

def ocr_pdf(pdf_path: str, lang: str = None, dpi: int = 200) -> str:
    """
    Simple wrapper so older code can call ocr_pdf(...) directly.
    Internally uses OCRHandler.extract_text_from_pdf.
    """
    handler = OCRHandler()
    return handler.extract_text_from_pdf(pdf_path, lang=lang, dpi=dpi)
