# app/core/chunker.py
"""
Chunker utilities for the multimodal RAG project.

Provides:
- Chunker class for text and code with token-window + overlap behavior.
- Simpler functions for paragraph-aware chunking.

This implementation is lightweight and deterministic (no external tokenizer dependency).
It uses a simple chars->tokens heuristic suitable for packing into LLM context windows.
"""

from typing import List
import re
import math
import logging

logger = logging.getLogger(__name__)


class Chunker:
    def __init__(self, max_tokens: int = 500, overlap: int = 50, chars_per_token: int = 4, file_type: str = "text"):
        """
        Args:
            max_tokens: target token length per chunk (approx)
            overlap: overlap in tokens between adjacent chunks
            chars_per_token: heuristic conversion factor (chars -> tokens)
            file_type: 'text' or 'code' (changes splitting strategy)
        """
        self.max_tokens = max_tokens
        self.overlap = overlap
        self.chars_per_token = chars_per_token
        self.file_type = file_type

    def estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        return max(1, math.ceil(len(text) / max(1, self.chars_per_token)))

    def chunk(self, content: str) -> List[str]:
        if not content:
            return []
        if self.file_type == "code":
            return self.chunk_code(content)
        return self.chunk_text(content)

    def chunk_text(self, text: str) -> List[str]:
        """
        Chunk natural language text by grouping sentence/paragraph pieces into windows
        controlled by max_tokens and overlap. Preserves paragraph boundaries where possible.
        """
        # Split on double newlines first (paragraphs)
        parts = [p.strip() for p in re.split(r'\n{2,}', text.strip()) if p.strip()]
        if not parts:
            # fallback to sentence split
            parts = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]

        # Convert parts to (text, tokens)
        tokens_parts = [(p, self.estimate_tokens(p)) for p in parts]
        chunks = []
        current = []
        current_tokens = 0
        i = 0
        while i < len(tokens_parts):
            part_text, part_tokens = tokens_parts[i]
            if current_tokens + part_tokens > self.max_tokens:
                # flush current chunk
                if current:
                    chunks.append(" ".join(current).strip())
                    # create overlap from end of current
                    overlap_tokens = 0
                    j = len(current) - 1
                    while j >= 0 and overlap_tokens < self.overlap:
                        overlap_tokens += self.estimate_tokens(current[j])
                        j -= 1
                    # keep tail for overlap
                    current = current[j + 1:]
                    current_tokens = sum(self.estimate_tokens(p) for p in current)
                else:
                    # single part exceeds chunk size — split it mechanically
                    subparts = self._split_long_text(part_text)
                    # insert subparts into the parts list
                    # replace current index with subparts (as separate parts)
                    tokens_parts.pop(i)
                    for sp in reversed(subparts):
                        tokens_parts.insert(i, (sp, self.estimate_tokens(sp)))
                    # continue without incrementing i
                    continue
            current.append(part_text)
            current_tokens += part_tokens
            i += 1

        if current:
            chunks.append(" ".join(current).strip())
        return chunks

    def _split_long_text(self, text: str) -> List[str]:
        """Split very long paragraph/text by characters into smaller windows (no smart split)."""
        approx_chars = self.max_tokens * self.chars_per_token
        parts = []
        start = 0
        L = len(text)
        while start < L:
            end = start + approx_chars
            parts.append(text[start:end].strip())
            start = max(0, end - (self.overlap * self.chars_per_token))
            if start >= L:
                break
        return parts

    def chunk_code(self, code: str) -> List[str]:
        """
        Chunk code by splitting on function/class boundaries where possible,
        while still respecting token budgets.
        """
        # split into blocks using def/class tokens (python-centric heuristic)
        blocks = re.split(r'(?=^\s*(def|class)\s)', code, flags=re.MULTILINE)
        blocks = [b.strip() for b in blocks if b and b.strip()]
        chunks = []
        current = []
        current_tokens = 0
        for b in blocks:
            b_tokens = self.estimate_tokens(b)
            if current_tokens + b_tokens > self.max_tokens:
                if current:
                    chunks.append("\n\n".join(current).strip())
                    # overlap
                    overlap_tokens = 0
                    j = len(current) - 1
                    while j >= 0 and overlap_tokens < self.overlap:
                        overlap_tokens += self.estimate_tokens(current[j])
                        j -= 1
                    current = current[j + 1:]
                    current_tokens = sum(self.estimate_tokens(x) for x in current)
                else:
                    chunks.append(b)
                    continue
            current.append(b)
            current_tokens += b_tokens
        if current:
            chunks.append("\n\n".join(current).strip())
        return chunks

def chunk_paragraphs(text: str, min_length: int = 50) -> list[str]:
    """
    Naive paragraph splitter: splits text by double newlines, then yields paragraphs
    that have at least min_length characters.
    """
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    return [p for p in paras if len(p) >= min_length]

# --- Convenience functional APIs (wrappers around Chunker) ---

def chunk_text(text: str, max_tokens: int = 500, overlap: int = 50, chars_per_token: int = 4) -> list[str]:
    """
    Functional wrapper for text chunking (uses Chunker under the hood).
    """
    return Chunker(max_tokens=max_tokens, overlap=overlap, chars_per_token=chars_per_token, file_type="text").chunk_text(text)


def estimate_tokens(text: str, chars_per_token: int = 4) -> int:
    """
    Functional wrapper for token estimation.
    """
    return max(1, (len(text) + chars_per_token - 1) // chars_per_token)
