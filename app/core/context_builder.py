# app/core/context_builder.py
from typing import List, Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger("asklyne.context_builder")


@dataclass
class ContextBuilder:
    """
    Build a compact context string for LLM prompts from retrieved candidate chunks.

    Primary entry points:
      - build_context_from_candidates(candidates, query, max_total_chars=3000, per_chunk_chars=800, session_id=None)
        -> str  (final context text to inject into prompt)
    """

    max_total_chars: int = 3000
    per_chunk_chars: int = 800
    safety_margin: float = 0.95

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate (1 token ~= 4 chars)."""
        return max(1, len(text) // 4)

    def _truncate_chunk(self, text: str, max_chars: int) -> str:
        """Truncate long chunks without breaking words."""
        if len(text) <= max_chars:
            return text
        cut = text[:max_chars]
        last_nl = cut.rfind("\n")
        if last_nl > max_chars // 2:
            cut = cut[:last_nl]
        else:
            last_space = cut.rfind(" ")
            if last_space > max_chars // 2:
                cut = cut[:last_space]
        return cut.rstrip() + "..."

    def build_context_from_candidates(
        self,
        candidates: List[Dict],
        query: str,
        max_total_chars: Optional[int] = None,
        per_chunk_chars: Optional[int] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """
        Build a context string suitable for insertion into an LLM prompt.

        Args:
            candidates: list of retrieved chunks.
            query: user query string.
            max_total_chars: override global context limit.
            per_chunk_chars: override per-chunk truncation.
            session_id: optional session tag for logging/tracking.

        Returns:
            Formatted string:
                "Context:\n<chunk1>\n---\n<chunk2>...\n\nQuestion: <query>\nAnswer:"
        """
        max_chars = int(max_total_chars or self.max_total_chars)
        per_chunk = int(per_chunk_chars or self.per_chunk_chars)
        budget = int(max_chars * float(self.safety_margin))

        if not candidates:
            logger.info("ContextBuilder: empty candidate list (session=%s)", session_id)
            return f"Context:\n\nQuestion: {query}\nAnswer:"

        normalized = []
        for c in candidates:
            score = c.get("score") if isinstance(c.get("score"), (int, float)) else 0.0

            # Extract text
            text = None
            if isinstance(c.get("text"), str) and c.get("text").strip():
                text = c.get("text").strip()
            else:
                payload = c.get("payload") or {}
                if isinstance(payload, dict):
                    t = payload.get("text") or payload.get("content") or payload.get("body")
                    if isinstance(t, str):
                        text = t.strip()
            if not text:
                text = str(c)

            c_id = c.get("id")
            payload = c.get("payload") or {}
            source = payload.get("filename") or payload.get("source") or payload.get("source_path")

            normalized.append({"score": float(score), "id": c_id, "text": text, "source": source})

        # Sort by descending score
        normalized.sort(key=lambda x: -x["score"])

        selected_parts = []
        used_chars = 0

        for entry in normalized:
            chunk_text = self._truncate_chunk(entry["text"], per_chunk)
            header_parts = []
            if entry.get("id"):
                header_parts.append(f"id:{entry['id']}")
            if entry.get("source"):
                header_parts.append(f"src:{entry['source']}")
            header = ("[" + " ".join(header_parts) + "] ") if header_parts else ""

            piece = header + chunk_text

            if used_chars + len(piece) > budget:
                if not selected_parts:
                    remaining = max(64, budget)
                    piece = header + self._truncate_chunk(chunk_text, remaining - len(header))
                    selected_parts.append(piece)
                break

            selected_parts.append(piece)
            used_chars += len(piece)

        context_body = "\n\n---\n\n".join(selected_parts)
        final_context = f"Context:\n\n{context_body}\n\nQuestion: {query}\nAnswer:"

        logger.debug(
            "ContextBuilder: built context for session=%s | chars=%d / budget=%d | chunks=%d",
            session_id,
            used_chars,
            budget,
            len(selected_parts),
        )
        return final_context
