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
      - build_context_from_candidates(candidates, query, max_total_chars=3000, per_chunk_chars=800)
        -> str  (final context text to inject into prompt)
    """

    # maximum characters allowed in the whole assembled context (not tokens) - keeps prompt size bounded
    max_total_chars: int = 3000

    # chars to allow per chunk before truncating that chunk (keeps each chunk small)
    per_chunk_chars: int = 800

    # safety margin (fraction of max_total_chars used)
    safety_margin: float = 0.95

    def _estimate_tokens(self, text: str) -> int:
        """
        Lightweight token estimate: 1 token ~= 4 characters (common heuristic).
        Not used as authoritative — mainly for debugging.
        """
        return max(1, len(text) // 4)

    def _truncate_chunk(self, text: str, max_chars: int) -> str:
        """
        Truncate a chunk to `max_chars` characters without breaking words where possible.
        """
        if len(text) <= max_chars:
            return text
        # try to cut at last newline or space before the limit
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
    ) -> str:
        """
        Build a context string suitable for insertion into an LLM prompt.

        - candidates: list of dicts returned by retriever; expected keys:
            'id' (optional), 'score' (optional), 'payload' or 'text' (text under 'text' key)
            For payload-based hits the text is usually in hit['payload']['text'].
        - query: original user query (used to finalize prompt string).
        - max_total_chars: override default max_total_chars for this call.
        - per_chunk_chars: override per_chunk_chars for this call.

        Returns: formatted string "Context:\n<chunk1>\n---\n<chunk2>...\n\nQuestion: <query>\nAnswer:"
        """

        max_chars = int(max_total_chars or self.max_total_chars)
        per_chunk = int(per_chunk_chars or self.per_chunk_chars)
        budget = int(max_chars * float(self.safety_margin))

        if not candidates:
            return f"Context:\n\nQuestion: {query}\nAnswer:"

        # normalise candidates into (score, id, text, source) and sort by score desc
        normalized = []
        for c in candidates:
            score = c.get("score") if isinstance(c.get("score"), (int, float)) else 0.0

            # text may be in c['text'] or c['payload']['text']
            text = None
            if isinstance(c.get("text"), str) and c.get("text").strip():
                text = c.get("text").strip()
            else:
                payload = c.get("payload") or {}
                if isinstance(payload, dict):
                    t = payload.get("text") or payload.get("content") or payload.get("body")
                    if isinstance(t, str):
                        text = t.strip()

            # fallback to entire candidate repr
            if not text:
                text = str(c)

            c_id = c.get("id") or (payload.get("id") if isinstance(payload, dict) and payload.get("id") else None)
            # source metadata for citation (filename, page, etc) if present in payload
            source = None
            if isinstance(payload, dict):
                source = payload.get("filename") or payload.get("source") or payload.get("source_path")

            normalized.append({"score": float(score), "id": c_id, "text": text, "source": source})

        # sort by score descending
        normalized.sort(key=lambda x: -x["score"])

        selected_parts = []
        used_chars = 0

        for entry in normalized:
            chunk_text = entry["text"]
            # truncate chunk if too long
            chunk_text = self._truncate_chunk(chunk_text, per_chunk)

            # prepare header - include ID and optional source for traceability
            header_parts = []
            if entry.get("id"):
                header_parts.append(f"id:{entry['id']}")
            if entry.get("source"):
                header_parts.append(f"src:{entry['source']}")
            header = ("[" + " ".join(header_parts) + "] ") if header_parts else ""

            piece = header + chunk_text

            # if adding this piece will exceed our budget, stop (we keep highest scored first)
            if used_chars + len(piece) > budget:
                # if nothing selected yet, still include a truncated piece to avoid empty context
                if not selected_parts:
                    remaining = max(64, budget)  # ensure small useful piece
                    piece = header + self._truncate_chunk(chunk_text, remaining - len(header))
                    selected_parts.append(piece)
                break

            selected_parts.append(piece)
            used_chars += len(piece)

        context_body = "\n\n---\n\n".join(selected_parts)

        final = (
            f"Context:\n\n{context_body}\n\n"
            f"Question: {query}\n"
            f"Answer:"
        )
        logger.debug("Built context: chars=%d / budget=%d selected_chunks=%d", used_chars, budget, len(selected_parts))
        return final
