# app/extractors/asr_handler.py
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ASRUnavailable(RuntimeError):
    """Raised when no ASR backend can be initialized."""


class ASRHandler:
    """
    ASR handler supporting faster-whisper and whisper backends.

    - No heavy import at module import time.
    - _init_model() performs the heavy work; call it from background preload or on demand.
    """

    def __init__(self, model_name: Optional[str] = None, backend_preference: Optional[List[str]] = None):
        self.model_name = model_name
        self.backend_preference = backend_preference or ["faster_whisper", "whisper"]
        self._backend: Optional[str] = None
        self._model: Optional[Any] = None

    def _init_model(self) -> None:
        """Lazy initialize model (if not already)."""
        if self._model is not None:
            return

        last_exc = None
        for backend in self.backend_preference:
            try:
                if backend == "faster_whisper":
                    # local import to avoid heavy dependency at module import time
                    from faster_whisper import WhisperModel  # type: ignore

                    model_name = self.model_name or "tiny"
                    logger.info("ASRHandler: loading faster_whisper model=%s", model_name)
                    # use CPU by default; change device arg if you have GPU
                    # compute_type may be optional depending on your faster-whisper version
                    self._model = WhisperModel(model_name, device="cpu")
                    self._backend = "faster_whisper"
                    return

                elif backend == "whisper":
                    import whisper  # type: ignore

                    model_name = self.model_name or "small"
                    logger.info("ASRHandler: loading whisper model=%s", model_name)
                    self._model = whisper.load_model(model_name)
                    self._backend = "whisper"
                    return

            except Exception as e:
                logger.warning("ASRHandler: backend %s failed to load: %s", backend, e)
                last_exc = e
                continue

        raise ASRUnavailable(
            "No ASR backend available. Install faster-whisper (recommended) or whisper. "
            f"Last error: {last_exc}"
        )

    def transcribe(self, audio_path: str, language: Optional[str] = None, task: str = "transcribe", **kwargs) -> Dict[str, Any]:
        """
        Transcribe an audio file to text.
        """
        try:
            self._init_model()
        except Exception as e:
            raise ASRUnavailable(f"ASR initialization failed: {e}") from e

        if self._backend == "faster_whisper":
            try:
                segments, info = self._model.transcribe(audio_path, language=language, task=task, **kwargs)
                collected = []
                full_text = []
                for seg in segments:
                    collected.append(
                        {
                            "start": float(getattr(seg, "start", 0.0)),
                            "end": float(getattr(seg, "end", 0.0)),
                            "text": getattr(seg, "text", "").strip(),
                            "confidence": getattr(seg, "avg_logprob", None),
                        }
                    )
                    full_text.append(getattr(seg, "text", "").strip())
                return {"text": " ".join(full_text).strip(), "segments": collected, "backend": "faster_whisper"}
            except Exception as e:
                logger.exception("faster_whisper transcription failed: %s", e)
                raise

        elif self._backend == "whisper":
            try:
                out = self._model.transcribe(audio_path, language=language, task=task, **kwargs)
                text = out.get("text", "") if isinstance(out, dict) else getattr(out, "text", "")
                segments = out.get("segments", []) if isinstance(out, dict) else []
                return {"text": text.strip(), "segments": segments, "backend": "whisper"}
            except Exception as e:
                logger.exception("whisper transcription failed: %s", e)
                raise

        else:
            raise ASRUnavailable("No valid ASR backend initialized")

    def is_available(self) -> bool:
        try:
            self._init_model()
            return True
        except Exception:
            return False
