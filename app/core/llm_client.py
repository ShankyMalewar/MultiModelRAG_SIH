# app/core/llm_client.py
import os
import asyncio
import logging
from typing import Any, Dict, Optional, List

import httpx

logger = logging.getLogger("asklyne.llm_client")
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))

class LLMClientError(Exception):
    pass

class LLMClient:
    """
    HTTP client for a local Ollama-like LLM endpoint.
    Public methods:
      - ping(path="/") -> bool
      - generate(messages, max_tokens=None, temperature=0.0, model=None, stream=False, extra=None) -> dict
      - close() -> None
    Environment variables:
      - LLM_HOST (default: http://localhost:11434)
      - LLM_CHAT_PATH (default: /api/chat)
      - LLM_TIMEOUT_SECONDS (default: 60)
      - LLM_RETRIES (default: 2)
      - LLM_BACKOFF_INITIAL (default: 0.5)
      - LLM_BACKOFF_FACTOR (default: 2)
      - LLM_MODEL (default: qwen2.5:7b-instruct)
    """

    def __init__(self, host: Optional[str] = None):
        self.host = host or os.getenv("LLM_HOST", "http://localhost:11434")
        self.chat_endpoint = os.getenv("LLM_CHAT_PATH", "/api/chat")
        self.timeout_seconds = int(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
        self.retries = int(os.getenv("LLM_RETRIES", "2"))
        self.backoff_initial = float(os.getenv("LLM_BACKOFF_INITIAL", "0.5"))
        self.backoff_factor = float(os.getenv("LLM_BACKOFF_FACTOR", "2"))
        self.default_model = os.getenv("LLM_MODEL", "qwen2.5:7b-instruct")
        self._client: Optional[httpx.AsyncClient] = None

    def _api_url(self) -> str:
        base = self.host.rstrip("/")
        path = self.chat_endpoint.lstrip("/")
        return f"{base}/{path}"

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            timeout = httpx.Timeout(
                self.timeout_seconds,
                connect=10.0,
                read=self.timeout_seconds,
                write=self.timeout_seconds,
            )
            self._client = httpx.AsyncClient(timeout=timeout)
        return self._client

    async def close(self) -> None:
        """Close persistent httpx client."""
        if self._client:
            try:
                await self._client.aclose()
            except Exception:
                logger.exception("Error closing httpx client")
            finally:
                self._client = None

    async def ping(self, path: str = "/") -> bool:
        """
        Check that the host is reachable.
        Returns True on 2xx/3xx response; False otherwise.
        """
        url = self.host.rstrip("/") + ("/" + path.lstrip("/")) if path else self.host
        try:
            client = await self._ensure_client()
            resp = await client.get(url, timeout=5.0)
            logger.debug("LLM ping %s -> %s", url, resp.status_code)
            return 200 <= resp.status_code < 400
        except Exception as e:
            logger.warning("LLM ping failed: %s", e)
            return False

    async def generate(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
        model: Optional[str] = None,
        stream: bool = False,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Send the request to the LLM and return structured response.
        Retries automatically on timeouts and transient errors.
        Returns: {"status_code": int, "raw": parsed_json, "text": Optional[str]}
        """

        api_url = self._api_url()

        # Compose payload in the Ollama-compatible shape
        payload: Dict[str, Any] = {
            "model": model or self.default_model,
            "messages": messages if isinstance(messages, list) else [{"role": "user", "content": str(messages)}],
            "stream": bool(stream),
        }

        # Ollama-style options (putting generation params under "options")
        options: Dict[str, Any] = {}
        if max_tokens is not None:
            # Ollama uses "num_predict" (approx.) — keep it in options to avoid unknown top-level keys.
            options["num_predict"] = int(max_tokens)
        if temperature is not None:
            options["temperature"] = float(temperature)
        if options:
            payload["options"] = options

        if extra:
            # Merge safely, avoid clobbering model/messages
            for k, v in extra.items():
                if k in ("model", "messages"):
                    continue
                payload[k] = v

        logger.debug("LLM request -> %s (payload keys=%s)", api_url, list(payload.keys()))

        last_exc: Optional[Exception] = None
        for attempt in range(0, self.retries + 1):
            client = await self._ensure_client()
            try:
                logger.debug("LLM generate attempt %d/%d timeout=%ds", attempt + 1, self.retries + 1, self.timeout_seconds)
                resp = await client.post(api_url, json=payload)
                # treat 5xx as transient
                if 500 <= resp.status_code < 600:
                    last_exc = LLMClientError(f"LLM returned status {resp.status_code}: {resp.text[:400]}")
                    logger.warning("LLM host returned %s (attempt %d) - will retry", resp.status_code, attempt + 1)
                    raise last_exc

                # parse JSON response if possible
                try:
                    data = resp.json()
                except Exception as e:
                    text = resp.text
                    logger.debug("LLM returned non-json response (status=%s): %.300s", resp.status_code, text)
                    raise LLMClientError(f"Non-JSON response from LLM: status={resp.status_code} body={text[:1000]}") from e

                # Extract a convenient text if possible
                extracted_text = None
                if isinstance(data, dict):
                    # Common Ollama-like shapes: data may contain "message" {"role","content"} or custom keys
                    msg = data.get("message") or data.get("response") or data.get("text") or data.get("_raw_text")
                    if isinstance(msg, dict) and "content" in msg:
                        extracted_text = msg.get("content")
                    elif isinstance(msg, str):
                        extracted_text = msg
                    else:
                        # Some Ollama streams produce nested raw parts, try to find a raw text key
                        extracted_text = data.get("_raw_text") or data.get("text")

                logger.debug("LLM response OK (status=%s)", resp.status_code)
                return {"status_code": resp.status_code, "raw": data, "text": extracted_text}

            except httpx.ReadTimeout as e:
                last_exc = e
                logger.warning("LLM ReadTimeout on attempt %d/%d: %s", attempt + 1, self.retries + 1, e)
            except httpx.RequestError as e:
                last_exc = e
                logger.warning("LLM network error on attempt %d/%d: %s", attempt + 1, self.retries + 1, e)
            except LLMClientError as e:
                last_exc = e
                logger.warning("LLM client error on attempt %d/%d: %s", attempt + 1, self.retries + 1, e)
            except Exception as e:
                last_exc = e
                logger.exception("Unexpected error calling LLM on attempt %d/%d", attempt + 1, self.retries + 1)

            # backoff if we'll retry
            if attempt < self.retries:
                backoff = self.backoff_initial * (self.backoff_factor ** attempt)
                logger.debug("LLM backoff waiting %.2fs before next attempt", backoff)
                await asyncio.sleep(backoff)

        msg = f"Error while calling LLM host {self.host}: {last_exc}"
        logger.error(msg)
        raise LLMClientError(msg)
