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
    Environment variables:
      - LLM_HOST (default: http://localhost:11434)
      - LLM_CHAT_PATH (default: /api/chat)
      - LLM_TIMEOUT_SECONDS (default: 60)
      - LLM_RETRIES (default: 2)
      - LLM_BACKOFF_INITIAL (default: 0.5)
      - LLM_BACKOFF_FACTOR (default: 2)
      - LLM_MODEL (default: qwen2.5:7b-instruct)  # primary preferred model
      - LLM_MODEL_FALLBACKS (optional, comma-separated models)
    """
    def __init__(self, host: Optional[str] = None):
        self.host = host or os.getenv("LLM_HOST", "http://localhost:11434")
        self.chat_endpoint = os.getenv("LLM_CHAT_PATH", "/api/chat")
        self.timeout_seconds = int(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
        self.retries = int(os.getenv("LLM_RETRIES", "2"))
        self.backoff_initial = float(os.getenv("LLM_BACKOFF_INITIAL", "0.5"))
        self.backoff_factor = float(os.getenv("LLM_BACKOFF_FACTOR", "2"))
        primary = os.getenv("LLM_MODEL", "qwen2.5:1.5b-instruct")
        fallbacks = os.getenv("LLM_MODEL_FALLBACKS", "")
        fallback_list = [m.strip() for m in ([primary] + fallbacks.split(",")) if m and m.strip()]
        seen = set()
        self.model_candidates = []
        for m in fallback_list:
            if m not in seen:
                self.model_candidates.append(m)
                seen.add(m)

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
        if self._client:
            try:
                await self._client.aclose()
            except Exception:
                logger.exception("Error closing httpx client")
            finally:
                self._client = None

    async def ping(self, path: str = "/") -> bool:
        url = self.host.rstrip("/") + ("/" + path.lstrip("/")) if path else self.host
        try:
            client = await self._ensure_client()
            resp = await client.get(url, timeout=5.0)
            logger.debug("LLM ping %s -> %s", url, resp.status_code)
            return 200 <= resp.status_code < 400
        except Exception as e:
            logger.warning("LLM ping failed: %s", e)
            return False

    def _is_memory_error(self, resp_text: str) -> bool:
        text = (resp_text or "").lower()
        keywords = [
            "requires more system memory",
            "out of memory",
            "insufficient memory",
            "memory",
            "not enough memory",
            "oom",
        ]
        return any(k in text for k in keywords)

    async def generate(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
        model: Optional[str] = None,
        stream: bool = False,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        api_url = self._api_url()

        candidates = [model] if model else list(self.model_candidates)
        if not candidates:
            candidates = [os.getenv("LLM_MODEL", "qwen2.5:1.5b-instruct")]

        last_exc: Optional[Exception] = None

        for model_candidate in candidates:
            payload: Dict[str, Any] = {
                "model": model_candidate,
                "messages": messages if isinstance(messages, list) else [{"role": "user", "content": str(messages)}],
                "stream": bool(stream),
            }

            options: Dict[str, Any] = {}
            if max_tokens is not None:
                options["num_predict"] = int(max_tokens)
            if temperature is not None:
                options["temperature"] = float(temperature)
            if options:
                payload["options"] = options

            if extra:
                for k, v in extra.items():
                    if k in ("model", "messages"):
                        continue
                    payload[k] = v

            logger.debug("LLM request -> %s model=%s payload_keys=%s", api_url, model_candidate, list(payload.keys()))

            # try/retry loop per candidate
            for attempt in range(0, self.retries + 1):
                client = await self._ensure_client()
                try:
                    logger.debug("LLM generate attempt %d/%d model=%s timeout=%ds",
                                 attempt + 1, self.retries + 1, model_candidate, self.timeout_seconds)
                    resp = await client.post(api_url, json=payload)

                    # If 404: model not found on LLM host -> try next candidate
                    if resp.status_code == 404:
                        text = resp.text or ""
                        logger.warning("LLM model '%s' not found (404): %s", model_candidate, text[:200])
                        last_exc = LLMClientError(f"Model {model_candidate} not found: {text[:200]}")
                        break  # break retry loop, move to next candidate

                    # If 5xx, check memory error heuristics
                    if 500 <= resp.status_code < 600:
                        text = resp.text or ""
                        if self._is_memory_error(text):
                            logger.warning("LLM model '%s' memory error: %s", model_candidate, text[:400])
                            last_exc = LLMClientError(f"Model {model_candidate} memory error: {text[:400]}")
                            break  # try next candidate
                        else:
                            last_exc = LLMClientError(f"LLM returned status {resp.status_code}: {resp.text[:400]}")
                            logger.warning("LLM host returned %s (attempt %d) - will retry", resp.status_code, attempt + 1)
                            raise last_exc

                    # parse JSON
                    try:
                        data = resp.json()
                    except Exception as e:
                        text = resp.text
                        logger.debug("LLM returned non-json response (status=%s): %.300s", resp.status_code, text)
                        raise LLMClientError(f"Non-JSON response from LLM: status={resp.status_code} body={text[:1000]}") from e

                    # Extract text from common response shapes
                    extracted_text = None
                    if isinstance(data, dict):
                        msg = data.get("message") or data.get("response") or data.get("text") or data.get("_raw_text")
                        if isinstance(msg, dict) and "content" in msg:
                            extracted_text = msg.get("content")
                        elif isinstance(msg, str):
                            extracted_text = msg
                        else:
                            if isinstance(data.get("choices"), list) and data["choices"]:
                                first = data["choices"][0]
                                if isinstance(first, dict):
                                    m = first.get("message") or first.get("text")
                                    if isinstance(m, dict) and "content" in m:
                                        extracted_text = m.get("content")
                                    elif isinstance(m, str):
                                        extracted_text = m

                    logger.debug("LLM response OK (status=%s) model=%s", resp.status_code, model_candidate)
                    result = {"status_code": resp.status_code, "raw": data, "text": extracted_text, "model_used": model_candidate}
                    return result

                except httpx.ReadTimeout as e:
                    last_exc = e
                    logger.warning("LLM ReadTimeout on attempt %d/%d model=%s: %s", attempt + 1, self.retries + 1, model_candidate, e)
                except httpx.RequestError as e:
                    last_exc = e
                    logger.warning("LLM network error on attempt %d/%d model=%s: %s", attempt + 1, self.retries + 1, model_candidate, e)
                except LLMClientError as e:
                    last_exc = e
                    logger.warning("LLM client error on attempt %d/%d model=%s: %s", attempt + 1, self.retries + 1, model_candidate, e)
                except Exception as e:
                    last_exc = e
                    logger.exception("Unexpected error calling LLM on attempt %d/%d model=%s", attempt + 1, self.retries + 1, model_candidate)

                if attempt < self.retries:
                    backoff = self.backoff_initial * (self.backoff_factor ** attempt)
                    logger.debug("LLM backoff waiting %.2fs before next attempt", backoff)
                    await asyncio.sleep(backoff)

            # End attempts for this model_candidate
            # If we broke due to memory or 404, last_exc will contain an LLMClientError; proceed to next candidate
            if isinstance(last_exc, LLMClientError) and ("memory" in str(last_exc).lower() or "not found" in str(last_exc).lower()):
                logger.info("Switching LLM model candidate due to issue: tried=%s", model_candidate)
                last_exc = None
                continue

        msg = f"Error while calling LLM host {self.host}: {last_exc}"
        logger.error(msg)
        raise LLMClientError(msg)
