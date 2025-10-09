# # app/services/qdrant_service.py
# """
# Robust QdrantService used by the ingest & retrieval pipeline.

# Goals/fixes implemented:
# - Handles qdrant-client when available, otherwise falls back to REST (requests).
# - Detects named-vector collections and maps upserts / searches to the correct REST/client formats.
# - Provides multiple REST payload formats (some Qdrant versions expect "vector": {"name":..., "vector": [...]}
#   while some expect "vector": {"name":..., "values": [...]} or simply the list).
# - Defensive error handling and more helpful logs for debugging.
# """

# import time
# import logging
# import json
# from typing import List, Tuple, Any, Optional, Dict

# # network
# try:
#     import requests
# except Exception:
#     requests = None

# # try qdrant client
# _qdrant_client_available = False
# try:
#     from qdrant_client import QdrantClient
#     from qdrant_client.http import models as qdrant_models
#     _qdrant_client_available = True
# except Exception:
#     QdrantClient = None
#     qdrant_models = None

# DEFAULT_COLLECTION = "asklyne_collection"


# class QdrantService:
#     def __init__(self, host: str = "http://localhost:6333", collection: str = DEFAULT_COLLECTION, api_key: str = None):
#         # logger
#         self.logger = logging.getLogger("asklyne.qdrant")
#         if not self.logger.handlers:
#             # avoid interfering with global logging configuration but ensure something is present
#             logging.basicConfig(level=logging.INFO)

#         self.host = host.rstrip("/")
#         self.collection_name = collection
#         self.api_key = api_key
#         self.client: Optional[Any] = None
#         self._use_qdrant_client = _qdrant_client_available

#         # Try to initialize qdrant-client if available
#         if self._use_qdrant_client:
#             try:
#                 try:
#                     # new client accepts url=
#                     self.client = QdrantClient(url=self.host, api_key=self.api_key)
#                 except TypeError:
#                     # older signature host=
#                     self.client = QdrantClient(host=self.host, api_key=self.api_key)
#                 self.logger.info(
#                     "QdrantService initialized host=%s collection=%s (qdrant-client)",
#                     self.host,
#                     self.collection_name,
#                 )
#             except Exception as e:
#                 self.logger.warning("Failed to init qdrant-client, will fallback to HTTP: %s", e)
#                 self.client = None
#                 self._use_qdrant_client = False
#         else:
#             if requests is None:
#                 self.logger.error("Neither qdrant-client nor requests available; Qdrant unavailable")
#             else:
#                 self.logger.info("QdrantService initialized host=%s collection=%s (HTTP)", self.host, self.collection_name)

#         # Best-effort create/check collection
#         try:
#             self.create_collection_if_not_exists()
#         except Exception as e:
#             self.logger.debug("create_collection_if_not_exists raised: %s", e)

#     # ----------------------
#     # Collection helpers
#     # ----------------------
#     def _collections_url(self):
#         return f"{self.host}/collections"

#     def create_collection_if_not_exists(self) -> bool:
#         """
#         Best-effort: check collection and create minimal schema if missing.
#         Note: we try multiple client signatures to maximize compatibility.
#         """
#         try:
#             # Try qdrant-client introspection
#             if self.client:
#                 try:
#                     cols = self.client.get_collections()
#                     names = []
#                     # qdrant-client shape may vary
#                     if hasattr(cols, "collections"):
#                         names = [c.name for c in cols.collections]
#                     elif isinstance(cols, dict):
#                         names = [c.get("name") for c in cols.get("result", {}).get("collections", [])]
#                     if self.collection_name in names:
#                         self.logger.debug("Collection exists: %s", self.collection_name)
#                         return True
#                 except Exception:
#                     # fall through to REST check
#                     pass

#             # REST check
#             if requests is None:
#                 return False
#             r = requests.get(f"{self.host}/collections", timeout=5)
#             if r.status_code == 200:
#                 try:
#                     data = r.json()
#                     collections = [c.get("name") for c in data.get("result", {}).get("collections", [])]
#                     if self.collection_name in collections:
#                         self.logger.debug("Collection exists (REST): %s", self.collection_name)
#                         return True
#                 except Exception:
#                     pass

#             # Create a minimal collection (vector size may be overridden by upsert)
#             self.logger.info("Creating collection %s (best-effort)", self.collection_name)
#             body = {"name": self.collection_name, "vectors": {"size": 1024, "distance": "Cosine"}}

#             if self.client:
#                 try:
#                     # Try a few qdrant-client create signatures
#                     try:
#                         self.client.recreate_collection(self.collection_name, vectors_config=body["vectors"])
#                         self.logger.debug("Recreated collection via qdrant-client.")
#                         return True
#                     except Exception:
#                         self.client.create_collection(collection_name=self.collection_name, vectors=body["vectors"])
#                         self.logger.debug("Created collection via qdrant-client (alternative signature).")
#                         return True
#                 except Exception as e:
#                     self.logger.debug("Failed to create collection via qdrant-client: %s", e)

#             if requests:
#                 url = f"{self.host}/collections/{self.collection_name}"
#                 headers = {"Content-Type": "application/json"}
#                 if self.api_key:
#                     headers["X-API-Key"] = self.api_key
#                 resp = requests.put(url, json=body, headers=headers, timeout=10)
#                 if resp.status_code in (200, 201):
#                     self.logger.debug("Created collection via REST.")
#                     return True
#                 else:
#                     self.logger.debug("Create collection REST failed: %s %s", resp.status_code, resp.text)
#             return False
#         except Exception as e:
#             self.logger.exception("create_collection_if_not_exists error: %s", e)
#             return False

#     # ----------------------
#     # Upsert
#     # ----------------------
#     def upsert_chunks(self, chunk_objs, batch_size: int = 64) -> Tuple[int, List[Any]]:
#         """
#         Upsert a list of chunk objects (dict or object) into Qdrant.
#         Returns (num_upserted, failed_list)
#         """
#         points = []
#         failed = []

#         # Normalize and prepare points
#         for idx, c in enumerate(chunk_objs):
#             emb = None
#             if isinstance(c, dict):
#                 emb = c.get("embedding") or c.get("vector") or c.get("embedding_vector") or c.get("vec")
#             else:
#                 emb = getattr(c, "embedding", None) or getattr(c, "vector", None) or getattr(c, "embedding_vector", None)

#             if emb is None:
#                 self.logger.debug("Chunk missing embedding: %s", repr(c))
#                 failed.append(c)
#                 continue

#             # Convert embedding to list[float]
#             try:
#                 if hasattr(emb, "tolist"):
#                     emb = emb.tolist()
#                 emb = [float(x) for x in emb]
#             except Exception as e:
#                 self.logger.warning("Failed to normalize embedding: %s", e)
#                 failed.append(c)
#                 continue

#             # Basic payload extraction
#             payload = {}
#             try:
#                 if isinstance(c, dict):
#                     # copy most fields but exclude embedding/vector
#                     payload = {k: v for k, v in c.items() if k not in ("embedding", "vector")}
#                 else:
#                     for name in ("text", "filename", "source_path", "meta"):
#                         if hasattr(c, name):
#                             payload[name] = getattr(c, name)
#             except Exception:
#                 pass

#             # Create ID
#             cid = None
#             if isinstance(c, dict):
#                 cid = c.get("id") or c.get("chunk_id") or c.get("uid")
#             else:
#                 cid = getattr(c, "id", None) or getattr(c, "chunk_id", None)
#             if not cid:
#                 cid = f"chunk-{int(time.time()*1000)}-{idx}"

#             points.append({"id": cid, "vector": emb, "payload": payload})

#         if not points:
#             return 0, failed

#         # Detect named vector key (best-effort)
#         named_vector_key = None
#         try:
#             if self.client:
#                 info = self.client.get_collection(self.collection_name)
#                 cfg = getattr(info, "config", None)
#                 if cfg is not None and hasattr(cfg, "params"):
#                     vectors = getattr(cfg.params, "vectors", None)
#                     if isinstance(vectors, dict) and vectors:
#                         named_vector_key = next(iter(vectors.keys()))
#         except Exception:
#             named_vector_key = None

#         if named_vector_key:
#             self.logger.info("Detected named vector collection: using key '%s'", named_vector_key)
#             # wrap vectors under name for all points for REST compatibility later
#             for p in points:
#                 p["vector"] = {named_vector_key: p["vector"]}

#         upserted = 0
#         try:
#             for i in range(0, len(points), batch_size):
#                 batch = points[i : i + batch_size]
#                 if self.client:
#                     try:
#                         # qdrant-client often accepts plain dicts as points in newer versions
#                         self.client.upsert(collection_name=self.collection_name, points=batch)
#                         upserted += len(batch)
#                         continue
#                     except TypeError:
#                         # attempt building PointStruct objects if qdrant_models available
#                         if qdrant_models is not None:
#                             point_objs = []
#                             for p in batch:
#                                 vec = p["vector"]
#                                 # If named vector, create NamedVector etc when models support it
#                                 try:
#                                     point_objs.append(qdrant_models.PointStruct(id=p["id"], vector=vec, payload=p["payload"]))
#                                 except Exception:
#                                     # fallback to raw dict
#                                     point_objs.append(p)
#                             try:
#                                 self.client.upsert(collection_name=self.collection_name, points=point_objs)
#                                 upserted += len(batch)
#                                 continue
#                             except Exception as e:
#                                 self.logger.debug("qdrant-client upsert fallback failed: %s", e)
#                                 # fallthrough to REST fallback
#                         else:
#                             # fallthrough to REST fallback
#                             pass
#                     except Exception as e:
#                         # If any other client-level error, try to continue or fallback to REST
#                         self.logger.debug("qdrant-client upsert error: %s", e)
#                         # try REST below if requests available

#                 # REST fallback
#                 if requests is None:
#                     raise RuntimeError("No available Qdrant client (qdrant-client not usable and requests missing).")
#                 url = f"{self.host}/collections/{self.collection_name}/points?wait=true"
#                 headers = {"Content-Type": "application/json"}
#                 if self.api_key:
#                     headers["X-API-Key"] = self.api_key

#                 # REST expects per-point shape: {"id":..., "vector": <list or named dict>, "payload": {...}}
#                 body = {"points": []}
#                 for p in batch:
#                     # For some Qdrant versions the named vector must be in "vector": {"name":"default","vector":[..]}
#                     vec = p["vector"]
#                     if isinstance(vec, dict):
#                         # already wrapped as {named_key: [..]} earlier -> convert to named vector for REST
#                         # choose the first item
#                         k, v = next(iter(vec.items()))
#                         # try both possible named-vector fields, prefer 'vector'
#                         named_vec_variant = {"name": k, "vector": v}
#                         # include as-is; we'll let server validate exact shape
#                         body["points"].append({"id": p["id"], "vector": named_vec_variant, "payload": p["payload"]})
#                     else:
#                         body["points"].append({"id": p["id"], "vector": vec, "payload": p["payload"]})

#                 resp = requests.post(url, json=body, headers=headers, timeout=30)
#                 if resp.status_code not in (200, 201):
#                     self.logger.error("Qdrant REST upsert failed: %s %s", resp.status_code, resp.text)
#                     for p in batch:
#                         failed.append(p)
#                 else:
#                     upserted += len(batch)
#         except Exception as e:
#             self.logger.exception("Upsert failed: %s", e)
#             return upserted, failed

#         return upserted, failed

#     # ----------------------
#     # Search
#     # ----------------------
#     def search_by_vector(
#         self,
#         query_vector,
#         top_k: int = 5,
#         filter_meta: Optional[Dict[str, Any]] = None,
#         with_payload: bool = True,
#     ) -> List[Dict[str, Any]]:
#         """
#         Perform a similarity search in Qdrant using the provided query vector.
#         Returns list of dicts: {id, score, payload, vector}
#         """
#         logger = self.logger

#         # Basic client guard
#         if (not hasattr(self, "client")) or self.client is None:
#             # we still allow REST fallback, so only log here
#             logger.debug("Qdrant client object not present; using REST if available.")

#         try:
#             # --- detect named vector key (best-effort) ---
#             named_vector_key = None
#             try:
#                 if self.client:
#                     info = self.client.get_collection(self.collection_name)
#                     cfg = getattr(info, "config", None)
#                     if cfg is not None and hasattr(cfg, "params"):
#                         vectors = getattr(cfg.params, "vectors", None)
#                         if isinstance(vectors, dict) and vectors:
#                             named_vector_key = next(iter(vectors.keys()))
#             except Exception:
#                 named_vector_key = None

#             # Normalize query vector if user passed named dict
#             qvec = query_vector
#             if isinstance(qvec, dict) and named_vector_key and named_vector_key in qvec:
#                 qvec = qvec.get(named_vector_key)

#             # Ensure qvec is a list of floats
#             if hasattr(qvec, "tolist"):
#                 qvec = qvec.tolist()
#             if not isinstance(qvec, list):
#                 # cannot search with non-list
#                 logger.error("Invalid query vector type for search: %s", type(qvec))
#                 return []

#             # --- Try qdrant-client search first if available ---
#             if self.client:
#                 try:
#                     # Many qdrant-client versions accept query_vector + vector_name args
#                     if named_vector_key:
#                         results = self.client.search(
#                             collection_name=self.collection_name,
#                             query_vector=qvec,
#                             limit=top_k,
#                             with_payload=with_payload,
#                             vector_name=named_vector_key,
#                         )
#                     else:
#                         results = self.client.search(
#                             collection_name=self.collection_name,
#                             query_vector=qvec,
#                             limit=top_k,
#                             with_payload=with_payload,
#                         )
#                     out = []
#                     for r in results:
#                         out.append(
#                             {
#                                 "id": getattr(r, "id", None),
#                                 "score": getattr(r, "score", None),
#                                 "payload": getattr(r, "payload", None),
#                                 "vector": getattr(r, "vector", None),
#                             }
#                         )
#                     return out
#                 except Exception as e_client:
#                     logger.debug("qdrant-client search failed (falling back to REST): %s", e_client)

#             # --- REST fallback ---
#             if requests is None:
#                 logger.error("No REST client available for Qdrant fallback.")
#                 return []

#             url = f"{self.host}/collections/{self.collection_name}/points/search"
#             headers = {"Content-Type": "application/json"}
#             if self.api_key:
#                 headers["X-API-Key"] = self.api_key

#             # REST vector formats vary between versions; try a few shapes.
#             tried_payloads = []

#             # 1) Named vector shape the server often accepts
#             if named_vector_key:
#                 body1 = {"vector": {"name": named_vector_key, "vector": qvec}, "limit": top_k, "with_payload": with_payload}
#                 tried_payloads.append(body1)

#                 body2 = {"vector": {"name": named_vector_key, "values": qvec}, "limit": top_k, "with_payload": with_payload}
#                 tried_payloads.append(body2)

#             # 2) Plain vector as list (unnamed collection)
#             body_plain = {"vector": qvec, "limit": top_k, "with_payload": with_payload}
#             tried_payloads.append(body_plain)

#             # 3) Some older/newer endpoints accept "query": { ... } wrapper - include as last resort
#             body_query_wrapper = {"query": {"vector": qvec, "top": top_k}, "with_payload": with_payload}
#             tried_payloads.append(body_query_wrapper)

#             # If filter_meta is provided - try to attach it where appropriate (server may ignore unsupported keys)
#             if filter_meta:
#                 # place under "filter" if shape supports it
#                 for pb in tried_payloads:
#                     if isinstance(pb, dict):
#                         pb["filter"] = filter_meta

#             response = None
#             results_json = None
#             last_error = None
#             for body in tried_payloads:
#                 try:
#                     logger.debug("Qdrant REST search trying payload keys=%s named=%s", list(body.keys()), bool(named_vector_key))
#                     r = requests.post(url, json=body, headers=headers, timeout=15)
#                     if r.status_code != 200:
#                         # log and try next payload
#                         logger.debug("Qdrant REST search returned status %s for payload (len=%d). body preview: %.200s", r.status_code, len(json.dumps(body)), r.text[:400])
#                         last_error = (r.status_code, r.text)
#                         continue
#                     # got OK
#                     response = r
#                     results_json = r.json()
#                     break
#                 except Exception as re:
#                     last_error = re
#                     logger.debug("Qdrant REST search attempt failed: %s", re)
#                     continue

#             if response is None:
#                 logger.error("Qdrant REST search failed (all payloads tried). last_error=%s", last_error)
#                 return []

#             # Normalize response
#             hits = []
#             if isinstance(results_json, list):
#                 # some older server shapes return top-level list
#                 hits = results_json
#             elif isinstance(results_json, dict):
#                 # look for various shapes
#                 maybe = results_json.get("result")
#                 if isinstance(maybe, dict):
#                     hits = maybe.get("hits") or []
#                 elif isinstance(maybe, list):
#                     hits = maybe
#                 if not hits:
#                     hits = results_json.get("hits") or []
#                 if not hits:
#                     # sometimes "result" is already list
#                     for v in results_json.values():
#                         if isinstance(v, list):
#                             hits = v
#                             break
#             else:
#                 logger.error("Unexpected Qdrant REST search response type: %s", type(results_json))
#                 return []

#             if not isinstance(hits, list):
#                 logger.error("Qdrant search hits normalization failed; type=%s", type(hits))
#                 return []

#             out = []
#             for h in hits:
#                 if not isinstance(h, dict):
#                     continue
#                 hid = h.get("id")
#                 score = h.get("score")
#                 payload = h.get("payload")
#                 vector = h.get("vector")
#                 # some shapes: {"point": {...}, "score":...}
#                 if hid is None and isinstance(h.get("point"), dict):
#                     point = h.get("point")
#                     hid = point.get("id")
#                     if payload is None:
#                         payload = point.get("payload")
#                     if vector is None:
#                         vector = point.get("vector")
#                 out.append({"id": hid, "score": score, "payload": payload, "vector": vector})

#             return out

#         except Exception as e:
#             logger.exception("Qdrant search failed (unexpected): %s", e)
#             return []

#===============================================================================================================================================================================


# # app/services/qdrant_service.py
# """
# Resilient Qdrant HTTP client for this project.

# Goals:
# - Detect collection's named vectors and vector dimension on init (if exists).
# - Create collection with robust payload shapes (try several).
# - Upsert points using the correct 'vectors' (named) field.
# - Search using multiple compatible payload shapes and fallback gracefully.
# - Keep last_payload_sent and last_server_response for diagnostics.
# """

# from typing import Any, Dict, List, Optional, Tuple
# import uuid
# import json
# import time
# from loguru import logger
# import httpx

# HTTP_TIMEOUT = 20.0


# class QdrantService:
#     def __init__(
#         self,
#         host: str = "http://localhost:6333",
#         collection: str = "asklyne_collection",
#         default_vector_name: str = "default",
#     ):
#         self.host = host.rstrip("/")
#         self.collection = collection
#         self.default_vector_name = default_vector_name

#         # discovered collection state
#         self._named_vectors: Dict[str, int] = {}
#         self._vector_size: Optional[int] = None

#         # diagnostics
#         self.last_payload_sent: Optional[Dict[str, Any]] = None
#         self.last_server_response: Optional[Dict[str, Any]] = None

#         # http client
#         self._http = httpx.Client(timeout=HTTP_TIMEOUT)

#         # attempt to read collection metadata if exists
#         try:
#             self._probe_collection()
#         except Exception as e:
#             logger.exception("QdrantService init probe failed: %s", e)

#     # ---------------------------
#     # Helpers
#     # ---------------------------
#     def _url(self, path: str) -> str:
#         return f"{self.host}{path}"

#     def _set_last(self, endpoint: str, body: Any, resp: Optional[httpx.Response]):
#         self.last_payload_sent = {"endpoint": endpoint, "body": body}
#         if resp is None:
#             self.last_server_response = None
#         else:
#             try:
#                 txt = resp.text
#             except Exception:
#                 txt = None
#             self.last_server_response = {"status_code": resp.status_code, "text": txt}

#     def _safe_post(self, endpoint: str, body: Any) -> httpx.Response:
#         url = self._url(endpoint)
#         # store stringified body for quick debugging
#         body_to_send = body
#         self.last_payload_sent = {"endpoint": url, "body": body}
#         resp = self._http.post(url, json=body_to_send)
#         self._set_last(url, body_to_send, resp)
#         return resp

#     def _safe_put(self, endpoint: str, body: Any) -> httpx.Response:
#         url = self._url(endpoint)
#         self.last_payload_sent = {"endpoint": url, "body": body}
#         resp = self._http.put(url, json=body)
#         self._set_last(url, body, resp)
#         return resp

#     def _safe_get(self, endpoint: str) -> httpx.Response:
#         url = self._url(endpoint)
#         self.last_payload_sent = {"endpoint": url, "body": None}
#         resp = self._http.get(url)
#         self._set_last(url, None, resp)
#         return resp

#     # ---------------------------
#     # Collection probing & creation
#     # ---------------------------
#     def _probe_collection(self):
#         """Query GET /collections/<collection> and fill self._named_vectors and _vector_size."""
#         try:
#             resp = self._safe_get(f"/collections/{self.collection}")
#             if resp.status_code != 200:
#                 logger.debug("Collection not found during probe: %s", resp.status_code)
#                 return
#             data = resp.json().get("result") or resp.json()
#             params = data.get("config", {}).get("params", {}) if isinstance(data, dict) else {}
#             vectors = params.get("vectors") if params else None
#             # older qdrant variants may have 'params.vectors' or top-level 'vectors'
#             if not vectors:
#                 vectors = data.get("vectors") or data.get("params", {}).get("vectors")
#             if isinstance(vectors, dict):
#                 self._named_vectors = {}
#                 for name, info in vectors.items():
#                     # info may be a dict {'size': 1024, ...} or a number (legacy)
#                     if isinstance(info, dict):
#                         size = info.get("size")
#                     else:
#                         # if vectors: { "default": 1024 }
#                         size = int(info) if info else None
#                     if size:
#                         self._named_vectors[name] = int(size)
#                 # pick primary vector size if exists
#                 if self._named_vectors:
#                     # pick default name if present else first
#                     if self.default_vector_name in self._named_vectors:
#                         self._vector_size = self._named_vectors[self.default_vector_name]
#                     else:
#                         first_name = next(iter(self._named_vectors.keys()))
#                         self.default_vector_name = first_name
#                         self._vector_size = self._named_vectors[first_name]
#             else:
#                 # no vectors found; leave defaults
#                 logger.debug("No vectors info found in collection metadata.")
#         except Exception as e:
#             logger.debug("Exception probing collection: %s", e)

#     def _create_collection_try_variants(self, dim: int, distance: str = "Cosine") -> Tuple[bool, List[Tuple[Any, Dict[str, Any]]]]:
#         """
#         Try multiple payload shapes to create the collection and return success + list of attempts (payload, server_resp).
#         """
#         attempts = []
#         # Variant A: modern recommended shape
#         payload_a = {"vectors": {self.default_vector_name: {"size": dim, "distance": distance}}}
#         # Variant B: alternative with params wrapper
#         payload_b = {"params": {"vectors": {self.default_vector_name: {"size": dim, "distance": distance}}}}
#         # Variant C: very old numeric form
#         payload_c = {"vectors": {self.default_vector_name: dim}}
#         for body in (payload_a, payload_b, payload_c):
#             try:
#                 resp = self._safe_put(f"/collections/{self.collection}", body)
#                 attempts.append((body, {"status_code": resp.status_code, "text": resp.text}))
#                 if resp.status_code in (200, 201):
#                     # refresh probe
#                     time.sleep(0.05)
#                     self._probe_collection()
#                     return True, attempts
#                 # 409 means already exists (fine)
#                 if resp.status_code == 409:
#                     self._probe_collection()
#                     return True, attempts
#             except Exception as e:
#                 attempts.append((body, {"error": str(e)}))
#         return False, attempts

#     # ---------------------------
#     # Upsert
#     # ---------------------------
#     def upsert_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 64, wait: bool = True) -> Tuple[int, List[Dict[str, Any]]]:
#         """
#         chunks: list of {"id": optional, "embedding": [...], "payload": {...}}
#         Returns (ok_count, failed_list)
#         """
#         ok = 0
#         failed = []
#         # Ensure collection exists and vector size known
#         if not self._vector_size:
#             # Try to discover from first chunk if embedder provides dim
#             if chunks and "embedding" in chunks[0]:
#                 candidate_dim = len(chunks[0]["embedding"])
#                 # try create collection with that dim
#                 created, attempts = self._create_collection_try_variants(candidate_dim)
#                 if created:
#                     logger.info("Created collection using candidate dim=%s", candidate_dim)
#                 else:
#                     logger.warning("All attempts to create collection failed: %s", attempts)
#             # re-probe anyway
#             self._probe_collection()

#         # Prepare point bodies in batches
#         for i in range(0, len(chunks), batch_size):
#             batch = chunks[i : i + batch_size]
#             points = []
#             for c in batch:
#                 pid = None
#                 if isinstance(c, dict):
#                     pid = c.get("id")
#                     emb = c.get("embedding")
#                     payload = c.get("payload", {})
#                 else:
#                     # allow dataclass-like with attributes (defensive)
#                     pid = getattr(c, "id", None)
#                     emb = getattr(c, "embedding", None)
#                     payload = getattr(c, "payload", {}) or {}
#                 if pid is None:
#                     pid = str(uuid.uuid4())
#                 if emb is None:
#                     logger.warning("Skipping chunk without embedding: %s", pid)
#                     failed.append({"point": {"id": pid}, "error": "no embedding"})
#                     continue
#                 # normalize embedding to python list of floats
#                 try:
#                     if hasattr(emb, "tolist"):
#                         emb = emb.tolist()
#                 except Exception:
#                     pass
#                 emb = list(map(float, emb))
#                 # If collection uses named vectors -> use 'vectors': {name: [...]}
#                 if self._named_vectors:
#                     vectors_field = {self.default_vector_name: emb}
#                     point = {"id": pid, "vectors": vectors_field, "payload": payload}
#                 else:
#                     # legacy: use 'vector' top-level
#                     point = {"id": pid, "vector": emb, "payload": payload}
#                 points.append(point)

#             if not points:
#                 continue

#             params = "?wait=true" if wait else ""
#             endpoint = f"/collections/{self.collection}/points{params}"
#             body = {"points": points}
#             try:
#                 resp = self._safe_put(endpoint, body)
#                 # success: 200
#                 if resp.status_code in (200, 201):
#                     ok += len(points)
#                 else:
#                     # on 404 attempt to create collection and retry once
#                     if resp.status_code == 404:
#                         logger.info("Collection not found during upsert; attempting to create with dim=%s", len(points[0].get("vectors", {}).get(self.default_vector_name, []) if self._named_vectors else points[0].get("vector", [])))
#                         dim = len(points[0].get("vectors", {}).get(self.default_vector_name, []) if self._named_vectors else points[0].get("vector", []))
#                         created, attempts = self._create_collection_try_variants(dim)
#                         if created:
#                             resp2 = self._safe_put(endpoint, body)
#                             if resp2.status_code in (200, 201):
#                                 ok += len(points)
#                             else:
#                                 failed.append({"points": points, "error": resp2.text, "status": resp2.status_code})
#                         else:
#                             failed.append({"points": points, "error": "create_failed", "attempts": attempts})
#                     else:
#                         failed.append({"points": points, "error": resp.text, "status": resp.status_code})
#             except Exception as e:
#                 logger.exception("Raw HTTP upsert failed: %s", e)
#                 failed.append({"points": points, "error": str(e)})
#         return ok, failed

#     # ---------------------------
#     # Search / retrieve
#     # ---------------------------
#     def _try_search_shapes(self, vector: List[float], top_k: int = 10, filter_meta: Optional[Dict[str, Any]] = None, with_payload: bool = True) -> List[Dict[str, Any]]:
#         """
#         Try a few different JSON shapes for the search endpoint (Qdrant versions vary).
#         Returns list of hits (empty list if none).
#         Also sets last_payload_sent / last_server_response for debugging.
#         """
#         endpoint = f"/collections/{self.collection}/points/search"
#         # Normalize
#         if hasattr(vector, "tolist"):
#             vector = vector.tolist()
#         vector = list(map(float, vector))

#         # Build different shapes
#         tries = []

#         # shape 1: modern search wrapper with 'search': {'knn': {...}}
#         body1 = {
#             "limit": top_k,
#             "with_payload": with_payload,
#             "search": {"knn": {"field": self.default_vector_name, "vector": vector, "top": top_k}}
#         }
#         tries.append(body1)

#         # shape 2: older qdrant - top_vector_named: (vector_name + vector + top)
#         body2 = {"limit": top_k, "with_payload": with_payload, "vector_name": self.default_vector_name, "vector": vector, "top": top_k}
#         tries.append(body2)

#         # shape 3: top_vector (no name) - only valid for single unnamed vector collections
#         body3 = {"limit": top_k, "with_payload": with_payload, "vector": vector, "top": top_k}
#         tries.append(body3)

#         # shape 4: query.knn (experimental)
#         body4 = {"limit": top_k, "with_payload": with_payload, "query": {"knn": {"field": self.default_vector_name, "vector": vector, "top": top_k}}}
#         tries.append(body4)

#         # optionally add filter to each
#         if filter_meta:
#             for b in tries:
#                 b["filter"] = filter_meta

#         # try each
#         for body in tries:
#             try:
#                 resp = self._safe_post(endpoint, body)
#                 # If HTTP 200 parse response
#                 if resp.status_code == 200:
#                     # Qdrant sometimes returns {"result":[...]} or {"result":{"hits": [...]}} or raw list
#                     try:
#                         j = resp.json()
#                     except Exception:
#                         j = None
#                     if not j:
#                         return []
#                     # result could be list or dict
#                     if isinstance(j, dict):
#                         # modern: j["result"]["hits"]
#                         if "result" in j and isinstance(j["result"], dict) and "hits" in j["result"]:
#                             return j["result"]["hits"]
#                         # sometimes result itself is the list
#                         if "result" in j and isinstance(j["result"], list):
#                             return j["result"]
#                         # sometimes top-level list in 'result' not present, try top-level hits
#                         if "hits" in j:
#                             return j["hits"]
#                         # or raw list at top-level keys
#                         # fallback: if j is shaped as list-like inside other keys, find first list of dicts
#                         for v in j.values():
#                             if isinstance(v, list):
#                                 candidate = v
#                                 if candidate and isinstance(candidate[0], dict) and "id" in candidate[0]:
#                                     return candidate
#                         # no recognized shape
#                         return []
#                     elif isinstance(j, list):
#                         return j
#                 else:
#                     # if server returned error, log and continue to next shape
#                     logger.debug("Qdrant search attempt failed status=%s text=%s", resp.status_code, resp.text)
#                     # keep trying
#                     continue
#             except Exception as e:
#                 logger.debug("Qdrant search attempt exception: %s", e)
#                 continue
#         # all attempts failed
#         return []

#     def search_by_vector(self, vector: List[float], top_k: int = 10, filter_meta: Optional[Dict[str, Any]] = None, with_payload: bool = True) -> List[Dict[str, Any]]:
#         """
#         High-level search helper. Returns list of hits (each hit is a dict with id, score, payload).
#         """
#         # safety convert
#         if hasattr(vector, "tolist"):
#             vector = vector.tolist()
#         try:
#             vector = list(map(float, vector))
#         except Exception:
#             raise ValueError("Vector must be convertible to list of floats")

#         # verify shape if known
#         if self._vector_size and len(vector) != self._vector_size:
#             # try to recreate collection with new dim - only if it looks intentional (rare)
#             logger.warning("Vector dim mismatch (%s vs %s). Attempting to recreate collection to match incoming dim.", self._vector_size, len(vector))
#             created, _ = self._create_collection_try_variants(len(vector))
#             if created:
#                 # refresh
#                 time.sleep(0.05)
#                 self._probe_collection()
#             else:
#                 # if not created, return empty to avoid crashes
#                 logger.warning("Collection not recreated; returning empty search.")
#                 return []

#         hits = self._try_search_shapes(vector=vector, top_k=top_k, filter_meta=filter_meta, with_payload=with_payload)
#         # Normalize hit shape to a list of dicts with id/score/payload
#         normalized = []
#         for h in hits or []:
#             if not isinstance(h, dict):
#                 continue
#             # older Qdrant returns "payload" nested under "payload"
#             payload = h.get("payload") or h.get("payloads") or {}
#             score = h.get("score") or h.get("dist") or 0.0
#             normalized.append({"id": h.get("id"), "score": float(score), "payload": payload})
#         return normalized





