# app/services/qdrant_service.py
import uuid
import httpx
import time
from typing import List, Dict, Any, Optional, Tuple

DEFAULT_HOST = "http://localhost:6333"
DEFAULT_COLLECTION = "asklyne_collection"
DEFAULT_VECTOR_NAME = "default"
DEFAULT_VECTOR_SIZE = 1024
HTTP_TIMEOUT = 15.0

class QdrantService:
    def __init__(
        self,
        host: str = DEFAULT_HOST,
        collection_name: str = DEFAULT_COLLECTION,
        vector_name: str = DEFAULT_VECTOR_NAME,
    ):
        self.host = host.rstrip('/')
        self.collection_name = collection_name
        self.vector_name = vector_name
        self._http = httpx.Client(timeout=HTTP_TIMEOUT)
        self.last_payload_sent: Dict[str, Any] = {}
        self.last_server_response: Dict[str, Any] = {}
        # probe collection and set vector size / named vectors map
        self._named_vectors = {}
        self._vector_size: Optional[int] = None
        self._probe_or_create_collection()

    def _collection_url(self, suffix: str = "") -> str:
        base = f"{self.host}/collections/{self.collection_name}"
        return f"{base}{suffix}"

    def _probe_or_create_collection(self):
        # Probe
        url = self._collection_url("")
        try:
            r = self._http.get(url)
            self.last_server_response = {"status_code": r.status_code, "text": r.text}
            if r.status_code == 200:
                info = r.json().get("result", {})
                params = info.get("config", {}).get("params", {})
                vectors = params.get("vectors") or {}
                # pick first named vector, or fallback
                if isinstance(vectors, dict):
                    for name, spec in vectors.items():
                        size = spec.get("size") if isinstance(spec, dict) else None
                        self._named_vectors[name] = size
                    if self._named_vectors and self.vector_name in self._named_vectors:
                        self._vector_size = self._named_vectors[self.vector_name]
                    elif self._named_vectors:
                        # choose the first one
                        n, s = next(iter(self._named_vectors.items()))
                        self.vector_name = n
                        self._vector_size = s
                if self._vector_size is None:
                    # fallback
                    self._vector_size = DEFAULT_VECTOR_SIZE
                    self._named_vectors[self.vector_name] = self._vector_size
            elif r.status_code == 404:
                # create
                self._create_collection(DEFAULT_VECTOR_SIZE)
            else:
                # unknown: try to create
                self._create_collection(DEFAULT_VECTOR_SIZE)
        except Exception as exc:
            # if network down, set defaults and propagate later
            self._vector_size = DEFAULT_VECTOR_SIZE
            self._named_vectors[self.vector_name] = self._vector_size

    def _create_collection(self, size: int):
        url = self._collection_url("")
        payload = {"vectors": {self.vector_name: {"size": size, "distance": "Cosine"}}}
        self.last_payload_sent = {"endpoint": url, "body": payload}
        r = self._http.put(url, json=payload)
        self.last_server_response = {"status_code": r.status_code, "text": r.text}
        if r.status_code == 200:
            self._vector_size = size
            self._named_vectors[self.vector_name] = size
        else:
            # still set fallback
            self._vector_size = size
            self._named_vectors[self.vector_name] = size

    def upsert_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 64, wait: bool = True) -> Tuple[int, List[Any]]:
        """
        chunks: list of dicts { id: Optional[str|None], embedding: List[float], payload: dict }
        returns: (ok_count, failed_list)
        """
        points = []
        failed = []
        ok_count = 0

        for c in chunks:
            emb = c.get("embedding") if isinstance(c, dict) else getattr(c, "embedding", None)
            payload = c.get("payload") if isinstance(c, dict) else getattr(c, "payload", None)
            cid = c.get("id") if isinstance(c, dict) else getattr(c, "id", None)
            if emb is None:
                failed.append({"reason": "no_embedding", "chunk": c})
                continue
            # ensure plain python floats
            vector = [float(x) for x in emb]
            point_id = cid or str(uuid.uuid4())
            points.append({"id": point_id, "vectors": {self.vector_name: vector}, "payload": payload or {}})

        # chunk upload
        url = self._collection_url("/points")
        body = {"points": points}
        if wait:
            url = url + "?wait=true"
        self.last_payload_sent = {"endpoint": url, "body": body}
        r = self._http.put(url, json=body)
        self.last_server_response = {"status_code": r.status_code, "text": r.text}
        if r.status_code == 200:
            ok_count = len(points)
            # success for all points uploaded
            return ok_count, failed
        else:
            # treat all as failed if server returned something else
            return 0, [{"status_code": r.status_code, "text": r.text}]

    def _parse_search_response(self, r: httpx.Response) -> List[Dict[str, Any]]:
        self.last_server_response = {"status_code": r.status_code, "text": r.text}
        if r.status_code != 200:
            return []
        try:
            data = r.json()
        except Exception:
            return []
        # handle multiple server response shapes
        # common Qdrant responses:
        # - {"result": [ {id, score, payload}, ... ], "status":"ok"}
        # - {"result": {"result": [...], ...}, "status":"ok"}
        result = data.get("result")
        if isinstance(result, list):
            return result
        if isinstance(result, dict) and "result" in result and isinstance(result["result"], list):
            return result["result"]
        # some older variants might put hits under 'hits' or 'data'
        for key in ("hits", "data"):
            if isinstance(data.get(key), list):
                return data.get(key)
        return []

    # Put this function inside app/services/qdrant_service.py where your class QdrantService is defined.
    # Replace any existing search_by_vector / _try_search_shapes code with this (adjust indentation to match class).

    def search_by_vector(self, vector, top_k=5, filter_meta=None, with_payload=True, using=None, attempts=3):
        """
        Robust search wrapper that tries different JSON payload shapes until one succeeds.
        Returns list of hits (normalized) or [].
        """
        import time, requests, json, logging
        logger = getattr(self, "logger", logging.getLogger(__name__))

        host = getattr(self, "_host", getattr(self, "host", "http://localhost:6333"))
        collection = getattr(self, "collection_name", getattr(self, "_collection_name", "asklyne_collection"))
        url = f"{host}/collections/{collection}/points/search"
        vec = vector
        # ensure plain python list
        if hasattr(vec, "tolist"):
            vec = vec.tolist()

        # build filter if provided (Qdrant expects 'filter': <filter_obj>)
        qfilter = None
        if filter_meta:
            # We expect filter_meta to already be in qdrant 'must' format: {"must":[{"key":...}]}
            # Wrap to a top-level filter if necessary
            qfilter = filter_meta if isinstance(filter_meta, dict) else None

        # variant payloads to try (ordered)
        payload_variants = []

        # A: simple vector top/limit (old)
        payload_variants.append({"vector": vec, "limit": top_k, "with_payload": bool(with_payload)})

        # B: named vector (explicit name)
        payload_variants.append({"vector": {"name": using or "default", "vector": vec}, "limit": top_k, "with_payload": bool(with_payload)})

        # C: knn search wrapper
        payload_variants.append({"search": {"knn": {"field": using or "default", "vector": vec, "top": top_k}}, "with_payload": bool(with_payload)})

        # D: knn with top inside 'search' (alternative)
        payload_variants.append({"search": {"knn": {"field": using or "default", "vector": vec, "top": top_k}}, "with_payload": bool(with_payload), "filter": qfilter} if qfilter else {"search": {"knn": {"field": using or "default", "vector": vec, "top": top_k}}, "with_payload": bool(with_payload)})

        # E: using + vector + filter
        if qfilter:
            payload_variants.append({"vector": vec, "limit": top_k, "with_payload": bool(with_payload), "using": using or "default", "filter": qfilter})

        last_err = None
        for payload in payload_variants:
            # attach filter if missing and present
            if qfilter and "filter" not in payload:
                payload["filter"] = qfilter
            self.last_payload_sent = {"endpoint": url, "body": payload}
            try:
                resp = requests.post(url, json=payload, timeout=15)
                self.last_server_response = {"status_code": resp.status_code, "text": resp.text}
                if resp.status_code == 200:
                    # parse and return normalized hits
                    try:
                        data = resp.json()
                    except Exception:
                        logger.debug("search response not json: %s", resp.text)
                        return []
                    # normalize depending on returned shape
                    # Qdrant sometimes returns {"result": [...]} or {"result": {"ids": ...}} or direct list
                    hits = []
                    if isinstance(data, dict):
                        # common case: {"result":[{...},...], "status":"ok"}
                        if "result" in data and isinstance(data["result"], list):
                            hits = data["result"]
                        # sometimes result is dict with 'hits' or points list
                        elif "result" in data and isinstance(data["result"], dict):
                            # try to find list inside
                            for key in ("hits", "vectors", "points", "result"):
                                if isinstance(data["result"].get(key), list):
                                    hits = data["result"].get(key)
                                    break
                        elif "hits" in data and isinstance(data["hits"], list):
                            hits = data["hits"]
                    if not hits and isinstance(data, list):
                        hits = data
                    # final normalization: ensure each hit has id/payload/score where possible
                    normalized = []
                    for h in hits:
                        if isinstance(h, dict):
                            normalized.append(h)
                    return normalized
                else:
                    last_err = f"HTTP {resp.status_code} -> {resp.text[:400]}"
                    logger.debug("Qdrant search attempt failed status=%s text=%s", resp.status_code, resp.text[:800])
            except Exception as exc:
                last_err = str(exc)
                logger.exception("Qdrant search failed on payload attempt: %s", exc)
            # short pause between attempts
            time.sleep(0.1)
        # all failed
        logger.error("All search payload attempts failed. Last error: %s", last_err)
        self.last_server_response = {"error": last_err}
        return []
