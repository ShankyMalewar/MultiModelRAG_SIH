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

    def search_by_vector(self, vector: List[float], top_k: int = 10, filter_meta: Optional[Dict] = None, with_payload: bool = True) -> List[Dict[str, Any]]:
        """
        Tries several payload shapes to support different Qdrant versions:
         - new: {"vector": {"name":"default","vector":[...]}, "limit": N, "with_payload": true}
         - knn style: {"search": {"knn": {"field":"default","vector":[...],"top":N}}, "with_payload": true}
         - legacy: {"query":{"vector":[...],"top":N}, "with_payload": true}
        """
        vector = [float(x) for x in vector]
        url = self._collection_url("/points/search")
        candidates = []

        # shape 1: vector top-level (Qdrant 1.x shape)
        body1 = {"vector": {"name": self.vector_name, "vector": vector}, "limit": top_k, "with_payload": with_payload}
        if filter_meta:
            body1["filter"] = filter_meta
        candidates.append(body1)

        # shape 2: search.knn style
        body2 = {"search": {"knn": {"field": self.vector_name, "vector": vector, "top": top_k}}, "with_payload": with_payload}
        if filter_meta:
            body2["search"]["knn"]["filter"] = filter_meta
        candidates.append(body2)

        # shape 3: legacy query.vector
        body3 = {"query": {"vector": vector, "top": top_k}, "with_payload": with_payload}
        if filter_meta:
            body3["query"]["filter"] = filter_meta
        candidates.append(body3)

        for b in candidates:
            try:
                self.last_payload_sent = {"endpoint": url, "body": b}
                r = self._http.post(url, json=b)
                hits = self._parse_search_response(r)
                if hits:
                    return hits
                # if server returned 200 but empty result, keep trying other shapes
            except Exception:
                continue

        # last attempt: return empty list and keep last_server_response set
        return []
