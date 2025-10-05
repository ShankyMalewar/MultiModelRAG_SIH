# app/config.py
import os

class Config:
    TOKEN_LIMITS = {
        "free": 2048,
        "plus": 4096,
        "pro": 8192,
    }
    DEFAULT_TOKEN_LIMIT = TOKEN_LIMITS["free"]
    QDRANT_HOST = os.getenv("QDRANT_HOST", "http://localhost:6333")
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "asklyne_collection")
    TYPESENSE_HOST = os.getenv("TYPESENSE_HOST", "http://localhost:8108")
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
    MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "./models_cache")
