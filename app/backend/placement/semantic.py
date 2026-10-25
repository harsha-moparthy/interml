"""Local semantic search model. Uses fastembed (ONNX, CPU) with a model cached under app/data/models.
Run `python -m placement.semantic download` once. Everything works without it; semantic search is then off."""
import sys
import threading
from functools import lru_cache

import numpy as np

from . import config

import os

MODEL_NAME = os.environ.get("PLACEMENT_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
MODEL_DIR = config.DATA_DIR / "models"
_lock = threading.Lock()
_model = None


def available() -> bool:
    try:
        import fastembed  # noqa: F401
    except ImportError:
        return False
    return MODEL_DIR.exists() and any(MODEL_DIR.iterdir())


def _load():
    global _model
    with _lock:
        if _model is None:
            from fastembed import TextEmbedding
            _model = TextEmbedding(MODEL_NAME, cache_dir=str(MODEL_DIR))
        return _model


def embed_documents(texts: list[str]) -> np.ndarray:
    vecs = np.array(list(_load().embed(texts, batch_size=32)), dtype=np.float32)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


@lru_cache(maxsize=512)
def embed_query(text: str) -> np.ndarray:
    v = np.array(list(_load().query_embed([text]))[0], dtype=np.float32)
    return v / np.linalg.norm(v)


if __name__ == "__main__":
    if sys.argv[1:] == ["download"]:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        _load()
        print(f"model {MODEL_NAME} ready in {MODEL_DIR}")
    else:
        print("usage: python -m placement.semantic download")
