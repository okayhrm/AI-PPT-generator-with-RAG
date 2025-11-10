# embedder.py — local hashing embeddings with optional Google fallback (Py3.13)
from __future__ import annotations
import os, re, math, hashlib
from typing import List, Sequence

try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
except Exception:
    GoogleGenerativeAIEmbeddings = None  # optional

_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-']+")

def _tokenize(text: str) -> List[str]:
    return _WORD.findall((text or "").lower())

def _hash_to_index(s: str, dim: int) -> int:
    h = hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "little") % dim

def _hash_sign(s: str) -> int:
    b = hashlib.blake2b((s + "#").encode("utf-8"), digest_size=8).digest()[-1]
    return 1 if (b & 1) == 0 else -1

def _normalize(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(v*v for v in vec)) or 1.0
    return [v / norm for v in vec]

class LocalHashingEmbeddings:
    """Quota-free, deterministic feature-hashing embeddings."""
    def __init__(self, dim: int = 768):
        self.dim = int(dim)

    def _embed_one(self, text: str) -> List[float]:
        v = [0.0] * self.dim
        for tok in _tokenize(text):
            j = _hash_to_index(tok, self.dim)
            v[j] += _hash_sign(tok)  # signed hashing to reduce bias
        return _normalize(v)

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        return [self._embed_one(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_one(text)

def get_embedder():
    """
    Returns an object with:
      - embed_documents(List[str]) -> List[List[float]]
      - embed_query(str) -> List[float]
    Env:
      EMBED_PROVIDER=google|local|auto  (default: auto)
      EMBED_DIM=768                      (must match Pinecone index)
      GOOGLE_API_KEY / GEMINI_API_KEY    (for google provider)
    """
    provider = (os.getenv("EMBED_PROVIDER") or "auto").lower()
    dim = int(os.getenv("EMBED_DIM") or 768)

    if provider == "local":
        return LocalHashingEmbeddings(dim=dim)

    if GoogleGenerativeAIEmbeddings is not None:
        key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if key and provider in ("auto", "google"):
            try:
                return GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001",
                    google_api_key=key,
                )
            except Exception:
                pass  # fall back to local

    return LocalHashingEmbeddings(dim=dim)
