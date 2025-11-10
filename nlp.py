# nlp.py — Python 3.13 friendly (no torch)
import re
from typing import List, Dict, Tuple
import numpy as np
import yake
from nltk.corpus import wordnet as wn

# Use Google Embeddings via LangChain wrapper (already in reqs)
from langchain_google_genai import GoogleGenerativeAIEmbeddings
_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # 768-d

# ---------- Intent detection (regex) ----------
INTENTS = {
    "rag_request": [
        r"\baccording to (the )?(pdf|document|upload|paper)\b",
        r"\buse (my|the) (docs?|pdfs?|uploads?)\b",
        r"\bchapter\b|\bsection\b|\bpolicy\b|\bagreement\b|\breport\b",
        r"\bbalance sheet\b|\bfinancial documents?\b|\bincome tax returns?\b",
    ],
    "web_search": [
        r"\b(latest|today|trend|market|news|2025 report)\b",
    ],
}

def detect_intent(text: str) -> Dict[str, bool]:
    t = (text or "").lower()
    return {
        "rag_request": any(re.search(p, t) for p in INTENTS["rag_request"]),
        "web_search":  any(re.search(p, t) for p in INTENTS["web_search"]),
    }

# ---------- Topic extraction (YAKE) ----------
_kw_1 = yake.KeywordExtractor(lan="en", n=1, top=20)
_kw_2 = yake.KeywordExtractor(lan="en", n=2, top=20)
_kw_3 = yake.KeywordExtractor(lan="en", n=3, top=20)

def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()

def extract_topics(text: str, top_k: int = 5) -> List[str]:
    if not text: return []
    cands = set()
    for kw in (_kw_1, _kw_2, _kw_3):
        try:
            for phrase, _ in kw.extract_keywords(text):
                phrase = _clean(phrase)
                if 2 <= len(phrase) <= 60:
                    cands.add(phrase)
        except Exception:
            pass
    topics = sorted(cands, key=lambda x: (-len(x), x))
    return topics[:top_k]

# ---------- Query expansion (WordNet) ----------
def expand_query(q: str, extra: List[str] | None = None, max_syn=2) -> str:
    toks = [t for t in re.findall(r"[A-Za-z0-9\-]+", (q or "").lower()) if len(t) > 2]
    heads = set(toks)
    syns = set()
    for h in list(heads)[:8]:
        for syn in wn.synsets(h)[:max_syn]:
            for lemma in syn.lemmas()[:1]:
                s = lemma.name().replace("_", " ").lower()
                if s != h and len(s) > 2:
                    syns.add(s)
    pieces = [q] + list(heads) + list(syns) + (extra or [])
    seen, parts = set(), []
    for p in pieces:
        p = _clean(str(p))
        if p and p not in seen:
            seen.add(p); parts.append(p)
    return " ".join(parts)

# ---------- Embedding utilities ----------
def _emb_vecs(texts: List[str]) -> np.ndarray:
    if not texts: return np.zeros((0, 768), dtype=np.float32)
    vecs = _embeddings.embed_documents(texts)  # List[List[float]]
    return np.array(vecs, dtype=np.float32)

def _emb_query(q: str) -> np.ndarray:
    v = _embeddings.embed_query(q or "")
    return np.array(v, dtype=np.float32)

def _cosine(a: np.ndarray, b: np.ndarray, eps=1e-9) -> np.ndarray:
    a = a.astype(np.float32); b = b.astype(np.float32)
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + eps)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + eps)
    return a @ b.T

# ---------- Re-rank hits (cosine sim with Google embeddings) ----------
def rerank_hits(query: str, hits: List[Dict], top_k: int = 6) -> List[Dict]:
    """
    hits: [{text, filename, score, doc_id}]
    Returns hits sorted by cosine similarity to query embedding.
    """
    if not hits: return []
    qv = _emb_query(query)  # (768,)
    docs = [h.get("text", "") for h in hits]
    dv = _emb_vecs(docs)    # (n, 768)
    sims = _cosine(qv[None, :], dv).flatten().tolist()
    for h, s in zip(hits, sims):
        h["ce_score"] = float(s)  # reuse same key name
    hits.sort(key=lambda x: x.get("ce_score", 0.0), reverse=True)
    return hits[:top_k]

# ---------- Faithfulness proxy (max cosine vs sources) ----------
def nli_faithfulness(bullet: str, sources: List[str]) -> float:
    """
    Torch-free proxy: cosine(query=bullet, docs=sources) max score in [0,1].
    """
    if not sources: return 0.0
    qv = _emb_query(bullet)        # (768,)
    dv = _emb_vecs(sources)        # (m, 768)
    sims = _cosine(qv[None, :], dv).flatten()
    return float(np.clip(sims.max() if sims.size else 0.0, 0.0, 1.0))
