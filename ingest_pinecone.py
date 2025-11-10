import os, uuid
from typing import List, Dict
from pypdf import PdfReader
from pinecone import Pinecone, ServerlessSpec
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX   = os.getenv("PINECONE_INDEX", "ppt-rag")
PINECONE_CLOUD   = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION  = os.getenv("PINECONE_REGION", "us-east-1")
UPLOADS_DIR      = os.getenv("UPLOADS_DIR", "./data/uploads")
NAMESPACE        = os.getenv("NAMESPACE")  # optional; set from UI per session

pc = Pinecone(api_key=PINECONE_API_KEY)

def ensure_index(dim=768, metric="cosine"):
    if PINECONE_INDEX not in [i.name for i in pc.list_indexes()]:
        pc.create_index(
            name=PINECONE_INDEX, dimension=dim, metric=metric,
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION)
        )
    return pc.Index(PINECONE_INDEX)

index = ensure_index(768)
emb   = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # 768-d

def pdf_to_text(path: str) -> str:
    r = PdfReader(path)
    return "\n".join([(p.extract_text() or "") for p in r.pages])

def text_to_chunks(full: str, max_chars=1200, overlap=200) -> List[Dict]:
    chunks, i = [], 0
    while i < len(full):
        chunk = full[i:i+max_chars]
        chunks.append({"chunk_id": str(uuid.uuid4()), "content": chunk})
        i += max_chars - overlap
    return chunks

def upsert_chunks(doc_id: str, filename: str, chunks: List[Dict], namespace: str | None):
    texts = [c["content"] for c in chunks]
    vecs  = emb.embed_documents(texts)
    B = 100
    for i in range(0, len(chunks), B):
        batch = []
        for c, v in zip(chunks[i:i+B], vecs[i:i+B]):
            batch.append({
                "id": c["chunk_id"],
                "values": v,
                "metadata": {
                    "doc_id": doc_id,
                    "filename": filename,
                    "content": c["content"][:4000]
                }
            })
        index.upsert(vectors=batch, namespace=namespace)

if __name__ == "__main__":
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    pdfs = [f for f in os.listdir(UPLOADS_DIR) if f.lower().endswith(".pdf")]
    if not pdfs:
        print("No PDFs found in", UPLOADS_DIR)
    for fname in pdfs:
        path = os.path.join(UPLOADS_DIR, fname)
        txt = pdf_to_text(path)
        chunks = text_to_chunks(txt)
        upsert_chunks(str(uuid.uuid4()), fname, chunks, namespace=NAMESPACE)
        print(f"Ingested {fname} → {len(chunks)} chunks ✅ (namespace={NAMESPACE or 'default'})")
