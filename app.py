# app.py — Streamlit UI (Chat + RAG), Py3.13, Pinecone v5, local embeddings
from __future__ import annotations
import os, io, uuid
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

st.set_page_config(page_title="AI PPT Generator (Chat + RAG)", layout="wide")
st.title("AI PPT Generator (Local • Gemini • Pinecone)")
st.caption("Chat to refine the brief. Upload PDFs to ground with RAG. Then Build.")

load_dotenv()

# ---- try importing pipeline early but don't block UI
try:
    from ppt_gen import graph, TEMPLATES_DIR, DEPTH_DEFAULT
except Exception as e:
    st.error(f"Failed to import pipeline (ppt_gen): {e}")
    st.stop()

# ---------- cached heavy resources (Pinecone v5 + embedder) ----------
@st.cache_resource(show_spinner=False)
def get_pc_index_and_embeddings(_v: int = 2):
    """
    Initialize Pinecone index and the embedder (local hashing by default).
    Returns (index, emb) or raises RuntimeError with details.
    """
    from embedder import get_embedder
    try:
        from pinecone import Pinecone, ServerlessSpec
    except Exception as e:
        raise RuntimeError(f"Pinecone import failed: {e}")

    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX   = os.getenv("PINECONE_INDEX", "ppt-rag")
    PINECONE_CLOUD   = os.getenv("PINECONE_CLOUD", "aws")
    PINECONE_REGION  = os.getenv("PINECONE_REGION", "us-east-1")
    EMBED_DIM        = int(os.getenv("EMBED_DIM") or 768)

    if not PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY not set")

    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
    except Exception as e:
        raise RuntimeError(f"Pinecone client init failed: {e}")

    # v5-safe listing
    try:
        resp = pc.list_indexes()
        idx_names = [it.get("name") for it in (resp.get("indexes") or [])]
    except Exception as e:
        raise RuntimeError(f"Pinecone list_indexes failed: {e}")

    try:
        if os.getenv("RECREATE_INDEX") == "1" and (os.getenv("PINECONE_INDEX") in idx_names):
            pc.delete_index(PINECONE_INDEX)
            idx_names.remove(PINECONE_INDEX)
        if PINECONE_INDEX not in idx_names:
            pc.create_index(
                name=PINECONE_INDEX,
                dimension=EMBED_DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
            )
        index = pc.Index(PINECONE_INDEX)
    except Exception as e:
        raise RuntimeError(f"Pinecone create/open index failed: {e}")

    try:
        emb = get_embedder()  # local hashing by default (EMBED_PROVIDER=local|auto|google)
    except Exception as e:
        raise RuntimeError(f"Embedder init failed: {e}")

    return index, emb

# ---------- helpers ----------
def pdf_bytes_to_text(pdf_bytes: bytes) -> str:
    r = PdfReader(io.BytesIO(pdf_bytes))
    return "\n".join([(p.extract_text() or "") for p in r.pages])

def text_to_chunks(full: str, max_chars=1200, overlap=200):
    chunks, i = [], 0
    while i < len(full):
        chunk = full[i:i+max_chars]
        chunks.append({"chunk_id": str(uuid.uuid4()), "content": chunk})
        i += max_chars - overlap
    return chunks

def pinecone_upsert(index, emb, doc_id: str, filename: str, chunks, namespace: str):
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

# ---------- session ----------
if "project_id" not in st.session_state:
    st.session_state.project_id = str(uuid.uuid4())
if "history" not in st.session_state:
    st.session_state.history = []

namespace = st.session_state.project_id

# ---------- sidebar (uploads + settings) ----------
with st.sidebar:
    st.subheader("RAG Uploads")
    uploads = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    
    # Only initialize Pinecone if files are uploaded
    if uploads:
        try:
            index, emb = get_pc_index_and_embeddings()
        except Exception as e:
            st.error(f"Pinecone/Embeddings init failed: {e}")
            st.stop()

        total = 0
        for up in uploads:
            try:
                with st.spinner(f"Ingesting {up.name} …"):
                    txt = pdf_bytes_to_text(up.read())
                    chunks = text_to_chunks(txt)
                    pinecone_upsert(index, emb, str(uuid.uuid4()), up.name, chunks, namespace=namespace)
                    total += len(chunks)
            except Exception as e:
                st.error(f"Ingest {up.name} failed: {e}")
        st.success(f"Ingested {len(uploads)} file(s), {total} chunks ✅ (ns={namespace})")

    st.subheader("Content Depth")
    depth = st.radio("Depth", ["light", "detailed", "extensive"], index=1)

    st.subheader("Template")
    tdir = os.getenv("TEMPLATES_DIR", "./templates")
    try:
        tpls = [f for f in os.listdir(tdir) if f.endswith(".pptx")]
    except Exception as e:
        tpls = []
        st.error(f"Could not read templates dir '{tdir}': {e}")
    template = st.selectbox("Choose template", tpls or ["template1.pptx"], index=0)

    st.subheader("Retrieval Mode")
    force_rag = st.checkbox("Force RAG (use uploaded PDFs only)", value=False)

# ---------- tabs: Chat + Build ----------
tab_chat, tab_build = st.tabs(["💬 Chat", "🧱 Build Deck"])

with tab_chat:
    st.write("Use chat to refine the brief. Example: *\"Make tone formal and add an evaluation slide.\"*")

    for msg in st.session_state.history[-40:]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_msg = st.chat_input("Type your instructions…")
    if user_msg:
        st.session_state.history.append({"role": "user", "content": user_msg})
        with st.chat_message("assistant"):
            st.markdown("Noted ✅. These instructions will be applied to the next build.")
        st.session_state.history.append({"role": "assistant", "content": "Noted ✅. I will apply this in the next build."})
        st.rerun()

with tab_build:
    st.write("Final prompt/topic (used with your chat instructions):")
    topic = st.text_area(
        "Topic/Prompt",
        "Create a 10-slide deck on Retrieval-Augmented Generation (RAG) grounded in my uploads."
    )

    if st.button("Generate & Build PPT"):
        missing = [k for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY") if os.getenv(k)]
        # We allow slide gen with either GEMINI_API_KEY or GOOGLE_API_KEY set.
        if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
            st.error("Missing GEMINI_API_KEY or GOOGLE_API_KEY for Gemini slide generation.")
            st.stop()

        with st.spinner("Generating slides and building deck…"):
            try:
                # Initialize Pinecone here too if needed for RAG retrieval during generation
                if force_rag or uploads:
                    try:
                        index, emb = get_pc_index_and_embeddings()
                    except Exception as e:
                        st.error(f"Pinecone/Embeddings init failed: {e}")
                        st.stop()
                
                init_state = {
                    "prompt": topic,
                    "template": template,
                    "depth": depth,
                    "project_id": namespace,
                    "namespace": namespace,
                    "history": st.session_state.history,
                    "force_rag": bool(force_rag),
                }
                cfg = {
                    "model_name": "models/gemini-2.5-flash",
                    "selected_template": template,
                    "ui_mode": True,
                    "force_rag": bool(force_rag),
                }
                res = graph.invoke(init_state, config=cfg)
            except Exception as e:
                st.error(f"Graph failed: {e}")
                st.stop()

        st.session_state.history.append(
            {"role": "assistant", "content": f"Built deck (depth={depth}, force_rag={bool(force_rag)})."}
        )

        ppt_path = res.get("output_path")
        if not ppt_path:
            st.error("No output_path returned by the pipeline.")
        else:
            st.success("Deck ready!")
            st.code(ppt_path)
            try:
                with open(ppt_path, "rb") as f:
                    st.download_button(
                        "Download PPTX",
                        f,
                        file_name=os.path.basename(ppt_path),
                        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    )
            except Exception as e:
                st.warning(f"Download failed: {e}")