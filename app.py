import tempfile
from pathlib import Path

import streamlit as st
import faiss
import numpy as np
from groq import Groq
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "all-MiniLM-L6-v2"
GEN_MODEL = "llama-3.1-8b-instant"

def read_file(file_path: Path) -> str:
    if file_path.suffix.lower() == ".pdf":
        reader = PdfReader(str(file_path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return file_path.read_text(encoding="utf-8", errors="ignore")


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 200):
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(text), step):
        chunk = text[i:i + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


@st.cache_resource
def get_embedder():
    return SentenceTransformer(EMBED_MODEL)


@st.cache_data(show_spinner=False)
def build_rag(file_bytes_map: dict):
    """Cache key is a dict of {filename: bytes}, so we only rebuild when files change."""
    chunks = []
    meta = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for name, data in file_bytes_map.items():
            temp_path = Path(tmpdir) / name
            temp_path.write_bytes(data)

            text = read_file(temp_path)
            file_chunks = chunk_text(text)

            for i, chunk in enumerate(file_chunks):
                chunks.append(chunk)
                meta.append({"source": name, "chunk_id": i})

    embedder = get_embedder()
    embeddings = embedder.encode(chunks, show_progress_bar=True, convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return embedder, index, chunks, meta


def retrieve(query, embedder, index, chunks, meta, top_k=4):
    query_embedding = embedder.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1:
            results.append({
                "score": float(score),
                "text": chunks[idx],
                "source": meta[idx]["source"],
                "chunk_id": meta[idx]["chunk_id"],
            })
    return results


def generate_answer(query, results=None):
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])

    if results:
        context = "\n\n".join(
            f"[Source {i+1}: {r['source']} | chunk {r['chunk_id']}]\n{r['text']}"
            for i, r in enumerate(results)
        )
        prompt = f"""Answer the question using only the context below.

Question:
{query}

Context:
{context}

If the answer is not present in the context, say so clearly."""
    else:
        prompt = query

    response = client.chat.completions.create(
        model=GEN_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


# ── UI ──────────────────────────────────────────────────────────────────────

st.title("Student RAG Demo")

uploaded_files = st.file_uploader(
    "Upload documents (optional — upload files to ground answers in your content)",
    type=["pdf", "txt", "md"],
    accept_multiple_files=True,
)

question = st.text_input("Ask a question")
ask = st.button("Ask", type="primary")

if ask and question:
    if uploaded_files:
        file_bytes_map = {f.name: f.getvalue() for f in uploaded_files}

        with st.spinner("Building index (only runs once per file set)..."):
            embedder, index, chunks, meta = build_rag(file_bytes_map)

        st.caption(f"Indexed {len(chunks)} chunks from {len(uploaded_files)} file(s).")

        results = retrieve(question, embedder, index, chunks, meta)

        # ── Retrieved chunks ──────────────────────────────────────────────
        st.subheader("Retrieved Chunks")
        st.caption(
            "When you upload documents, the text is split into overlapping **chunks** "
            "and converted into numeric vectors (embeddings) that capture meaning. "
            "When you ask a question, the most semantically similar chunks are retrieved "
            "and handed to the LLM as context — this is **Retrieval-Augmented Generation (RAG)**."
        )

        for i, r in enumerate(results):
            score_pct = int(r["score"] * 100)
            # Color the badge: green ≥ 80, yellow ≥ 60, red below
            if score_pct >= 80:
                badge = f"🟢 {score_pct}% match"
            elif score_pct >= 60:
                badge = f"🟡 {score_pct}% match"
            else:
                badge = f"🔴 {score_pct}% match"

            with st.expander(f"Chunk {i+1} — {r['source']}  ·  {badge}", expanded=(i == 0)):
                col1, col2 = st.columns([1, 1])
                col1.metric("Source", r["source"])
                col2.metric("Chunk index", r["chunk_id"])
                st.progress(r["score"], text=f"Similarity score: {r['score']:.3f}")
                st.markdown("---")
                st.markdown(r["text"])

        st.subheader("Answer")
        with st.spinner("Generating answer..."):
            st.write(generate_answer(question, results))

    else:
        st.subheader("Answer")
        with st.spinner("Generating answer..."):
            st.write(generate_answer(question))

elif ask and not question:
    st.warning("Please enter a question.")
