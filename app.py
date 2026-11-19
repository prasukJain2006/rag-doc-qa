"""Streamlit UI for RAG Document Q&A. Run: streamlit run app.py"""

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from rag_doc_qa.chunking import load_and_chunk_docs
from rag_doc_qa.qa import answer_question
from rag_doc_qa.retrieval import Retriever

st.set_page_config(page_title="RAG Document Q&A", page_icon="📄")
st.title("📄 RAG Document Q&A")
st.write(
    "Ask a question about the documents in `docs/`. Answers come only from "
    "retrieved passages, with sources shown, and the app says "
    "'Not found in the documents' instead of guessing."
)


@st.cache_resource
def get_retriever():
    chunks = load_and_chunk_docs("docs", chunk_size=500, overlap=50)
    return Retriever(chunks), len(chunks)


retriever, n_chunks = get_retriever()
st.caption(f"Indexed {n_chunks} chunks from docs/ · Top-k = 3 · Provider: {os.environ.get('LLM_PROVIDER', 'mock')}")

question = st.text_input("Your question", placeholder="e.g. What is RAG and how does it reduce hallucinations?")
top_k = st.slider("Top-k retrieved chunks", 1, 5, 3)

if st.button("Ask") and question.strip():
    retrieved, latency_ms = retriever.retrieve(question, top_k=top_k)
    answer = answer_question(question, retrieved)
    st.subheader("Answer")
    st.write(answer)
    st.caption(f"Retrieval latency: {latency_ms:.1f} ms")
    st.subheader("Sources")
    for r in retrieved:
        st.markdown(f"**{r.chunk.source}** — {r.chunk.section} (score {r.score:.3f})")
        st.write(r.chunk.text)
