import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rag_doc_qa.chunking import load_and_chunk_docs
from rag_doc_qa.retrieval import Retriever


def test_retrieval_finds_rag_doc():
    chunks = load_and_chunk_docs(os.path.join(os.path.dirname(__file__), "..", "docs"))
    retriever = Retriever(chunks)
    retrieved, _ = retriever.retrieve("What is retrieval augmented generation?", top_k=3)
    assert "rag_and_llms.md" in [r.chunk.source for r in retrieved]
