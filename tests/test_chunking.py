import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rag_doc_qa.chunking import chunk_text, load_and_chunk_docs


def test_chunking_splits_long_paragraph():
    chunks = chunk_text("word " * 300, source="test.md", chunk_size=500, overlap=50)
    assert len(chunks) > 1
    assert all(c.text for c in chunks)


def test_docs_load():
    chunks = load_and_chunk_docs(os.path.join(os.path.dirname(__file__), "..", "docs"))
    assert len(chunks) >= 20
    assert {c.source for c in chunks} == {
        "machine_learning_basics.md",
        "neural_networks.md",
        "python_for_data.md",
        "rag_and_llms.md",
    }
