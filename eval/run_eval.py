"""Run the eval set and report retrieval hit rate + latency.

Usage: python eval/run_eval.py
Writes: eval/results.json

Definitions (stated so the numbers are checkable):
- hit@3: the expected document is among the top-3 retrieved chunks.
- keyword coverage: the extractive answer contains at least one expected
  keyword. This is NOT a human-judged answer-accuracy score.
- latency: retrieval time only, in milliseconds, averaged over questions.
"""

import json
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rag_doc_qa.chunking import load_and_chunk_docs
from rag_doc_qa.qa import answer_question
from rag_doc_qa.retrieval import Retriever

BASE = os.path.dirname(__file__)


def main():
    with open(os.path.join(BASE, "eval_questions.json"), encoding="utf-8") as f:
        questions = json.load(f)
    chunks = load_and_chunk_docs(os.path.join(BASE, "..", "docs"))
    retriever = Retriever(chunks)

    hits = 0
    keyword_hits = 0
    latencies = []
    failures = []
    for q in questions:
        retrieved, latency_ms = retriever.retrieve(q["question"], top_k=3)
        latencies.append(latency_ms)
        sources = [r.chunk.source for r in retrieved]
        if q["expected_doc"] in sources:
            hits += 1
        else:
            failures.append({"question": q["question"], "expected": q["expected_doc"], "got": sources})
        answer = answer_question(q["question"], retrieved)
        if any(kw.lower() in answer.lower() for kw in q["expected_keywords"]):
            keyword_hits += 1

    results = {
        "n_questions": len(questions),
        "n_chunks": len(chunks),
        "retrieval_hit_at_3": round(hits / len(questions), 3),
        "keyword_coverage": round(keyword_hits / len(questions), 3),
        "mean_retrieval_latency_ms": round(statistics.mean(latencies), 2),
        "failures": failures,
    }
    with open(os.path.join(BASE, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
