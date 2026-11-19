# RAG Document Q&A

Ask questions about your own documents, get answers grounded in retrieved passages — with sources, and an honest "Not found in the documents" instead of a hallucinated answer.

Built as a fresher AI Engineer portfolio project to replace a tutorial regression project: it shows chunking, retrieval, grounded generation, evaluation, and deployment, not just model training.

## Demo

Run locally (see below), or deploy with the included `Dockerfile` on Hugging Face Spaces / Render. Add your live demo link here after deploying.

## How it works

1. **Chunk** — `docs/*.md` are split by paragraph, then windowed to ~500 characters with 50 characters of overlap (`src/rag_doc_qa/chunking.py`).
2. **Retrieve** — TF-IDF + cosine similarity returns the top-k chunks (`src/rag_doc_qa/retrieval.py`, top-k = 3 by default).
3. **Answer** — In mock mode (default, no key needed) the answer is extracted from the top retrieved chunk, so it is always traceable. With `LLM_PROVIDER=gemini` and a `GEMINI_API_KEY`, a grounded prompt (temperature 0.1) answers only from context and abstains when the answer is not in the documents (`src/rag_doc_qa/qa.py`).
4. **Evaluate** — `eval/run_eval.py` runs 30 grounded questions and reports retrieval hit@3, keyword coverage, and mean retrieval latency.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Try: "What is RAG and how does it reduce hallucinations?" and "What is Adam's starting learning rate?" Then try an out-of-scope question, like "What is the capital of France?" — it should abstain, because that fact is not in `docs/`.

## Evaluation

Definitions, so the numbers are checkable and not marketing:

- **hit@3**: the expected document is among the top-3 retrieved chunks.
- **keyword coverage**: the extractive answer contains at least one expected keyword. This is *not* a human-judged answer-accuracy score.
- **latency**: retrieval time only, averaged over the eval questions.

Measured on 28 Oct 2026, on the 30-question set in `eval/eval_questions.json` against the 4 sample documents in `docs/` (24 chunks), with TF-IDF retrieval:

- Retrieval hit@3: **1.00** (30/30)
- Keyword coverage: **0.867** (26/30) — extractive-answer keyword check only, not human-judged accuracy
- Mean retrieval latency: **1.05 ms**
- Tests: 3 passed (`pytest tests/ -q`)

Full output is in `eval/results.json`. Re-run the eval after any change to chunking or retrieval — and do not claim these small-corpus numbers for a different, larger corpus.

```bash
python eval/run_eval.py
python -m pytest tests/ -q
```

## Trade-offs and limitations (say these in an interview)

- **TF-IDF is a baseline, not semantic search.** It matches keywords, so paraphrased questions can fail where embeddings would succeed. That is a deliberate choice for a fast, dependency-light, credential-free demo. The next step is sentence-transformers embeddings + FAISS, keeping the same top-k and citation interface, then re-running this eval to show the difference.
- **Chunk size 500 / overlap 50 is a starting point for short notes**, not a universal optimum. Larger chunks keep more context but mix topics; smaller chunks lose context.
- **Mock answers are extractive**, so they read like quoted passages, not fluent LLM answers. That is the point in mock mode: no LLM call, no invented fact.
- **The sample corpus is small** (4 documents, ~30 chunks). Hit rates on this set do not prove performance on a large, messy, real corpus.

## Next steps

1. Swap TF-IDF for `sentence-transformers` embeddings and FAISS, re-run the eval, and report both numbers side by side.
2. Add document upload in the UI and a persistent vector store (ChromaDB) instead of re-indexing on every start.
3. Add reranking and a human-judged answer-accuracy sample on top of hit@3.

## Project structure

```
app.py                    Streamlit UI
src/rag_doc_qa/
  chunking.py             Paragraph + windowed chunking
  retrieval.py            TF-IDF + cosine retriever
  qa.py                   Mock extractive + Gemini grounded answers
docs/                     4 sample documents (ML, neural nets, Python, RAG)
eval/
  eval_questions.json     30 questions with expected doc + keywords
  run_eval.py             Reports hit@3, keyword coverage, latency
tests/                    Chunking + retrieval tests
Dockerfile                Streamlit deployment
```

## Replace the corpus with your own

Drop your own `.md` files into `docs/`, add questions to `eval/eval_questions.json` with their expected document, and re-run the eval. Do not claim the sample-corpus numbers for a different corpus.
