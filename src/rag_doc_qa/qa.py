from __future__ import annotations

import os
import re

from .retrieval import RetrievedChunk


def _extractive_answer(question: str, retrieved: list[RetrievedChunk]) -> str:
    """
    Mock mode: no LLM call. Returns the most relevant sentences from the
    retrieved chunks, so the answer is always traceable to a source.
    This is the credential-free default (LLM_PROVIDER=mock).
    """
    if not retrieved or retrieved[0].score <= 0.0:
        return "Not found in the documents."
    # Pick sentences from the top chunk that share a word with the question.
    question_words = set(re.findall(r"[a-z]+", question.lower()))
    sentences = re.split(r"(?<=[.!?])\s+", retrieved[0].chunk.text)
    scored = sorted(
        sentences,
        key=lambda s: len(question_words & set(re.findall(r"[a-z]+", s.lower()))),
        reverse=True,
    )
    best = [s for s in scored if s.strip()][:2]
    return " ".join(best) if best else retrieved[0].chunk.text


def answer_question(
    question: str, retrieved: list[RetrievedChunk], provider: str | None = None
) -> str:
    provider = provider or os.environ.get("LLM_PROVIDER", "mock")
    if provider == "gemini":
        return _gemini_answer(question, retrieved)
    return _extractive_answer(question, retrieved)


def _gemini_answer(question: str, retrieved: list[RetrievedChunk]) -> str:
    """Gemini path: grounded prompt, explicit abstention, low temperature."""
    try:
        import google.generativeai as genai
    except ImportError:
        return _extractive_answer(question, retrieved) + "\n\n[Gemini library not installed; fell back to extractive answer.]"
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return _extractive_answer(question, retrieved) + "\n\n[No GEMINI_API_KEY set; fell back to extractive answer.]"
    if not retrieved or retrieved[0].score <= 0.0:
        return "Not found in the documents."
    context = "\n\n".join(f"[{r.chunk.source}] {r.chunk.text}" for r in retrieved)
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        "gemini-1.5-flash", generation_config={"temperature": 0.1}
    )
    prompt = (
        "Answer the question using ONLY the context below. "
        "If the context does not contain the answer, reply exactly: "
        "'Not found in the documents.'\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    return model.generate_content(prompt).text.strip()
