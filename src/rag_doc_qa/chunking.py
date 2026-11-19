from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    source: str  # filename, e.g. rag_and_llms.md
    section: str = ""


def _split_paragraphs(text: str) -> list[tuple[str, str]]:
    """
    Return (section, paragraph) pairs. A heading line sets the section
    for following paragraphs, whether or not a blank line follows it --
    the common Markdown style where a paragraph starts on the next line.
    """
    pairs: list[tuple[str, str]] = []
    section = ""
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            pairs.append((section, " ".join(buffer).strip()))
            buffer.clear()

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            flush()
            section = stripped.lstrip("#").strip()
        elif not stripped:
            flush()
        else:
            buffer.append(stripped)
    flush()
    return pairs


def chunk_text(text: str, source: str, chunk_size: int = 500, overlap: int = 50) -> list[Chunk]:
    """
    Chunk by paragraph first, then by characters if a paragraph is too long.

    Default 500 chars / 50 overlap is a starting point for short notes,
    not a universal optimum -- see README 'Trade-offs' for when to change it.
    """
    chunks: list[Chunk] = []
    for section, para in _split_paragraphs(text):
        # A paragraph longer than chunk_size is windowed with overlap.
        start = 0
        while start < len(para):
            piece = para[start : start + chunk_size].strip()
            if piece:
                chunks.append(Chunk(text=piece, source=source, section=section))
            if len(para) - start <= chunk_size:
                break
            start += chunk_size - overlap
    return chunks


def load_and_chunk_docs(docs_dir: str, chunk_size: int = 500, overlap: int = 50) -> list[Chunk]:
    import os

    all_chunks: list[Chunk] = []
    for name in sorted(os.listdir(docs_dir)):
        if not name.endswith(".md"):
            continue
        with open(os.path.join(docs_dir, name), encoding="utf-8") as f:
            all_chunks.extend(chunk_text(f.read(), source=name, chunk_size=chunk_size, overlap=overlap))
    return all_chunks
