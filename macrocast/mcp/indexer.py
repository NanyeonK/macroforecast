"""Chunk, embed, and index the Coulombe knowledge corpus into ChromaDB.

Three source types:
- papers: extracted PDF text, chunked by section header or paragraph
- blog:   plain-text blog posts, chunked by paragraph
- methodology: docs/research/coulombe-methodology.md, chunked by ## heading

Embedding model: nomic-ai/nomic-embed-text-v1.5 (8192-token context,
local inference via sentence-transformers, ~550 MB first download).

Run directly to build/rebuild the index:
    uv run python macrocast/mcp/indexer.py
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import chromadb

from macrocast.mcp.config import (
    BLOG_CACHE_DIR,
    CHROMA_COLLECTION,
    CHROMA_DIR,
    METHODOLOGY_DOC,
    PAPER_METADATA,
    PAPERS_CACHE_DIR,
    make_embedding_function,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chunking parameters
# ---------------------------------------------------------------------------

TARGET_CHUNK_CHARS = 4000  # ~800-1000 tokens for nomic-embed
OVERLAP_CHARS = 400

# ---------------------------------------------------------------------------
# Chunking utilities
# ---------------------------------------------------------------------------


def _split_by_paragraphs(text: str, target: int = TARGET_CHUNK_CHARS) -> list[str]:
    """Split text into chunks roughly `target` chars long, at paragraph breaks."""
    paragraphs = re.split(r"\n{2,}", text.strip())
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) + 2 > target and current:
            chunks.append(current.strip())
            # start next chunk with overlap: last paragraph of previous chunk
            last_para = current.rsplit("\n\n", 1)[-1] if "\n\n" in current else ""
            current = last_para + "\n\n" + para if last_para else para
        else:
            current = current + "\n\n" + para if current else para
    if current.strip():
        chunks.append(current.strip())
    return chunks


def _detect_section(line: str) -> bool:
    """Return True if `line` looks like a section heading in an academic paper."""
    # Numbered heading: "1 Introduction", "2.1 Data", "A Appendix"
    if re.match(r"^[1-9A-Z][\.\d]*\s+[A-Z]", line):
        return True
    # All-caps short lines (common in PDF extraction)
    stripped = line.strip()
    if stripped.isupper() and 3 < len(stripped) < 60:
        return True
    return False


def chunk_paper(text: str, paper_key: str) -> list[dict[str, Any]]:
    """Split paper text into section-aware chunks with metadata."""
    lines = text.splitlines()
    sections: list[tuple[str, list[str]]] = []  # (section_title, lines)
    current_title = "Preamble"
    current_lines: list[str] = []

    for line in lines:
        if _detect_section(line):
            if current_lines:
                sections.append((current_title, current_lines))
            current_title = line.strip()
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_title, current_lines))

    chunks = []
    for section_title, sec_lines in sections:
        section_text = "\n".join(sec_lines).strip()
        if not section_text:
            continue
        sub_chunks = _split_by_paragraphs(section_text)
        for i, chunk in enumerate(sub_chunks):
            if not chunk.strip():
                continue
            chunks.append(
                {
                    "text": chunk,
                    "source_type": "paper",
                    "paper_key": paper_key,
                    "section": section_title,
                    "chunk_index": i,
                }
            )
    return chunks


def chunk_blog_post(post: dict) -> list[dict[str, Any]]:
    """Split a single blog post into paragraph-based chunks."""
    text = post["text"]
    sub_chunks = _split_by_paragraphs(text)
    return [
        {
            "text": chunk,
            "source_type": "blog",
            "paper_key": "",
            "section": post["title"],
            "chunk_index": i,
            "blog_url_id": post.get("url_id", ""),
            "blog_date": post.get("published_date", ""),
        }
        for i, chunk in enumerate(sub_chunks)
        if chunk.strip()
    ]


def chunk_methodology(path: Path) -> list[dict[str, Any]]:
    """Split the methodology markdown by ## sections."""
    text = path.read_text(encoding="utf-8")
    # Split on ## headings
    parts = re.split(r"(?m)^(#{1,3} .+)$", text)

    chunks = []
    current_heading = "Overview"
    current_text = ""
    for part in parts:
        if re.match(r"^#{1,3} ", part):
            if current_text.strip():
                sub = _split_by_paragraphs(current_text)
                for i, c in enumerate(sub):
                    if c.strip():
                        chunks.append(
                            {
                                "text": c,
                                "source_type": "methodology",
                                "paper_key": "",
                                "section": current_heading,
                                "chunk_index": i,
                            }
                        )
            current_heading = part.strip("# ").strip()
            current_text = ""
        else:
            current_text += part
    # flush last section
    if current_text.strip():
        sub = _split_by_paragraphs(current_text)
        for i, c in enumerate(sub):
            if c.strip():
                chunks.append(
                    {
                        "text": c,
                        "source_type": "methodology",
                        "paper_key": "",
                        "section": current_heading,
                        "chunk_index": i,
                    }
                )
    return chunks


# ---------------------------------------------------------------------------
# ChromaDB indexing
# ---------------------------------------------------------------------------


def _get_collection(
    chroma_dir: Path = CHROMA_DIR,
) -> chromadb.Collection:
    """Return (or create) the persistent ChromaDB collection."""
    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))
    ef = make_embedding_function()
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def _unique_id(
    source_type: str, key: str, section: str, chunk_index: int, global_idx: int
) -> str:
    """Generate a stable, unique document ID for ChromaDB."""
    safe = re.sub(
        r"[^a-zA-Z0-9_-]",
        "_",
        f"{source_type}_{key}_{section}_{chunk_index}_{global_idx}",
    )
    return safe[:512]


def build_index(force: bool = False) -> None:
    """Build or rebuild the ChromaDB index from cached sources.

    Parameters
    ----------
    force:
        If True, delete and recreate the collection before indexing.
    """
    chroma_dir = CHROMA_DIR
    chroma_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(chroma_dir))
    ef = make_embedding_function()

    if force:
        try:
            client.delete_collection(CHROMA_COLLECTION)
            logger.info("Deleted existing collection '%s'.", CHROMA_COLLECTION)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    all_chunks: list[dict[str, Any]] = []

    # --- Papers ---
    for key in PAPER_METADATA:
        cache_file = PAPERS_CACHE_DIR / f"{key}.json"
        if not cache_file.exists():
            logger.warning("Paper cache missing, skipping: %s", key)
            continue
        record = json.loads(cache_file.read_text(encoding="utf-8"))
        chunks = chunk_paper(record["text"], key)
        logger.info("Paper %s: %d chunks", key, len(chunks))
        all_chunks.extend(chunks)

    # --- Blog ---
    blog_index = BLOG_CACHE_DIR / "index.json"
    if blog_index.exists():
        posts = json.loads(blog_index.read_text(encoding="utf-8"))
        for post in posts:
            chunks = chunk_blog_post(post)
            logger.info("Blog '%s': %d chunks", post["title"][:50], len(chunks))
            all_chunks.extend(chunks)
    else:
        logger.warning("Blog cache missing. Run ingest_blog.py first.")

    # --- Methodology ---
    if METHODOLOGY_DOC.exists():
        chunks = chunk_methodology(METHODOLOGY_DOC)
        logger.info("Methodology doc: %d chunks", len(chunks))
        all_chunks.extend(chunks)
    else:
        logger.warning("Methodology doc not found: %s", METHODOLOGY_DOC)

    logger.info("Total chunks to index: %d", len(all_chunks))

    # Batch-upsert into ChromaDB
    batch_size = 64
    for batch_start in range(0, len(all_chunks), batch_size):
        batch = all_chunks[batch_start : batch_start + batch_size]
        ids = [
            _unique_id(
                c["source_type"],
                c.get("paper_key", ""),
                c.get("section", ""),
                c["chunk_index"],
                batch_start + i,
            )
            for i, c in enumerate(batch)
        ]
        documents = [c["text"] for c in batch]
        metadatas = [
            {
                k: v
                for k, v in c.items()
                if k != "text" and isinstance(v, (str, int, float, bool))
            }
            for c in batch
        ]
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        logger.info("  Upserted batch %d-%d", batch_start, batch_start + len(batch) - 1)

    final_count = collection.count()
    logger.info(
        "Index complete. Collection '%s' has %d documents.",
        CHROMA_COLLECTION,
        final_count,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build_index(force=False)
