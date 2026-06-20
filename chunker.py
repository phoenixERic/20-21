"""
chunker.py
----------
Turns page records (from pdf_extractor.py) into "chunks" ready for embedding.

CHUNKING STRATEGY (and why):
This system uses a HYBRID chunking approach, which is the best practice for
documents that mix prose and tables:

1. TABLES ARE NEVER SPLIT. Each table becomes its OWN chunk, kept fully intact.
   Splitting a financial table mid-way (e.g. cutting "iPhone $65,775" from its
   row) destroys the meaning of the numbers. A table is a single semantic unit,
   so we treat it as one chunk regardless of its length.
   We also prefix each table chunk with a bit of context (the source file,
   page number, and the prose text immediately preceding it on the page, e.g.
   the table's title like "Note 2 - Revenue") so the embedding captures what
   the table is ABOUT, not just raw numbers.

2. PROSE TEXT is split using a RECURSIVE, TOKEN-AWARE strategy:
   - Split on paragraph breaks first, then sentences, then words, only as a
     last resort — this keeps semantically related sentences together.
   - Chunks target ~400 tokens, with ~80 tokens of overlap between
     consecutive chunks, so context isn't lost at chunk boundaries.
   - 400 tokens is a good middle ground for financial filings: small enough
     for precise retrieval, large enough to keep a full discussion (like
     "Why iPhone revenue declined") in one piece.

3. EVERY chunk carries its source_file + page_number as metadata, which is
   what enables exact page-number citations in the final answer.
"""

import tiktoken
from typing import List, Dict

# Using GPT's tokenizer as a reasonably accurate, fast, free token counter.
# (We're not using OpenAI models — this is just a free, offline token-counting tool.)
_ENCODER = tiktoken.get_encoding("cl100k_base")

CHUNK_SIZE_TOKENS = 400
CHUNK_OVERLAP_TOKENS = 80


def _count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


def _split_text_recursive(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Recursive text splitter: tries to split on the largest "natural" boundary
    first (paragraphs), and only falls back to smaller boundaries (sentences,
    then words) if a piece is still too big.
    """
    if _count_tokens(text) <= chunk_size:
        return [text] if text.strip() else []

    separators = ["\n\n", "\n", ". ", " "]

    def split_by(text, seps):
        if not seps:
            # Last resort: hard token slice
            tokens = _ENCODER.encode(text)
            return [_ENCODER.decode(tokens[i:i + chunk_size])
                    for i in range(0, len(tokens), chunk_size)]

        sep = seps[0]
        pieces = text.split(sep)
        chunks = []
        current = ""

        for piece in pieces:
            candidate = (current + sep + piece) if current else piece
            if _count_tokens(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # piece itself might still be too big -> recurse with smaller separator
                if _count_tokens(piece) > chunk_size:
                    chunks.extend(split_by(piece, seps[1:]))
                    current = ""
                else:
                    current = piece
        if current:
            chunks.append(current)
        return chunks

    raw_chunks = split_by(text, separators)

    # Add overlap between consecutive chunks so context isn't lost at boundaries
    overlapped = []
    for i, chunk in enumerate(raw_chunks):
        if i == 0:
            overlapped.append(chunk)
        else:
            prev_tokens = _ENCODER.encode(overlapped[-1])
            overlap_text = _ENCODER.decode(prev_tokens[-overlap:]) if len(prev_tokens) > overlap else overlapped[-1]
            overlapped.append(overlap_text + " " + chunk)

    return [c.strip() for c in overlapped if c.strip()]


def chunk_pages(page_records: List[Dict]) -> List[Dict]:
    """
    Convert page records into final chunks ready for embedding.

    Returns a list of dicts:
    {
        "chunk_id": "2023_Q1_AAPL.pdf_p10_table0",
        "content": "...",              # the actual text to embed
        "source_file": "2023_Q1_AAPL.pdf",
        "page_number": 10,
        "chunk_type": "table" | "text",
    }
    """
    all_chunks = []

    for record in page_records:
        source_file = record["source_file"]
        page_number = record["page_number"]
        text = record["text"]
        tables = record["tables"]

        # --- 1. Table chunks: each table is its own chunk, never split ---
        for idx, table_md in enumerate(tables):
            # Give the table a bit of surrounding context so the embedding
            # knows what the table is about (helps retrieval accuracy a lot).
            context_snippet = text[:300].strip() if text else ""
            content = (
                f"[Table from {source_file}, page {page_number}]\n"
                f"Context: {context_snippet}\n\n"
                f"{table_md}"
            )
            all_chunks.append({
                "chunk_id": f"{source_file}_p{page_number}_table{idx}",
                "content": content,
                "source_file": source_file,
                "page_number": page_number,
                "chunk_type": "table",
            })

        # --- 2. Prose text chunks: recursively split if needed ---
        if text and text.strip():
            text_pieces = _split_text_recursive(text, CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS)
            for idx, piece in enumerate(text_pieces):
                content = f"[Text from {source_file}, page {page_number}]\n{piece}"
                all_chunks.append({
                    "chunk_id": f"{source_file}_p{page_number}_text{idx}",
                    "content": content,
                    "source_file": source_file,
                    "page_number": page_number,
                    "chunk_type": "text",
                })

    return all_chunks


if __name__ == "__main__":
    # Quick manual test
    from pdf_extractor import extract_all_pdfs

    pages = extract_all_pdfs("../data/pdfs")
    chunks = chunk_pages(pages)
    print(f"\nTotal chunks created: {len(chunks)}")
    table_chunks = [c for c in chunks if c["chunk_type"] == "table"]
    text_chunks = [c for c in chunks if c["chunk_type"] == "text"]
    print(f"  - Table chunks: {len(table_chunks)}")
    print(f"  - Text chunks: {len(text_chunks)}")
    if table_chunks:
        print(f"\nSample table chunk:\n{table_chunks[0]['content'][:500]}")
