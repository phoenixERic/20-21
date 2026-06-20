"""
pdf_extractor.py
----------------
Turns a PDF into a list of "page records" where tables and prose text
are extracted SEPARATELY and kept clearly labeled.

Why separate tables from text?
Financial PDFs (10-Qs, 10-Ks, etc.) pack their most important numbers into
tables. If you just pull plain text out of a PDF, tables either:
  (a) get mangled into unreadable strings of numbers with no column meaning, or
  (b) get silently dropped by simpler extraction tools.

pdfplumber lets us detect table regions on each page and extract them as
structured rows/columns, which we then convert into clean Markdown tables.
Markdown tables are great for RAG because:
  - they're plain text (easy to embed and store),
  - they preserve row/column structure so the LLM can actually read them,
  - they're compact compared to raw table JSON.

Every record keeps track of which PDF file and which page number it came
from, which is what lets us cite "Source: filename.pdf, page X" later.
"""

import pdfplumber
from pathlib import Path
from typing import List, Dict


def _table_to_markdown(table: List[List[str]]) -> str:
    """Convert a pdfplumber-extracted table (list of rows) into a Markdown table string."""
    if not table or len(table) == 0:
        return ""

    # Clean None / whitespace in cells
    cleaned = [
        [(cell or "").strip().replace("\n", " ") for cell in row]
        for row in table
    ]

    header = cleaned[0]
    rows = cleaned[1:]

    md_lines = []
    md_lines.append("| " + " | ".join(header) + " |")
    md_lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in rows:
        # Pad row if it's shorter than header (ragged tables happen often in PDFs)
        if len(row) < len(header):
            row = row + [""] * (len(header) - len(row))
        md_lines.append("| " + " | ".join(row[: len(header)]) + " |")

    return "\n".join(md_lines)


def extract_pdf(pdf_path: str) -> List[Dict]:
    """
    Extract every page of a PDF into a structured record.

    Returns a list of dicts, one per page, shaped like:
    {
        "source_file": "2023_Q1_AAPL.pdf",
        "page_number": 4,            # 1-indexed, matches what a human sees in a PDF viewer
        "text": "...",               # prose text on the page (tables removed to avoid duplication)
        "tables": ["| a | b |\n...", ...],  # each table on the page as a Markdown string
    }
    """
    pdf_path = Path(pdf_path)
    records = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_number = i + 1  # human-friendly, 1-indexed

            # 1. Extract tables first, as structured data
            raw_tables = page.find_tables()
            markdown_tables = []
            for t in raw_tables:
                extracted = t.extract()
                md = _table_to_markdown(extracted)
                if md.strip():
                    markdown_tables.append(md)

            # 2. Extract plain text, but exclude the regions where tables were found.
            #    This avoids the same numbers appearing twice (once as garbled text,
            #    once as a clean table), which would confuse the embedding model.
            if raw_tables:
                # Build a page copy with table bounding boxes blanked out
                table_bboxes = [t.bbox for t in raw_tables]
                page_no_tables = page
                for bbox in table_bboxes:
                    page_no_tables = page_no_tables.outside_bbox(bbox)
                text = page_no_tables.extract_text() or ""
            else:
                text = page.extract_text() or ""

            records.append(
                {
                    "source_file": pdf_path.name,
                    "page_number": page_number,
                    "text": text.strip(),
                    "tables": markdown_tables,
                }
            )

    return records


def extract_all_pdfs(pdf_dir: str) -> List[Dict]:
    """Run extract_pdf() over every .pdf file in a directory. Returns one combined list."""
    pdf_dir = Path(pdf_dir)
    all_records = []
    pdf_files = sorted(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"⚠️  No PDF files found in {pdf_dir}. Add some PDFs there first.")
        return []

    for pdf_file in pdf_files:
        print(f"📄 Extracting: {pdf_file.name}")
        records = extract_pdf(pdf_file)
        all_records.extend(records)
        print(f"   -> {len(records)} pages extracted")

    return all_records


if __name__ == "__main__":
    # Quick manual test: run `python pdf_extractor.py` to sanity-check extraction
    results = extract_all_pdfs("../data/pdfs")
    print(f"\nTotal pages extracted across all PDFs: {len(results)}")
    if results:
        sample = results[3] if len(results) > 3 else results[0]
        print(f"\nSample record (page {sample['page_number']} of {sample['source_file']}):")
        print(f"Text preview: {sample['text'][:200]}")
        print(f"Tables found on this page: {len(sample['tables'])}")
