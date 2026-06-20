"""
ingest.py
---------
Run this ONCE (and again whenever you add new PDFs) to:
  1. Extract text + tables from every PDF in data/pdfs/
  2. Chunk the content (table-aware)
  3. Embed and store everything in the local ChromaDB

Usage:
    python ingest.py
"""

from pdf_extractor import extract_all_pdfs
from chunker import chunk_pages
from vector_store import VectorStore

PDF_DIR = ""


def main():
    print("=" * 60)
    print("STEP 1: Extracting PDFs (text + tables, per page)")
    print("=" * 60)
    page_records = extract_all_pdfs(PDF_DIR)

    if not page_records:
        print("\n❌ No PDFs found. Put your PDF files in data/pdfs/ and re-run this script.")
        return

    print("\n" + "=" * 60)
    print("STEP 2: Chunking (tables kept intact, text split smartly)")
    print("=" * 60)
    chunks = chunk_pages(page_records)
    print(f"Created {len(chunks)} chunks total.")

    print("\n" + "=" * 60)
    print("STEP 3: Embedding + storing in local vector database")
    print("=" * 60)
    store = VectorStore()
    store.add_chunks(chunks)

    print("\n🎉 Ingestion complete! You can now run: python query.py")


if __name__ == "__main__":
    main()
