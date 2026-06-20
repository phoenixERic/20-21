"""
query.py
--------
Interactive command-line chat with your PDFs.

Usage:
    python query.py

Make sure you've run `python ingest.py` at least once before this.
"""

from rag_engine import RAGEngine


def print_answer(result: dict):
    print("\n" + "-" * 60)
    print("ANSWER:")
    print(result["answer"])

    if result["sources"]:
        print("\n📚 Sources:")
        for s in result["sources"]:
            print(f"   - {s['source_file']}, page {s['page_number']} (similarity: {s['similarity']})")
    print("-" * 60 + "\n")


def main():
    print("🔧 Starting up (loading embedding model + connecting to Gemini)...")
    engine = RAGEngine()

    doc_count = engine.store.count()
    if doc_count == 0:
        print("\n❌ Your vector database is empty. Run `python ingest.py` first!")
        return

    print(f"\n✅ Ready! {doc_count} chunks indexed and searchable.")
    print("Ask a question about your PDFs (type 'exit' or 'quit' to stop).\n")

    while True:
        question = input("❓ Your question: ").strip()
        if question.lower() in ("exit", "quit", "q"):
            print("👋 Bye!")
            break
        if not question:
            continue

        result = engine.answer(question)
        print_answer(result)


if __name__ == "__main__":
    main()
