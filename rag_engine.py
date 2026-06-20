"""
rag_engine.py
-------------
The core RAG logic:
  1. Retrieve the most relevant chunks for a question from ChromaDB.
  2. Feed ONLY those chunks (plus their page numbers) to Gemini.
  3. Force Gemini, via the prompt, to:
       - answer using ONLY the provided chunks (no outside knowledge),
       - cite the exact source file + page number(s) it used,
       - explicitly say "I cannot find this in the provided documents"
         if the retrieved chunks don't actually contain the answer.

This "grounding" + "explicit refusal" prompt design is what prevents the
classic RAG failure mode where the LLM just makes up a plausible-sounding
answer instead of admitting the documents don't cover it.

We also apply a SIMILARITY THRESHOLD before even calling the LLM: if the
top retrieved chunk isn't similar enough to the question, we skip the LLM
call entirely and return "not found" — this catches cases where the
question is about a totally unrelated topic.
"""

import os
import google.generativeai as genai
from dotenv import load_dotenv
from vector_store import VectorStore

load_dotenv()  # reads GEMINI_API_KEY from your .env file

GEMINI_MODEL_NAME = "gemini-1.5-flash"  # fast + free-tier friendly; swap for "gemini-1.5-pro" for higher quality
TOP_K = 6
# If the best match's similarity is below this, we don't even bother asking the LLM.
MIN_SIMILARITY_THRESHOLD = 0.25

SYSTEM_PROMPT = """You are a precise document-analysis assistant. You answer questions ONLY using the
excerpts provided below, which come from PDF documents. Each excerpt is labeled with its
source file and page number.

STRICT RULES:
1. Use ONLY the information in the excerpts below. Do not use any outside knowledge.
2. Every factual claim in your answer MUST be traceable to a specific excerpt.
3. At the end of your answer, list the sources you actually used, in this exact format:
   Sources: [filename, page X], [filename, page Y]
4. If the excerpts do NOT contain enough information to answer the question, you MUST say:
   "I cannot find this information in the provided documents."
   Do not guess, estimate, or fill gaps with outside knowledge. Partial information is fine
   to share, but be explicit about what is and isn't covered by the excerpts.
5. Tables are provided in Markdown format — read rows and columns carefully before answering
   any numeric question.
"""


class RAGEngine:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key or api_key == "your_api_key_here":
            raise ValueError(
                "GEMINI_API_KEY not set. Copy .env.example to .env and add your real key "
                "(get one free at https://aistudio.google.com/app/apikey)"
            )
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        self.store = VectorStore()

    def answer(self, question: str, top_k: int = TOP_K) -> dict:
        """
        Returns:
        {
            "answer": str,
            "sources": [{"source_file": ..., "page_number": ..., "similarity": ...}, ...],
            "grounded": bool,   # False if we short-circuited due to low similarity
        }
        """
        hits = self.store.search(question, top_k=top_k)

        if not hits or hits[0]["similarity"] < MIN_SIMILARITY_THRESHOLD:
            return {
                "answer": "I cannot find this information in the provided documents.",
                "sources": [],
                "grounded": False,
            }

        # Build the context block fed to Gemini, with clear excerpt labeling
        context_blocks = []
        for i, hit in enumerate(hits):
            context_blocks.append(
                f"--- Excerpt {i+1} (Source: {hit['source_file']}, Page {hit['page_number']}) ---\n"
                f"{hit['content']}\n"
            )
        context_text = "\n".join(context_blocks)

        full_prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"DOCUMENT EXCERPTS:\n{context_text}\n\n"
            f"QUESTION: {question}\n\n"
            f"ANSWER:"
        )

        response = self.model.generate_content(full_prompt)
        answer_text = response.text.strip()

        # De-duplicate sources for a clean citation list, preserving order
        seen = set()
        sources = []
        for hit in hits:
            key = (hit["source_file"], hit["page_number"])
            if key not in seen:
                seen.add(key)
                sources.append({
                    "source_file": hit["source_file"],
                    "page_number": hit["page_number"],
                    "similarity": round(hit["similarity"], 3),
                })

        return {
            "answer": answer_text,
            "sources": sources,
            "grounded": True,
        }
