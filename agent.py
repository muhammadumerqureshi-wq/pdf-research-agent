import os
from groq import Groq
from rag_engine import hybrid_search

CONFIDENCE_THRESHOLD = 0.60  # fused hybrid score threshold
from dotenv import load_dotenv
load_dotenv()

def answer_question(question, collection, bm25, chunks):
    """
    Updated to use hybrid_search (BM25 + Semantic) instead of pure semantic search.
    bm25   → BM25 index built from chunks (pass from app.py/eval.py)
    chunks → original chunk list (needed for BM25 scoring)
    """
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    # ── Hybrid Retrieval (BM25 + ChromaDB fused) ────────────────────────────
    top_chunks, best_score = hybrid_search(
        query=question,
        collection=collection,
        bm25=bm25,
        chunks=chunks,
        n_results=3,
        alpha=0.7,          # 70% semantic + 30% BM25 — reduces false BM25 keyword matches
    )

    print(f"[agent] Best hybrid score: {best_score:.4f} | Threshold: {CONFIDENCE_THRESHOLD}")

    # ── Route based on fused score ───────────────────────────────────────────
    if best_score >= CONFIDENCE_THRESHOLD:
        mode = "pdf"
        context = "\n\n---\n\n".join(top_chunks)
        prompt = (
            f"Answer the question using ONLY the context below. "
            f"Do not use outside knowledge.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}"
        )
    else:
        mode = "web"
        prompt = (
            f"Answer the following question using your general knowledge:\n\n"
            f"Question: {question}"
        )

    print(f"[agent] Routing to mode: {mode.upper()}")

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
    )

    answer = response.choices[0].message.content
    return answer, mode