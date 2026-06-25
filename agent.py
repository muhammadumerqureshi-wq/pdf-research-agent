import os
from groq import Groq
from dotenv import load_dotenv
import rag_engine

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = "llama-3.3-70b-versatile"
CONFIDENCE_THRESHOLD = 0.35

client = Groq(api_key=GROQ_API_KEY)

def call_groq(prompt: str) -> str:
    """Send a prompt to Groq and return the response text."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.7,
    )
    return response.choices[0].message.content


def answer_from_pdf(question: str, chunks_with_scores) -> str:
    """Answer using context retrieved from the uploaded PDF."""
    context = "\n\n---\n\n".join(chunk for chunk, _ in chunks_with_scores)
    prompt = f"""Answer the question using ONLY the context below.
If the context genuinely doesn't contain the answer, say so plainly instead of guessing.

Context:
{context}

Question: {question}"""
    return call_groq(prompt)


def answer_from_web(question: str) -> str:
    """Answer using the LLM's own knowledge (no PDF context)."""
    prompt = f"""You are a helpful assistant. Answer the following question clearly and concisely 
using your knowledge. If you are unsure, say so.

Question: {question}"""
    return call_groq(prompt)


def answer_question(question: str, collection, k: int = 4):
    """
    Route the question:
    - If PDF has relevant content (score >= threshold) → answer from PDF
    - Otherwise → answer from LLM knowledge
    """
    results = rag_engine.retrieve(question, collection, k=k)
    best_score = results[0][1] if results else 0

    if best_score >= CONFIDENCE_THRESHOLD:
        return answer_from_pdf(question, results), "pdf"
    else:
        return answer_from_web(question), "web"