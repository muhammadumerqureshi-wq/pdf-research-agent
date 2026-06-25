"""
eval.py

Produces the one number every recruiter-facing project needs: an accuracy
score. Fill in sample_qa.json with real questions about your test PDF,
then run this.

Usage:
  python eval.py path/to/your.pdf
"""

import json
import sys

from agent import answer_question
from rag_engine import build_vector_store, chunk_text, load_pdf_text


def run_eval(pdf_path: str, qa_path: str = "sample_qa.json"):
    with open(qa_path) as f:
        qa_pairs = json.load(f)

    text = load_pdf_text(pdf_path)
    chunks = chunk_text(text)
    collection = build_vector_store(chunks)

    correct = 0
    for item in qa_pairs:
        answer, mode = answer_question(item["question"], collection)
        is_correct = item["expected_keyword"].lower() in answer.lower()
        correct += int(is_correct)

        print(f"\nQ: {item['question']}")
        print(f"Mode: {mode}")
        print(f"Answer: {answer[:200]}")
        print(f"Expected keyword found: {is_correct}")

    accuracy = correct / len(qa_pairs) * 100
    print(f"\n=== Accuracy: {correct}/{len(qa_pairs)} ({accuracy:.0f}%) ===")
    print("Put this number in your README.")


if __name__ == "__main__":
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "test.pdf"
    run_eval(pdf_path)
