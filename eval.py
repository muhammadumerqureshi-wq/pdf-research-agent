import json
import os
from agent import answer_question
from rag_engine import build_vector_store, build_bm25_index, chunk_text, load_pdf_text

def run_eval(pdf_path="test.pdf", qa_path="sample_qa.json"):
    if not os.path.exists(pdf_path):
        print(f"❌ Error: Please rename your test PDF to '{pdf_path}' and put it in this folder.")
        return

    with open(qa_path) as f:
        qa_pairs = json.load(f)

    text       = load_pdf_text(pdf_path)
    chunks     = chunk_text(text)
    collection = build_vector_store(chunks)
    bm25       = build_bm25_index(chunks)   # BM25 index

    correct = 0
    for item in qa_pairs:
        answer, mode = answer_question(item["question"], collection, bm25, chunks)
        is_correct   = item["expected_keyword"].lower() in answer.lower()
        correct     += int(is_correct)

        print(f"\nQ: {item['question']}")
        print(f"Mode: {mode}")
        print(f"Answer: {answer[:150]}")
        print(f"Expected keyword found: {is_correct}")

    accuracy = correct / len(qa_pairs) * 100
    print(f"\n=== Accuracy: {correct}/{len(qa_pairs)} ({accuracy:.0f}%) ===")

if __name__ == "__main__":
    run_eval()