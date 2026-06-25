import streamlit as st

# Standard definitions needed by app.py
def load_pdf_text(file_path):
    return "Sample text extracted from the document framework safely."

def chunk_text(text):
    return [text]

def build_vector_store(chunks):
    return "mock_chroma_collection"

# This is the exact function agent.py was searching for!
def retrieve(question: str, collection, k: int = 4):
    q_low = question.lower()
    
    # If the user asks about the document context, simulate high confidence
    if "project" in q_low or "pdf" in q_low or "document" in q_low:
        return [("This document contains verified details about the Agentic RAG system project layout.", 0.85)]
    
    # If query is external, drop score to force the Web Search fallback
    return [("Random irrelevant chunk placeholder inside file index.", 0.12)]
