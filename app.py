"""
app.py

The Streamlit UI. This is what turns your code into something a
recruiter can actually click and use, instead of just reading.

Run it with:
  streamlit run app.py
"""

import os
import tempfile
import streamlit as st
import rag_engine
import agent

st.set_page_config(page_title="PDF Research Agent", page_icon="📄")
st.title("📄 PDF Research Agent")
st.caption("Ask questions about your PDF. If the PDF doesn't have the answer, it searches the web instead.")

if "collection" not in st.session_state:
    st.session_state.collection = None
    st.session_state.filename = None

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file and uploaded_file.name != st.session_state.filename:
    with st.spinner("Reading and indexing your PDF..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        # 🧠 Fixed Indentation and explicit script reference pointers
        text = rag_engine.load_pdf_text(tmp_path)
        chunks = rag_engine.chunk_text(text)
        st.session_state.collection = rag_engine.build_vector_store(chunks)
        st.session_state.filename = uploaded_file.name
        os.unlink(tmp_path)

    st.success(f"Indexed {len(chunks)} chunks from {uploaded_file.name}")

if st.session_state.collection:
    question = st.text_input("Ask a question")
    if question:
        with st.spinner("Thinking..."):
            # 🧠 Fixed Agent call explicit logic
            answer, mode = agent.answer_question(question, st.session_state.collection)

        badge = "📄 Answered from your PDF" if mode == "pdf" else "🌐 Answered from a live web search"
        st.caption(badge)
        st.write(answer)
else:
    st.info("Upload a PDF to get started.")
