import os
import chromadb
from chromadb.utils import embedding_functions
import pypdf
from rank_bm25 import BM25Okapi

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

COLLECTION_NAME = "pdf_chunks"


def load_pdf_text(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = pypdf.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ""
    print(f"[load_pdf_text] Total characters extracted: {len(text):,}")
    return text


def chunk_text(text, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)
    chunks = [c.strip() for c in chunks if c.strip()]
    print(f"[chunk_text] Chunks created: {len(chunks)}")
    return chunks


def build_vector_store(chunks, persist_directory="./chroma_db"):
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    client = chromadb.PersistentClient(path=persist_directory)

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [f"chunk_{i}" for i in range(len(chunks))]

    BATCH_SIZE = 500
    for start in range(0, len(chunks), BATCH_SIZE):
        collection.add(
            documents=chunks[start : start + BATCH_SIZE],
            ids=ids[start : start + BATCH_SIZE],
        )

    print(f"[build_vector_store] Stored {collection.count()} vectors in '{COLLECTION_NAME}'")
    return collection


def build_bm25_index(chunks):
    """
    BM25 NEW: Builds a keyword-based BM25 index from the same chunks.
    BM25 tokenizes each chunk into words and creates an inverted index.
    This catches exact keyword matches that semantic search sometimes misses.
    e.g. query "GPT-4" → BM25 finds exact "GPT-4" mentions instantly.
    """
    tokenized = [chunk.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized)
    print(f"[build_bm25_index] BM25 index built for {len(chunks)} chunks")
    return bm25


def hybrid_search(query, collection, bm25, chunks, n_results=3, alpha=0.5):
    """
    HYBRID SEARCH: Combines BM25 (keyword) + ChromaDB (semantic) scores using RRF fusion.

    alpha = weight for semantic score (0.0 → pure BM25, 1.0 → pure semantic)
    Default alpha=0.5 means equal weight to both.

    How RRF (Reciprocal Rank Fusion) works:
    - BM25 ranks all chunks by keyword match → gives rank positions
    - ChromaDB ranks all chunks by semantic similarity → gives rank positions
    - Final score = alpha * semantic_score + (1-alpha) * bm25_normalized_score
    - Top n_results chunks returned
    """
    # ── Step 1: Semantic search via ChromaDB ────────────────────────────────
    semantic_results = collection.query(
        query_texts=[query],
        n_results=len(chunks),          # get ALL chunks ranked
        include=["documents", "distances"],
    )
    semantic_docs      = semantic_results["documents"][0]
    semantic_distances = semantic_results["distances"][0]
    semantic_scores    = [1 - d for d in semantic_distances]  # convert distance → similarity

    # Map chunk text → semantic score
    semantic_map = {doc: score for doc, score in zip(semantic_docs, semantic_scores)}

    # ── Step 2: BM25 keyword search ─────────────────────────────────────────
    tokenized_query = query.lower().split()
    bm25_raw_scores = bm25.get_scores(tokenized_query)  # score for every chunk

    # Normalize BM25 scores to 0→1 range
    bm25_max = max(bm25_raw_scores) if max(bm25_raw_scores) > 0 else 1
    bm25_normalized = [s / bm25_max for s in bm25_raw_scores]

    # ── Step 3: Fuse scores ─────────────────────────────────────────────────
    fused = []
    for i, chunk in enumerate(chunks):
        sem_score = semantic_map.get(chunk, 0.0)
        bm25_score = bm25_normalized[i]
        final_score = alpha * sem_score + (1 - alpha) * bm25_score
        fused.append((chunk, final_score, sem_score, bm25_score))

    # Sort by final fused score descending
    fused.sort(key=lambda x: x[1], reverse=True)
    top = fused[:n_results]

    print(f"[hybrid_search] Top {n_results} chunks:")
    for i, (_, final, sem, bm) in enumerate(top):
        print(f"  #{i+1} → fused={final:.4f} | semantic={sem:.4f} | bm25={bm:.4f}")

    best_score   = top[0][1]
    top_chunks   = [t[0] for t in top]

    return top_chunks, best_score


def load_vector_store(persist_directory="./chroma_db"):
    """Load an existing collection at query time."""
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    client = chromadb.PersistentClient(path=persist_directory)
    return client.get_collection(name=COLLECTION_NAME, embedding_function=ef)