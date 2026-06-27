# 📄 PDF Research Agent — Hybrid RAG System

A production-ready **Hybrid Retrieval-Augmented Generation (RAG)** agent that answers questions from uploaded PDFs using a smart confidence-threshold routing system. When the PDF doesn't contain the answer, it automatically falls back to LLM general knowledge.

---

## 🚀 Live Demo

> Upload any PDF → Ask questions → Watch the agent decide: PDF or Web?

![Demo](assets/demo.gif)

---

## 🧠 How It Works

```
User Query
    │
    ▼
┌─────────────────────────────────────────┐
│           Hybrid Retrieval              │
│                                         │
│  BM25 (Keyword)  +  ChromaDB (Semantic) │
│       30%        +       70%            │
│                                         │
│    Fused Score via RRF Fusion           │
└──────────────────┬──────────────────────┘
                   │
         ┌─────────▼─────────┐
         │  Score >= 0.65?   │
         └────┬─────────┬────┘
              │YES       │NO
              ▼          ▼
         📄 PDF       🌐 Web
         Answer      Fallback
              │          │
              └────┬─────┘
                   ▼
            Groq LLaMA 3.1
               Response
```

### Routing Logic
| Fused Hybrid Score | Mode | Source |
|---|---|---|
| `>= 0.65` | 📄 PDF | Retrieved from your document |
| `< 0.65` | 🌐 Web | LLM general knowledge fallback |

---

## ⚙️ Tech Stack

| Component | Technology |
|---|---|
| UI | Streamlit |
| Vector Store | ChromaDB |
| Embeddings | Sentence-Transformers (`all-MiniLM-L6-v2`) |
| Keyword Search | BM25 (`rank-bm25`) |
| Score Fusion | RRF (Reciprocal Rank Fusion) |
| LLM | Groq LLaMA 3.1 8B Instant |
| PDF Parsing | PyPDF |
| Text Splitting | LangChain RecursiveCharacterTextSplitter |

---

## 📊 Evaluation Results

```
=== Eval on 5 Q&A pairs ===

Q: What is the Turing Test?         Mode: PDF  ✅  keyword found: True
Q: Who are the Godfathers of AI?    Mode: PDF  ✅  keyword found: True
Q: What are three core elements?    Mode: PDF  ✅  keyword found: True
Q: What is a hallucination in AI?   Mode: PDF  ✅  keyword found: True
Q: When was ChatGPT released?       Mode: PDF  ✅  keyword found: True

=== Accuracy: 5/5 (100%) ===
```

---

## 🔍 Why Hybrid Search?

Pure semantic search fails on exact keyword queries like `"GPT-4"`, `"Section 3.2"`, or `"Algorithms"` because embeddings capture *meaning*, not exact tokens. BM25 catches these instantly.

| Query Type | Semantic Only | BM25 Only | Hybrid |
|---|---|---|---|
| "What is AI?" | ✅ | ⚠️ | ✅ |
| "three core elements" | ⚠️ 0.49 | ✅ 1.00 | ✅ 0.74 |
| "GPT-4 release date" | ⚠️ | ✅ | ✅ |
| Out-of-PDF question | ❌ false PDF | ❌ false PDF | ✅ Web |

---

## 📁 Project Structure

```
pdf-research-agent/
│
├── app.py              # Streamlit UI
├── agent.py            # Routing logic + Groq LLM calls
├── rag_engine.py       # PDF parsing, chunking, ChromaDB, BM25
├── eval.py             # Accuracy evaluation script
├── sample_qa.json      # Test Q&A pairs
├── requirements.txt    # Dependencies
└── README.md
```

---

## 🛠️ Setup & Run

### 1. Clone the repo
```bash
git clone https://github.com/muhammadumerqureshi-wq/pdf-research-agent.git
cd pdf-research-agent
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set your Groq API key
```bash
# Windows PowerShell
$env:GROQ_API_KEY = "your_groq_api_key_here"

# Or create a .env file
echo "GROQ_API_KEY=your_groq_api_key_here" > .env
```

Get your free API key at [console.groq.com](https://console.groq.com)

### 4. Run the app
```bash
streamlit run app.py
```

### 5. Run evaluation
```bash
# Add test.pdf to project folder first
python eval.py
```

---

## 🔑 Key Features

- **Hybrid Retrieval** — BM25 + Semantic search fused with weighted RRF scoring
- **Smart Routing** — Confidence threshold decides PDF vs Web automatically
- **224 Chunks** — Proper character-based chunking (500 chars, 50 overlap)
- **Cosine Similarity** — Correctly converts ChromaDB distance to similarity
- **Batch Upsert** — Handles large PDFs safely with batched ChromaDB inserts
- **100% Eval Accuracy** — Verified on domain-specific Q&A pairs

---

## 📦 Requirements

```
streamlit
chromadb
sentence-transformers
pypdf
groq
rank-bm25
langchain-text-splitters
python-dotenv
```

---

## 👨‍💻 Author

**Muhammad Umer Qureshi**  
AI/ML Engineer
[GitHub](https://github.com/muhammadumerqureshi-wq)  
[Fiverr](https://www.fiverr.com/s/38D79yB)
[Linkedin](https://www.linkedin.com/in/muhammad-umer-qureshi-243b12387/) 
[Gmail] <muhammadumerqueshi39@gmail.com>

---

## 📄 License

MIT License — free to use, modify, and distribute.
