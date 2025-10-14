## 🚀 PLP RAG – Pet Care Retrieval-Augmented Generation

RAG system for pet care Q&A with hybrid retrieval, reranking, and LLM/baseline answer generation. Includes a Streamlit app and a simple CLI test runner.

### Key Components
- **Retrieval**: BM25 + dense embeddings (all‑MiniLM‑L6‑v2) fused via RRF
- **Reranking**: Cross‑encoder MiniLM on top‑k results
- **Answer Generation**: `free_llm_generator.py` with providers (Groq, DeepSeek) and a basic fallback
- **Vector Store**: ChromaDB via LangChain

### Current Repo Highlights
- Dedup ingestion: PDF files are skipped when a same‑name `.txt` exists
- Chunking tuned for markdown: larger chunks with overlap
- Batch add to Chroma to avoid “batch size exceeds maximum”
- Simple test runner `test_questions.py` to try questions quickly

---

## 📦 Setup

### 1) Create and activate venv
```bash
cd "/Users/dotsnoise/PLP RAG"
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies
Use the pinned versions in `requirements.txt` (these were validated together):
```bash
pip install -r requirements.txt
```

If you hit transformer/Hub compatibility issues, ensure versions align with:
- numpy 1.24.x, torch 2.1.x, transformers 4.36.x, sentence-transformers 2.2.2, huggingface_hub 0.19.4

### 3) Environment variables (recommended)
Do not commit API keys. Export them locally:
```bash
export GROQ_API_KEY="..."
export DEEPSEEK_API_KEY="..."
```

The system automatically prefers Groq → DeepSeek → basic fallback.

---

## ▶️ Usage

### A) Streamlit app
```bash
source .venv/bin/activate
streamlit run proposed_app.py --server.port 8501
```

### B) Python API
```python
from proposed_rag_system import ProposedRAGManager

rag = ProposedRAGManager(collection_name="proposed_rag_documents", use_openai=False)
rag.add_directory("documents")
result = rag.ask("What can I feed my cat?")
print(result["answer"])   # final answer
print(result["confidence"])  # float 0..1
print(result["sources"])  # list of source dicts
```

### C) Simple test runner
Edit questions in `test_questions.py`:
```python
questions = [
    "What can I feed my dog?",
    "What vaccines does my kitten need?",
]
```
Run it:
```bash
source .venv/bin/activate
python test_questions.py
```

---

## 🧠 How It Works

High‑level flow:
```
User Question
  → BM25 and Dense Retrieval
  → RRF Fusion of results
  → Cross‑encoder Reranking
  → FreeLLM/Basic generation with citations
  → Final Answer
```

### Retrieval
- `bm25_retriever.py`: BM25 with NLTK tokenization and stopwords
- `vector_store.py`: LangChain + Chroma vector store (all‑MiniLM‑L6‑v2)
- `rrf_fusion.py`: Reciprocal Rank Fusion to merge BM25 + dense

### Reranking
- `cross_encoder_reranker.py`: `cross-encoder/ms-marco-MiniLM-L-6-v2` reranks top results, thresholded

### Generation
- `free_llm_generator.py`: providers (Groq, DeepSeek, Hugging Face) with robust fallback to a basic extractive summary

### Ingestion
- `proposed_rag_system.py` → `ingest_directory` collects supported files and skips `.pdf` when same‑name `.txt` exists
- `document_processor.py` uses a markdown‑friendly splitter and larger chunk sizes with overlap

---

## 📁 Repository Structure
```
bm25_retriever.py
cross_encoder_reranker.py
document_processor.py
free_llm_generator.py
proposed_app.py
proposed_rag_system.py
rrf_fusion.py
test_questions.py
vector_store.py
documents/  # your corpus (txt, md, etc.)
```

Notes:
- Local artifacts like `.venv/` and `chroma_db/` are ignored and should not be committed
- PDF ingestion is supported but will be skipped if a `.txt` with the same basename exists

---

## ⚙️ Configuration

`config.py` and in‑code defaults control:
- Chunk size/overlap
- RRF fusion parameter `k`
- Reranker threshold and `max_rerank`

You can pass parameters via `ProposedRAGManager.ask(question, use_reranking=True, rerank_threshold=0.1, max_rerank=20)`.

---

## 🧪 Evaluation (optional)

For quick sanity checks, rely on:
- Confidence score in results
- Source list and answer preview in `test_questions.py`

BLEU/ROUGE are available but often under‑reflect RAG answer utility (due to free‑form, contextual responses).

---

## 🛠️ Troubleshooting

- Missing packages: activate venv then `pip install -r requirements.txt`
- SentenceTransformers import failures: ensure compatible `huggingface_hub==0.19.4`
- Chroma schema errors: delete `chroma_db/` to re‑initialize
- Large batch errors: document adds are already batched in `vector_store.py`
- NLTK tokenizer warnings: install/download required data (e.g., `punkt`)

---

## 🔒 Security & Git Hygiene

- Do not commit secrets. Keep `api_keys.py` out of git (`.gitignore`) or use environment variables
- If a secret was committed, rotate it and rewrite history before pushing

---

## 📜 License

Provided as‑is for educational and practical RAG use cases.

---

**🐾 Built for clear, sourced pet care answers.**
