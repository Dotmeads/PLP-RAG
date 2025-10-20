# 🗂️ Project Organization Summary (Developer Reference)

This document complements `README.md` with developer-focused notes: workflows, environment tips, internal scripts, and operational guidance not needed on the public homepage.

## 📦 Packages and Environments

- Python: 3.9–3.11 compatible; current venv located at `.venv/`.
- NumPy: use `numpy<2` to avoid compiled-extension incompatibilities with some deps.
- Prefer installing with:
  - `requirements_stable.txt` for deterministic installs
  - `requirements.txt` for the lighter core

Activation:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements_stable.txt
```

## 🧩 Repository Structure (quick view)

```
rag_system/          # RAG components (BM25, embeddings, reranker, Chroma)
chatbot_system/      # Intent classifier, NER, pipeline, responses, synonyms
apps/                # Streamlit apps (unified + optimized)
tests/               # Integration tests
src/                 # Azure pet search components from teammate
documents/           # Knowledge base
models/              # Local intent model artifacts
chroma_db/           # Chroma persistence dir
AZURE_SETUP.md       # Azure setup guide
README.md            # Main project docs (homepage)
```

## 🔑 Secrets and Keys

- LLM keys are read from `api_keys.py` or env vars.
- Azure credentials should live in `.streamlit/secrets.toml` (see `AZURE_SETUP.md`).
- Never commit secrets; use env or Streamlit secrets.

## ☁️ Azure Notes (Pet Search)

- Required secrets in `.streamlit/secrets.toml`:
  - `AZURE_CONNECTION_STRING`, `ML_ARTIFACTS_CONTAINER`, `PETS_CONTAINER`, `NER_PREFIX`, `MR_PREFIX`, `PETS_CSV_BLOB`
- Local cache: `artifacts/ner`, `artifacts/mr`, `artifacts/pets.csv`
- Quick connectivity check:
```bash
python test_azure_connection.py
```

## 🔗 RAG Components

- `rag_system/proposed_rag_system.py` exposes `ProposedRAGManager`
- Typical usage:
```python
from rag_system.proposed_rag_system import ProposedRAGManager
rag = ProposedRAGManager(collection_name="pet_care", use_openai=False)
rag.add_directory("documents")
rag.ask("What can I feed my dog?")
```
- Embeddings: SentenceTransformers `all-MiniLM-L6-v2`
- Vector store: Chroma (persisted under `chroma_db/`)

## 🤖 Chatbot Components

- `chatbot_system/chatbot_pipeline.py` orchestrates intent + NER + routing
- Intent model artifacts in `models/intent/*`
- NER pipeline downloads from HF hub (teammate model) on first run

## 💬 Multi-turn Scripts

- `multi_turn_demo.py` – automated scenarios (adoption, care, mixed)
- `multi_turn_chat.py` – interactive CLI with session persistence

## 🧪 Testing

Run the integration test that validates imports and minimal flows:
```bash
python tests/test_unified_integration.py
```

## ⚠️ Common Pitfalls

- If LangChain warnings appear, ensure `langchain_community` imports are used.
- If HF downloads time out, re-run; the cache will persist after first success.
- If `grpcio`/`pyarrow` builds are slow, prefer wheels and ensure Xcode tools are installed.
- If FAISS build fails, use prebuilt `faiss-cpu` wheel.

## 🔧 Developer Tasks

- Keep `README.md` as single source of truth for users.
- Use this summary for internal notes, shortcuts, and ops guidance.
- Update `requirements_stable.txt` when pinning/upgrading infra-sensitive deps.


