# Project Organization Summary

## 📁 **Folder Structure:**

### **🔧 Core Configuration (Root Level):**
- `api_keys.py` - API keys for LLM providers
- `config.py` - General configuration settings
- `requirements.txt` - Dependencies
- `requirements_stable.txt` - Stable dependencies
- `README.md` - Project documentation
- `AZURE_SETUP.md` - Azure setup guide

### **🧠 RAG System (`rag_system/`):**
- `bm25_retriever.py` - BM25 keyword-based retrieval
- `cross_encoder_reranker.py` - Cross-encoder reranking
- `document_processor.py` - Document chunking and processing
- `free_llm_generator.py` - LLM answer generation (Groq, DeepSeek)
- `proposed_rag_system.py` - Main RAG orchestration
- `rrf_fusion.py` - Reciprocal Rank Fusion
- `vector_store.py` - ChromaDB vector storage

### **🤖 Chatbot System (`chatbot_system/`):**
- `chatbot_pipeline.py` - Main chatbot logic
- `entity_extractor.py` - Named Entity Recognition
- `intent_classifier.py` - Intent classification
- `responses.py` - Predefined response templates
- `synonyms.py` - Entity normalization and synonyms

### **☁️ Azure System (`azure_system/`):**
- `azure_petbot_app.py` - Azure-powered pet search app

### **🚀 Applications (`apps/`):**
- `optimized_unified_app.py` - Optimized unified app
- `unified_petbot_app.py` - Main unified application

### **🧪 Tests (`tests/`):**
- `test_unified_integration.py` - Integration tests

### **📚 Data & Models:**
- `documents/` - Pet care knowledge base
- `models/` - Pre-trained ML models
- `src/` - Azure components (from teammate)
- `chroma_db/` - Vector database storage

## 🔄 **Import Updates Needed:**

The following files need import path updates:
1. `apps/unified_petbot_app.py` - Update RAG imports
2. `apps/optimized_unified_app.py` - Update RAG imports  
3. `chatbot_system/chatbot_pipeline.py` - Update RAG imports
4. `tests/test_unified_integration.py` - Update all imports

## 📋 **Next Steps:**
1. Update import statements in affected files
2. Test the reorganized structure
3. Update documentation
