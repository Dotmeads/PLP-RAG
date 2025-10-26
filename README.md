# 🐾 Pawfect Match - Advanced Pet Care & Adoption System

A comprehensive AI-powered system that combines **Retrieval-Augmented Generation (RAG)** for pet care questions with **intelligent chatbot capabilities** for pet adoption assistance. Features hybrid retrieval, intent classification, entity extraction, and seamless integration with Azure cloud services.

## ✨ Key Features

### 🧠 **Advanced RAG System**
- **Hybrid Retrieval**: BM25 + Dense embeddings (all-MiniLM-L6-v2) with RRF fusion
- **Smart Reranking**: Cross-encoder reranking for improved relevance
- **Multi-format Support**: PDF, TXT, MD, DOCX document processing
- **Free LLM Integration**: Groq, DeepSeek, and Hugging Face with intelligent fallbacks
- **941 Document Chunks**: Comprehensive pet care knowledge base

### 🤖 **Intelligent Chatbot**
- **Intent Classification**: Distinguishes between pet adoption and pet care queries
- **Entity Extraction**: NER model extracts pet types, breeds, locations, and attributes
- **Multi-Turn Conversations**: Context-aware responses with session tracking
- **Entity Accumulation**: Builds up pet preferences across conversation turns
- **Smart Routing**: Automatically directs queries to appropriate systems
- **Conversation State Management**: Maintains context across multiple interactions

### ☁️ **Azure Integration** ⚡ **OPTIMIZED**
- **Cloud Storage**: Azure Blob Storage for models and data
- **Advanced Search**: FAISS-based similarity search with BM25
- **Scalable Architecture**: Production-ready cloud deployment
- **Instant Search**: Pre-loaded components for <1 second response times
- **Smart Caching**: 5-minute result caching for repeated queries
- **6,956 Pet Database**: Comprehensive adoption database with 401 breeds

## 🏗️ System Architecture

```
User Query
    ↓
Intent Classification (Adoption vs Care)
    ↓
┌─────────────────┬─────────────────┐
│   Pet Adoption  │   Pet Care      │
│   (Azure Search)│   (RAG System)  │
└─────────────────┴─────────────────┘
    ↓
Entity Extraction & Response Generation
    ↓
Unified Response
```

## 📁 Project Structure

```
PLP RAG/
├── 🧠 **RAG System** (`rag_system/`)
│   ├── bm25_retriever.py          # BM25 keyword retrieval
│   ├── cross_encoder_reranker.py  # Document reranking
│   ├── document_processor.py      # Multi-format document processing
│   ├── free_llm_generator.py     # LLM integration (Groq, DeepSeek)
│   ├── proposed_rag_system.py    # Main RAG orchestrator
│   ├── rrf_fusion.py             # Rank fusion algorithm
│   └── vector_store.py           # ChromaDB vector storage
│
├── 🤖 **Chatbot Flow** (`chatbot_flow/`)
│   ├── chatbot_pipeline.py       # Main chatbot logic
│   ├── entity_extractor.py       # NER for pet entities
│   ├── intent_classifier.py      # Intent classification
│   ├── responses.py              # Response templates
│   └── synonyms.py               # Entity normalization
│
├── ☁️ **Azure System** (`azure_system/`)
│   └── azure_petbot_app.py       # Azure pet search app
│
├── 🚀 **Applications** (`apps/`)
│   ├── unified_petbot_app.py     # Main unified application
│   └── ui_style.py              # UI styling module
│
├── 🧪 **Tests** (`tests/`)
│   ├── test_unified_integration.py # Integration tests
│   ├── test_azure_connection.py    # Azure connection testing
│   ├── test_enhanced_retrieval.py  # Enhanced retrieval testing
│   └── evaluate_rag_quality.py    # RAG quality evaluation
│
├── 💬 **Multi-Turn Chat** (`multi_turn/`)
│   ├── multi_turn_chat.py          # Interactive multi-turn chat
│   └── multi_turn_demo.py          # Automated conversation demos
│
├── 📚 **Data & Models**
│   ├── documents/                # Pet care knowledge base
│   ├── models/                   # Pre-trained ML models
│   ├── pet_retrieval/            # Azure components
│   └── chroma_db/                # Vector database
│
└── 📖 **Documentation**
    ├── AZURE_SETUP.md                    # Azure configuration guide
    └── ORGANIZATION_SUMMARY.md           # Project organization summary
```

## 🚀 Quick Start

### 1. **Environment Setup**
```bash
# Clone and navigate to project
cd "PLP RAG"

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. **API Keys Configuration**
Create `api_keys.py` or set environment variables:
```python
# api_keys.py
GROQ_API_KEY = "your_groq_key_here"
DEEPSEEK_API_KEY = "your_deepseek_key_here"
```

Or export environment variables:
```bash
export GROQ_API_KEY="your_groq_key_here"
export DEEPSEEK_API_KEY="your_deepseek_key_here"
```

### 3. **Run the Unified Application** ⚡ **OPTIMIZED**
```bash
# Activate virtual environment
source .venv/bin/activate

# Run the optimized unified app (pre-loads all components)
streamlit run apps/unified_petbot_app.py --server.port 8503

# The app will show "🚀 Initializing PetBot systems..." during startup
# After initialization, all searches are instant!
```

## 🎯 Usage Examples

### **Pet Care Questions (RAG System)**
```python
from rag_system.proposed_rag_system import ProposedRAGManager

# Initialize RAG system
rag = ProposedRAGManager(collection_name="pet_care", use_openai=False)
rag.add_directory("documents")

# Ask pet care questions
result = rag.ask("What can I feed my cat?")
print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']}")
print(f"Sources: {len(result['sources'])} documents")
```

### **Pet Adoption Queries (Chatbot)**
```python
from chatbot_flow.chatbot_pipeline import ChatbotPipeline
from rag_system.proposed_rag_system import ProposedRAGManager

# Initialize chatbot with RAG integration
rag = ProposedRAGManager(collection_name="pet_care", use_openai=False)
rag.add_directory("documents")
chatbot = ChatbotPipeline(rag)

# Handle adoption queries
response = chatbot.handle_message("I want to adopt a golden retriever puppy in Selangor")
print(response)
```

### **Multi-Turn Conversations**
```python
from chatbot_flow.chatbot_pipeline import ChatbotPipeline
from rag_system.proposed_rag_system import ProposedRAGManager

# Initialize chatbot with RAG integration
rag = ProposedRAGManager(collection_name="pet_care", use_openai=False)
rag.add_directory("documents")
chatbot = ChatbotPipeline(rag)

# Multi-turn conversation example
responses = []
responses.append(chatbot.handle_message("I want to adopt a pet"))
responses.append(chatbot.handle_message("I prefer dogs"))
responses.append(chatbot.handle_message("Golden retrievers are nice"))
responses.append(chatbot.handle_message("I live in Selangor"))

for i, response in enumerate(responses, 1):
    print(f"Turn {i}: {response}")
```

### **Intent Classification & Entity Extraction**
```python
from chatbot_flow.intent_classifier import IntentClassifier
from chatbot_flow.entity_extractor import EntityExtractor

# Intent classification
intent_classifier = IntentClassifier()
intent = intent_classifier.predict("I want to adopt a dog")
print(f"Intent: {intent[0]} (confidence: {intent[1]:.2f})")

# Entity extraction
entity_extractor = EntityExtractor()
entities = entity_extractor.extract("I want a golden retriever puppy in Selangor")
print(f"Entities: {entities}")
```

## 💬 Multi-Turn Conversation Features

### **🎭 Conversation Capabilities**
- **Session State Management**: Maintains conversation context across multiple turns
- **Entity Accumulation**: Builds up pet preferences progressively (breed → location → age)
- **Intent Persistence**: Remembers user's primary goal throughout conversation
- **Context-Aware Responses**: References previous conversation elements
- **Smart Intent Switching**: Seamlessly transitions between adoption and care topics

### **📝 Example Multi-Turn Flow**
```
👤 User: "I want to adopt a pet"
🤖 Bot: "Which state or area are you in?"

👤 User: "I prefer dogs" 
🤖 Bot: "Which state or area are you in?"
📊 State: Intent=find_pet, Entities={'PET_TYPE': 'dog'}

👤 User: "Golden retrievers are nice"
🤖 Bot: "Added breed: Golden Retriever. Which state or area are you in?"
📊 State: Intent=find_pet, Entities={'PET_TYPE': 'dog', 'BREED': 'Golden Retriever'}

👤 User: "I live in Selangor"
🤖 Bot: "Got it! Searching for Golden Retriever dog in Selangor..."
📊 State: Intent=find_pet, Entities={'PET_TYPE': 'dog', 'BREED': 'Golden Retriever', 'STATE': 'Selangor'}
```

### **🚀 Testing Multi-Turn Features**
```bash
# Automated demo scenarios
python multi_turn/multi_turn_demo.py

# Interactive chat mode
python multi_turn/multi_turn_chat.py

# Test specific conversation flows
python -c "
from chatbot_flow.chatbot_pipeline import ChatbotPipeline
from rag_system.proposed_rag_system import ProposedRAGManager

rag = ProposedRAGManager('test', use_openai=False)
rag.add_directory('documents')
chatbot = ChatbotPipeline(rag)

# Test multi-turn conversation
print(chatbot.handle_message('I want to adopt a dog'))
print(chatbot.handle_message('Golden retrievers'))
print(chatbot.handle_message('In Selangor'))
"
```

## ⚙️ Configuration

### **RAG System Parameters**
```python
# Custom configuration
rag = ProposedRAGManager(
    collection_name="custom_collection",
    use_openai=False,
    chunk_size=1000,
    chunk_overlap=200
)

# Query with custom parameters
result = rag.ask(
    "What vaccines does my kitten need?",
    use_reranking=True,
    rerank_threshold=0.1,
    max_rerank=20
)
```

### **Azure Configuration**
For Azure integration, create `.streamlit/secrets.toml`:
```toml
[azure]
connection_string = "your_azure_connection_string"
ml_container = "ml-artifacts"
pets_container = "pets-data"
```

See `AZURE_SETUP.md` for detailed Azure configuration instructions.

## 📊 Performance Metrics

- **RAG System**: 941 document chunks, ~2s response time, 0.9+ confidence
- **Pet Search**: <1s search time, 6,956 pets, BM25 + FAISS hybrid search
- **Chatbot**: 98.6% intent accuracy, multi-turn conversation support
- **System**: Pre-loading, smart caching, 5-min result cache

## 🔧 Development

### **Adding New Document Types**
1. Update `document_processor.py` with new file type support
2. Add processing logic in `load_document()` method
3. Test with sample files

### **Extending Chatbot Capabilities**
1. Add new intents in `intent_classifier.py`
2. Update entity types in `entity_extractor.py`
3. Add response templates in `responses.py`

### **Customizing RAG Parameters**
1. Modify `config.py` for global settings
2. Pass parameters to `ProposedRAGManager`
3. Adjust reranking thresholds as needed

## 📈 Future Enhancements

- [x] **Multi-Turn Conversations**: ✅ Implemented with session state management
- [x] **Entity Accumulation**: ✅ Progressive building of pet preferences
- [x] **Intent Switching**: ✅ Seamless transitions between adoption and care
- [ ] **Multi-language Support**: Extend to support multiple languages
- [ ] **Voice Interface**: Add speech-to-text and text-to-speech
- [ ] **Mobile App**: React Native mobile application
- [ ] **Advanced Analytics**: User interaction tracking and insights
- [ ] **Pet Health Monitoring**: Integration with health tracking devices

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📜 License

This project is provided as-is for educational and practical use cases in pet care and adoption assistance.

## 🙏 Acknowledgments

- **RAG System**: Built on LangChain, ChromaDB, and Sentence Transformers
- **Chatbot Components**: Integrated from pet adoption chatbot project
- **Azure Integration**: Cloud services integration for scalable deployment
- **Documentation**: Comprehensive guides for easy setup and usage

---

**🐾 Built with ❤️ for pet lovers and their furry friends.**

*For detailed setup instructions, see `AZURE_SETUP.md` and `PROJECT_ORGANIZATION.md`*