# PLP RAG System

A comprehensive Retrieval-Augmented Generation (RAG) system built with Python, LangChain, and Streamlit. This system allows you to upload documents, create a searchable knowledge base, and ask questions to get AI-powered answers based on your documents.

## Features

- **Document Processing**: Support for PDF, TXT, DOCX, MD, and HTML files
- **Vector Storage**: ChromaDB for efficient similarity search
- **Multiple Embeddings**: OpenAI embeddings or SentenceTransformer embeddings
- **Retrieval System**: Advanced document retrieval with similarity search
- **Generation**: OpenAI GPT models for answer generation
- **Web Interface**: Beautiful Streamlit-based chat interface
- **Conversation Memory**: Maintains context across multiple questions
- **Source Citation**: Shows which documents were used for each answer

## Installation

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   CHROMA_PERSIST_DIRECTORY=./chroma_db
   ```

## Quick Start

### 1. Using the Web Interface

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

### 2. Using the Python API

```python
from rag_pipeline import RAGManager

# Initialize the system
rag_manager = RAGManager()

# Add documents
rag_manager.add_directory("documents/")

# Ask questions
response = rag_manager.ask("What is machine learning?")
print(response['answer'])
```

### 3. Run Examples

```bash
python examples/basic_usage.py
```

## Project Structure

```
PLP RAG/
├── app.py                 # Streamlit web interface
├── config.py             # Configuration settings
├── document_processor.py # Document loading and chunking
├── vector_store.py       # Vector database operations
├── retrieval_system.py   # Document retrieval logic
├── generation_system.py  # LLM integration for answers
├── rag_pipeline.py       # Main RAG pipeline
├── requirements.txt      # Python dependencies
├── documents/           # Sample documents
├── examples/           # Usage examples
└── README.md          # This file
```

## Configuration

Edit `config.py` to customize:

- **Chunk size and overlap**: How documents are split
- **Model settings**: Which LLM and embedding models to use
- **File types**: Which document formats to support
- **Search parameters**: Number of documents to retrieve

## Usage Examples

### Basic Usage

```python
from rag_pipeline import RAGManager

# Initialize
rag = RAGManager()

# Add documents
rag.add_documents(["document1.pdf", "document2.txt"])

# Ask questions
response = rag.ask("What is the main topic?")
print(response['answer'])
```

### Chat with Memory

```python
# Chat maintains conversation context
response1 = rag.chat("What is machine learning?")
response2 = rag.chat("What are its main types?")  # Remembers previous context
```

### Search Without Generation

```python
# Get relevant documents without generating an answer
results = rag.search_context("machine learning algorithms")
print(f"Found {results['total_sources']} relevant sources")
```

## API Reference

### RAGManager

Main class for interacting with the RAG system.

#### Methods

- `add_documents(file_paths)`: Add multiple documents
- `add_directory(directory_path)`: Add all documents from a directory
- `ask(question, use_memory=False)`: Ask a question
- `chat(question)`: Chat with conversation memory
- `get_suggestions()`: Get suggested questions
- `get_stats()`: Get system statistics
- `reset()`: Reset the system
- `search_context(query)`: Search for relevant context

### DocumentProcessor

Handles document loading and processing.

#### Supported Formats

- **PDF**: `.pdf` files
- **Text**: `.txt` files
- **Word**: `.docx` files
- **Markdown**: `.md` files
- **HTML**: `.html` files

### VectorStore

Manages the vector database and embeddings.

#### Features

- Persistent storage with ChromaDB
- Multiple embedding models
- Similarity search with scores
- Metadata filtering
- Collection management

## Web Interface Features

### Document Management
- Upload multiple documents at once
- Support for various file formats
- Automatic processing and chunking
- System statistics display

### Chat Interface
- Real-time question answering
- Source citation and references
- Conversation history
- Suggested questions
- Memory management

### Configuration
- API key management
- System initialization
- Reset functionality
- Status monitoring

## Advanced Features

### Custom Embeddings

You can use different embedding models by modifying the `VectorStore` initialization:

```python
# Use SentenceTransformer instead of OpenAI
vector_manager = VectorStoreManager(use_openai=False)
```

### Compression Retrieval

Enable contextual compression for better retrieval:

```python
retrieval_manager = RetrievalManager(vector_manager, use_compression=True)
```

### Custom Prompts

Modify the generation prompts in `generation_system.py` to customize how answers are generated.

## Troubleshooting

### Common Issues

1. **OpenAI API Key Error**
   - Make sure your API key is set in the environment or `.env` file
   - Verify the key is valid and has sufficient credits

2. **Document Processing Errors**
   - Check that files are in supported formats
   - Ensure files are not corrupted or password-protected

3. **Memory Issues**
   - Large documents may require more RAM
   - Consider reducing chunk size in `config.py`

4. **Vector Store Issues**
   - Delete the `chroma_db` directory to reset the vector store
   - Check disk space for vector storage

### Performance Tips

- Use smaller chunk sizes for better precision
- Increase chunk overlap for better context
- Use compression retrieval for better relevance
- Monitor system stats to optimize performance

## Contributing

Feel free to contribute to this project by:

1. Adding support for new document formats
2. Improving the retrieval algorithms
3. Enhancing the web interface
4. Adding new features or optimizations

## License

This project is open source and available under the MIT License.

## Support

For questions or issues:

1. Check the troubleshooting section
2. Review the examples in the `examples/` directory
3. Examine the code documentation
4. Create an issue with detailed information about your problem

---

**Happy RAG-ing!** 🤖📚
