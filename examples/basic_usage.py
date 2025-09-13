"""
Basic usage examples for the RAG system
"""
import os
from pathlib import Path
from rag_pipeline import RAGManager

# Set up OpenAI API key (replace with your actual key)
# os.environ["OPENAI_API_KEY"] = "your-api-key-here"

def basic_example():
    """Basic example of using the RAG system"""
    print("=== Basic RAG System Example ===\n")
    
    # Initialize the RAG manager
    print("1. Initializing RAG system...")
    rag_manager = RAGManager()
    
    # Get system stats
    print("2. Getting system statistics...")
    stats = rag_manager.get_stats()
    print(f"   - Collection: {stats.get('vector_store', {}).get('collection_name', 'N/A')}")
    print(f"   - Documents: {stats.get('vector_store', {}).get('document_count', 0)}")
    print(f"   - Generation available: {stats.get('generation_available', False)}")
    print()
    
    # Add documents
    print("3. Adding sample documents...")
    documents_dir = Path("documents")
    if documents_dir.exists():
        results = rag_manager.add_directory(str(documents_dir))
        print(f"   - Successfully processed: {len(results['success'])} directories")
        print(f"   - Total chunks created: {results['total_chunks']}")
        if results['failed']:
            print(f"   - Failed: {results['failed']}")
    else:
        print("   - No documents directory found")
    print()
    
    # Ask questions (if OpenAI key is available)
    if os.getenv("OPENAI_API_KEY"):
        print("4. Asking questions...")
        
        questions = [
            "What is machine learning?",
            "What are the types of machine learning?",
            "What is artificial intelligence?",
            "What are the benefits of AI?"
        ]
        
        for question in questions:
            print(f"   Q: {question}")
            response = rag_manager.ask(question)
            print(f"   A: {response['answer'][:200]}...")
            print(f"   Sources: {len(response.get('sources', []))}")
            print()
    else:
        print("4. OpenAI API key not set - skipping question answering")
        print("   Set OPENAI_API_KEY environment variable to enable generation")
        print()
    
    # Search for context without generation
    print("5. Searching for context...")
    search_results = rag_manager.search_context("machine learning algorithms")
    print(f"   - Found {search_results['total_sources']} relevant sources")
    if search_results['context']:
        print(f"   - Context preview: {search_results['context'][:200]}...")
    print()
    
    # Get suggested questions
    print("6. Getting suggested questions...")
    suggestions = rag_manager.get_suggestions()
    if suggestions:
        print("   Suggested questions:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"   {i}. {suggestion}")
    else:
        print("   No suggestions available")
    print()
    
    print("=== Example completed ===")


def chat_example():
    """Example of using the chat functionality"""
    print("=== Chat Example ===\n")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("OpenAI API key not set - chat example skipped")
        return
    
    rag_manager = RAGManager()
    
    # Simulate a conversation
    conversation = [
        "What is machine learning?",
        "What are the main types?",
        "Can you give me examples of supervised learning?",
        "What about unsupervised learning examples?"
    ]
    
    print("Simulating a conversation:")
    for i, question in enumerate(conversation, 1):
        print(f"\n{i}. User: {question}")
        response = rag_manager.chat(question)
        print(f"   Assistant: {response['answer'][:300]}...")
        print(f"   Sources: {len(response.get('sources', []))}")


def document_processing_example():
    """Example of document processing"""
    print("=== Document Processing Example ===\n")
    
    from document_processor import DocumentProcessor
    
    processor = DocumentProcessor()
    
    # Process a single document
    sample_doc = Path("documents/sample_document.txt")
    if sample_doc.exists():
        print(f"Processing: {sample_doc}")
        chunks = processor.process_file(str(sample_doc))
        print(f"Created {len(chunks)} chunks")
        
        for i, chunk in enumerate(chunks[:2], 1):  # Show first 2 chunks
            print(f"\nChunk {i}:")
            print(f"Content: {chunk.page_content[:200]}...")
            print(f"Metadata: {chunk.metadata}")
    else:
        print("Sample document not found")


if __name__ == "__main__":
    # Run examples
    basic_example()
    print("\n" + "="*50 + "\n")
    
    if os.getenv("OPENAI_API_KEY"):
        chat_example()
        print("\n" + "="*50 + "\n")
    
    document_processing_example()
