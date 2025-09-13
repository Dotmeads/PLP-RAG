"""
Test script for the pet-focused RAG system
"""
import os
from rag_pipeline import RAGManager

def test_pet_rag_system():
    """Test the RAG system with pet care questions"""
    print("🐾 Testing Pet Care RAG System 🐾\n")
    
    # Initialize the RAG manager
    print("1. Initializing RAG system...")
    rag_manager = RAGManager()
    
    # Add the pet care documents
    print("2. Adding pet care documents...")
    documents_dir = "documents"
    results = rag_manager.add_directory(documents_dir)
    
    print(f"   ✅ Successfully processed: {len(results['success'])} directories")
    print(f"   📄 Total chunks created: {results['total_chunks']}")
    if results['failed']:
        print(f"   ❌ Failed: {results['failed']}")
    print()
    
    # Get system stats
    stats = rag_manager.get_stats()
    print(f"3. System Statistics:")
    print(f"   📊 Documents in vector store: {stats.get('vector_store', {}).get('document_count', 0)}")
    print(f"   🤖 Generation available: {stats.get('generation_available', False)}")
    print()
    
    # Test questions about pet care
    test_questions = [
        "What should I feed my cat?",
        "How often should I take my dog to the vet?",
        "What are the signs of a healthy pet?",
        "How do I care for a rabbit?",
        "What vaccinations does my dog need?",
        "How can I prevent obesity in my pet?",
        "What should I do if my pet has an emergency?",
        "How do I groom my pet properly?"
    ]
    
    print("4. Testing Pet Care Questions:")
    print("=" * 50)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n❓ Question {i}: {question}")
        
        if os.getenv("OPENAI_API_KEY"):
            # Test with generation
            response = rag_manager.ask(question)
            print(f"🤖 AI Answer: {response['answer'][:200]}...")
            print(f"📚 Sources used: {len(response.get('sources', []))}")
            
            # Show source previews
            if response.get('sources'):
                print("   📖 Source previews:")
                for j, source in enumerate(response['sources'][:2], 1):  # Show first 2 sources
                    print(f"      {j}. {source.get('content_preview', 'No preview')[:100]}...")
        else:
            # Test without generation (just retrieval)
            search_results = rag_manager.search_context(question)
            print(f"🔍 Found {search_results['total_sources']} relevant sources")
            if search_results['context']:
                print(f"📝 Context preview: {search_results['context'][:200]}...")
        
        print("-" * 30)
    
    # Test suggested questions
    print("\n5. Getting Suggested Questions:")
    suggestions = rag_manager.get_suggestions()
    if suggestions:
        print("💡 Suggested follow-up questions:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"   {i}. {suggestion}")
    else:
        print("   No suggestions available")
    
    print("\n🎉 Pet Care RAG System Test Complete!")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("\n💡 Tip: Set OPENAI_API_KEY environment variable to enable AI-powered answers")
        print("   Example: export OPENAI_API_KEY='your-api-key-here'")

if __name__ == "__main__":
    test_pet_rag_system()
