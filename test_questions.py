#!/usr/bin/env python3
"""
Simple RAG Test Script - Modify questions as needed
"""
from proposed_rag_system import ProposedRAGManager
import time

def test_rag_questions():
    """Test RAG system with custom questions"""
    
    # ========================================
    # MODIFY THESE QUESTIONS AS NEEDED
    # ========================================
    questions = [
        "What can I feed my dog?",
        "What vaccines does my kitten need?",
        "How can I tell if my dog is sick?"
        # "How often should I walk my dog?",
        # "What are signs of a sick cat?",
        # "How do I train my puppy?",
    ]
    
    print('🚀 RAG System Test - Custom Questions')
    print('=' * 60)
    
    # Initialize RAG system
    print('📚 Initializing RAG system...')
    rag = ProposedRAGManager(collection_name='proposed_rag_documents', use_openai=False)
    
    # Load documents
    print('📚 Loading documents...')
    result = rag.add_directory('documents')
    print(f'✅ Loaded {result["documents_processed"]} documents from {result["files_processed"]} files')
    
    # Test each question
    for i, question in enumerate(questions, 1):
        print(f'\n{"="*60}')
        print(f'🔍 Question {i}: {question}')
        print('-' * 60)
        
        try:
            start_time = time.time()
            answer_result = rag.ask(question)
            end_time = time.time()
            
            print(f'⏱️  Response time: {end_time - start_time:.2f}s')
            print(f'📊 Confidence: {answer_result["confidence"]:.3f}')
            print(f'🔗 Sources used: {len(answer_result["sources"])}')
            
            print('\n💬 Answer:')
            print('-' * 40)
            print(answer_result['answer'])
            
            print('\n📚 Sources:')
            for j, source in enumerate(answer_result['sources'], 1):
                source_name = source.get('source', 'Unknown').split('/')[-1]
                print(f'  {j}. {source_name}')
                
        except Exception as e:
            print(f'❌ Error processing question: {e}')
    
    print(f'\n{"="*60}')
    print('🎉 All tests completed!')

if __name__ == "__main__":
    test_rag_questions()
