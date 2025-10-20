#!/usr/bin/env python3
"""
Test script for Unified PetBot integration
Tests both RAG system and Azure components (if available)
"""
import os
import sys
import traceback

# Add the project root to Python path so we can import our organized modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def test_imports():
    """Test that all required modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        # Test RAG components
        from rag_system.proposed_rag_system import ProposedRAGManager
        print("✅ RAG system imported successfully")
        
        from chatbot_system.chatbot_pipeline import ChatbotPipeline
        from chatbot_system.intent_classifier import IntentClassifier
        from chatbot_system.entity_extractor import EntityExtractor
        print("✅ Chatbot components imported successfully")
        
        # Test Azure components
        from src.config import get_blob_settings, local_ner_dir, local_mr_dir, local_pets_csv_path
        from src.azure_io import download_prefix_flat, smart_download_single_blob
        from src.models import load_ner_pipeline, load_mr_model, load_faiss_index
        from src.retrieval import only_text, BM25, parse_facets_from_text, entity_spans_to_facets
        print("✅ Azure components imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_rag_system():
    """Test RAG system initialization"""
    print("\n🧠 Testing RAG system...")
    
    try:
        from rag_system.proposed_rag_system import ProposedRAGManager
        
        # Initialize RAG system
        rag = ProposedRAGManager(collection_name="test_rag_documents", use_openai=False)
        print("✅ RAG system initialized")
        
        # Test document ingestion
        if os.path.exists("documents"):
            rag.add_directory("documents")
            print(f"✅ Documents ingested successfully")
        else:
            print("⚠️ No documents directory found")
        
        # Test query
        result = rag.ask("What can I feed my dog?")
        print(f"✅ RAG query successful: {len(result['answer'])} characters")
        
        return True
        
    except Exception as e:
        print(f"❌ RAG system error: {e}")
        traceback.print_exc()
        return False

def test_chatbot_components():
    """Test chatbot components"""
    print("\n🤖 Testing chatbot components...")
    
    try:
        from chatbot_system.intent_classifier import IntentClassifier
        from chatbot_system.entity_extractor import EntityExtractor
        from chatbot_system.chatbot_pipeline import ChatbotPipeline
        from rag_system.proposed_rag_system import ProposedRAGManager
        
        # Initialize components
        intent_classifier = IntentClassifier()
        entity_extractor = EntityExtractor()
        rag = ProposedRAGManager(collection_name="test_chatbot_documents", use_openai=False)
        
        if os.path.exists("documents"):
            rag.add_directory("documents")
        
        chatbot = ChatbotPipeline(rag)
        print("✅ Chatbot pipeline initialized")
        
        # Test intent classification
        intent = intent_classifier.predict("I want to adopt a dog")
        print(f"✅ Intent classification: {intent}")
        
        # Test entity extraction
        entities = entity_extractor.extract("I want a golden retriever puppy in Selangor")
        print(f"✅ Entity extraction: {entities}")
        
        # Test chatbot response
        response = chatbot.handle_message("What can I feed my cat?")
        print(f"✅ Chatbot response: {len(response)} characters")
        
        return True
        
    except Exception as e:
        print(f"❌ Chatbot components error: {e}")
        traceback.print_exc()
        return False

def test_azure_components():
    """Test Azure components (if configured)"""
    print("\n☁️ Testing Azure components...")
    
    try:
        from src.config import get_blob_settings, local_ner_dir, local_mr_dir, local_pets_csv_path
        
        # Test configuration
        try:
            blob_settings = get_blob_settings()
            print("✅ Azure configuration loaded")
            print(f"   ML Container: {blob_settings['ml_container']}")
            print(f"   Pets Container: {blob_settings['pets_container']}")
        except Exception as e:
            print(f"⚠️ Azure configuration not available: {e}")
            print("   This is expected if Azure is not configured")
            return True
        
        # Test local directories
        ner_dir = local_ner_dir()
        mr_dir = local_mr_dir()
        pets_csv = local_pets_csv_path()
        
        print(f"✅ Local directories created:")
        print(f"   NER dir: {ner_dir}")
        print(f"   MR dir: {mr_dir}")
        print(f"   Pets CSV: {pets_csv}")
        
        # Test Azure I/O functions
        from src.azure_io import list_blobs_with_prefix, try_download_single_blob
        
        # This would require actual Azure credentials
        print("✅ Azure I/O functions available")
        
        return True
        
    except Exception as e:
        print(f"❌ Azure components error: {e}")
        traceback.print_exc()
        return False

def test_requirements():
    """Test that all required packages are installed"""
    print("\n📦 Testing requirements...")
    
    required_packages = [
        'streamlit',
        'pandas',
        'numpy',
        'transformers',
        'sentence-transformers',
        'torch',
        'chromadb',
        'rank-bm25',
        'nltk',
        'scikit-learn',
        'rapidfuzz',
        'joblib',
        'azure-storage-blob',
        'faiss-cpu'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {missing_packages}")
        print("Run: pip install -r requirements_stable.txt")
        return False
    
    print("✅ All required packages are installed")
    return True

def main():
    """Run all tests"""
    print("🐾 Unified PetBot Integration Test")
    print("=" * 50)
    
    tests = [
        ("Requirements", test_requirements),
        ("Imports", test_imports),
        ("RAG System", test_rag_system),
        ("Chatbot Components", test_chatbot_components),
        ("Azure Components", test_azure_components),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your Unified PetBot is ready to go!")
        print("\nTo run the app:")
        print("   streamlit run unified_petbot_app.py")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        print("\nCommon fixes:")
        print("   1. Install missing packages: pip install -r requirements_stable.txt")
        print("   2. Ensure documents directory exists with pet care files")
        print("   3. Configure Azure secrets if using Azure components")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
