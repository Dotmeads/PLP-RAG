"""
Test script to verify all imports work for Streamlit app
"""
import sys
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")

try:
    import streamlit as st
    print("✅ Streamlit import successful")
except ImportError as e:
    print(f"❌ Streamlit import failed: {e}")

try:
    import sentence_transformers
    print("✅ sentence-transformers import successful")
except ImportError as e:
    print(f"❌ sentence-transformers import failed: {e}")

try:
    from rank_bm25 import BM25Okapi
    print("✅ rank-bm25 import successful")
except ImportError as e:
    print(f"❌ rank-bm25 import failed: {e}")

try:
    import nltk
    print("✅ nltk import successful")
except ImportError as e:
    print(f"❌ nltk import failed: {e}")

try:
    from proposed_rag_system import ProposedRAGManager
    print("✅ ProposedRAGManager import successful")
except ImportError as e:
    print(f"❌ ProposedRAGManager import failed: {e}")

try:
    from bm25_retriever import BM25Retriever
    print("✅ BM25Retriever import successful")
except ImportError as e:
    print(f"❌ BM25Retriever import failed: {e}")

try:
    from rrf_fusion import RRFFusion
    print("✅ RRFFusion import successful")
except ImportError as e:
    print(f"❌ RRFFusion import failed: {e}")

try:
    from cross_encoder_reranker import CrossEncoderReranker
    print("✅ CrossEncoderReranker import successful")
except ImportError as e:
    print(f"❌ CrossEncoderReranker import failed: {e}")

try:
    from extractive_generator import ExtractiveAnswerGenerator
    print("✅ ExtractiveAnswerGenerator import successful")
except ImportError as e:
    print(f"❌ ExtractiveAnswerGenerator import failed: {e}")

print("\n🎉 All imports completed!")
