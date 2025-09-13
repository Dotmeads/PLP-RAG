"""
Main RAG pipeline that orchestrates all components
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from document_processor import DocumentProcessor
from vector_store import VectorStoreManager
from retrieval_system import RetrievalManager
from generation_system import GenerationManager
from config import MAX_CHUNKS, OPENAI_API_KEY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """Main RAG pipeline that orchestrates all components"""
    
    def __init__(self, collection_name: str = "rag_documents", use_openai: bool = True):
        self.collection_name = collection_name
        self.use_openai = use_openai
        
        # Initialize components
        self.document_processor = DocumentProcessor()
        self.vector_manager = VectorStoreManager(collection_name, use_openai)
        self.retrieval_manager = RetrievalManager(self.vector_manager, use_compression=False)
        
        # Initialize generation system if OpenAI key is available
        if OPENAI_API_KEY:
            self.generation_manager = GenerationManager(use_chat=True)
        else:
            self.generation_manager = None
            logger.warning("OpenAI API key not found. Generation will be disabled.")
        
        logger.info("RAG pipeline initialized successfully")
    
    def ingest_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        """Ingest multiple documents into the system"""
        results = {
            "success": [],
            "failed": [],
            "total_chunks": 0
        }
        
        for file_path in file_paths:
            try:
                logger.info(f"Ingesting document: {file_path}")
                
                # Process document
                chunks = self.document_processor.process_file(file_path)
                
                if chunks:
                    # Add to vector store
                    success = self.vector_manager.ingest_documents(chunks)
                    
                    if success:
                        results["success"].append(file_path)
                        results["total_chunks"] += len(chunks)
                        logger.info(f"Successfully ingested {file_path} ({len(chunks)} chunks)")
                    else:
                        results["failed"].append(f"{file_path}: Failed to add to vector store")
                else:
                    results["failed"].append(f"{file_path}: No chunks generated")
                    
            except Exception as e:
                error_msg = f"{file_path}: {str(e)}"
                results["failed"].append(error_msg)
                logger.error(f"Error ingesting {file_path}: {str(e)}")
        
        logger.info(f"Ingestion complete: {len(results['success'])} success, {len(results['failed'])} failed")
        return results
    
    def ingest_directory(self, directory_path: str) -> Dict[str, Any]:
        """Ingest all supported documents from a directory"""
        try:
            logger.info(f"Ingesting directory: {directory_path}")
            
            # Process all documents in directory
            chunks = self.document_processor.process_directory(directory_path)
            
            if chunks:
                # Add to vector store
                success = self.vector_manager.ingest_documents(chunks)
                
                if success:
                    return {
                        "success": [directory_path],
                        "failed": [],
                        "total_chunks": len(chunks)
                    }
                else:
                    return {
                        "success": [],
                        "failed": [f"{directory_path}: Failed to add to vector store"],
                        "total_chunks": 0
                    }
            else:
                return {
                    "success": [],
                    "failed": [f"{directory_path}: No documents found or processed"],
                    "total_chunks": 0
                }
                
        except Exception as e:
            logger.error(f"Error ingesting directory {directory_path}: {str(e)}")
            return {
                "success": [],
                "failed": [f"{directory_path}: {str(e)}"],
                "total_chunks": 0
            }
    
    def query(self, question: str, k: int = MAX_CHUNKS, 
              use_memory: bool = False) -> Dict[str, Any]:
        """Query the RAG system with a question"""
        try:
            logger.info(f"Processing query: {question[:100]}...")
            
            # Retrieve relevant documents
            context, sources = self.retrieval_manager.search_and_get_context(question, k)
            
            if not context:
                return {
                    "answer": "I couldn't find any relevant information to answer your question.",
                    "sources": [],
                    "context": "",
                    "query": question,
                    "error": "No relevant documents found"
                }
            
            # Generate answer if generation system is available
            if self.generation_manager:
                generation_result = self.generation_manager.answer_question(
                    question, context, sources, use_memory=use_memory
                )
                
                return {
                    "answer": generation_result.answer,
                    "sources": sources,
                    "context": context,
                    "query": question,
                    "model_used": generation_result.model_used
                }
            else:
                # Return context without generation
                return {
                    "answer": "Generation system not available. Here's the relevant context:",
                    "sources": sources,
                    "context": context,
                    "query": question,
                    "error": "No generation system available"
                }
                
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "answer": f"I encountered an error while processing your question: {str(e)}",
                "sources": [],
                "context": "",
                "query": question,
                "error": str(e)
            }
    
    def chat(self, question: str, k: int = MAX_CHUNKS) -> Dict[str, Any]:
        """Chat with the RAG system (with memory)"""
        if not self.generation_manager:
            return {
                "answer": "Chat functionality requires a generation system. Please set up OpenAI API key.",
                "sources": [],
                "context": "",
                "query": question,
                "error": "No generation system available"
            }
        
        return self.query(question, k, use_memory=True)
    
    def get_suggested_questions(self, context: str = None) -> List[str]:
        """Get suggested questions based on context or recent documents"""
        if not self.generation_manager:
            return []
        
        try:
            if context:
                return self.generation_manager.get_suggested_questions(context)
            else:
                # Get some recent documents to generate questions
                recent_docs = self.vector_manager.search("", k=3)
                if recent_docs:
                    context = "\n".join([doc.page_content for doc in recent_docs])
                    return self.generation_manager.get_suggested_questions(context)
                else:
                    return []
                    
        except Exception as e:
            logger.error(f"Error generating suggested questions: {str(e)}")
            return []
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        try:
            vector_stats = self.vector_manager.get_stats()
            
            stats = {
                "vector_store": vector_stats,
                "collection_name": self.collection_name,
                "generation_available": self.generation_manager is not None,
                "openai_configured": OPENAI_API_KEY is not None
            }
            
            if self.generation_manager:
                stats["conversation_memory"] = self.generation_manager.generation_system.get_memory_summary()
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting system stats: {str(e)}")
            return {"error": str(e)}
    
    def reset_system(self):
        """Reset the entire system (clear vector store and memory)"""
        try:
            # Reset vector store
            self.vector_manager.vector_store.reset_collection()
            
            # Reset generation memory
            if self.generation_manager:
                self.generation_manager.reset_conversation()
            
            logger.info("System reset successfully")
            
        except Exception as e:
            logger.error(f"Error resetting system: {str(e)}")
            raise
    
    def export_context(self, query: str, k: int = MAX_CHUNKS) -> Dict[str, Any]:
        """Export context and sources for a query without generation"""
        try:
            context, sources = self.retrieval_manager.search_and_get_context(query, k)
            
            return {
                "query": query,
                "context": context,
                "sources": sources,
                "total_sources": len(sources)
            }
            
        except Exception as e:
            logger.error(f"Error exporting context: {str(e)}")
            return {
                "query": query,
                "context": "",
                "sources": [],
                "total_sources": 0,
                "error": str(e)
            }


class RAGManager:
    """High-level manager for RAG operations"""
    
    def __init__(self, collection_name: str = "rag_documents", use_openai: bool = True):
        self.pipeline = RAGPipeline(collection_name, use_openai)
    
    def add_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        """Add documents to the RAG system"""
        return self.pipeline.ingest_documents(file_paths)
    
    def add_directory(self, directory_path: str) -> Dict[str, Any]:
        """Add all documents from a directory"""
        return self.pipeline.ingest_directory(directory_path)
    
    def ask(self, question: str, use_memory: bool = False) -> Dict[str, Any]:
        """Ask a question to the RAG system"""
        return self.pipeline.query(question, use_memory=use_memory)
    
    def chat(self, question: str) -> Dict[str, Any]:
        """Chat with the RAG system"""
        return self.pipeline.chat(question)
    
    def get_suggestions(self) -> List[str]:
        """Get suggested questions"""
        return self.pipeline.get_suggested_questions()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        return self.pipeline.get_system_stats()
    
    def reset(self):
        """Reset the system"""
        self.pipeline.reset_system()
    
    def search_context(self, query: str) -> Dict[str, Any]:
        """Search for context without generation"""
        return self.pipeline.export_context(query)


if __name__ == "__main__":
    # Example usage
    rag_manager = RAGManager()
    
    # Get system stats
    stats = rag_manager.get_stats()
    print(f"System stats: {stats}")
    
    # Example: Add documents
    # results = rag_manager.add_documents(["document1.pdf", "document2.txt"])
    # print(f"Added documents: {results}")
    
    # Example: Ask a question
    # if OPENAI_API_KEY:
    #     response = rag_manager.ask("What is machine learning?")
    #     print(f"Answer: {response['answer']}")
    
    print("RAG pipeline ready!")
