"""
Retrieval system for finding relevant documents and context
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

# LangChain components
from langchain_core.documents import Document as LangChainDocument
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import ChatOpenAI

from vector_store import VectorStoreManager
from config import MAX_CHUNKS, DEFAULT_MODEL, OPENAI_API_KEY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Container for retrieval results"""
    documents: List[LangChainDocument]
    scores: List[float]
    query: str
    total_found: int


class RetrievalSystem:
    """Handles document retrieval and context preparation"""
    
    def __init__(self, vector_manager: VectorStoreManager, use_compression: bool = False):
        self.vector_manager = vector_manager
        self.use_compression = use_compression
        
        # Initialize compression retriever if requested
        if use_compression and OPENAI_API_KEY:
            self._setup_compression_retriever()
        else:
            self.compression_retriever = None
    
    def _setup_compression_retriever(self):
        """Set up contextual compression retriever"""
        try:
            llm = ChatOpenAI(
                openai_api_key=OPENAI_API_KEY,
                model_name=DEFAULT_MODEL,
                temperature=0
            )
            
            compressor = LLMChainExtractor.from_llm(llm)
            self.compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=self.vector_manager.vector_store.vectorstore.as_retriever()
            )
            
            logger.info("Compression retriever initialized")
            
        except Exception as e:
            logger.error(f"Error setting up compression retriever: {str(e)}")
            self.compression_retriever = None
    
    def retrieve_documents(self, query: str, k: int = MAX_CHUNKS, 
                          with_scores: bool = True, 
                          filter_dict: Optional[Dict[str, Any]] = None) -> RetrievalResult:
        """Retrieve relevant documents for a query"""
        try:
            logger.info(f"Retrieving documents for query: {query[:100]}...")
            
            # Use compression retriever if available
            if self.compression_retriever and not filter_dict:
                documents = self.compression_retriever.get_relevant_documents(query)
                scores = [1.0] * len(documents)  # Compression doesn't provide scores
                
            else:
                # Use regular vector search
                if with_scores:
                    results = self.vector_manager.vector_store.vectorstore.similarity_search_with_score(query, k)
                    documents = [doc for doc, score in results]
                    scores = [score for doc, score in results]
                else:
                    documents = self.vector_manager.vector_store.vectorstore.similarity_search(query, k)
                    scores = [1.0] * len(documents)
            
            # Apply filters if specified
            if filter_dict:
                documents = self._filter_documents(documents, filter_dict)
                scores = scores[:len(documents)]
            
            result = RetrievalResult(
                documents=documents,
                scores=scores,
                query=query,
                total_found=len(documents)
            )
            
            logger.info(f"Retrieved {len(documents)} documents")
            return result
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return RetrievalResult([], [], query, 0)
    
    def _filter_documents(self, documents: List[LangChainDocument], 
                         filter_dict: Dict[str, Any]) -> List[LangChainDocument]:
        """Filter documents based on metadata"""
        filtered_docs = []
        
        for doc in documents:
            if all(doc.metadata.get(key) == value for key, value in filter_dict.items()):
                filtered_docs.append(doc)
        
        return filtered_docs
    
    def get_context(self, retrieval_result: RetrievalResult, 
                   max_context_length: int = 4000) -> str:
        """Extract and format context from retrieved documents"""
        context_parts = []
        current_length = 0
        
        for i, doc in enumerate(retrieval_result.documents):
            doc_text = doc.page_content
            doc_length = len(doc_text)
            
            # Check if adding this document would exceed max length
            if current_length + doc_length > max_context_length:
                # Truncate the last document if needed
                remaining_length = max_context_length - current_length
                if remaining_length > 100:  # Only add if meaningful length remains
                    doc_text = doc_text[:remaining_length] + "..."
                    context_parts.append(f"Document {i+1}:\n{doc_text}")
                break
            
            context_parts.append(f"Document {i+1}:\n{doc_text}")
            current_length += doc_length
        
        context = "\n\n".join(context_parts)
        logger.info(f"Generated context of {len(context)} characters")
        return context
    
    def get_sources(self, retrieval_result: RetrievalResult) -> List[Dict[str, Any]]:
        """Extract source information from retrieved documents"""
        sources = []
        
        for i, doc in enumerate(retrieval_result.documents):
            source_info = {
                "index": i + 1,
                "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "metadata": doc.metadata,
                "score": retrieval_result.scores[i] if i < len(retrieval_result.scores) else 0.0
            }
            sources.append(source_info)
        
        return sources
    
    def hybrid_search(self, query: str, k: int = MAX_CHUNKS) -> RetrievalResult:
        """Perform hybrid search combining multiple retrieval strategies"""
        try:
            # Regular similarity search
            regular_result = self.retrieve_documents(query, k, with_scores=True)
            
            # TODO: Add keyword search, semantic search variations, etc.
            # For now, return regular result
            return regular_result
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {str(e)}")
            return RetrievalResult([], [], query, 0)
    
    def rerank_results(self, retrieval_result: RetrievalResult, 
                      query: str) -> RetrievalResult:
        """Rerank retrieved documents based on additional criteria"""
        try:
            # Simple reranking based on query-document similarity
            # In a more sophisticated system, you might use cross-encoders or other reranking models
            
            documents_with_scores = list(zip(retrieval_result.documents, retrieval_result.scores))
            
            # Sort by score (higher is better)
            documents_with_scores.sort(key=lambda x: x[1], reverse=True)
            
            reranked_docs = [doc for doc, score in documents_with_scores]
            reranked_scores = [score for doc, score in documents_with_scores]
            
            return RetrievalResult(
                documents=reranked_docs,
                scores=reranked_scores,
                query=query,
                total_found=len(reranked_docs)
            )
            
        except Exception as e:
            logger.error(f"Error reranking results: {str(e)}")
            return retrieval_result


class RetrievalManager:
    """High-level manager for retrieval operations"""
    
    def __init__(self, vector_manager: VectorStoreManager, use_compression: bool = False):
        self.retrieval_system = RetrievalSystem(vector_manager, use_compression)
    
    def search_and_get_context(self, query: str, k: int = MAX_CHUNKS, 
                              max_context_length: int = 4000) -> Tuple[str, List[Dict[str, Any]]]:
        """Search for documents and return formatted context and sources"""
        try:
            # Retrieve documents
            retrieval_result = self.retrieval_system.retrieve_documents(query, k)
            
            # Get context
            context = self.retrieval_system.get_context(retrieval_result, max_context_length)
            
            # Get sources
            sources = self.retrieval_system.get_sources(retrieval_result)
            
            return context, sources
            
        except Exception as e:
            logger.error(f"Error in search and context retrieval: {str(e)}")
            return "", []
    
    def search_with_reranking(self, query: str, k: int = MAX_CHUNKS) -> RetrievalResult:
        """Search with reranking applied"""
        try:
            # Initial retrieval
            retrieval_result = self.retrieval_system.retrieve_documents(query, k)
            
            # Rerank results
            reranked_result = self.retrieval_system.rerank_results(retrieval_result, query)
            
            return reranked_result
            
        except Exception as e:
            logger.error(f"Error in search with reranking: {str(e)}")
            return RetrievalResult([], [], query, 0)


if __name__ == "__main__":
    # Example usage
    from vector_store import VectorStoreManager
    
    vector_manager = VectorStoreManager()
    retrieval_manager = RetrievalManager(vector_manager)
    
    # Example search
    # context, sources = retrieval_manager.search_and_get_context("What is machine learning?")
    # print(f"Context: {context[:200]}...")
    # print(f"Sources: {len(sources)}")
    
    print("Retrieval system ready!")
