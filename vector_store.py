"""
Vector database and embedding storage for RAG system
"""
import os
import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings

# LangChain components
try:
    from langchain_community.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings
    from langchain_core.documents import Document as LangChainDocument
    from langchain_community.embeddings import SentenceTransformerEmbeddings
except ImportError:
    # Fallback for older versions
    from langchain.vectorstores import Chroma
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.schema import Document as LangChainDocument
    from langchain.embeddings import SentenceTransformerEmbeddings

from config import (
    CHROMA_PERSIST_DIRECTORY, 
    OPENAI_API_KEY, 
    EMBEDDING_MODEL,
    MAX_CHUNKS
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    """Manages vector database operations for RAG system"""
    
    def __init__(self, collection_name: str = "rag_documents", use_openai: bool = True):
        self.collection_name = collection_name
        self.use_openai = use_openai and OPENAI_API_KEY is not None
        
        # Initialize embeddings
        if self.use_openai:
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=OPENAI_API_KEY,
                model=EMBEDDING_MODEL
            )
            logger.info("Using OpenAI embeddings")
        else:
            self.embeddings = SentenceTransformerEmbeddings(
                model_name="all-MiniLM-L6-v2"
            )
            logger.info("Using SentenceTransformer embeddings")
        
        # Initialize ChromaDB
        self._initialize_chroma()
    
    def _initialize_chroma(self):
        """Initialize ChromaDB client and collection"""
        try:
            # Create persist directory if it doesn't exist
            os.makedirs(CHROMA_PERSIST_DIRECTORY, exist_ok=True)
            
            # Initialize ChromaDB client
            self.chroma_client = chromadb.PersistentClient(
                path=CHROMA_PERSIST_DIRECTORY,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Initialize or get collection
            self.vectorstore = Chroma(
                client=self.chroma_client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=CHROMA_PERSIST_DIRECTORY
            )
            
            logger.info(f"Vector store initialized with collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            raise
    
    def add_documents(self, documents: List[LangChainDocument]) -> List[str]:
        """Add documents to the vector store"""
        try:
            if not documents:
                logger.warning("No documents to add")
                return []
            
            # Add documents in smaller batches to avoid batch size limits
            batch_size = 100
            all_ids = []
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                batch_ids = self.vectorstore.add_documents(batch)
                all_ids.extend(batch_ids)
                logger.info(f"Added batch {i//batch_size + 1}: {len(batch)} documents")
            
            # Persist the changes
            self.vectorstore.persist()
            
            logger.info(f"Added {len(documents)} documents to vector store in {len(all_ids)} batches")
            return all_ids
            
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise
    
    def add_texts(self, texts: List[str], metadatas: List[Dict[str, Any]] = None) -> List[str]:
        """Add texts directly to the vector store"""
        try:
            if metadatas is None:
                metadatas = [{}] * len(texts)
            
            ids = self.vectorstore.add_texts(texts, metadatas)
            self.vectorstore.persist()
            
            logger.info(f"Added {len(texts)} texts to vector store")
            return ids
            
        except Exception as e:
            logger.error(f"Error adding texts: {str(e)}")
            raise
    
    def similarity_search(self, query: str, k: int = MAX_CHUNKS) -> List[LangChainDocument]:
        """Perform similarity search"""
        try:
            results = self.vectorstore.similarity_search(query, k=k)
            logger.info(f"Found {len(results)} similar documents for query")
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            raise
    
    def similarity_search_with_score(self, query: str, k: int = MAX_CHUNKS) -> List[tuple]:
        """Perform similarity search with scores"""
        try:
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            logger.info(f"Found {len(results)} similar documents with scores")
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search with score: {str(e)}")
            raise
    
    def similarity_search_by_vector(self, embedding: List[float], k: int = MAX_CHUNKS) -> List[LangChainDocument]:
        """Search by vector embedding"""
        try:
            results = self.vectorstore.similarity_search_by_vector(embedding, k=k)
            logger.info(f"Found {len(results)} similar documents by vector")
            return results
            
        except Exception as e:
            logger.error(f"Error in vector similarity search: {str(e)}")
            raise
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        try:
            collection = self.chroma_client.get_collection(self.collection_name)
            count = collection.count()
            
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "embedding_model": EMBEDDING_MODEL if self.use_openai else "all-MiniLM-L6-v2",
                "persist_directory": CHROMA_PERSIST_DIRECTORY
            }
            
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            return {}
    
    def delete_collection(self):
        """Delete the entire collection"""
        try:
            self.chroma_client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
            raise
    
    def reset_collection(self):
        """Reset the collection (delete and recreate)"""
        try:
            self.delete_collection()
            self._initialize_chroma()
            logger.info("Collection reset successfully")
            
        except Exception as e:
            logger.error(f"Error resetting collection: {str(e)}")
            raise
    
    def query_with_filters(self, query: str, filter_dict: Dict[str, Any], k: int = MAX_CHUNKS) -> List[LangChainDocument]:
        """Search with metadata filters"""
        try:
            results = self.vectorstore.similarity_search(
                query, 
                k=k, 
                filter=filter_dict
            )
            logger.info(f"Found {len(results)} filtered documents")
            return results
            
        except Exception as e:
            logger.error(f"Error in filtered search: {str(e)}")
            raise


class VectorStoreManager:
    """High-level manager for vector store operations"""
    
    def __init__(self, collection_name: str = "rag_documents", use_openai: bool = True):
        self.vector_store = VectorStore(collection_name, use_openai)
    
    def ingest_documents(self, documents: List[LangChainDocument]) -> bool:
        """Ingest documents into the vector store"""
        try:
            if not documents:
                logger.warning("No documents to ingest")
                return False
            
            self.vector_store.add_documents(documents)
            logger.info(f"Successfully ingested {len(documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Error ingesting documents: {str(e)}")
            return False
    
    def search(self, query: str, k: int = MAX_CHUNKS, with_scores: bool = False) -> List:
        """Search for relevant documents"""
        try:
            if with_scores:
                return self.vector_store.vectorstore.similarity_search_with_score(query, k)
            else:
                return self.vector_store.vectorstore.similarity_search(query, k)
                
        except Exception as e:
            logger.error(f"Error searching: {str(e)}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        return self.vector_store.get_collection_info()


if __name__ == "__main__":
    # Example usage
    vector_manager = VectorStoreManager()
    
    # Get collection info
    stats = vector_manager.get_stats()
    print(f"Vector store stats: {stats}")
    
    print("Vector store ready!")
