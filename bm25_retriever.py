"""
BM25 Retrieval System for the proposed RAG configuration
"""
import logging
import nltk
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
import re

# Download required NLTK data with SSL fix
try:
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    logger.warning(f"Could not download NLTK data: {e}")
    pass

logger = logging.getLogger(__name__)

class BM25Retriever:
    """BM25-based retrieval system for keyword matching"""
    
    def __init__(self, documents: List[str], document_metadata: List[Dict[str, Any]] = None):
        """
        Initialize BM25 retriever
        
        Args:
            documents: List of document texts
            document_metadata: List of metadata for each document
        """
        self.documents = documents
        self.document_metadata = document_metadata or [{}] * len(documents)
        self.tokenized_docs = self._tokenize_documents(documents)
        self.bm25 = BM25Okapi(self.tokenized_docs)
        
        logger.info(f"BM25 retriever initialized with {len(documents)} documents")
    
    def _tokenize_documents(self, documents: List[str]) -> List[List[str]]:
        """Tokenize documents for BM25 indexing"""
        try:
            from nltk.corpus import stopwords
            from nltk.tokenize import word_tokenize
            
            stop_words = set(stopwords.words('english'))
            tokenized = []
            
            for doc in documents:
                # Clean and tokenize
                doc_clean = re.sub(r'[^\w\s]', ' ', doc.lower())
                tokens = word_tokenize(doc_clean)
                
                # Remove stopwords and short tokens
                tokens = [token for token in tokens 
                         if token not in stop_words and len(token) > 2]
                
                tokenized.append(tokens)
            
            logger.info("Using NLTK tokenization with stopwords")
            return tokenized
            
        except Exception as e:
            logger.warning(f"Error in NLTK tokenization, using simple tokenization: {e}")
            # Fallback to simple tokenization
            tokenized = []
            for doc in documents:
                tokens = re.findall(r'\b\w+\b', doc.lower())
                # Simple stopword removal
                common_stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
                tokens = [token for token in tokens if len(token) > 2 and token not in common_stopwords]
                tokenized.append(tokens)
            
            logger.info("Using simple tokenization fallback")
            return tokenized
    
    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for relevant documents using BM25
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of search results with scores
        """
        try:
            # Tokenize query
            query_tokens = self._tokenize_documents([query])[0]
            
            if not query_tokens:
                logger.warning("Empty query tokens after tokenization")
                return []
            
            # Get BM25 scores
            scores = self.bm25.get_scores(query_tokens)
            
            # Get top k results (include all results, not just positive scores)
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
            
            results = []
            for idx in top_indices:
                # Include all results, even with zero scores
                result = {
                    'document_id': idx,
                    'content': self.documents[idx],
                    'score': float(scores[idx]),
                    'source': self.document_metadata[idx].get('source', f'Document {idx}'),
                    'metadata': self.document_metadata[idx],
                    'retrieval_method': 'bm25'
                }
                results.append(result)
            
            logger.info(f"BM25 search returned {len(results)} results for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error in BM25 search: {str(e)}")
            return []
    
    def get_document_count(self) -> int:
        """Get the number of indexed documents"""
        return len(self.documents)
    
    def update_documents(self, new_documents: List[str], new_metadata: List[Dict[str, Any]] = None):
        """Update the document index with new documents"""
        self.documents.extend(new_documents)
        self.document_metadata.extend(new_metadata or [{}] * len(new_documents))
        self.tokenized_docs.extend(self._tokenize_documents(new_documents))
        self.bm25 = BM25Okapi(self.tokenized_docs)
        
        logger.info(f"BM25 index updated with {len(new_documents)} new documents")


class HybridRetriever:
    """Combines BM25 and dense vector retrieval"""
    
    def __init__(self, dense_retriever, documents: List[str], document_metadata: List[Dict[str, Any]] = None):
        """
        Initialize hybrid retriever
        
        Args:
            dense_retriever: Dense vector retriever (your existing system)
            documents: List of document texts
            document_metadata: List of metadata for each document
        """
        self.dense_retriever = dense_retriever
        self.bm25_retriever = BM25Retriever(documents, document_metadata)
        
        logger.info("Hybrid retriever initialized with BM25 and dense components")
    
    def search(self, query: str, k: int = 20, bm25_weight: float = 0.5, dense_weight: float = 0.5) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining BM25 and dense retrieval
        
        Args:
            query: Search query
            k: Number of results to return
            bm25_weight: Weight for BM25 results
            dense_weight: Weight for dense results
            
        Returns:
            List of combined search results
        """
        try:
            # Get results from both retrievers
            bm25_results = self.bm25_retriever.search(query, k)
            dense_results = self.dense_retriever.similarity_search(query, k)
            
            # Convert dense results to our format
            dense_formatted = []
            for i, doc in enumerate(dense_results):
                dense_formatted.append({
                    'document_id': f"dense_{i}",
                    'content': doc.page_content,
                    'score': 1.0,  # Dense retriever doesn't provide scores
                    'source': doc.metadata.get('source', f'Dense Document {i}'),
                    'metadata': doc.metadata,
                    'retrieval_method': 'dense'
                })
            
            # Combine results using simple score combination
            combined_results = self._combine_results(bm25_results, dense_formatted, bm25_weight, dense_weight)
            
            # Sort by combined score and return top k
            combined_results.sort(key=lambda x: x['combined_score'], reverse=True)
            
            logger.info(f"Hybrid search returned {len(combined_results)} results")
            return combined_results[:k]
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {str(e)}")
            return []
    
    def _combine_results(self, bm25_results: List[Dict], dense_results: List[Dict], 
                        bm25_weight: float, dense_weight: float) -> List[Dict]:
        """Combine BM25 and dense results"""
        # Create a mapping of content to results
        content_map = {}
        
        # Add BM25 results
        for result in bm25_results:
            content_key = result['content'][:100]  # Use first 100 chars as key
            content_map[content_key] = {
                **result,
                'bm25_score': result['score'],
                'dense_score': 0.0,
                'combined_score': result['score'] * bm25_weight
            }
        
        # Add dense results
        for result in dense_results:
            content_key = result['content'][:100]
            if content_key in content_map:
                # Update existing result
                content_map[content_key]['dense_score'] = result['score']
                content_map[content_key]['combined_score'] += result['score'] * dense_weight
            else:
                # Add new result
                content_map[content_key] = {
                    **result,
                    'bm25_score': 0.0,
                    'dense_score': result['score'],
                    'combined_score': result['score'] * dense_weight
                }
        
        return list(content_map.values())


if __name__ == "__main__":
    # Test the BM25 retriever
    test_docs = [
        "Cats are obligate carnivores and need high-protein diets.",
        "Dogs should visit the veterinarian annually for checkups.",
        "Quality pet food should have meat as the first ingredient.",
        "Regular exercise is important for pet health and well-being."
    ]
    
    retriever = BM25Retriever(test_docs)
    results = retriever.search("What should I feed my cat?", k=3)
    
    print("BM25 Test Results:")
    for result in results:
        print(f"Score: {result['score']:.3f} - {result['content'][:50]}...")
