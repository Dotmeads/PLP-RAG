"""
Cross-encoder Reranker for the proposed RAG system
"""
import logging
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

class CrossEncoderReranker:
    """Cross-encoder based reranker for improving retrieval precision"""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize cross-encoder reranker
        
        Args:
            model_name: Name of the cross-encoder model to use
        """
        self.model_name = model_name
        self.model = None
        self._load_model()
        
        logger.info(f"Cross-encoder reranker initialized with model: {model_name}")
    
    def _load_model(self):
        """Load the cross-encoder model"""
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name)
            logger.info("Cross-encoder model loaded successfully")
        except ImportError:
            logger.warning("sentence-transformers not available, using mock reranker")
            self.model = None
        except Exception as e:
            logger.error(f"Error loading cross-encoder model: {str(e)}")
            self.model = None
    
    def rerank(self, query: str, documents: List[Dict[str, Any]], 
               top_k: int = 5, batch_size: int = 32) -> List[Dict[str, Any]]:
        """
        Rerank documents using cross-encoder
        
        Args:
            query: Search query
            documents: List of documents to rerank
            top_k: Number of top documents to return
            batch_size: Batch size for processing
            
        Returns:
            Reranked documents with new scores
        """
        try:
            if not documents:
                return []
            
            if self.model is None:
                # Use mock reranking if model not available
                return self._mock_rerank(query, documents, top_k)
            
            # Prepare query-document pairs
            pairs = [(query, doc.get('content', '')) for doc in documents]
            
            # Get relevance scores in batches
            scores = []
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i + batch_size]
                batch_scores = self.model.predict(batch_pairs)
                scores.extend(batch_scores.tolist())
            
            # Combine documents with new scores
            reranked_docs = []
            for doc, score in zip(documents, scores):
                reranked_doc = doc.copy()
                reranked_doc['rerank_score'] = float(score)
                reranked_doc['original_score'] = doc.get('score', 0.0)
                reranked_docs.append(reranked_doc)
            
            # Sort by rerank score
            reranked_docs.sort(key=lambda x: x['rerank_score'], reverse=True)
            
            logger.info(f"Reranked {len(documents)} documents, returning top {min(top_k, len(reranked_docs))}")
            return reranked_docs[:top_k]
            
        except Exception as e:
            logger.error(f"Error in cross-encoder reranking: {str(e)}")
            return documents[:top_k]  # Return original order if error
    
    def _mock_rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """Mock reranking when cross-encoder is not available"""
        logger.info("Using mock reranking (cross-encoder not available)")
        
        query_words = set(query.lower().split())
        reranked_docs = []
        
        for doc in documents:
            content = doc.get('content', '').lower()
            content_words = set(content.split())
            
            # Simple overlap-based scoring
            overlap = len(query_words.intersection(content_words))
            max_words = max(len(query_words), len(content_words))
            
            if max_words > 0:
                score = overlap / max_words
            else:
                score = 0.0
            
            # Add some randomness to simulate neural model behavior
            noise = np.random.normal(0, 0.05)
            final_score = max(0.0, min(1.0, score + noise))
            
            reranked_doc = doc.copy()
            reranked_doc['rerank_score'] = final_score
            reranked_doc['original_score'] = doc.get('score', 0.0)
            reranked_docs.append(reranked_doc)
        
        # Sort by rerank score
        reranked_docs.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        return reranked_docs[:top_k]
    
    def rerank_with_threshold(self, query: str, documents: List[Dict[str, Any]], 
                            threshold: float = 0.5, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Rerank documents and filter by threshold
        
        Args:
            query: Search query
            documents: List of documents to rerank
            threshold: Minimum rerank score threshold (for logits, use negative values)
            top_k: Maximum number of documents to return
            
        Returns:
            Reranked documents above threshold
        """
        reranked = self.rerank(query, documents, top_k=len(documents))
        
        # For cross-encoder logits, we need to use a different threshold approach
        # Since logits can be negative, we'll use a percentile-based approach
        if reranked:
            scores = [doc.get('rerank_score', -float('inf')) for doc in reranked]
            # Use top percentile instead of absolute threshold
            if threshold > 0:
                # Convert threshold to percentile (0.1 = top 10%)
                percentile_threshold = max(scores) - (max(scores) - min(scores)) * (1 - threshold)
                filtered = [doc for doc in reranked if doc.get('rerank_score', -float('inf')) >= percentile_threshold]
            else:
                # For negative thresholds, use as absolute threshold
                filtered = [doc for doc in reranked if doc.get('rerank_score', -float('inf')) >= threshold]
        else:
            filtered = []
        
        logger.info(f"Filtered {len(reranked)} documents to {len(filtered)} above threshold {threshold}")
        return filtered[:top_k]
    
    def batch_rerank(self, queries: List[str], document_sets: List[List[Dict[str, Any]]], 
                    top_k: int = 5) -> List[List[Dict[str, Any]]]:
        """
        Rerank multiple query-document sets in batch
        
        Args:
            queries: List of queries
            document_sets: List of document sets (one per query)
            top_k: Number of top documents to return per query
            
        Returns:
            List of reranked document sets
        """
        results = []
        
        for query, documents in zip(queries, document_sets):
            reranked = self.rerank(query, documents, top_k)
            results.append(reranked)
        
        return results


class AdaptiveReranker(CrossEncoderReranker):
    """Adaptive reranker that adjusts based on query characteristics"""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        super().__init__(model_name)
        self.query_stats = {}
    
    def rerank_adaptive(self, query: str, documents: List[Dict[str, Any]], 
                       top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Adaptive reranking based on query characteristics
        """
        # Analyze query characteristics
        query_length = len(query.split())
        has_question_words = any(word in query.lower() for word in ['what', 'how', 'why', 'when', 'where'])
        
        # Adjust parameters based on query characteristics
        if query_length <= 3:
            # Short queries: be more lenient
            threshold = 0.3
        elif has_question_words:
            # Question queries: focus on precision
            threshold = 0.6
        else:
            # Regular queries: balanced approach
            threshold = 0.5
        
        # Use threshold-based reranking
        return self.rerank_with_threshold(query, documents, threshold, top_k)
    
    def update_query_stats(self, query: str, results: List[Dict[str, Any]], 
                          user_feedback: Optional[Dict[str, Any]] = None):
        """Update statistics for query optimization"""
        if query not in self.query_stats:
            self.query_stats[query] = {
                'count': 0,
                'avg_rerank_score': 0.0,
                'feedback': []
            }
        
        stats = self.query_stats[query]
        stats['count'] += 1
        
        if results:
            avg_score = sum(doc.get('rerank_score', 0) for doc in results) / len(results)
            stats['avg_rerank_score'] = (stats['avg_rerank_score'] * (stats['count'] - 1) + avg_score) / stats['count']
        
        if user_feedback:
            stats['feedback'].append(user_feedback)


if __name__ == "__main__":
    # Test the cross-encoder reranker
    test_docs = [
        {'document_id': '1', 'content': 'Cats are obligate carnivores and need high-protein diets', 'score': 0.8},
        {'document_id': '2', 'content': 'Dogs should visit the veterinarian annually for checkups', 'score': 0.7},
        {'document_id': '3', 'content': 'Quality pet food should have meat as the first ingredient', 'score': 0.6},
    ]
    
    reranker = CrossEncoderReranker()
    reranked = reranker.rerank("What should I feed my cat?", test_docs, top_k=2)
    
    print("Cross-encoder Reranking Test Results:")
    for doc in reranked:
        print(f"Rerank Score: {doc['rerank_score']:.3f} - {doc['content'][:50]}...")
