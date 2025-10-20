"""
Reciprocal Rank Fusion (RRF) for combining multiple retrieval results
"""
import logging
from typing import List, Dict, Any, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)

class RRFFusion:
    """Reciprocal Rank Fusion for combining multiple retrieval systems"""
    
    def __init__(self, k: int = 60):
        """
        Initialize RRF fusion
        
        Args:
            k: RRF parameter (typically 60)
        """
        self.k = k
        logger.info(f"RRF fusion initialized with k={k}")
    
    def fuse_results(self, result_sets: List[List[Dict[str, Any]]], 
                    weights: Optional[List[float]] = None) -> List[Dict[str, Any]]:
        """
        Fuse multiple result sets using Reciprocal Rank Fusion
        
        Args:
            result_sets: List of result sets from different retrievers
            weights: Optional weights for each result set
            
        Returns:
            Fused and ranked results
        """
        try:
            if not result_sets:
                return []
            
            # Default weights if not provided
            if weights is None:
                weights = [1.0] * len(result_sets)
            
            # Ensure weights sum to 1
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
            
            # Collect all unique documents with their RRF scores
            document_scores = defaultdict(lambda: {
                'scores': [],
                'ranks': [],
                'methods': [],
                'content': '',
                'metadata': {},
                'source': ''
            })
            
            # Process each result set
            for set_idx, (results, weight) in enumerate(zip(result_sets, weights)):
                for rank, result in enumerate(results):
                    # Create a unique key for the document
                    doc_key = self._create_document_key(result)
                    
                    # Calculate RRF score
                    rrf_score = weight / (self.k + rank + 1)
                    
                    # Store document information
                    if not document_scores[doc_key]['content']:
                        document_scores[doc_key].update({
                            'content': result.get('content', ''),
                            'metadata': result.get('metadata', {}),
                            'source': result.get('source', ''),
                            'document_id': result.get('document_id', doc_key)
                        })
                    
                    document_scores[doc_key]['scores'].append(rrf_score)
                    document_scores[doc_key]['ranks'].append(rank)
                    document_scores[doc_key]['methods'].append(result.get('retrieval_method', f'set_{set_idx}'))
            
            # Calculate final scores and create fused results
            fused_results = []
            for doc_key, doc_info in document_scores.items():
                final_score = sum(doc_info['scores'])
                avg_rank = sum(doc_info['ranks']) / len(doc_info['ranks'])
                
                fused_result = {
                    'document_id': doc_info['document_id'],
                    'content': doc_info['content'],
                    'score': final_score,
                    'rrf_score': final_score,
                    'avg_rank': avg_rank,
                    'source': doc_info['source'],
                    'metadata': doc_info['metadata'],
                    'retrieval_methods': doc_info['methods'],
                    'num_retrievers': len(doc_info['scores']),
                    'individual_scores': doc_info['scores']
                }
                
                fused_results.append(fused_result)
            
            # Sort by final RRF score
            fused_results.sort(key=lambda x: x['rrf_score'], reverse=True)
            
            logger.info(f"RRF fusion combined {len(result_sets)} result sets into {len(fused_results)} unique documents")
            return fused_results
            
        except Exception as e:
            logger.error(f"Error in RRF fusion: {str(e)}")
            return []
    
    def _create_document_key(self, result: Dict[str, Any]) -> str:
        """Create a unique key for a document"""
        # Try to use document_id first
        if 'document_id' in result:
            return str(result['document_id'])
        
        # Use content hash as fallback
        content = result.get('content', '')
        return str(hash(content[:200]))  # Use first 200 chars for hash
    
    def fuse_bm25_dense(self, bm25_results: List[Dict[str, Any]], 
                       dense_results: List[Dict[str, Any]],
                       bm25_weight: float = 0.5, dense_weight: float = 0.5) -> List[Dict[str, Any]]:
        """
        Convenience method for fusing BM25 and dense results
        
        Args:
            bm25_results: Results from BM25 retriever
            dense_results: Results from dense retriever
            bm25_weight: Weight for BM25 results
            dense_weight: Weight for dense results
            
        Returns:
            Fused results
        """
        return self.fuse_results([bm25_results, dense_results], [bm25_weight, dense_weight])


class AdvancedRRFFusion(RRFFusion):
    """Advanced RRF with additional features"""
    
    def __init__(self, k: int = 60, normalize_scores: bool = True):
        """
        Initialize advanced RRF fusion
        
        Args:
            k: RRF parameter
            normalize_scores: Whether to normalize scores before fusion
        """
        super().__init__(k)
        self.normalize_scores = normalize_scores
    
    def fuse_results_with_normalization(self, result_sets: List[List[Dict[str, Any]]], 
                                      weights: Optional[List[float]] = None) -> List[Dict[str, Any]]:
        """
        Fuse results with score normalization
        """
        if not result_sets:
            return []
        
        # Normalize scores if requested
        if self.normalize_scores:
            result_sets = self._normalize_scores(result_sets)
        
        return self.fuse_results(result_sets, weights)
    
    def _normalize_scores(self, result_sets: List[List[Dict[str, Any]]]) -> List[List[Dict[str, Any]]]:
        """Normalize scores within each result set"""
        normalized_sets = []
        
        for results in result_sets:
            if not results:
                normalized_sets.append([])
                continue
            
            # Get all scores
            scores = [r.get('score', 0) for r in results]
            
            if not scores or max(scores) == min(scores):
                # No variation in scores, keep as is
                normalized_sets.append(results)
                continue
            
            # Normalize to 0-1 range
            min_score, max_score = min(scores), max(scores)
            score_range = max_score - min_score
            
            normalized_results = []
            for result in results:
                normalized_result = result.copy()
                original_score = result.get('score', 0)
                normalized_score = (original_score - min_score) / score_range
                normalized_result['score'] = normalized_score
                normalized_result['original_score'] = original_score
                normalized_results.append(normalized_result)
            
            normalized_sets.append(normalized_results)
        
        return normalized_sets
    
    def fuse_with_confidence(self, result_sets: List[List[Dict[str, Any]]], 
                           confidence_scores: List[float]) -> List[Dict[str, Any]]:
        """
        Fuse results using confidence scores as weights
        """
        # Normalize confidence scores
        total_confidence = sum(confidence_scores)
        if total_confidence > 0:
            weights = [c / total_confidence for c in confidence_scores]
        else:
            weights = [1.0 / len(confidence_scores)] * len(confidence_scores)
        
        return self.fuse_results(result_sets, weights)


if __name__ == "__main__":
    # Test RRF fusion
    bm25_results = [
        {'document_id': '1', 'content': 'Cats need protein', 'score': 0.8, 'source': 'Pet Guide'},
        {'document_id': '2', 'content': 'Dogs need exercise', 'score': 0.6, 'source': 'Pet Guide'},
    ]
    
    dense_results = [
        {'document_id': '1', 'content': 'Cats need protein', 'score': 0.7, 'source': 'Pet Guide'},
        {'document_id': '3', 'content': 'Fish need clean water', 'score': 0.5, 'source': 'Pet Guide'},
    ]
    
    rrf = RRFFusion()
    fused = rrf.fuse_bm25_dense(bm25_results, dense_results)
    
    print("RRF Fusion Test Results:")
    for result in fused:
        print(f"Score: {result['rrf_score']:.3f} - {result['content']}")
