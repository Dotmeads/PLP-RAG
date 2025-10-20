"""
Proposed RAG System: BM25 + Dense + RRF + Cross-encoder + Extractive Generation
"""
import logging
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .bm25_retriever import BM25Retriever, HybridRetriever
from .rrf_fusion import RRFFusion
from .cross_encoder_reranker import CrossEncoderReranker
from .free_llm_generator import FreeLLMGenerator, LLMAnswerResult
from .vector_store import VectorStoreManager
from .document_processor import DocumentProcessor

logger = logging.getLogger(__name__)

@dataclass
class ProposedRAGResult:
    """Result from the proposed RAG system"""
    answer: str
    citations: List[Dict[str, Any]]
    confidence: float
    sources_used: List[str]
    performance_metrics: Dict[str, Any]
    retrieval_breakdown: Dict[str, Any]

class ProposedRAGSystem:
    """Complete proposed RAG system implementation"""
    
    def __init__(self, collection_name: str = "proposed_rag_documents", use_openai: bool = True):
        """
        Initialize the proposed RAG system
        
        Args:
            collection_name: Name for the vector collection
            use_openai: Whether to use OpenAI embeddings
        """
        self.collection_name = collection_name
        self.use_openai = use_openai
        
        # Initialize components
        self.document_processor = DocumentProcessor()
        self.vector_manager = VectorStoreManager(collection_name, use_openai)
        self.bm25_retriever = None
        self.rrf_fusion = RRFFusion(k=60)
        self.reranker = CrossEncoderReranker()
        # Try free LLM providers in order of preference
        try:
            self.answer_generator = FreeLLMGenerator(provider="groq")
            logger.info("Using free LLM generator (Groq)")
        except Exception:
            try:
                self.answer_generator = FreeLLMGenerator(provider="deepseek")
                logger.info("Using free LLM generator (DeepSeek)")
            except Exception:
                # Fallback to basic generation (no LLM)
                self.answer_generator = FreeLLMGenerator(provider="basic")
                logger.info("Using basic answer generation (no LLM)")
        
        # Performance tracking
        self.query_count = 0
        self.performance_history = []
        
        logger.info("Proposed RAG system initialized successfully")
    
    def ingest_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        """
        Ingest documents into the system
        
        Args:
            file_paths: List of file paths to ingest
            
        Returns:
            Ingestion results
        """
        try:
            logger.info(f"Ingesting {len(file_paths)} documents")
            
            # Process documents
            all_documents = []
            all_metadata = []
            
            for file_path in file_paths:
                docs = self.document_processor.process_file(file_path)
                all_documents.extend(docs)
                
                # Extract metadata
                for doc in docs:
                    metadata = doc.metadata.copy()
                    metadata['file_path'] = file_path
                    all_metadata.append(metadata)
            
            # Add to vector store
            self.vector_manager.ingest_documents(all_documents)
            
            # Prepare documents for BM25
            document_texts = [doc.page_content for doc in all_documents]
            
            # Initialize BM25 retriever
            self.bm25_retriever = BM25Retriever(document_texts, all_metadata)
            
            logger.info(f"Successfully ingested {len(all_documents)} documents")
            
            return {
                'success': True,
                'documents_processed': len(all_documents),
                'files_processed': len(file_paths),
                'bm25_indexed': len(document_texts)
            }
            
        except Exception as e:
            logger.error(f"Error ingesting documents: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'documents_processed': 0
            }
    
    def ingest_directory(self, directory_path: str) -> Dict[str, Any]:
        """
        Ingest all documents from a directory
        
        Args:
            directory_path: Path to directory containing documents
            
        Returns:
            Ingestion results
        """
        try:
            # Get all supported files from directory
            import os
            import glob
            
            supported_extensions = ['.pdf', '.txt', '.docx', '.md', '.html']
            file_paths = []
            
            # First, collect all files
            all_files = []
            for ext in supported_extensions:
                pattern = os.path.join(directory_path, f"**/*{ext}")
                all_files.extend(glob.glob(pattern, recursive=True))
            
            # Filter out PDF files if TXT version exists
            file_paths = self._filter_duplicate_files(all_files)
            
            if not file_paths:
                logger.warning(f"No supported documents found in {directory_path}")
                return {
                    'success': False,
                    'error': 'No supported documents found',
                    'documents_processed': 0
                }
            
            return self.ingest_documents(file_paths)
            
        except Exception as e:
            logger.error(f"Error ingesting directory: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'documents_processed': 0
            }
    
    def _filter_duplicate_files(self, file_paths: List[str]) -> List[str]:
        """
        Filter out PDF files if a corresponding TXT file exists with the same name
        
        Args:
            file_paths: List of all file paths found
            
        Returns:
            Filtered list of file paths
        """
        import os
        from pathlib import Path
        
        # Group files by their base name (without extension)
        file_groups = {}
        for file_path in file_paths:
            path_obj = Path(file_path)
            base_name = path_obj.stem  # filename without extension
            if base_name not in file_groups:
                file_groups[base_name] = []
            file_groups[base_name].append(file_path)
        
        filtered_files = []
        skipped_files = []
        
        for base_name, files in file_groups.items():
            # Check if we have both PDF and TXT versions
            pdf_files = [f for f in files if f.lower().endswith('.pdf')]
            txt_files = [f for f in files if f.lower().endswith('.txt')]
            
            if pdf_files and txt_files:
                # Prefer TXT files over PDF files
                filtered_files.extend(txt_files)
                skipped_files.extend(pdf_files)
                logger.info(f"Skipping PDF files {pdf_files} in favor of TXT files {txt_files}")
            else:
                # No conflict, include all files
                filtered_files.extend(files)
        
        if skipped_files:
            logger.info(f"Filtered out {len(skipped_files)} duplicate PDF files")
            logger.info(f"Processing {len(filtered_files)} files after deduplication")
        
        return filtered_files
    
    def query(self, question: str, use_reranking: bool = True, 
              rerank_threshold: float = 0.1, max_rerank: int = 20) -> ProposedRAGResult:
        """
        Process a query through the complete proposed RAG pipeline
        
        Args:
            question: User question
            use_reranking: Whether to use cross-encoder reranking
            rerank_threshold: Minimum score threshold for reranking
            max_rerank: Maximum number of documents to rerank
            
        Returns:
            ProposedRAGResult with answer and metadata
        """
        start_time = time.time()
        self.query_count += 1
        
        try:
            logger.info(f"Processing query #{self.query_count}: {question[:100]}...")
            
            # Step 1: Hybrid Retrieval (BM25 + Dense + RRF)
            retrieval_start = time.time()
            bm25_results, dense_results = self._hybrid_retrieval(question)
            retrieval_time = (time.time() - retrieval_start) * 1000
            
            # Step 2: RRF Fusion
            fusion_start = time.time()
            fused_results = self._rrf_fusion(bm25_results, dense_results)
            fusion_time = (time.time() - fusion_start) * 1000
            
            # Step 3: Cross-encoder Reranking (optional)
            rerank_start = time.time()
            if use_reranking and fused_results:
                reranked_results = self._rerank_documents(question, fused_results, 
                                                        rerank_threshold, max_rerank)
            else:
                reranked_results = fused_results[:5]  # Take top 5 without reranking
            rerank_time = (time.time() - rerank_start) * 1000
            
            # Step 4: Extractive Answer Generation
            generation_start = time.time()
            answer_result = self._generate_answer(question, reranked_results)
            generation_time = (time.time() - generation_start) * 1000
            
            total_time = (time.time() - start_time) * 1000
            
            # Compile performance metrics
            performance_metrics = {
                'total_time_ms': total_time,
                'retrieval_time_ms': retrieval_time,
                'fusion_time_ms': fusion_time,
                'rerank_time_ms': rerank_time,
                'generation_time_ms': generation_time,
                'query_count': self.query_count
            }
            
            # Compile retrieval breakdown
            retrieval_breakdown = {
                'bm25_results': len(bm25_results),
                'dense_results': len(dense_results),
                'fused_results': len(fused_results),
                'reranked_results': len(reranked_results),
                'use_reranking': use_reranking
            }
            
            # Create result
            result = ProposedRAGResult(
                answer=answer_result.answer,
                citations=answer_result.citations,
                confidence=answer_result.confidence,
                sources_used=answer_result.sources_used,
                performance_metrics=performance_metrics,
                retrieval_breakdown=retrieval_breakdown
            )
            
            # Store performance history
            self.performance_history.append({
                'query': question,
                'metrics': performance_metrics,
                'confidence': answer_result.confidence,
                'timestamp': time.time()
            })
            
            logger.info(f"Query processed successfully in {total_time:.1f}ms (confidence: {answer_result.confidence:.3f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return self._create_error_result(str(e))
    
    def _hybrid_retrieval(self, question: str) -> tuple:
        """Perform hybrid retrieval using BM25 and dense search"""
        try:
            # BM25 retrieval
            bm25_results = self.bm25_retriever.search(question, k=20) if self.bm25_retriever else []
            
            # Dense retrieval
            dense_results = self.vector_manager.vector_store.vectorstore.similarity_search(question, k=20)
            
            # Convert dense results to our format
            dense_formatted = []
            for i, doc in enumerate(dense_results):
                dense_formatted.append({
                    'document_id': f"dense_{i}",
                    'content': doc.page_content,
                    'score': 1.0,
                    'source': doc.metadata.get('source', f'Dense Document {i}'),
                    'metadata': doc.metadata,
                    'retrieval_method': 'dense'
                })
            
            return bm25_results, dense_formatted
            
        except Exception as e:
            logger.error(f"Error in hybrid retrieval: {str(e)}")
            return [], []
    
    def _rrf_fusion(self, bm25_results: List[Dict], dense_results: List[Dict]) -> List[Dict]:
        """Fuse BM25 and dense results using RRF"""
        try:
            if not bm25_results and not dense_results:
                return []
            
            # Use RRF fusion
            fused_results = self.rrf_fusion.fuse_bm25_dense(
                bm25_results, dense_results, 
                bm25_weight=0.5, dense_weight=0.5
            )
            
            return fused_results
            
        except Exception as e:
            logger.error(f"Error in RRF fusion: {str(e)}")
            return bm25_results + dense_results  # Fallback to simple concatenation
    
    def _rerank_documents(self, question: str, documents: List[Dict], 
                         threshold: float, max_docs: int) -> List[Dict]:
        """Rerank documents using cross-encoder"""
        try:
            if not documents:
                return []
            
            # Limit documents for reranking
            docs_to_rerank = documents[:max_docs]
            
            # Rerank with threshold
            reranked = self.reranker.rerank_with_threshold(
                question, docs_to_rerank, threshold, top_k=5
            )
            
            return reranked
            
        except Exception as e:
            logger.error(f"Error in reranking: {str(e)}")
            return documents[:5]  # Fallback to top 5
    
    def _generate_answer(self, question: str, documents: List[Dict]) -> LLMAnswerResult:
        """Generate hybrid answer from documents"""
        try:
            return self.answer_generator.generate_answer(question, documents)
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return self.answer_generator._create_empty_answer()
    
    def _create_error_result(self, error_message: str) -> ProposedRAGResult:
        """Create error result"""
        return ProposedRAGResult(
            answer=f"I encountered an error while processing your question: {error_message}",
            citations=[],
            confidence=0.0,
            sources_used=[],
            performance_metrics={'error': error_message},
            retrieval_breakdown={'error': error_message}
        )
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        try:
            vector_stats = self.vector_manager.get_stats()
            bm25_count = self.bm25_retriever.get_document_count() if self.bm25_retriever else 0
            
            return {
                'vector_store': vector_stats,
                'bm25_documents': bm25_count,
                'total_queries': self.query_count,
                'avg_confidence': self._calculate_avg_confidence(),
                'avg_response_time': self._calculate_avg_response_time()
            }
        except Exception as e:
            logger.error(f"Error getting system stats: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_avg_confidence(self) -> float:
        """Calculate average confidence from recent queries"""
        if not self.performance_history:
            return 0.0
        
        recent_queries = self.performance_history[-10:]  # Last 10 queries
        confidences = [q['confidence'] for q in recent_queries]
        return sum(confidences) / len(confidences)
    
    def _calculate_avg_response_time(self) -> float:
        """Calculate average response time from recent queries"""
        if not self.performance_history:
            return 0.0
        
        recent_queries = self.performance_history[-10:]  # Last 10 queries
        times = [q['metrics']['total_time_ms'] for q in recent_queries]
        return sum(times) / len(times)
    
    def reset_system(self):
        """Reset the system"""
        try:
            # Reset vector store
            self.vector_manager = VectorStoreManager(self.collection_name, self.use_openai)
            
            # Reset BM25
            self.bm25_retriever = None
            
            # Reset performance tracking
            self.query_count = 0
            self.performance_history = []
            
            logger.info("System reset successfully")
            
        except Exception as e:
            logger.error(f"Error resetting system: {str(e)}")


class ProposedRAGManager:
    """High-level manager for the proposed RAG system"""
    
    def __init__(self, collection_name: str = "proposed_rag_documents", use_openai: bool = True):
        self.system = ProposedRAGSystem(collection_name, use_openai)
    
    def add_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        """Add documents to the system"""
        return self.system.ingest_documents(file_paths)
    
    def add_directory(self, directory_path: str) -> Dict[str, Any]:
        """Add all documents from a directory"""
        return self.system.ingest_directory(directory_path)
    
    def ask(self, question: str, **kwargs) -> Dict[str, Any]:
        """Ask a question to the system"""
        result = self.system.query(question, **kwargs)
        
        return {
            'answer': result.answer,
            'citations': result.citations,
            'confidence': result.confidence,
            'sources': result.sources_used,
            'performance': result.performance_metrics,
            'retrieval_info': result.retrieval_breakdown
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        return self.system.get_system_stats()
    
    def reset(self):
        """Reset the system"""
        self.system.reset_system()


if __name__ == "__main__":
    # Test the proposed RAG system
    system = ProposedRAGManager()
    
    # Add documents
    result = system.add_directory("documents")
    print(f"Documents added: {result}")
    
    # Test query
    response = system.ask("What should I feed my cat?")
    print(f"Answer: {response['answer']}")
    print(f"Confidence: {response['confidence']:.3f}")
    print(f"Sources: {', '.join(response['sources'])}")
