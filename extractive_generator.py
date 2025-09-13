"""
Extractive Answer Generation System for the proposed RAG configuration
"""
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class ExtractiveAnswer:
    """Extractive answer with citations"""
    answer: str
    citations: List[Dict[str, Any]]
    confidence: float
    sources_used: List[str]
    answer_sentences: List[Dict[str, Any]]

class ExtractiveAnswerGenerator:
    """Generate answers by extracting relevant sentences from retrieved documents"""
    
    def __init__(self, max_answer_length: int = 500, min_confidence: float = 0.3):
        """
        Initialize extractive answer generator
        
        Args:
            max_answer_length: Maximum length of generated answer
            min_confidence: Minimum confidence threshold for answers
        """
        self.max_answer_length = max_answer_length
        self.min_confidence = min_confidence
        self.sentence_splitter = re.compile(r'[.!?]+')
        
        logger.info(f"Extractive answer generator initialized (max_length={max_answer_length})")
    
    def generate_answer(self, query: str, documents: List[Dict[str, Any]]) -> ExtractiveAnswer:
        """
        Generate an extractive answer from retrieved documents
        
        Args:
            query: User query
            documents: Retrieved documents with scores
            
        Returns:
            ExtractiveAnswer object
        """
        try:
            if not documents:
                return self._create_empty_answer()
            
            # Extract relevant sentences
            relevant_sentences = self._extract_relevant_sentences(query, documents)
            
            if not relevant_sentences:
                return self._create_empty_answer()
            
            # Combine sentences into coherent answer
            answer_text = self._combine_sentences(relevant_sentences)
            
            # Generate citations
            citations = self._generate_citations(documents)
            
            # Calculate confidence
            confidence = self._calculate_confidence(documents, relevant_sentences)
            
            # Get sources used
            sources_used = list(set(doc.get('source', 'Unknown') for doc in documents[:3]))
            
            return ExtractiveAnswer(
                answer=answer_text,
                citations=citations,
                confidence=confidence,
                sources_used=sources_used,
                answer_sentences=relevant_sentences
            )
            
        except Exception as e:
            logger.error(f"Error generating extractive answer: {str(e)}")
            return self._create_empty_answer()
    
    def _extract_relevant_sentences(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract relevant sentences from documents"""
        query_words = set(query.lower().split())
        relevant_sentences = []
        
        for doc in documents:
            content = doc.get('content', '')
            doc_score = doc.get('rerank_score', doc.get('score', 0.0))
            source = doc.get('source', 'Unknown')
            
            # Split into sentences
            sentences = self.sentence_splitter.split(content)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 10:  # Skip very short sentences
                    continue
                
                # Calculate sentence relevance
                sentence_words = set(sentence.lower().split())
                overlap = len(query_words.intersection(sentence_words))
                
                if overlap > 0:
                    relevance_score = overlap / len(query_words)
                    
                    # Boost score based on document quality
                    final_score = relevance_score * (0.5 + 0.5 * doc_score)
                    
                    relevant_sentences.append({
                        'text': sentence,
                        'relevance_score': final_score,
                        'source': source,
                        'document_score': doc_score,
                        'overlap_count': overlap
                    })
        
        # Sort by relevance score
        relevant_sentences.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Limit number of sentences
        return relevant_sentences[:5]
    
    def _combine_sentences(self, sentences: List[Dict[str, Any]]) -> str:
        """Combine sentences into a coherent answer"""
        if not sentences:
            return "I couldn't find relevant information to answer your question."
        
        # Simple combination strategy
        answer_parts = []
        current_length = 0
        
        for sent in sentences:
            sentence_text = sent['text']
            
            # Check if adding this sentence would exceed max length
            if current_length + len(sentence_text) > self.max_answer_length:
                break
            
            answer_parts.append(sentence_text)
            current_length += len(sentence_text)
        
        if not answer_parts:
            # If no sentences fit, take the first one and truncate
            first_sentence = sentences[0]['text']
            if len(first_sentence) > self.max_answer_length:
                first_sentence = first_sentence[:self.max_answer_length-3] + "..."
            answer_parts = [first_sentence]
        
        # Join sentences with proper punctuation
        answer = ' '.join(answer_parts)
        
        # Ensure proper ending
        if not answer.endswith(('.', '!', '?')):
            answer += '.'
        
        return answer
    
    def _generate_citations(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate citations for the answer"""
        citations = []
        
        for i, doc in enumerate(documents[:3]):  # Top 3 sources
            citation = {
                'source': doc.get('source', f'Document {i+1}'),
                'relevance_score': doc.get('rerank_score', doc.get('score', 0.0)),
                'content_preview': doc.get('content', '')[:200] + '...',
                'rank': i + 1,
                'document_id': doc.get('document_id', f'doc_{i}'),
                'retrieval_methods': doc.get('retrieval_methods', ['unknown'])
            }
            citations.append(citation)
        
        return citations
    
    def _calculate_confidence(self, documents: List[Dict[str, Any]], 
                            sentences: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for the answer"""
        if not documents or not sentences:
            return 0.0
        
        # Base confidence on top document score
        top_doc_score = documents[0].get('rerank_score', documents[0].get('score', 0.0))
        
        # Normalize cross-encoder scores (they can be negative)
        if top_doc_score < 0:
            # Convert negative logits to positive confidence
            top_doc_score = max(0.0, min(1.0, (top_doc_score + 10) / 20))  # Rough normalization
        
        # Boost confidence if multiple documents support the answer
        doc_diversity = len(set(doc.get('source', '') for doc in documents[:3]))
        diversity_boost = min(0.2, doc_diversity * 0.1)
        
        # Boost confidence based on sentence relevance
        avg_sentence_relevance = np.mean([s['relevance_score'] for s in sentences])
        sentence_boost = avg_sentence_relevance * 0.3
        
        # Calculate final confidence
        confidence = top_doc_score + diversity_boost + sentence_boost
        
        return min(1.0, max(0.0, confidence))
    
    def _create_empty_answer(self) -> ExtractiveAnswer:
        """Create an empty answer when no relevant information is found"""
        return ExtractiveAnswer(
            answer="I couldn't find relevant information to answer your question. Please try rephrasing your query or ask about a different topic.",
            citations=[],
            confidence=0.0,
            sources_used=[],
            answer_sentences=[]
        )
    
    def generate_with_context(self, query: str, documents: List[Dict[str, Any]], 
                            context: Optional[str] = None) -> ExtractiveAnswer:
        """
        Generate answer with additional context
        
        Args:
            query: User query
            documents: Retrieved documents
            context: Additional context to consider
            
        Returns:
            ExtractiveAnswer object
        """
        # If context is provided, add it as an additional document
        if context:
            context_doc = {
                'content': context,
                'score': 0.8,  # High score for context
                'source': 'Additional Context',
                'document_id': 'context'
            }
            documents = [context_doc] + documents
        
        return self.generate_answer(query, documents)
    
    def generate_multiple_answers(self, query: str, documents: List[Dict[str, Any]], 
                                num_answers: int = 3) -> List[ExtractiveAnswer]:
        """
        Generate multiple answer variations
        
        Args:
            query: User query
            documents: Retrieved documents
            num_answers: Number of answer variations to generate
            
        Returns:
            List of ExtractiveAnswer objects
        """
        answers = []
        
        # Generate different answer lengths
        length_variations = [300, 500, 700]
        
        for i in range(min(num_answers, len(length_variations))):
            # Temporarily change max length
            original_length = self.max_answer_length
            self.max_answer_length = length_variations[i]
            
            answer = self.generate_answer(query, documents)
            answers.append(answer)
            
            # Restore original length
            self.max_answer_length = original_length
        
        return answers


class AdvancedExtractiveGenerator(ExtractiveAnswerGenerator):
    """Advanced extractive generator with additional features"""
    
    def __init__(self, max_answer_length: int = 500, min_confidence: float = 0.3):
        super().__init__(max_answer_length, min_confidence)
        self.query_patterns = self._load_query_patterns()
    
    def _load_query_patterns(self) -> Dict[str, str]:
        """Load patterns for different query types"""
        return {
            'what': 'definition',
            'how': 'procedure',
            'why': 'explanation',
            'when': 'temporal',
            'where': 'location',
            'which': 'comparison',
            'who': 'person',
            'should': 'recommendation'
        }
    
    def generate_typed_answer(self, query: str, documents: List[Dict[str, Any]]) -> ExtractiveAnswer:
        """Generate answer based on query type"""
        query_type = self._classify_query(query)
        
        if query_type == 'definition':
            return self._generate_definition_answer(query, documents)
        elif query_type == 'procedure':
            return self._generate_procedure_answer(query, documents)
        elif query_type == 'recommendation':
            return self._generate_recommendation_answer(query, documents)
        else:
            return self.generate_answer(query, documents)
    
    def _classify_query(self, query: str) -> str:
        """Classify query type"""
        query_lower = query.lower()
        
        for pattern, query_type in self.query_patterns.items():
            if pattern in query_lower:
                return query_type
        
        return 'general'
    
    def _generate_definition_answer(self, query: str, documents: List[Dict[str, Any]]) -> ExtractiveAnswer:
        """Generate definition-style answer"""
        # Look for sentences that define or explain concepts
        definition_sentences = []
        
        for doc in documents:
            content = doc.get('content', '')
            sentences = self.sentence_splitter.split(content)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if any(word in sentence.lower() for word in ['is', 'are', 'means', 'refers to', 'defined as']):
                    definition_sentences.append({
                        'text': sentence,
                        'relevance_score': 1.0,
                        'source': doc.get('source', 'Unknown'),
                        'document_score': doc.get('score', 0.0),
                        'overlap_count': 1
                    })
        
        if definition_sentences:
            # Use definition sentences
            answer_text = self._combine_sentences(definition_sentences[:3])
        else:
            # Fall back to regular generation
            return self.generate_answer(query, documents)
        
        citations = self._generate_citations(documents)
        confidence = self._calculate_confidence(documents, definition_sentences)
        sources_used = list(set(doc.get('source', 'Unknown') for doc in documents[:3]))
        
        return ExtractiveAnswer(
            answer=answer_text,
            citations=citations,
            confidence=confidence,
            sources_used=sources_used,
            answer_sentences=definition_sentences
        )
    
    def _generate_procedure_answer(self, query: str, documents: List[Dict[str, Any]]) -> ExtractiveAnswer:
        """Generate procedure-style answer"""
        # Look for step-by-step or procedural information
        procedure_sentences = []
        
        for doc in documents:
            content = doc.get('content', '')
            sentences = self.sentence_splitter.split(content)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if any(word in sentence.lower() for word in ['first', 'then', 'next', 'finally', 'step', 'process']):
                    procedure_sentences.append({
                        'text': sentence,
                        'relevance_score': 1.0,
                        'source': doc.get('source', 'Unknown'),
                        'document_score': doc.get('score', 0.0),
                        'overlap_count': 1
                    })
        
        if procedure_sentences:
            answer_text = self._combine_sentences(procedure_sentences[:5])
        else:
            return self.generate_answer(query, documents)
        
        citations = self._generate_citations(documents)
        confidence = self._calculate_confidence(documents, procedure_sentences)
        sources_used = list(set(doc.get('source', 'Unknown') for doc in documents[:3]))
        
        return ExtractiveAnswer(
            answer=answer_text,
            citations=citations,
            confidence=confidence,
            sources_used=sources_used,
            answer_sentences=procedure_sentences
        )
    
    def _generate_recommendation_answer(self, query: str, documents: List[Dict[str, Any]]) -> ExtractiveAnswer:
        """Generate recommendation-style answer"""
        # Look for recommendation or advice sentences
        recommendation_sentences = []
        
        for doc in documents:
            content = doc.get('content', '')
            sentences = self.sentence_splitter.split(content)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if any(word in sentence.lower() for word in ['should', 'recommend', 'suggest', 'advise', 'best', 'important']):
                    recommendation_sentences.append({
                        'text': sentence,
                        'relevance_score': 1.0,
                        'source': doc.get('source', 'Unknown'),
                        'document_score': doc.get('score', 0.0),
                        'overlap_count': 1
                    })
        
        if recommendation_sentences:
            answer_text = self._combine_sentences(recommendation_sentences[:4])
        else:
            return self.generate_answer(query, documents)
        
        citations = self._generate_citations(documents)
        confidence = self._calculate_confidence(documents, recommendation_sentences)
        sources_used = list(set(doc.get('source', 'Unknown') for doc in documents[:3]))
        
        return ExtractiveAnswer(
            answer=answer_text,
            citations=citations,
            confidence=confidence,
            sources_used=sources_used,
            answer_sentences=recommendation_sentences
        )


if __name__ == "__main__":
    # Test the extractive answer generator
    test_docs = [
        {
            'content': 'Cats are obligate carnivores and need high-protein diets. Quality cat food should have meat as the first ingredient.',
            'score': 0.9,
            'source': 'Pet Care Guide',
            'document_id': '1'
        },
        {
            'content': 'Dogs should visit the veterinarian annually for routine checkups and vaccinations.',
            'score': 0.8,
            'source': 'Veterinary Guide',
            'document_id': '2'
        }
    ]
    
    generator = ExtractiveAnswerGenerator()
    answer = generator.generate_answer("What should I feed my cat?", test_docs)
    
    print("Extractive Answer Generation Test:")
    print(f"Answer: {answer.answer}")
    print(f"Confidence: {answer.confidence:.3f}")
    print(f"Sources: {', '.join(answer.sources_used)}")
    print(f"Citations: {len(answer.citations)}")
