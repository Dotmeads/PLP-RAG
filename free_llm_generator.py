#!/usr/bin/env python3
"""
Free LLM Answer Generator - Supports multiple free LLM providers
"""
import logging
import requests
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Import API keys
try:
    from api_keys import DEEPSEEK_API_KEY, GROQ_API_KEY, HUGGINGFACE_API_KEY
except ImportError:
    DEEPSEEK_API_KEY = None
    GROQ_API_KEY = None
    HUGGINGFACE_API_KEY = None

logger = logging.getLogger(__name__)

@dataclass
class LLMAnswerResult:
    """Result from LLM answer generation"""
    answer: str
    confidence: float
    sources_used: List[Dict[str, Any]]
    generation_method: str
    citations: List[str]

class FreeLLMGenerator:
    """Free LLM answer generator supporting multiple providers"""
    
    def __init__(self, provider: str = "deepseek"):
        self.provider = provider.lower()
        self.api_key = self._get_api_key()
        
        if not self.api_key:
            logger.warning(f"No API key found for {provider}, using basic generation")
            self.provider = "basic"
        
        logger.info(f"Free LLM generator initialized with provider: {self.provider}")
    
    def _get_api_key(self) -> Optional[str]:
        """Get API key for the selected provider"""
        if self.provider == "deepseek":
            return DEEPSEEK_API_KEY
        elif self.provider == "groq":
            return GROQ_API_KEY
        elif self.provider == "huggingface":
            return HUGGINGFACE_API_KEY
        return None
    
    def generate_answer(self, question: str, documents: List[Dict[str, Any]]) -> LLMAnswerResult:
        """Generate answer using free LLM"""
        try:
            if not documents:
                return self._create_empty_answer()
            
            # Extract context from documents
            context = self._extract_context(documents)
            
            if not context:
                return self._create_empty_answer()
            
            # Generate answer based on provider
            if self.provider == "deepseek":
                return self._generate_with_deepseek(question, context, documents)
            elif self.provider == "groq":
                return self._generate_with_groq(question, context, documents)
            elif self.provider == "huggingface":
                return self._generate_with_huggingface(question, context, documents)
            else:
                return self._generate_basic(question, context, documents)
                
        except Exception as e:
            logger.error(f"Error in LLM generation: {str(e)}")
            return self._create_empty_answer()
    
    def _extract_context(self, documents: List[Dict[str, Any]]) -> str:
        """Extract relevant context from documents"""
        context_parts = []
        
        for doc in documents[:3]:  # Use top 3 documents
            content = doc.get('content', '')
            source = doc.get('source', 'Unknown')
            
            # Clean content
            content = self._clean_content(content)
            
            if content and len(content) > 50:
                context_parts.append(f"Source: {source}\nContent: {content[:500]}...")
        
        return "\n\n".join(context_parts)
    
    def _generate_with_deepseek(self, question: str, context: str, documents: List[Dict[str, Any]]) -> LLMAnswerResult:
        """Generate answer using DeepSeek API"""
        try:
            # Try the correct DeepSeek API endpoint
            url = "https://api.deepseek.com/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            prompt = self._create_prompt(question, context)
            
            data = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "You are a helpful veterinary assistant. Provide clear, accurate, and professional advice about pet care based on the given context."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 400,
                "temperature": 0.3,
                "stream": False
            }
            
            response = requests.post(url, headers=headers, json=data, timeout=30)
            
            # Check if it's a payment error
            if response.status_code == 402:
                logger.warning("DeepSeek API requires payment, falling back to basic generation")
                return self._generate_basic(question, context, documents)
            
            response.raise_for_status()
            
            result = response.json()
            answer = result['choices'][0]['message']['content'].strip()
            
            return LLMAnswerResult(
                answer=answer,
                confidence=0.9,
                sources_used=[{'source': doc.get('source', 'Unknown'), 'content': doc.get('content', '')[:100]} for doc in documents[:3]],
                generation_method="deepseek",
                citations=[doc.get('source', 'Unknown') for doc in documents[:3]]
            )
            
        except Exception as e:
            logger.error(f"DeepSeek API error: {e}")
            return self._generate_basic(question, context, documents)
    
    def _generate_with_groq(self, question: str, context: str, documents: List[Dict[str, Any]]) -> LLMAnswerResult:
        """Generate answer using Groq API"""
        try:
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            prompt = self._create_prompt(question, context)
            
            data = {
                "model": "llama-3.1-8b-instant",
                "messages": [
                    {"role": "system", "content": "You are a helpful veterinary assistant. Provide clear, accurate, and professional advice about pet care based on the given context."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 400,
                "temperature": 0.3
            }
            
            response = requests.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            answer = result['choices'][0]['message']['content'].strip()
            
            return LLMAnswerResult(
                answer=answer,
                confidence=0.9,
                sources_used=[{'source': doc.get('source', 'Unknown'), 'content': doc.get('content', '')[:100]} for doc in documents[:3]],
                generation_method="groq",
                citations=[doc.get('source', 'Unknown') for doc in documents[:3]]
            )
            
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            return self._generate_basic(question, context, documents)
    
    def _generate_with_huggingface(self, question: str, context: str, documents: List[Dict[str, Any]]) -> LLMAnswerResult:
        """Generate answer using Hugging Face API"""
        try:
            url = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            prompt = f"Question: {question}\nContext: {context}\nAnswer:"
            
            data = {
                "inputs": prompt,
                "parameters": {
                    "max_length": 200,
                    "temperature": 0.7
                }
            }
            
            response = requests.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            answer = result[0]['generated_text'].replace(prompt, "").strip()
            
            return LLMAnswerResult(
                answer=answer,
                confidence=0.8,
                sources_used=[{'source': doc.get('source', 'Unknown'), 'content': doc.get('content', '')[:100]} for doc in documents[:3]],
                generation_method="huggingface",
                citations=[doc.get('source', 'Unknown') for doc in documents[:3]]
            )
            
        except Exception as e:
            logger.error(f"Hugging Face API error: {e}")
            return self._generate_basic(question, context, documents)
    
    def _generate_basic(self, question: str, context: str, documents: List[Dict[str, Any]]) -> LLMAnswerResult:
        """Generate basic answer without LLM"""
        # Extract key information from context
        answer_parts = []
        
        # Add contextual introduction
        if "feed" in question.lower() or "food" in question.lower():
            answer_parts.append("Based on veterinary guidelines and nutritional research:")
        elif "healthy" in question.lower() or "signs" in question.lower():
            answer_parts.append("Here are the key indicators of pet health:")
        elif "train" in question.lower() or "training" in question.lower():
            answer_parts.append("Here's what you need to know about pet training:")
        else:
            answer_parts.append("Based on the available information:")
        
        # Extract relevant sentences from context
        if context:
            # Split context into sentences and pick the most relevant ones
            sentences = context.split('.')
            relevant_sentences = []
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 20 and any(word in sentence.lower() for word in question.lower().split()):
                    relevant_sentences.append(sentence)
            
            # Add up to 3 most relevant sentences
            for sentence in relevant_sentences[:3]:
                if sentence:
                    answer_parts.append(f"• {sentence}.")
        
        # If no relevant context found, provide general advice
        if len(answer_parts) == 1:
            if "feed" in question.lower() or "food" in question.lower():
                answer_parts.append("• Feed your pet a balanced, high-quality commercial diet appropriate for their age and health status.")
                answer_parts.append("• Avoid feeding human foods that are toxic to pets, such as chocolate, onions, grapes, and raisins.")
                answer_parts.append("• Consult with your veterinarian for specific dietary recommendations.")
            elif "healthy" in question.lower() or "signs" in question.lower():
                answer_parts.append("• Look for bright, clear eyes and clean ears without discharge.")
                answer_parts.append("• Check for a shiny, healthy coat and normal energy levels.")
                answer_parts.append("• Monitor eating habits, bowel movements, and overall behavior.")
            else:
                answer_parts.append("• Regular veterinary check-ups are essential for your pet's health.")
                answer_parts.append("• Provide proper nutrition, exercise, and mental stimulation.")
                answer_parts.append("• Watch for any changes in behavior or physical condition.")
        
        # Add professional note
        answer_parts.append("\nNote: Always consult with your veterinarian for personalized advice regarding your pet's specific needs.")
        
        answer = "\n".join(answer_parts)
        
        return LLMAnswerResult(
            answer=answer,
            confidence=0.7,
            sources_used=[{'source': doc.get('source', 'Unknown'), 'content': doc.get('content', '')[:100]} for doc in documents[:3]],
            generation_method="basic",
            citations=[doc.get('source', 'Unknown') for doc in documents[:3]]
        )
    
    def _create_prompt(self, question: str, context: str) -> str:
        """Create prompt for LLM"""
        return f"""Question: {question}

Context from veterinary sources:
{context}

Please provide a clear, comprehensive answer based on the context above. Structure your response professionally and include specific recommendations where appropriate. Keep the answer concise but informative."""
    
    def _clean_content(self, content: str) -> str:
        """Clean content by removing formatting artifacts"""
        import re
        
        # Remove excessive whitespace and newlines
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
        
        # Fix common formatting issues
        content = re.sub(r'com\s*\n\s*\d+\s*\n\s*[A-Z]+\s*\n', '', content)
        content = re.sub(r'^\s*\d+\s*$', '', content, flags=re.MULTILINE)
        content = re.sub(r'^[A-Z]{2,}\s*$', '', content, flags=re.MULTILINE)
        
        # Remove single character lines
        content = re.sub(r'^\s*[a-zA-Z]\s*$', '', content, flags=re.MULTILINE)
        
        # Clean up remaining whitespace
        content = re.sub(r'\s+', ' ', content)
        
        return content.strip()
    
    def _create_empty_answer(self) -> LLMAnswerResult:
        """Create empty answer result"""
        return LLMAnswerResult(
            answer="I couldn't find relevant information to answer your question. Please try rephrasing your query or ask about a different topic.",
            confidence=0.0,
            sources_used=[],
            generation_method="none",
            citations=[]
        )
