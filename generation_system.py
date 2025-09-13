"""
Generation system for creating answers using LLMs
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

# LangChain components
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain.memory import ConversationBufferMemory

from config import (
    OPENAI_API_KEY, 
    DEFAULT_MODEL, 
    TEMPERATURE, 
    MAX_TOKENS
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Container for generation results"""
    answer: str
    sources: List[Dict[str, Any]]
    query: str
    context_used: str
    model_used: str
    tokens_used: Optional[int] = None


class GenerationSystem:
    """Handles answer generation using LLMs"""
    
    def __init__(self, model_name: str = DEFAULT_MODEL, use_chat: bool = True):
        self.model_name = model_name
        self.use_chat = use_chat
        
        if not OPENAI_API_KEY:
            raise ValueError("OpenAI API key is required for generation")
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            model_name=model_name,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
        
        # Initialize memory for conversation
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Create prompt templates
        self._create_prompts()
        
        logger.info(f"Generation system initialized with model: {model_name}")
    
    def _create_prompts(self):
        """Create prompt templates for different use cases"""
        
        # Main RAG prompt
        self.rag_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a helpful AI assistant that answers questions based on the provided context.

Context:
{context}

Question: {question}

Instructions:
1. Answer the question based ONLY on the information provided in the context above.
2. If the context doesn't contain enough information to answer the question, say so clearly.
3. Be concise but comprehensive in your answer.
4. If you reference specific information, mention which document or source it came from.
5. Do not make up information that isn't in the context.

Answer:"""
        )
        
        # Chat prompt for conversational RAG
        self.chat_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a helpful AI assistant that answers questions based on provided context. 
            Always base your answers on the given context and cite sources when possible. 
            If the context doesn't contain enough information, say so clearly."""),
            HumanMessage(content="Context: {context}\n\nQuestion: {question}")
        ])
        
        # Summarization prompt
        self.summarize_prompt = PromptTemplate(
            input_variables=["text"],
            template="""Please provide a concise summary of the following text:

{text}

Summary:"""
        )
        
        # Question generation prompt
        self.question_prompt = PromptTemplate(
            input_variables=["context"],
            template="""Based on the following context, generate 3 relevant questions that could be asked:

Context:
{context}

Questions:"""
        )
    
    def generate_answer(self, query: str, context: str, 
                       sources: List[Dict[str, Any]] = None) -> GenerationResult:
        """Generate an answer based on query and context"""
        try:
            logger.info(f"Generating answer for query: {query[:100]}...")
            
            # Use chat model
            messages = self.chat_prompt.format_messages(
                context=context,
                question=query
            )
            response = self.llm(messages)
            answer = response.content
            
            result = GenerationResult(
                answer=answer,
                sources=sources or [],
                query=query,
                context_used=context,
                model_used=self.model_name
            )
            
            logger.info("Answer generated successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return GenerationResult(
                answer="I apologize, but I encountered an error while generating an answer.",
                sources=[],
                query=query,
                context_used=context,
                model_used=self.model_name
            )
    
    def generate_with_memory(self, query: str, context: str, 
                           sources: List[Dict[str, Any]] = None) -> GenerationResult:
        """Generate answer with conversation memory"""
        try:
            # Add current context to memory
            self.memory.chat_memory.add_user_message(f"Context: {context}")
            self.memory.chat_memory.add_user_message(f"Question: {query}")
            
            # Get conversation history
            chat_history = self.memory.chat_memory.messages
            
            # Create messages for chat model
            messages = [
                SystemMessage(content="""You are a helpful AI assistant that answers questions based on provided context. 
                You have access to conversation history. Always base your answers on the given context and cite sources when possible."""),
                *chat_history[-10:],  # Last 10 messages for context
                HumanMessage(content=f"Please answer: {query}")
            ]
            
            response = self.llm(messages)
            answer = response.content
            
            # Add AI response to memory
            self.memory.chat_memory.add_ai_message(answer)
            
            result = GenerationResult(
                answer=answer,
                sources=sources or [],
                query=query,
                context_used=context,
                model_used=self.model_name
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating answer with memory: {str(e)}")
            return self.generate_answer(query, context, sources)
    
    def summarize_text(self, text: str) -> str:
        """Summarize a given text"""
        try:
            prompt = self.summarize_prompt.format(text=text)
            response = self.llm.invoke(prompt)
            return response.content
            
        except Exception as e:
            logger.error(f"Error summarizing text: {str(e)}")
            return "Error generating summary."
    
    def generate_questions(self, context: str) -> List[str]:
        """Generate relevant questions based on context"""
        try:
            prompt = self.question_prompt.format(context=context)
            response = self.llm.invoke(prompt)
            
            # Parse questions from response
            questions = []
            for line in response.content.split('\n'):
                line = line.strip()
                if line and (line.startswith(('1.', '2.', '3.', '-', '•')) or '?' in line):
                    # Clean up the question
                    question = line.split('.', 1)[-1].strip() if '.' in line else line
                    question = question.lstrip('-• ').strip()
                    if question.endswith('?'):
                        questions.append(question)
            
            return questions[:3]  # Return up to 3 questions
            
        except Exception as e:
            logger.error(f"Error generating questions: {str(e)}")
            return []
    
    def clear_memory(self):
        """Clear conversation memory"""
        self.memory.clear()
        logger.info("Conversation memory cleared")
    
    def get_memory_summary(self) -> str:
        """Get a summary of the conversation memory"""
        try:
            messages = self.memory.chat_memory.messages
            if not messages:
                return "No conversation history."
            
            # Create a simple summary
            user_messages = [msg.content for msg in messages if isinstance(msg, HumanMessage)]
            ai_messages = [msg.content for msg in messages if isinstance(msg, AIMessage)]
            
            return f"Conversation has {len(user_messages)} user messages and {len(ai_messages)} AI responses."
            
        except Exception as e:
            logger.error(f"Error getting memory summary: {str(e)}")
            return "Error retrieving conversation history."


class GenerationManager:
    """High-level manager for generation operations"""
    
    def __init__(self, model_name: str = DEFAULT_MODEL, use_chat: bool = True):
        self.generation_system = GenerationSystem(model_name, use_chat)
    
    def answer_question(self, query: str, context: str, 
                       sources: List[Dict[str, Any]] = None,
                       use_memory: bool = False) -> GenerationResult:
        """Answer a question with optional memory"""
        try:
            if use_memory:
                return self.generation_system.generate_with_memory(query, context, sources)
            else:
                return self.generation_system.generate_answer(query, context, sources)
                
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return GenerationResult(
                answer="I apologize, but I encountered an error while processing your question.",
                sources=[],
                query=query,
                context_used=context,
                model_used=self.generation_system.model_name
            )
    
    def chat_with_context(self, query: str, context: str, 
                         sources: List[Dict[str, Any]] = None) -> GenerationResult:
        """Chat with context and memory"""
        return self.answer_question(query, context, sources, use_memory=True)
    
    def get_suggested_questions(self, context: str) -> List[str]:
        """Get suggested questions based on context"""
        return self.generation_system.generate_questions(context)
    
    def summarize_context(self, context: str) -> str:
        """Summarize the provided context"""
        return self.generation_system.summarize_text(context)
    
    def reset_conversation(self):
        """Reset the conversation memory"""
        self.generation_system.clear_memory()


if __name__ == "__main__":
    # Example usage
    if OPENAI_API_KEY:
        generation_manager = GenerationManager()
        
        # Example generation
        # result = generation_manager.answer_question(
        #     "What is machine learning?",
        #     "Machine learning is a subset of artificial intelligence..."
        # )
        # print(f"Answer: {result.answer}")
        
        print("Generation system ready!")
    else:
        print("OpenAI API key not found. Please set OPENAI_API_KEY in your environment.")
