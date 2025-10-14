"""
Configuration settings for the RAG system
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Vector Database Settings
CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")

# Document Processing Settings
CHUNK_SIZE = 2000  # Increased to keep related content together
CHUNK_OVERLAP = 300  # Increased overlap for better context
MAX_CHUNKS = 5

# LLM Settings
DEFAULT_MODEL = "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-ada-002"
TEMPERATURE = 0.7
MAX_TOKENS = 1000

# Application Settings
APP_NAME = os.getenv("APP_NAME", "PLP RAG System")
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# Supported file types
SUPPORTED_EXTENSIONS = ['.pdf', '.txt', '.docx', '.md', '.html']
