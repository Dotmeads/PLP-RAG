# pet_retrieval/config.py
from __future__ import annotations
import os
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def sget(key: str, default: str | None = None) -> str:
    """Get a value from Streamlit secrets with a helpful error."""
    try:
        v = st.secrets[key]
    except Exception:
        if default is not None:
            return default
        raise KeyError(
            f"Missing '{key}' in .streamlit/secrets.toml. "
            f"Please add it (see docs)."
        )
    if isinstance(v, str):
        return v
    # allow nested dict to be flattened by caller if needed
    return v

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

def get_blob_settings() -> dict:
    conn = sget("AZURE_CONNECTION_STRING")
    ml_container = sget("ML_ARTIFACTS_CONTAINER")
    pets_container = sget("PETS_CONTAINER")

    # allow "ner" or "ner/"
    def _norm_prefix(p: str) -> str:
        p = (p or "").replace("\\", "/").strip("/")
        return p  # no trailing slash; we'll handle both
    ner_prefix = _norm_prefix(sget("NER_PREFIX"))
    mr_prefix  = _norm_prefix(sget("MR_PREFIX"))

    pets_csv_blob = sget("PETS_CSV_BLOB")

    return {
        "connection_string": conn,
        "ml_container": ml_container,
        "pets_container": pets_container,
        "ner_prefix": ner_prefix,
        "mr_prefix": mr_prefix,
        "pets_csv_blob": pets_csv_blob,
    }

# Local cache dirs/files
def _cache_root() -> str:
    root = os.path.join(os.getcwd(), "artifacts")
    os.makedirs(root, exist_ok=True)
    return root

def local_ner_dir() -> str:
    p = os.path.join(_cache_root(), "ner")
    os.makedirs(p, exist_ok=True)
    return p

def local_mr_dir() -> str:
    p = os.path.join(_cache_root(), "mr")
    os.makedirs(p, exist_ok=True)
    return p

def local_pets_csv_path() -> str:
    p = os.path.join(_cache_root(), "pets.csv")
    return p


# --- NER source selection (HF vs Azure/local) ---
def ner_source() -> str:
    import streamlit as st
    try:
        return str(st.secrets.get("NER_SOURCE", "hf")).strip().lower()
    except Exception:
        return "hf"

def ner_hf_repo() -> str:
    import streamlit as st
    try:
        return str(st.secrets.get("NER_HF_REPO", "kerrringuo/pet-adoption-ner")).strip()
    except Exception:
        return "kerrringuo/pet-adoption-ner"
