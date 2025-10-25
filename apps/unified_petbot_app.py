# -*- coding: utf-8 -*-
"""
Pawfect Match - Single Chat Interface with Intent Classification
Automatically routes pet care questions to RAG and pet adoption queries to Azure search
"""
import streamlit as st
st.set_page_config(page_title="Pawfect Match", layout="wide")

import os, time, ast, re, json
import sys
from typing import List, Dict, Any, Tuple, Optional

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import pandas as pd
import numpy as np

# Import your existing RAG components
from rag_system.proposed_rag_system import ProposedRAGManager
from chatbot_flow.chatbot_pipeline import ChatbotPipeline
from chatbot_flow.intent_classifier import IntentClassifier
from chatbot_flow.entity_extractor import EntityExtractor

# Import Azure components
from pet_retrieval.config import get_blob_settings, local_ner_dir, local_mr_dir, local_pets_csv_path
from pet_retrieval.azure_io import download_prefix_flat, smart_download_single_blob
from pet_retrieval.models import load_ner_pipeline, load_mr_model, load_faiss_index
from pet_retrieval.retrieval import (
    only_text, BM25,
    parse_facets_from_text, entity_spans_to_facets, sanitize_facets_ner_light,
    filter_with_relaxation, make_boosted_query,
    emb_search, mmr_rerank
)
from pet_retrieval.ui import sidebar_controls

# Optional fuzzy breed mapping
try:
    from rapidfuzz import process, fuzz
    _HAS_FUZZ = True
except Exception:
    _HAS_FUZZ = False
    process = fuzz = None

# -------------------------------------------
# CONFIGURATION
# -------------------------------------------
RELAX_ORDER = ["colors_any", "state", "gender"]
MIN_CAND_FLOOR_BASE = 300

# -------------------------------------------
# UTILITY FUNCTIONS
# -------------------------------------------
def safe_merge(a, b):
    """Safely merge two lists, handling None values"""
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    return list(set(a + b))

def resolve_overlaps_longest(spans):
    """Resolve overlapping entity spans by keeping the longest one"""
    if not spans:
        return spans
    
    # Sort by start position, then by length (descending)
    sorted_spans = sorted(spans, key=lambda x: (x['start'], -(x['end'] - x['start'])))
    
    resolved = []
    for span in sorted_spans:
        # Check if this span overlaps with any already resolved span
        overlaps = False
        for resolved_span in resolved:
            if (span['start'] < resolved_span['end'] and span['end'] > resolved_span['start']):
                overlaps = True
                break
        
        if not overlaps:
            resolved.append(span)
    
    return resolved

# -------------------------------------------
# BOOTSTRAP FUNCTIONS
# -------------------------------------------
@st.cache_resource
def bootstrap_rag_system():
    """Initialize RAG system and chatbot pipeline"""
    try:
        # Initialize RAG system with free embeddings (no OpenAI required)
        rag = ProposedRAGManager(use_openai=False)
        
        # Load documents
        documents_dir = os.path.join(project_root, "documents")
        if os.path.exists(documents_dir):
            result = rag.add_directory(documents_dir)
            if not result.get('success', False):
                st.warning(f"⚠️ Document loading had issues: {result.get('error', 'Unknown error')}")
        else:
            st.warning(f"⚠️ Documents directory not found: {documents_dir}")
            # Try to create it
            try:
                os.makedirs(documents_dir, exist_ok=True)
            except Exception as e:
                st.error(f"❌ Could not create documents directory: {e}")
        
        # Initialize chatbot pipeline with RAG (Azure components will be added later)
        chatbot = ChatbotPipeline(rag)
        
        return rag, chatbot
    except Exception as e:
        st.error(f"❌ Failed to initialize RAG system: {str(e)}")
        return None, None

@st.cache_resource
def bootstrap_azure_components():
    """Initialize Azure pet search components"""
    try:
        # Get Azure settings
        settings = get_blob_settings()
        
        # Download models and data (only if not already cached)
        ner_dir = local_ner_dir()
        mr_dir = local_mr_dir()
        pets_csv = local_pets_csv_path()
        
        # Check if files already exist to avoid re-downloading
        if not os.path.exists(ner_dir) or len(os.listdir(ner_dir)) == 0:
            print("📥 Downloading NER models...")
            download_prefix_flat(settings["connection_string"], settings["ml_container"], "ner", ner_dir)
        else:
            print("✅ NER models already cached")
            
        if not os.path.exists(mr_dir) or len(os.listdir(mr_dir)) == 0:
            print("📥 Downloading MR models...")
            download_prefix_flat(settings["connection_string"], settings["ml_container"], "mr", mr_dir)
        else:
            print("✅ MR models already cached")
            
        if not os.path.exists(pets_csv):
            print("📥 Downloading pet data...")
            smart_download_single_blob(settings["connection_string"], settings["pets_container"], "all_pet_details_clean.csv", pets_csv)
        else:
            print("✅ Pet data already cached")
        
        # Load models
        print("🔄 Loading NER pipeline...")
        ner = load_ner_pipeline(local_ner_dir())
        print("✅ NER pipeline loaded")
        
        print("🔄 Loading sentence transformer model (437MB - this may take 1-2 minutes)...")
        student, doc_ids, doc_vecs = load_mr_model(local_mr_dir())
        print("✅ Sentence transformer model loaded")
        
        print("🔄 Getting model dimensions...")
        model_dim = student.get_sentence_embedding_dimension()
        print("🔄 Loading FAISS index...")
        faiss_index = load_faiss_index(local_mr_dir(), model_dim)
        print("✅ FAISS index loaded")
        
        # Load pet data
        dfp = pd.read_csv(local_pets_csv_path())
        
        # Initialize BM25
        bm25 = BM25()
        doc_map = {i: text for i, text in enumerate(dfp["description_clean"].fillna("").tolist())}
        bm25.fit(doc_map)
        
        # Create breed catalog
        breed_catalog = dfp["breed"].dropna().unique().tolist()
        breed_to_animal = dfp.groupby("breed")["animal"].first().to_dict()
        
        return ner, student, doc_ids, doc_vecs, faiss_index, dfp, bm25, breed_catalog, breed_to_animal
        
    except Exception as e:
        st.error(f"Failed to initialize Azure components: {str(e)}")
        return None, None, None, None, None, None, None, None, None

# -------------------------------------------
# PET SEARCH FUNCTIONS
# -------------------------------------------
def perform_pet_search(query, azure_components, topk=12):
    """Perform pet search using Azure components - OPTIMIZED with caching"""
    if azure_components[0] is None:
        return None, "Pet search not available"
    
    ner, student, doc_ids, doc_vecs, faiss_index, dfp, bm25, breed_catalog, breed_to_animal = azure_components
    
    try:
        # Process query - limit NER processing for speed
        query_short = query[:300] if len(query) > 300 else query
        raw_spans = ner([query_short])[0] if query else []
        spans = resolve_overlaps_longest(raw_spans)
        mf = entity_spans_to_facets(spans)
        rf = parse_facets_from_text(query)
        
        # Age facet parsing
        age_floor_mo, age_ceil_mo = None, None
        tq = only_text(query)
        
        # Check for "younger than X years" or "less than X years"
        younger_than = re.search(r"(younger than|less than|under)\s+([1-9][0-9]?)\s*y(ear)?s?", tq, re.IGNORECASE)
        older_than = re.search(r"(older than|more than|over)\s+([1-9][0-9]?)\s*y(ear)?s?", tq, re.IGNORECASE)
        
        # Check for exact age patterns
        m_age_mo = re.search(r"\b([1-9][0-9]?)\s*mo(nth)?s?\b", tq)
        m_age_yr = re.search(r"\b([1-9][0-9]?)\s*y(ear)?s?\b", tq)
        
        if "puppy" in tq or "kitten" in tq:
            age_floor_mo, age_ceil_mo = 0, 12
        elif younger_than:
            val = int(younger_than.group(2)); age_ceil_mo = 12*val
        elif older_than:
            val = int(older_than.group(2)); age_floor_mo = 12*val
        elif m_age_mo:
            val = int(m_age_mo.group(1)); age_floor_mo, age_ceil_mo = max(0, val-3), val+3
        elif m_age_yr:
            val = int(m_age_yr.group(1)); mo = 12*val; age_floor_mo, age_ceil_mo = max(0, mo-6), mo+6
        
        # Handle PET_TYPE extraction (cats -> Cat, dogs -> Dog)
        pet_type = None
        if mf.get("animal"):
            pet_type = mf.get("animal")
        elif rf.get("animal"):
            pet_type = rf.get("animal")
        elif "cats" in query.lower() or "cat" in query.lower():
            pet_type = ["Cat"]
        elif "dogs" in query.lower() or "dog" in query.lower():
            pet_type = ["Dog"]
        
        # Ensure pet_type is a list
        if isinstance(pet_type, str):
            pet_type = [pet_type]
        
        # Handle state case sensitivity and abbreviations
        ner_state = mf.get("state")
        text_state = rf.get("state")
        
        # Convert to lists if they're strings
        if isinstance(ner_state, str):
            ner_state = [ner_state]
        if isinstance(text_state, str):
            text_state = [text_state]
            
        state_list = safe_merge(ner_state, text_state)
        
        # Also check for KL abbreviation in the query
        if not state_list and "kl" in query.lower():
            state_list = ["Kuala Lumpur"]
        
        if state_list:
            # Convert to proper case for database matching
            state_map = {
                "kuala lumpur": "Kuala Lumpur",
                "kl": "Kuala Lumpur", 
                "selangor": "Selangor",
                "johor": "Johor",
                "penang": "Penang",
                "perak": "Perak"
            }
            state_list = [state_map.get(state.lower(), state.title()) for state in state_list]
        
        facets = {
            "animal": pet_type,
            "breed":  safe_merge(mf.get("breed"),  rf.get("breed")),
            "gender": safe_merge(mf.get("gender"), rf.get("gender")),
            "colors_any": safe_merge(mf.get("colors_any"), rf.get("colors_any")),
            "state": state_list,
            "furlength": safe_merge(mf.get("furlength"), rf.get("furlength")),
            "age_floor_mo": age_floor_mo,
            "age_ceil_mo": age_ceil_mo
        }
        
        
        # Apply filters
        df_filtered = dfp.copy()
        for k, v in facets.items():
            if v is None or (isinstance(v, list) and len(v) == 0):
                continue
            if k == "age_floor_mo":
                df_filtered = df_filtered[df_filtered["age_months"] >= v]
            elif k == "age_ceil_mo":
                df_filtered = df_filtered[df_filtered["age_months"] <= v]
            elif k == "colors_any":
                df_filtered = df_filtered[df_filtered["color"].str.contains("|".join(v), case=False, na=False)]
            elif k == "breed":
                if _HAS_FUZZ:
                    # Fuzzy breed matching
                    breed_matches = []
                    for breed in v:
                        matches = process.extract(breed, breed_catalog, limit=3, scorer=fuzz.ratio)
                        breed_matches.extend([match[0] for match in matches if match[1] > 70])
                    if breed_matches:
                        df_filtered = df_filtered[df_filtered["breed"].isin(breed_matches)]
                else:
                    df_filtered = df_filtered[df_filtered["breed"].str.contains("|".join(v), case=False, na=False)]
            else:
                # Handle case-insensitive filtering for specific fields
                if k == "animal":
                    # Convert to proper case for database matching
                    animal_map = {"dog": "Dog", "cat": "Cat", "puppy": "Dog", "kitten": "Cat"}
                    proper_case_animals = [animal_map.get(animal.lower(), animal) for animal in v]
                    df_filtered = df_filtered[df_filtered[k].isin(proper_case_animals)]
                elif k == "gender":
                    # Convert to proper case for database matching
                    gender_map = {"male": "Male", "female": "Female", "mixed": "Mixed"}
                    proper_case_genders = [gender_map.get(gender.lower(), gender) for gender in v]
                    df_filtered = df_filtered[df_filtered[k].isin(proper_case_genders)]
                elif k == "breed":
                    # Use case-insensitive string matching for breeds
                    df_filtered = df_filtered[df_filtered[k].str.contains("|".join(v), case=False, na=False)]
                else:
                    df_filtered = df_filtered[df_filtered[k].str.contains("|".join(v), case=False, na=False)]
        
        if len(df_filtered) == 0:
            return None, "No pets found matching your criteria. Try relaxing your search terms."
        
        # Hybrid search: BM25 + embeddings
        bm25_results = bm25.search(only_text(query), topk=topk)
        bm25_scores = {idx: score for idx, score in bm25_results}
        emb_results = emb_search(query, student, doc_ids, doc_vecs, pool_topn=topk, faiss_index=faiss_index)
        emb_scores = {idx: score for idx, score in emb_results}
        
        # Combine scores
        combined_scores = {}
        for idx, score in bm25_scores.items():
            combined_scores[idx] = score * 0.3
        for idx, score in emb_scores.items():
            combined_scores[idx] = combined_scores.get(idx, 0) + score * 0.7
        
        # Sort by combined score
        sorted_indices = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in sorted_indices[:topk]]
        
        # Filter indices to only include those that exist in the filtered DataFrame
        valid_indices = [idx for idx in top_indices if idx in df_filtered.index]
        
        # Get results - use valid indices if available, otherwise take first few rows
        if len(valid_indices) > 0:
            results = df_filtered.loc[valid_indices]
        else:
            # Fallback: take first few rows from filtered DataFrame
            results = df_filtered.head(topk)
        
        return results, None
        
    except Exception as e:
        return None, f"Search error: {str(e)}"

def _safe_list_from_cell(x):
    """Parse strings like "['a','b']" or '["a","b"]' or comma strings into list."""
    if isinstance(x, list): return x
    if x is None: return []
    s = str(x).strip()
    if not s: return []
    if s.startswith("[") and s.endswith("]"):
        try:
            import json
            obj = json.loads(s)
            if isinstance(obj, list): return obj
        except Exception:
            pass
        try:
            import ast
            obj = ast.literal_eval(s)
            if isinstance(obj, list): return obj
        except Exception:
            return []
    if "," in s: return [t.strip() for t in s.split(",") if t.strip()]
    return [s]

def _first_photo_url(row) -> str:
    """Get the first photo URL from photo_links"""
    photos = row.get("photo_links")
    if not isinstance(photos, list):
        photos = _safe_list_from_cell(photos)
    if photos:
        url = str(photos[0]).strip().strip('"').strip("'")
        return url if url else None
    return None

def _age_years_from_months(age_months) -> str:
    """Convert age in months to years/months format"""
    try:
        m = float(age_months)
        y = m/12.0
        if m < 12: 
            return f"{int(round(m))} mo"
        return f"{y:.1f} yrs"
    except Exception:
        return "—"

def _badge_bool(x, label):
    """Create a badge for boolean values"""
    v = str(x or "").strip().lower()
    if v in {"true","yes","y","1"}: return f"✅ {label}"
    if v in {"false","no","n","0"}: return f"❌ {label}"
    if v in {"unknown", "nan", ""}: return f"➖ {label}"
    return f"ℹ️ {label}: {x}"

def _comma_join_listlike(x):
    """Join list-like values with commas"""
    if isinstance(x, list):
        return ", ".join([str(t) for t in x if str(t).strip()]) or "—"
    if isinstance(x, str) and x.strip():
        return x
    return "—"

def render_pet_card(row: pd.Series):
    """Render a single pet card"""
    pid = int(row.get("pet_id", 0))
    name = str(row.get("name") or f"Pet {pid}")
    url = str(row.get("url") or "")
    animal = (row.get("animal") or "").title()
    breed = str(row.get("breed") or "—")
    gender = (row.get("gender") or "—").title()
    state = (row.get("state") or "—").title()
    color = str(row.get("color") or "—")
    age_mo = row.get("age_months")
    age_yrs_txt = _age_years_from_months(age_mo)
    size = str(row.get("size") or "—").title()
    fur = str(row.get("fur_length") or "—").title()
    cond = str(row.get("condition") or "—").title()
    vacc = _badge_bool(row.get("vaccinated"), "vaccinated")
    dewm = _badge_bool(row.get("dewormed"), "dewormed")
    neut = _badge_bool(row.get("neutered"), "neutered")
    spay = _badge_bool(row.get("spayed"), "spayed")

    # Pet name with link
    if url: 
        st.markdown(f"### [{name}]({url})")
    else:   
        st.markdown(f"### {name}")

    # Photo
    img_url = _first_photo_url(row)
    if img_url:
        try:
            st.image(img_url, width=300)
        except Exception:
            st.markdown(f"![photo]({img_url})")
    else:
        st.info("No photo available.")

    # Basic info
    st.write(
        f"**{animal}** • **Breed:** {breed} • **Gender:** {gender} • "
        f"**Age:** {age_yrs_txt} • **State:** {state}"
    )
    st.write(
        f"**Color(s):** {color} • **Size:** {size} • **Fur:** {fur} • **Condition:** {cond}"
    )
    
    # Status badges
    st.markdown(" | ".join([vacc, dewm, neut, spay]))

    # Description
    desc = str(row.get("description_clean") or "").strip()
    if desc:
        excerpt = desc if len(desc) < 300 else (desc[:300].rsplit(" ", 1)[0] + "…")
        with st.expander("Description", expanded=False):
            st.write(excerpt)

def display_pet_results(results, error_msg=None):
    """Display pet search results in 2-column card format"""
    if error_msg:
        st.markdown(f'<div class="status-card status-error">❌ {error_msg}</div>', unsafe_allow_html=True)
        return
    
    if results is None or len(results) == 0:
        st.markdown('<div class="status-card status-warning">⚠️ No pets found matching your criteria.</div>', unsafe_allow_html=True)
        return
    
    st.markdown(f'<div class="status-card">🎉 Found {len(results)} pets matching your criteria!</div>', unsafe_allow_html=True)
    
    # Display pets in 2-column grid
    rows = [r for _, r in results.iterrows()]
    n = len(rows)
    
    for i in range(0, n, 2):  # Process 2 pets at a time
        cols = st.columns(2, gap="medium")
        for j, col in enumerate(cols):
            idx = i + j
            if idx >= n: 
                continue
            with col:
                with st.container(border=True):
                    render_pet_card(rows[idx])

# -------------------------------------------
# MAIN APP
# -------------------------------------------
def main():
    # Add enhanced styling
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #ffb6c1 0%, #ffc0cb 50%, #ffd1dc 100%);
        background-attachment: fixed;
    }
    
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(45deg, #ff6b9d, #ff8fab);
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(255, 107, 157, 0.3);
    }
    
    .main-header h1 {
        color: white;
        font-size: 3rem;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: white;
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    .chat-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
    }
    
    .pet-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 5px solid #ff6b9d;
        transition: transform 0.3s ease;
    }
    
    .pet-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .status-card {
        background: linear-gradient(45deg, #4CAF50, #45a049);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        text-align: center;
    }
    
    .status-error {
        background: linear-gradient(45deg, #f44336, #d32f2f);
    }
    
    .status-warning {
        background: linear-gradient(45deg, #ff9800, #f57c00);
    }
    
    .chat-message {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .user-message {
        background: linear-gradient(45deg, #ff6b9d, #ff8fab);
        color: white;
        margin-left: 2rem;
    }
    
    .assistant-message {
        background: linear-gradient(45deg, #e3f2fd, #f3e5f5);
        margin-right: 2rem;
    }
    
    .status-bar {
        background: linear-gradient(135deg, #ff6b9d, #ff8fab);
        color: white;
        padding: 1rem 2rem;
        margin: -1rem -1rem 2rem -1rem;
        border-radius: 0 0 20px 20px;
        box-shadow: 0 4px 20px rgba(255, 107, 157, 0.3);
    }
    
    .status-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        align-items: center;
    }
    
    .status-item {
        background: rgba(255, 255, 255, 0.2);
        padding: 0.8rem;
        border-radius: 10px;
        text-align: center;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    .status-item.success {
        background: rgba(76, 175, 80, 0.3);
        border-color: rgba(76, 175, 80, 0.5);
    }
    
    .status-item.warning {
        background: rgba(255, 152, 0, 0.3);
        border-color: rgba(255, 152, 0, 0.5);
    }
    
    .status-item.error {
        background: rgba(244, 67, 54, 0.3);
        border-color: rgba(244, 67, 54, 0.5);
    }
    
    .status-icon {
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
        display: block;
    }
    
    .status-text {
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    
    .status-detail {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Enhanced header
    st.markdown("""
    <div class="main-header">
        <h1>🐾 Pawfect Match</h1>
        <p>Your Intelligent Pet Assistant - Ask about pet care or find your perfect pet match</p>
    </div>
    """, unsafe_allow_html=True)

    # Pre-load all components at startup
    with st.spinner("🚀 Initializing Pawfect Match systems..."):
        # Load RAG system
        rag, chatbot = bootstrap_rag_system()
        
        # Load Azure components
        azure_components = bootstrap_azure_components()
        
        # Add Azure components to chatbot if both systems loaded successfully
        if chatbot is not None and azure_components[0] is not None:
            chatbot.azure_components = azure_components
            chatbot.pet_search_func = perform_pet_search

    # Status bar at the top
    def get_status_class(status_type):
        if status_type == "success":
            return "success"
        elif status_type == "warning":
            return "warning"
        elif status_type == "error":
            return "error"
        return ""

    def get_status_icon(status_type):
        if status_type == "success":
            return "✅"
        elif status_type == "warning":
            return "⚠️"
        elif status_type == "error":
            return "❌"
        return "ℹ️"

    # RAG System Status
    rag_status = "success" if rag is not None else "error"
    rag_icon = get_status_icon(rag_status)
    rag_text = "RAG System Ready" if rag is not None else "RAG System Failed"
    rag_detail = "941 documents loaded" if rag is not None else "Check configuration"

    # Azure System Status
    azure_status = "success" if azure_components[0] is not None else "warning"
    azure_icon = get_status_icon(azure_status)
    azure_text = "Pet Search Ready" if azure_components[0] is not None else "Pet Search Limited"
    azure_detail = f"{len(azure_components[5])} pets available" if azure_components[0] is not None and azure_components[5] is not None else "Azure not configured"

    # Overall System Status
    overall_status = "success" if rag is not None else "error"
    overall_icon = get_status_icon(overall_status)
    overall_text = "All Systems Ready" if rag is not None else "System Issues"
    overall_detail = "Ready to help!" if rag is not None else "Please check status"

    st.markdown(f"""
    <div class="status-bar">
        <div class="status-grid">
            <div class="status-item {get_status_class(overall_status)}">
                <span class="status-icon">{overall_icon}</span>
                <div class="status-text">{overall_text}</div>
                <div class="status-detail">{overall_detail}</div>
            </div>
            <div class="status-item {get_status_class(rag_status)}">
                <span class="status-icon">{rag_icon}</span>
                <div class="status-text">{rag_text}</div>
                <div class="status-detail">{rag_detail}</div>
            </div>
            <div class="status-item {get_status_class(azure_status)}">
                <span class="status-icon">{azure_icon}</span>
                <div class="status-text">{azure_text}</div>
                <div class="status-detail">{azure_detail}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Simplified sidebar with just tips and controls
    with st.sidebar:
        st.markdown("### 🎯 What I Can Help With")
        st.markdown("""
        <div style="background: rgba(255,255,255,0.8); padding: 1rem; border-radius: 10px; margin: 0.5rem 0;">
            <p><strong>🔍 Pet Care Questions</strong><br>
            Health, feeding, grooming, training, nutrition, vaccinations</p>
            
            <p><strong>🏠 Pet Adoption</strong><br>
            Find pets by breed, location, age, gender, colors, etc.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        # Add some fun elements
        st.markdown("---")
        st.markdown("### 🎨 Quick Tips")
        st.markdown("""
        - Ask about **pet health** for medical advice
        - Search for **your pawfect match** by describing what you want
        - Use **specific breeds** for better search results
        - Ask **follow-up questions** for more details
        """)

    # Main content area - Enhanced Chat Interface
    st.markdown("""
    <div style="text-align: center; margin: 2rem 0;">
        <h2 style="color: #ff6b9d; margin-bottom: 0.5rem;">💬 Chat with Pawfect Match</h2>
        <p style="color: #666; font-size: 1.1rem;">Ask about pet care or find your perfect pet match! I'll automatically understand what you need.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if systems are available
    if rag is None or chatbot is None:
        st.error("RAG system not available. Please check the sidebar for errors.")
        return
    
    if azure_components[0] is None:
        st.warning("Pet search not available. I can only help with pet care questions.")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Example prompts
    st.markdown("### 💡 Try asking me:")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🐕 What should I feed my puppy?", use_container_width=True):
            st.session_state.example_prompt = "What should I feed my puppy?"
    
    with col2:
        if st.button("🏠 Find my pawfect golden retriever", use_container_width=True):
            st.session_state.example_prompt = "I want to adopt a golden retriever"
    
    with col3:
        if st.button("🏥 My cat is sick, what should I do?", use_container_width=True):
            st.session_state.example_prompt = "My cat is sick, what should I do?"
    
    # Chat input
    prompt = st.chat_input("Ask me anything about pets...")
    
    # Handle example prompts
    if hasattr(st.session_state, 'example_prompt'):
        prompt = st.session_state.example_prompt
        delattr(st.session_state, 'example_prompt')
    
    if prompt:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response using intent classification
        with st.chat_message("assistant"):
            with st.spinner("🔍 Finding your pawfect match..."):
                try:
                    response = chatbot.handle_message(prompt)
                    
                    # Check if this is a pet search result marker
                    if isinstance(response, str) and "PET_SEARCH_RESULTS:" in response:
                        # Extract the search query from the user's prompt and perform search
                        search_query = prompt
                        results, error = perform_pet_search(search_query, azure_components, topk=12)
                        
                        if error:
                            st.error(f"Search error: {error}")
                            st.session_state.messages.append({"role": "assistant", "content": f"Search error: {error}"})
                        else:
                            display_pet_results(results, error)
                            # Store the response for chat history
                            st.session_state.messages.append({"role": "assistant", "content": f"Found {len(results) if results is not None else 0} pets matching your criteria!"})
                    
                    # Check if response is structured pet data
                    elif isinstance(response, dict) and "pets" in response:
                        # Display pet search results with enhanced UI
                        st.markdown(response["message"])
                        
                        # Helper functions from friend's code
                        def _safe_list_from_cell(x):
                            """Parse strings like "['a','b']" or '["a","b"]' or comma strings into list."""
                            if isinstance(x, list): return x
                            if x is None: return []
                            s = str(x).strip()
                            if not s: return []
                            if s.startswith("[") and s.endswith("]"):
                                try:
                                    import json
                                    obj = json.loads(s)
                                    if isinstance(obj, list): return obj
                                except Exception:
                                    pass
                                try:
                                    import ast
                                    obj = ast.literal_eval(s)
                                    if isinstance(obj, list): return obj
                                except Exception:
                                    return []
                            if "," in s: return [t.strip() for t in s.split(",") if t.strip()]
                            return [s]
                        
                        def _first_photo_url(photo_links):
                            """Extract first photo URL safely"""
                            if not photo_links:
                                return None
                            photos = _safe_list_from_cell(photo_links)
                            if photos:
                                url = str(photos[0]).strip().strip('"').strip("'")
                                return url if url else None
                            return None
                        
                        def _age_years_from_months(age_months):
                            """Convert months to readable age format"""
                            try:
                                m = float(age_months)
                                y = m/12.0
                                if m < 12: return f"{int(round(m))} mo (puppy/kitten)"
                                return f"{y:.1f} yrs"
                            except Exception:
                                return "—"
                        
                        def _badge_bool(x, label):
                            """Create status badges"""
                            v = str(x or "").strip().lower()
                            if v in {"true","yes","y","1"}: return f"✅ {label}"
                            if v in {"false","no","n","0"}: return f"❌ {label}"
                            if v in {"unknown", "nan", ""}: return f"➖ {label}"
                            return f"ℹ️ {label}: {x}"
                        
                        # Display pets in a grid layout
                        pets = response["pets"][:6]  # Limit to 6 pets for performance
                        cols_per_row = 2
                        
                        for i in range(0, len(pets), cols_per_row):
                            cols = st.columns(cols_per_row, gap="medium")
                            for j, col in enumerate(cols):
                                if i + j >= len(pets):
                                    break
                                pet = pets[i + j]
                                
                                with col:
                                    with st.container(border=True):
                                        # Pet name and link
                                        name = pet.get('name', f"Pet {i+j+1}")
                                        adoption_url = pet.get('adoption_url', '')
                                        if adoption_url:
                                            st.markdown(f"### [{name}]({adoption_url})")
                                        else:
                                            st.markdown(f"### {name}")
                                        
                                        # Display first photo
                                        photo_url = _first_photo_url(pet.get('photo_urls', []))
                                        if photo_url:
                                            try:
                                                st.image(photo_url, width=300)
                                            except Exception:
                                                st.markdown(f"![photo]({photo_url})")
                                        else:
                                            st.info("No photo available.")
                                        
                                        # Pet details
                                        animal = pet.get('animal', 'Unknown').title()
                                        breed = pet.get('breed', '—')
                                        gender = pet.get('gender', '—').title()
                                        state = pet.get('state', '—').title()
                                        color = pet.get('color', '—')
                                        age_text = _age_years_from_months(pet.get('age_months'))
                                        size = pet.get('size', '—').title()
                                        
                                        st.write(
                                            f"**{animal}** • **Breed:** {breed} • **Gender:** {gender} • "
                                            f"**Age:** {age_text} • **State:** {state}"
                                        )
                                        st.write(f"**Color:** {color} • **Size:** {size}")
                                        
                                        # Status badges (if available)
                                        if pet.get('n_photos', 0) > 0:
                                            st.write(f"📸 {pet['n_photos']} photo(s) available")
                                        
                                        # Description
                                        desc = pet.get('description', '')
                                        if desc:
                                            with st.expander("Description", expanded=False):
                                                excerpt = desc if len(desc) < 200 else (desc[:200] + "…")
                                                st.write(excerpt)
                        
                        # Store the formatted response for chat history
                        formatted_response = response["message"] + "\n\n" + "\n".join([
                            f"**{i+1}. {pet['name']}** - {pet['breed']}, {pet['age_months']} months, {pet['gender']}, {pet['color']}, {pet['size']}"
                            for i, pet in enumerate(response["pets"])
                        ])
                        st.session_state.messages.append({"role": "assistant", "content": formatted_response})
                    else:
                        # Regular text response
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()