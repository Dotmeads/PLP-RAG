# -*- coding: utf-8 -*-
"""
Unified PetBot App - Single Chat Interface with Intent Classification
Automatically routes pet care questions to RAG and pet adoption queries to Azure search
"""
import streamlit as st
st.set_page_config(page_title="Unified PetBot", layout="wide")

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
        # Initialize RAG system
        rag = ProposedRAGManager()
        
        # Load documents
        documents_dir = os.path.join(project_root, "documents")
        if os.path.exists(documents_dir):
            rag.add_directory(documents_dir)
        
        # Initialize chatbot pipeline with RAG (Azure components will be added later)
        chatbot = ChatbotPipeline(rag)
        
        return rag, chatbot
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {str(e)}")
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
@st.cache_data(ttl=300)  # Cache search results for 5 minutes
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
        m_age_mo = re.search(r"\b([1-9][0-9]?)\s*mo(nth)?s?\b", tq)
        m_age_yr = re.search(r"\b([1-9][0-9]?)\s*y(ear)?s?\b", tq)
        if "puppy" in tq or "kitten" in tq:
            age_floor_mo, age_ceil_mo = 0, 12
        elif m_age_mo:
            val = int(m_age_mo.group(1)); age_floor_mo, age_ceil_mo = max(0, val-3), val+3
        elif m_age_yr:
            val = int(m_age_yr.group(1)); mo = 12*val; age_floor_mo, age_ceil_mo = max(0, mo-6), mo+6
        
        facets = {
            "animal": safe_merge(mf.get("animal"), rf.get("animal")),
            "breed":  safe_merge(mf.get("breed"),  rf.get("breed")),
            "gender": safe_merge(mf.get("gender"), rf.get("gender")),
            "colors_any": safe_merge(mf.get("colors_any"), rf.get("colors_any")),
            "state": safe_merge(mf.get("state"), rf.get("state")),
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
                df_filtered = df_filtered[df_filtered["colors"].str.contains("|".join(v), case=False, na=False)]
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
                df_filtered = df_filtered[df_filtered[k].str.contains("|".join(v), case=False, na=False)]
        
        if len(df_filtered) == 0:
            return None, "No pets found matching your criteria. Try relaxing your search terms."
        
        # Hybrid search: BM25 + embeddings
        bm25_scores = bm25.get_scores(only_text(query))
        emb_scores = emb_search(student, query, doc_ids, doc_vecs, faiss_index, topk=topk)
        
        # Combine scores
        combined_scores = {}
        for idx, score in bm25_scores.items():
            combined_scores[idx] = score * 0.3
        for idx, score in emb_scores.items():
            combined_scores[idx] = combined_scores.get(idx, 0) + score * 0.7
        
        # Sort by combined score
        sorted_indices = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in sorted_indices[:topk]]
        
        # Get results
        results = df_filtered.iloc[top_indices] if len(top_indices) > 0 else df_filtered.head(topk)
        
        return results, None
        
    except Exception as e:
        return None, f"Search error: {str(e)}"

def display_pet_results(results, error_msg=None):
    """Display pet search results"""
    if error_msg:
        st.error(error_msg)
        return
    
    if results is None or len(results) == 0:
        st.warning("No pets found matching your criteria.")
        return
    
    st.write(f"Found {len(results)} pets:")
    
    # Display pet cards
    for idx, row in results.iterrows():
        with st.container():
            col1, col2 = st.columns([1, 3])
            
            with col1:
                # Display first photo
                if pd.notna(row.get('photo_url_1')) and row['photo_url_1']:
                    try:
                        st.image(row['photo_url_1'], width=200, caption=f"{row.get('name', 'Unknown')} - {row.get('breed', 'Unknown breed')}")
                    except:
                        st.write("📷 Photo unavailable")
                else:
                    st.write("📷 No photo available")
            
            with col2:
                st.write(f"**{row.get('name', 'Unknown')}** - {row.get('breed', 'Unknown breed')}")
                st.write(f"**Age**: {row.get('age_months', 'Unknown')} months")
                st.write(f"**Gender**: {row.get('gender', 'Unknown')}")
                st.write(f"**Location**: {row.get('state', 'Unknown')}")
                st.write(f"**Colors**: {row.get('colors', 'Unknown')}")
                if pd.notna(row.get('description')):
                    st.write(f"**Description**: {row['description'][:200]}...")
            
            st.divider()

# -------------------------------------------
# MAIN APP
# -------------------------------------------
def main():
    st.title("🐾 Unified PetBot")
    st.caption("Intelligent Pet Assistant - Ask about pet care or find pets for adoption")

    # Pre-load all components at startup
    with st.spinner("🚀 Initializing PetBot systems..."):
        # Load RAG system
        rag, chatbot = bootstrap_rag_system()
        
        # Load Azure components
        azure_components = bootstrap_azure_components()
        
        # Add Azure components to chatbot if both systems loaded successfully
        if chatbot is not None and azure_components[0] is not None:
            chatbot.azure_components = azure_components

    # Display system status
    with st.sidebar:
        st.header("System Status")
        
        # RAG System Status
        st.subheader("RAG System")
        if rag is None:
            st.error("❌ RAG system failed to load")
        else:
            st.success("✅ RAG system ready")
            try:
                doc_count = len(rag.system.vector_manager.vector_store.get()['ids'])
                st.write(f"**Documents**: {doc_count}")
            except:
                st.write(f"**Documents**: Available")
        
        # Azure System Status
        st.subheader("Pet Search")
        if azure_components[0] is None:
            st.error("❌ Azure pet search failed to load")
        else:
            st.success("✅ Pet search ready")
            st.write(f"**Pet Database**: {len(azure_components[5])} pets")
        
        st.divider()
        
        # Show intent classification info
        st.subheader("Intent Classification")
        st.write("🔍 **Pet Care Questions**: Health, feeding, grooming, training")
        st.write("🏠 **Pet Adoption**: Find pets by breed, location, age, etc.")
        st.write("💬 **General Chat**: Greetings, thanks, general conversation")
        
        st.divider()
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

    # Main content area - Unified Chat Interface
    st.header("💬 Chat with PetBot")
    st.caption("Ask about pet care, find pets for adoption, or just chat! I'll automatically understand what you need.")
    
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
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about pets..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response using intent classification
        with st.chat_message("assistant"):
            with st.spinner("Analyzing your request..."):
                try:
                    response = chatbot.handle_message(prompt)
                    
                    # Check if response is structured pet data
                    if isinstance(response, dict) and "pets" in response:
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