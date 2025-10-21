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
        
        # Initialize chatbot pipeline with RAG
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
        
        # Download models and data
        download_prefix_flat(settings["connection_string"], settings["ml_container"], "ner", local_ner_dir())
        download_prefix_flat(settings["connection_string"], settings["ml_container"], "mr", local_mr_dir())
        smart_download_single_blob(settings["connection_string"], settings["pets_container"], "all_pet_details_clean.csv", local_pets_csv_path())
        
        # Load models
        ner = load_ner_pipeline(local_ner_dir())
        student, doc_ids, doc_vecs = load_mr_model(local_mr_dir())
        model_dim = student.get_sentence_embedding_dimension()
        faiss_index = load_faiss_index(local_mr_dir(), model_dim)
        
        # Load pet data
        dfp = pd.read_csv(local_pets_csv_path())
        
        # Initialize BM25
        bm25 = BM25(dfp["description_clean"].fillna("").tolist())
        
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
    """Perform pet search using Azure components"""
    if azure_components[0] is None:
        return None, "Pet search not available"
    
    ner, student, doc_ids, doc_vecs, faiss_index, dfp, bm25, breed_catalog, breed_to_animal = azure_components
    
    try:
        # Process query
        raw_spans = ner([query[:300]])[0] if query else []
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

    # Load all components
    with st.sidebar:
        st.header("System Status")
        
        # Load RAG system
        st.subheader("RAG System")
        rag, chatbot = bootstrap_rag_system()
        if rag is None:
            st.error("❌ RAG system failed to load")
        else:
            st.success("✅ RAG system ready")
            try:
                doc_count = len(rag.system.vector_manager.vector_store.get()['ids'])
                st.write(f"**Documents**: {doc_count}")
            except:
                st.write(f"**Documents**: Available")
        
        # Load Azure components
        st.subheader("Pet Search")
        azure_components = bootstrap_azure_components()
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
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()