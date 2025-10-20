# -*- coding: utf-8 -*-
"""
Optimized Unified PetBot App - Maximum Performance
Uses all essential components properly integrated for best results:
- Intent Classification + Entity Extraction for routing
- RAG System for pet care questions  
- Advanced Search for pet adoption
- Proper error handling and fallbacks
"""

import streamlit as st
import json
import os
import sys
from typing import List, Dict, Any, Optional
import logging

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set page config first
st.set_page_config(
    page_title="PetBot - AI-Powered Pet Assistant", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------
# CORE IMPORTS WITH PROPER ERROR HANDLING
# -------------------------------------------
try:
    from chatbot_pipeline import ChatbotPipeline
    from proposed_rag_system import ProposedRAGSystem
    from intent_classifier import IntentClassifier
    from entity_extractor import EntityExtractor
    from synonyms import SYNONYMS, canonicalize
    from responses import get_response
    CORE_AVAILABLE = True
    logger.info("✅ All core components loaded successfully")
except ImportError as e:
    logger.error(f"❌ Core components not available: {e}")
    CORE_AVAILABLE = False

# -------------------------------------------
# DEMO DATA (Enhanced with more realistic data)
# -------------------------------------------
DEMO_PETS = [
    {
        'pet_id': 1, 'name': 'Buddy', 'animal': 'dog', 'breed': 'golden retriever',
        'gender': 'male', 'state': 'selangor', 'age_months': 24, 'color': 'golden',
        'size': 'large', 'fur_length': 'long', 'condition': 'healthy',
        'description': 'Friendly golden retriever looking for a loving home. Great with kids and other pets. Loves to play fetch and go on walks.',
        'vaccinated': True, 'dewormed': True, 'neutered': True, 'spayed': False,
        'url': 'https://example.com/buddy', 'photo_links': ['https://example.com/buddy1.jpg'],
        'special_needs': False, 'good_with_kids': True, 'good_with_pets': True
    },
    {
        'pet_id': 2, 'name': 'Luna', 'animal': 'cat', 'breed': 'persian',
        'gender': 'female', 'state': 'johor', 'age_months': 18, 'color': 'white',
        'size': 'medium', 'fur_length': 'long', 'condition': 'healthy',
        'description': 'Beautiful Persian cat, very gentle and calm. Perfect for apartment living. Loves to be brushed and cuddled.',
        'vaccinated': True, 'dewormed': True, 'neutered': False, 'spayed': True,
        'url': 'https://example.com/luna', 'photo_links': ['https://example.com/luna1.jpg'],
        'special_needs': False, 'good_with_kids': True, 'good_with_pets': False
    },
    {
        'pet_id': 3, 'name': 'Max', 'animal': 'dog', 'breed': 'german shepherd',
        'gender': 'male', 'state': 'penang', 'age_months': 36, 'color': 'brown',
        'size': 'large', 'fur_length': 'short', 'condition': 'healthy',
        'description': 'Loyal German Shepherd, great with kids. Needs active family. Excellent guard dog and very intelligent.',
        'vaccinated': True, 'dewormed': True, 'neutered': True, 'spayed': False,
        'url': 'https://example.com/max', 'photo_links': ['https://example.com/max1.jpg'],
        'special_needs': False, 'good_with_kids': True, 'good_with_pets': True
    },
    {
        'pet_id': 4, 'name': 'Bella', 'animal': 'cat', 'breed': 'siamese',
        'gender': 'female', 'state': 'kuala lumpur', 'age_months': 12, 'color': 'cream',
        'size': 'medium', 'fur_length': 'short', 'condition': 'healthy',
        'description': 'Elegant Siamese cat, very social and playful. Loves attention and will follow you around the house.',
        'vaccinated': True, 'dewormed': True, 'neutered': False, 'spayed': True,
        'url': 'https://example.com/bella', 'photo_links': ['https://example.com/bella1.jpg'],
        'special_needs': False, 'good_with_kids': True, 'good_with_pets': True
    },
    {
        'pet_id': 5, 'name': 'Charlie', 'animal': 'dog', 'breed': 'labrador',
        'gender': 'male', 'state': 'selangor', 'age_months': 30, 'color': 'black',
        'size': 'large', 'fur_length': 'short', 'condition': 'healthy',
        'description': 'Active Labrador, loves to play and exercise. Great family dog. Needs daily walks and playtime.',
        'vaccinated': True, 'dewormed': True, 'neutered': True, 'spayed': False,
        'url': 'https://example.com/charlie', 'photo_links': ['https://example.com/charlie1.jpg'],
        'special_needs': False, 'good_with_kids': True, 'good_with_pets': True
    },
    {
        'pet_id': 6, 'name': 'Milo', 'animal': 'cat', 'breed': 'maine coon',
        'gender': 'male', 'state': 'johor', 'age_months': 8, 'color': 'orange',
        'size': 'large', 'fur_length': 'long', 'condition': 'healthy',
        'description': 'Fluffy Maine Coon kitten, very playful and curious. Will grow to be a large, majestic cat.',
        'vaccinated': True, 'dewormed': True, 'neutered': False, 'spayed': False,
        'url': 'https://example.com/milo', 'photo_links': ['https://example.com/milo1.jpg'],
        'special_needs': False, 'good_with_kids': True, 'good_with_pets': True
    }
]

# -------------------------------------------
# ENHANCED HELPER FUNCTIONS
# -------------------------------------------
def _age_years_from_months(age_months) -> str:
    """Convert age in months to readable format"""
    try:
        m = float(age_months)
        y = m / 12.0
        if m < 12:
            return f"{int(round(m))} mo (puppy/kitten)"
        return f"{y:.1f} yrs"
    except Exception:
        return "—"

def _badge_bool(x, label):
    """Create status badges for boolean values"""
    v = str(x or "").strip().lower()
    if v in {"true", "yes", "y", "1"}:
        return f"✅ {label}"
    if v in {"false", "no", "n", "0"}:
        return f"❌ {label}"
    if v in {"unknown", "nan", ""}:
        return f"➖ {label}"
    return f"ℹ️ {label}: {x}"

def _calculate_match_score(pet: Dict[str, Any], query: str, entities: Dict[str, Any]) -> float:
    """Calculate how well a pet matches the search criteria"""
    score = 0.0
    query_lower = query.lower()
    pet_text = f"{pet.get('animal', '')} {pet.get('breed', '')} {pet.get('description', '')} {pet.get('state', '')} {pet.get('color', '')} {pet.get('gender', '')} {pet.get('size', '')}".lower()
    
    # Exact phrase matching (highest priority)
    if query_lower in pet_text:
        score += 100
    
    # Entity-based matching
    if entities.get('PET_TYPE') and entities['PET_TYPE'].lower() in pet.get('animal', '').lower():
        score += 50
    
    if entities.get('BREED') and entities['BREED'].lower() in pet.get('breed', '').lower():
        score += 40
    
    if entities.get('STATE') and entities['STATE'].lower() in pet.get('state', '').lower():
        score += 30
    
    if entities.get('COLOR') and entities['COLOR'].lower() in pet.get('color', '').lower():
        score += 20
    
    if entities.get('GENDER') and entities['GENDER'].lower() in pet.get('gender', '').lower():
        score += 15
    
    # Keyword matching
    keywords = [w for w in query_lower.split() if len(w) > 2 and w not in {'the', 'and', 'or', 'in', 'for', 'with'}]
    for keyword in keywords:
        if keyword in pet_text:
            score += 10
    
    return score

# -------------------------------------------
# ENHANCED PET CARD RENDERING
# -------------------------------------------
def render_pet_card(pet: Dict[str, Any], show_score: bool = False, score: float = 0.0):
    """Render a single pet card with enhanced styling and information"""
    pid = pet.get("pet_id")
    name = pet.get("name", f"Pet {pid}")
    animal = pet.get("animal", "").title()
    breed = pet.get("breed", "—")
    gender = pet.get("gender", "—").title()
    state = pet.get("state", "—").title()
    color = pet.get("color", "—")
    age_mo = pet.get("age_months")
    age_yrs_txt = _age_years_from_months(age_mo)
    size = pet.get("size", "—").title()
    fur = pet.get("fur_length", "—").title()
    cond = pet.get("condition", "—").title()
    
    # Health status
    vacc = _badge_bool(pet.get("vaccinated"), "vaccinated")
    dewm = _badge_bool(pet.get("dewormed"), "dewormed")
    neut = _badge_bool(pet.get("neutered"), "neutered")
    spay = _badge_bool(pet.get("spayed"), "spayed")
    
    # Compatibility
    good_kids = _badge_bool(pet.get("good_with_kids"), "good with kids")
    good_pets = _badge_bool(pet.get("good_with_pets"), "good with pets")
    special_needs = _badge_bool(pet.get("special_needs"), "special needs")

    with st.container(border=True):
        # Header with name and score
        header_col1, header_col2 = st.columns([4, 1])
        with header_col1:
            st.markdown(f"### 🐾 {name}")
        with header_col2:
            if show_score:
                st.metric("Match Score", f"{score:.1f}")
        
        # Photo placeholder
        st.info("📷 Photo placeholder - would show pet image here")
        
        # Basic info in two columns
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**{animal}** • **{breed}**")
            st.write(f"**Gender:** {gender} • **Age:** {age_yrs_txt}")
        with col2:
            st.write(f"**Location:** {state}")
            st.write(f"**Color:** {color} • **Size:** {size}")
        
        # Health status
        st.markdown("**Health Status:**")
        st.markdown(" | ".join([vacc, dewm, neut, spay]))
        
        # Compatibility
        st.markdown("**Compatibility:**")
        st.markdown(" | ".join([good_kids, good_pets, special_needs]))
        
        # Description
        desc = pet.get("description", "").strip()
        if desc:
            with st.expander("Description", expanded=False):
                st.write(desc)
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("💖 Interested", key=f"interested_{pid}"):
                st.success(f"Great choice! {name} is a wonderful {animal.lower()}!")
        with col2:
            if st.button("📞 Contact", key=f"contact_{pid}"):
                st.info(f"Contact info for {name} would be displayed here")
        with col3:
            if st.button("ℹ️ More Info", key=f"info_{pid}"):
                st.info(f"More details about {name} would be shown here")

def render_results_grid(pets: List[Dict[str, Any]], max_cols: int = 3, show_scores: bool = False, scores: Dict[int, float] = None):
    """Render pets in a responsive grid layout with optional scoring"""
    if not pets:
        st.warning("No results to show.")
        return
    
    n = len(pets)
    col_count = max(1, min(max_cols, n))
    for i in range(0, n, col_count):
        cols = st.columns(col_count, gap="medium")
        for j, col in enumerate(cols):
            idx = i + j
            if idx >= n:
                continue
            with col:
                pet = pets[idx]
                score = scores.get(pet['pet_id'], 0.0) if scores else 0.0
                render_pet_card(pet, show_score=show_scores, score=score)

# -------------------------------------------
# ENHANCED SEARCH FUNCTIONS
# -------------------------------------------
def advanced_search_with_entities(pets: List[Dict[str, Any]], query: str, entities: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """Advanced search using both query and extracted entities"""
    if not query.strip():
        return pets
    
    if entities is None:
        entities = {}
    
    # Calculate scores for each pet
    scores = {}
    for pet in pets:
        score = _calculate_match_score(pet, query, entities)
        if score > 0:
            scores[pet['pet_id']] = score
    
    # Sort by score (highest first)
    results = sorted(pets, key=lambda x: scores.get(x['pet_id'], 0), reverse=True)
    return results, scores

# -------------------------------------------
# RAG INTEGRATION WITH CACHING
# -------------------------------------------
@st.cache_resource
def load_rag_system():
    """Load RAG system with caching"""
    if not CORE_AVAILABLE:
        return None
    
    try:
        rag = ProposedRAGSystem()
        # Load documents if they exist
        if os.path.exists("documents"):
            rag.ingest_directory("documents")
            logger.info("✅ RAG system loaded with documents")
        else:
            logger.warning("⚠️ No documents directory found, RAG system loaded without knowledge base")
        return rag
    except Exception as e:
        logger.error(f"❌ Failed to load RAG system: {e}")
        return None

@st.cache_resource
def load_chatbot_pipeline():
    """Load chatbot pipeline with RAG integration"""
    if not CORE_AVAILABLE:
        return None
    
    try:
        # Load RAG system
        rag_system = load_rag_system()
        
        # Create chatbot with RAG integration
        chatbot = ChatbotPipeline(rag_system=rag_system)
        logger.info("✅ Chatbot pipeline loaded with RAG integration")
        return chatbot
    except Exception as e:
        logger.error(f"❌ Failed to load chatbot pipeline: {e}")
        return None

# -------------------------------------------
# MAIN APP
# -------------------------------------------
def main():
    st.title("🐾 PetBot - AI-Powered Pet Assistant")
    st.caption("Powered by AI: Intent Classification, Entity Extraction, RAG, and Advanced Search")
    
    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "current_mode" not in st.session_state:
        st.session_state.current_mode = "chat"
    if "search_entities" not in st.session_state:
        st.session_state.search_entities = {}
    
    # Sidebar for mode selection and controls
    with st.sidebar:
        st.header("🎛️ Controls")
        mode = st.radio(
            "Choose your mode:",
            ["💬 Chat Mode", "🔍 Search Mode", "🧠 RAG Mode"],
            index=0 if st.session_state.current_mode == "chat" else 1
        )
        
        if "Chat" in mode:
            st.session_state.current_mode = "chat"
        elif "Search" in mode:
            st.session_state.current_mode = "search"
        else:
            st.session_state.current_mode = "rag"
        
        st.divider()
        
        # System status
        st.subheader("🔧 System Status")
        st.write(f"**Core Components:** {'✅ Available' if CORE_AVAILABLE else '❌ Not Available'}")
        
        if CORE_AVAILABLE:
            # Load components to check their status
            chatbot = load_chatbot_pipeline()
            rag_system = load_rag_system()
            
            st.write(f"**Chatbot:** {'✅ Loaded' if chatbot else '❌ Failed'}")
            st.write(f"**RAG System:** {'✅ Loaded' if rag_system else '❌ Failed'}")
            st.write(f"**Demo Data:** ✅ {len(DEMO_PETS)} pets loaded")
        else:
            st.error("Core components not available. Please check dependencies.")
        
        # Search controls (only in search mode)
        if st.session_state.current_mode == "search":
            st.subheader("🔍 Search Controls")
            view_mode = st.radio("View Mode", ["Cards", "Table"], index=0, horizontal=True)
            grid_cols = st.slider("Cards per row", 1, 4, 3, 1)
            topk = st.slider("Max Results", min_value=3, max_value=20, value=12, step=1)
            show_scores = st.checkbox("Show Match Scores", value=True)
    
    # Main content area
    if st.session_state.current_mode == "chat":
        # CHAT MODE
        st.header("💬 AI Chat Assistant")
        st.caption("Ask about pet care or tell me what kind of pet you're looking for!")
        
        # Load chatbot
        chatbot = load_chatbot_pipeline()
        
        if not CORE_AVAILABLE or chatbot is None:
            st.error("Chatbot is not available. Please check the system status.")
            st.info("You can still use Search Mode to find pets!")
        else:
            # Chat input
            user_input = st.chat_input("Type your message here...")
            
            if user_input:
                # Add user message to history
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                
                # Get bot response
                with st.spinner("AI is thinking..."):
                    try:
                        bot_response = chatbot.handle_message(user_input)
                    except Exception as e:
                        bot_response = f"Sorry, I encountered an error: {e}"
                        logger.error(f"Chatbot error: {e}")
                
                # Add bot response to history
                st.session_state.chat_history.append({"role": "bot", "content": bot_response})
            
            # Display chat history
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
            
            # Clear chat button
            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()
    
    elif st.session_state.current_mode == "search":
        # SEARCH MODE
        st.header("🔍 Advanced Pet Search")
        st.caption("Search for pets using natural language with AI-powered matching!")
        
        # Search input
        q = st.text_input("Your search", value="golden retriever puppy in selangor", key="search_input")
        
        if q.strip():
            # Extract entities if core components are available
            entities = {}
            if CORE_AVAILABLE:
                try:
                    # Load entity extractor
                    entity_extractor = EntityExtractor()
                    entities = entity_extractor.extract_entities(q)
                    st.session_state.search_entities = entities
                    
                    if entities:
                        st.info(f"**Detected entities:** {', '.join([f'{k}: {v}' for k, v in entities.items()])}")
                except Exception as e:
                    logger.error(f"Entity extraction error: {e}")
                    entities = st.session_state.search_entities
            
            # Search logic
            st.write("**Search Results:**")
            st.caption(f"Searching for: '{q}'")
            
            # Perform advanced search
            results, scores = advanced_search_with_entities(DEMO_PETS, q, entities)
            
            if len(results) > 0:
                st.success(f"Found {len(results)} results")
                
                # Display results
                if view_mode == "Cards":
                    render_results_grid(results[:topk], max_cols=grid_cols, show_scores=show_scores, scores=scores)
                else:
                    # Table view
                    st.write("**Pet List:**")
                    for i, pet in enumerate(results[:topk], 1):
                        score = scores.get(pet['pet_id'], 0.0)
                        with st.expander(f"{i}. {pet['name']} - {pet['animal'].title()} - {pet['breed']} (Score: {score:.1f})", expanded=False):
                            st.write(f"**Gender:** {pet['gender'].title()}")
                            st.write(f"**State:** {pet['state'].title()}")
                            st.write(f"**Age:** {_age_years_from_months(pet['age_months'])}")
                            st.write(f"**Color:** {pet['color'].title()}")
                            st.write(f"**Size:** {pet['size'].title()}")
                            st.write(f"**Description:** {pet['description']}")
            else:
                st.warning("No results found. Try different keywords or check spelling.")
        
        # Quick search suggestions
        st.subheader("💡 Quick Search Suggestions")
        suggestions = [
            "golden retriever puppy",
            "female cat in Johor", 
            "small dog for apartment",
            "brown dog good with kids",
            "kitten",
            "large dog",
            "white cat"
        ]
        
        cols = st.columns(len(suggestions))
        for i, suggestion in enumerate(suggestions):
            with cols[i]:
                if st.button(suggestion, key=f"suggestion_{i}"):
                    st.session_state.search_input = suggestion
                    st.rerun()
    
    else:
        # RAG MODE
        st.header("🧠 RAG-Powered Pet Care Assistant")
        st.caption("Ask detailed questions about pet care using our knowledge base!")
        
        # Load RAG system
        rag_system = load_rag_system()
        
        if rag_system is None:
            st.error("RAG system is not available. Please check the system status.")
            st.info("Make sure the RAG system and documents are properly set up.")
        else:
            # RAG query input
            rag_query = st.text_input("Ask about pet care:", value="what can I feed my dog?", key="rag_input")
            
            if rag_query.strip():
                with st.spinner("Searching knowledge base..."):
                    try:
                        result = rag_system.ask(rag_query)
                        
                        st.success("✅ Found relevant information!")
                        st.write("**Answer:**")
                        st.write(result.answer)
                        
                        if result.sources_used:
                            st.write("**Sources:**")
                            for source in result.sources_used:
                                if isinstance(source, dict) and 'source' in source:
                                    st.write(f"- {source['source']}")
                                else:
                                    st.write(f"- {source}")
                    
                    except Exception as e:
                        st.error(f"Error processing query: {e}")
                        logger.error(f"RAG error: {e}")
            
            # Sample questions
            st.subheader("💡 Sample Questions")
            sample_questions = [
                "What can I feed my dog?",
                "What vaccines does my kitten need?",
                "How often should I groom my cat?",
                "What are signs of a healthy pet?",
                "How much exercise does a dog need?"
            ]
            
            for question in sample_questions:
                if st.button(f"❓ {question}", key=f"sample_{question}"):
                    st.session_state.rag_input = question
                    st.rerun()

if __name__ == "__main__":
    main()
