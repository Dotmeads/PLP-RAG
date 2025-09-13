"""
Streamlit web interface for the RAG system
"""
import streamlit as st
import os
from pathlib import Path
from typing import List, Dict, Any
import logging

from rag_pipeline import RAGManager
from config import OPENAI_API_KEY, APP_NAME, DEBUG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title=APP_NAME,
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .source-item {
        background-color: #f5f5f5;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.25rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag_manager' not in st.session_state:
    st.session_state.rag_manager = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []


def initialize_rag_system():
    """Initialize the RAG system"""
    try:
        if st.session_state.rag_manager is None:
            with st.spinner("Initializing RAG system..."):
                st.session_state.rag_manager = RAGManager()
            st.success("RAG system initialized successfully!")
        return True
    except Exception as e:
        st.error(f"Error initializing RAG system: {str(e)}")
        return False


def display_chat_message(message: str, is_user: bool = True):
    """Display a chat message"""
    css_class = "user-message" if is_user else "assistant-message"
    st.markdown(f'<div class="chat-message {css_class}">{message}</div>', 
                unsafe_allow_html=True)


def display_sources(sources: List[Dict[str, Any]]):
    """Display source information"""
    if not sources:
        return
    
    st.markdown("**Sources:**")
    for i, source in enumerate(sources, 1):
        with st.expander(f"Source {i} (Score: {source.get('score', 'N/A'):.3f})"):
            st.write("**Content Preview:**")
            st.write(source.get('content_preview', 'No preview available'))
            
            st.write("**Metadata:**")
            metadata = source.get('metadata', {})
            for key, value in metadata.items():
                st.write(f"- **{key}:** {value}")


def main():
    """Main application"""
    st.markdown(f'<div class="main-header">{APP_NAME}</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # API Key input
        api_key = st.text_input(
            "OpenAI API Key",
            value=OPENAI_API_KEY or "",
            type="password",
            help="Enter your OpenAI API key to enable answer generation"
        )
        
        if api_key and not OPENAI_API_KEY:
            os.environ["OPENAI_API_KEY"] = api_key
            st.session_state.rag_manager = None  # Force reinitialization
        
        # Initialize system
        if st.button("Initialize RAG System", type="primary"):
            initialize_rag_system()
        
        st.divider()
        
        # Document upload
        st.header("📄 Document Management")
        
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=['pdf', 'txt', 'docx', 'md', 'html'],
            accept_multiple_files=True,
            help="Upload documents to add to the knowledge base"
        )
        
        if uploaded_files:
            # Save uploaded files
            upload_dir = Path("uploads")
            upload_dir.mkdir(exist_ok=True)
            
            for uploaded_file in uploaded_files:
                file_path = upload_dir / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.session_state.uploaded_files.append(str(file_path))
            
            st.success(f"Uploaded {len(uploaded_files)} files!")
        
        # Process uploaded files
        if st.session_state.uploaded_files and st.session_state.rag_manager:
            if st.button("Process Uploaded Files"):
                with st.spinner("Processing documents..."):
                    results = st.session_state.rag_manager.add_documents(st.session_state.uploaded_files)
                    
                    st.success(f"Processed {len(results['success'])} files successfully!")
                    if results['failed']:
                        st.warning(f"Failed to process {len(results['failed'])} files")
                        for failure in results['failed']:
                            st.error(failure)
                    
                    # Clear uploaded files after processing
                    st.session_state.uploaded_files = []
        
        st.divider()
        
        # System stats
        if st.session_state.rag_manager:
            st.header("📊 System Status")
            stats = st.session_state.rag_manager.get_stats()
            
            st.metric("Documents", stats.get('vector_store', {}).get('document_count', 0))
            st.metric("Generation", "Available" if stats.get('generation_available') else "Disabled")
            
            if st.button("Reset System"):
                st.session_state.rag_manager.reset()
                st.session_state.chat_history = []
                st.success("System reset!")
    
    # Main content area
    if st.session_state.rag_manager is None:
        st.info("👈 Please initialize the RAG system using the sidebar.")
        return
    
    # Chat interface
    st.header("💬 Chat with Your Documents")
    
    # Display chat history
    for message in st.session_state.chat_history:
        display_chat_message(message['content'], message['is_user'])
        
        if not message['is_user'] and 'sources' in message:
            display_sources(message['sources'])
    
    # Chat input
    user_input = st.chat_input("Ask a question about your documents...")
    
    if user_input:
        # Add user message to history
        st.session_state.chat_history.append({
            'content': user_input,
            'is_user': True
        })
        
        # Display user message
        display_chat_message(user_input, True)
        
        # Get response from RAG system
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.rag_manager.chat(user_input)
                
                # Add assistant response to history
                st.session_state.chat_history.append({
                    'content': response['answer'],
                    'is_user': False,
                    'sources': response.get('sources', [])
                })
                
                # Display assistant response
                display_chat_message(response['answer'], False)
                
                # Display sources
                if response.get('sources'):
                    display_sources(response['sources'])
                
                # Show error if any
                if 'error' in response:
                    st.error(f"Error: {response['error']}")
                    
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.session_state.chat_history.append({
                    'content': error_msg,
                    'is_user': False
                })
                display_chat_message(error_msg, False)
                st.error(f"Error: {str(e)}")
    
    # Suggested questions
    if st.session_state.rag_manager and st.session_state.chat_history:
        st.divider()
        st.header("💡 Suggested Questions")
        
        if st.button("Generate Suggestions"):
            suggestions = st.session_state.rag_manager.get_suggestions()
            if suggestions:
                for suggestion in suggestions:
                    if st.button(suggestion, key=f"suggestion_{suggestion}"):
                        st.session_state.chat_history.append({
                            'content': suggestion,
                            'is_user': True
                        })
                        st.rerun()
            else:
                st.info("No suggestions available. Try asking some questions first!")


if __name__ == "__main__":
    main()
