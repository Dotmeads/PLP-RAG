import streamlit as st
import pandas as pd
import html

# =========================================================
#  Theme
# =========================================================
PINK_CSS = """
<style>
/* -----------------------------------
   Base Layout & Font
----------------------------------- */
.stApp {
  background: linear-gradient(180deg, #fff7f5 0%, #fff3f0 100%);
  font-family: "Poppins", "Segoe UI", sans-serif;
}

/* Make main content full-width to match header/footer */
main .block-container {
  max-width: 100% !important;
  padding-left: 3rem !important;
  padding-right: 3rem !important;
  padding-bottom: 140px !important; /* space for fixed input bar */
}

/* -----------------------------------
   Header
----------------------------------- */
.main-header {
  text-align: center;
  padding: 2.5rem 0 1.2rem;
  border-radius: 20px;
  background: #ffcabd;  /* solid peach */
  box-shadow: 0 8px 24px rgba(255, 180, 160, 0.25);
}
.main-header h1 {
  font-size: 3.2rem;
  color: #5a5a5a;  /* soft grey */
  font-weight: 800;
  margin-bottom: 0.3rem;
  letter-spacing: 1px;
  text-shadow: none;
}
.main-header p {
  color: #6e6e6e;
  font-size: 1.1rem;
  margin: 0;
}

/* -----------------------------------
   Status Cards
----------------------------------- */
.status-bar {
  background: rgba(255, 255, 255, 0.45);
  backdrop-filter: blur(8px);
  padding: 0.8rem 1.2rem;
  border-radius: 14px;
  margin: 1.2rem 0 1.5rem;
}
.status-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 0.8rem;
}
.status-item {
  background: #fffaf8;
  border: 1px solid #ffe0d6;
  border-radius: 10px;
  padding: 0.5rem 0.7rem;
  text-align: center;
  font-size: 0.85rem;
  box-shadow: 0 1px 3px rgba(0,0,0,0.05);
  transition: all 0.2s ease;
}
.status-item:hover {
  transform: translateY(-2px);
  box-shadow: 0 3px 8px rgba(0,0,0,0.06);
}
.status-text {
  color: #444;
  font-weight: 600;
}
.status-detail {
  color: #777;
  font-size: 0.8rem;
}

/* -----------------------------------
   Buttons (global)
----------------------------------- */
button[kind="primary"] {
  background: linear-gradient(90deg, #ffc6bb, #ffe0d6);
  border: none;
  color: #fff !important;
  font-weight: 600;
  border-radius: 10px;
  padding: 0.4rem 1rem;
  transition: 0.2s ease;
}
button[kind="primary"]:hover {
  background: linear-gradient(90deg, #ffb8a8, #ffd0c0);
  transform: scale(1.03);
}

/* -----------------------------------
   Chat Area
----------------------------------- */
.stChatMessage {
  border-radius: 14px;
  padding: 0.8rem 1rem;
  margin-bottom: 0.8rem;
  max-width: 100% !important;
  box-shadow: 0 2px 6px rgba(0,0,0,0.05);
}
.stChatMessage[data-testid="stChatMessageUser"] {
  background-color: #fff;
  border: 1px solid #ffd6bf;
  margin-left: auto;
}
.stChatMessage[data-testid="stChatMessageAssistant"] {
  background-color: #fff8f6;
  border: 1px solid #ffe4d5;
}

/* -----------------------------------
   Fixed Input Bar - Aligned with Content
----------------------------------- */
.stChatFloatingInputContainer, 
div[data-testid="stChatFloatingInputContainer"], 
.stChatInputContainer {
  position: fixed !important;
  bottom: 0 !important;
  left: 0 !important;
  right: 0 !important;
  width: 100% !important;
  backdrop-filter: blur(10px);
  background-color: rgba(255, 247, 245, 0.5) !important;
  border-top: none !important;
  box-shadow: none !important;
  z-index: 9999 !important;
  padding: 0.4rem 0 !important; /* Remove horizontal padding from container */
}

/* Input wrapper - constrain width to match content */
[data-testid="stChatFloatingInputContainer"] > div,
.stChatInputContainer > div {
  max-width: calc(100% - 6rem) !important; /* Match main content padding (3rem each side) */
  margin: 0 auto !important;
  padding: 0 3rem !important; /* Match main content padding */
}

/* Input box itself */
.stChatInputContainer textarea,
[data-testid="stChatFloatingInputContainer"] textarea {
  border-radius: 12px !important;
  border: 1px solid #ffd6bf !important;
  background: #fff !important;
  font-size: 1rem !important;
  padding: 0.7rem 1rem !important;
  width: 100% !important;
}

/* Prevent content from hiding behind input bar */
main .block-container {
  padding-bottom: 100px !important;
}

/* -----------------------------------
   Footer (spacing consistency)
----------------------------------- */
.footer-section {
  margin-top: 1.8rem;
  padding-top: 0.8rem;
  border-top: 1px solid #ffe0d6;
  margin-bottom: 2rem !important;
}

/* Remove any ghost container below input */
div[data-testid="element-container"]:has(.stChatFloatingInputContainer) + div {
  display: none !important;
}

</style>
"""

# =========================================================
# Apply global UI
# =========================================================
def apply_global_ui():
    st.markdown(PINK_CSS, unsafe_allow_html=True)
    st.markdown("""
    <div class='main-header'>
      <h1>🐾 Pawfect Match</h1>
      <p>Your Intelligent Pet Assistant — Ask about pet care or find your pawfect pet!</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <style>
    [data-testid="stBottomBlockContainer"] {
        padding-bottom: 0rem !important;
        margin-bottom: -3rem !important;
    }
    .stChatInputContainer {
        margin-bottom: 2.2rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

# =========================================================
# Loading Box (Animated, auto-clear supported)
# =========================================================
def render_loading_box(container, message="Initializing systems...", emoji="🚀"):
    html = f"""
    <div style='
        display:flex;
        align-items:center;
        justify-content:center;
        gap:0.6rem;
        background:rgba(255, 250, 245, 0.9);
        border:1.5px solid #ffd6bf;
        border-radius:14px;
        padding:0.8rem 1.2rem;
        margin:1rem auto 1.2rem;
        width:fit-content;
        box-shadow:0 4px 10px rgba(0,0,0,0.04);
        animation: pulse 1.8s infinite;
    '>
        <span style='font-size:1.3rem;'>{emoji}</span>
        <span style='font-size:0.95rem;color:#555;font-weight:600;'>{message}</span>
    </div>
    <style>
    @keyframes pulse {{
      0% {{ opacity: 1; transform: scale(1); }}
      50% {{ opacity: 0.7; transform: scale(1.03); }}
      100% {{ opacity: 1; transform: scale(1); }}
    }}
    </style>
    """
    container.markdown(html, unsafe_allow_html=True)

# =========================================================
# Status Bar
# =========================================================
def render_status_bar(rag_ok, env_ok, app_status_ok, app_status_text, pets_detail):
    def icon(ok: bool) -> str:
        return "✔" if ok else "✖"
    
    st.markdown(
        f"""
        <div style="
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 1.6rem;
            font-size: 0.9rem;
            color: #3b3b3b;
            background: linear-gradient(180deg, #fffaf8 0%, #fff6f2 100%);
            border: 1px solid #ffd6c6;
            border-radius: 16px;
            padding: 0.55rem 1.6rem;
            width: fit-content;
            margin: 0.1rem auto 2.0rem auto;   /* top ↓ bottom ↑ */
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            backdrop-filter: blur(5px);
            text-align: center;
        ">
            <span>{icon(app_status_ok)} <strong>App:</strong> {app_status_text}</span>
            <span>{icon(rag_ok)} <strong>RAG:</strong> {'Online' if rag_ok else 'Unavailable'}</span>
            <span>{icon(env_ok)} <strong>Pets:</strong> {pets_detail}</span>
        </div>
        """,
        unsafe_allow_html=True
    )