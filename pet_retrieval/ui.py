# -*- coding: utf-8 -*-
"""
Created on Sat Oct 18 13:17:27 2025

@author: zheng
"""

# pet_retrieval/ui.py
import streamlit as st
from typing import Dict

# Fixed choices for "best model only"
_BEST_METHOD = "hybrid"   # hybrid (BM25 + embeddings)
_USE_MMR     = False      # ner_light_nom (no diversification)
_MMR_LAMBDA  = 0.35       # kept for compatibility; not used when _USE_MMR=False

def sidebar_controls() -> Dict:
    with st.sidebar:
        st.header("Search controls")

        # ---- View options ----
        view_mode = st.radio("View", ["Cards", "Table"], index=0, horizontal=True)
        grid_cols = st.slider("Cards per row", 1, 4, 3, 1)

        # ---- Retrieval knobs (keep Top-K; hide method/MMR) ----
        topk = st.slider("Top-K", min_value=3, max_value=50, value=12, step=1)

        # ---- Strictness ----
        strict_mode = st.checkbox("Strict Mode (smaller candidate floor)", value=False)

        # ---- Debug ----
        with st.expander("Advanced / Debug", expanded=False):
            debug_media = st.checkbox("Debug media URLs", value=False)

        # Persist debug toggle
        st.session_state["debug_media"] = debug_media

        # Return all keys expected by the app;
        # lock method/MMR to the best-model settings.
        return {
            "view_mode": view_mode,
            "grid_cols": grid_cols,
            "topk": topk,
            "method": _BEST_METHOD,      # locked
            "use_mmr": _USE_MMR,         # locked
            "mmr_lambda": _MMR_LAMBDA,   # compatibility only
            "strict_mode": strict_mode,
        }

def age_text_yr_mo(age_months) -> str:
    try:
        m = int(round(float(age_months)))
        if m < 12: return f"{m} mo (puppy/kitten)"
        y, r = divmod(m, 12)
        return f"{y} yr" if r == 0 else f"{y} yr {r} mo"
    except Exception:
        return "—"

def safe_colors_text(colors_val) -> str:
    txt = ", ".join(colors_val) if isinstance(colors_val, (list, tuple)) else str(colors_val or "").strip()
    if txt.lower() in {"unknown","nan","none",""}:
        return "—"
    return txt
