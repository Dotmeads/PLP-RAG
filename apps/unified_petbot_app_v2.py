# -*- coding: utf-8 -*-
"""
Pawfect Match - Single Chat Interface with Intent Classification
Automatically routes pet care questions to RAG and pet adoption queries to hybrid search (BM25 + Embeddings)

Changes in this version:
- Always route via IntentClassifier (pet_care vs find_pet) from your pipeline.
- Uses ChatbotPipeline (which embeds HF NER) to manage dialogue + entities.
- When ChatbotPipeline returns "Got it! Searching ...", we trigger the pet search.
- Keeps pink background, HF NER, hybrid search, 6-card max, fixed-size photos,
  green highlight for exact-match cards, and shows "facets used" banner.
"""
import streamlit as st
st.set_page_config(page_title="Pawfect Match", layout="wide")

import os, re, json, ast
import sys
from typing import List, Dict, Any, Tuple, Optional

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import pandas as pd
import numpy as np

# RAG + Chatbot (HF NER lives inside your ChatbotPipeline)
from rag_system.proposed_rag_system import ProposedRAGManager
from chatbot_flow.chatbot_pipeline import ChatbotPipeline  # uses IntentClassifier + EntityExtractor (HF)  :contentReference[oaicite:2]{index=2}
from chatbot_flow.intent_classifier import IntentClassifier                                   #           :contentReference[oaicite:3]{index=3}

# Retrieval stack (note: NER is NOT downloaded from Azure anymore)
from pet_retrieval.config import get_blob_settings, local_mr_dir, local_pets_csv_path
from pet_retrieval.azure_io import download_prefix_flat, smart_download_single_blob
from pet_retrieval.models import load_mr_model, load_faiss_index
from pet_retrieval.retrieval import (
    only_text, BM25,
    parse_facets_from_text,  # still available if needed
    emb_search
)

# Optional fuzzy breed mapping
try:
    from rapidfuzz import process, fuzz
    _HAS_FUZZ = True
except Exception:
    _HAS_FUZZ = False
    process = fuzz = None


# --------------------------
# Small helpers (UI/data)
# --------------------------
def _safe_list_from_cell(x):
    if isinstance(x, list): return x
    if x is None: return []
    s = str(x).strip()
    if not s: return []
    if s.startswith("[") and s.endswith("]"):
        try:
            obj = json.loads(s)
            if isinstance(obj, list): return obj
        except Exception:
            pass
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, list): return obj
        except Exception:
            return []
    if "," in s: return [t.strip() for t in s.split(",") if t.strip()]
    return [s]

def _first_photo_url_from_row(row) -> Optional[str]:
    photos = row.get("photo_links")
    if not isinstance(photos, list):
        photos = _safe_list_from_cell(photos)
    if photos:
        url = str(photos[0]).strip().strip('"').strip("'")
        return url if url else None
    return None

def _age_text_from_months(age_months) -> str:
    try:
        m = int(round(float(age_months)))
        if m < 12:
            return f"{m} mo"
        y, r = divmod(m, 12)
        return f"{y} yr" if r == 0 else f"{y} yr {r} mo"
    except Exception:
        return "—"

def _badge_bool(x, label):
    v = str(x or "").strip().lower()
    if v in {"true","yes","y","1"}: return f"✅ {label}"
    if v in {"false","no","n","0"}: return f"❌ {label}"
    if v in {"unknown","nan",""}: return f"➖ {label}"
    return f"ℹ️ {label}: {x}"

def _comma_join(x):
    if isinstance(x, list): return ", ".join([str(t) for t in x if str(t).strip()]) or "—"
    if isinstance(x, str) and x.strip(): return x
    return "—"


# --------------------------
# Bootstrap (RAG)
# --------------------------
@st.cache_resource(show_spinner=True)
def bootstrap_rag_system():
    """Initialize RAG system and chatbot pipeline (HF NER is inside ChatbotPipeline)."""
    try:
        rag = ProposedRAGManager()
        docs_dir = os.path.join(project_root, "documents")
        if os.path.exists(docs_dir):
            rag.add_directory(docs_dir)
        bot = ChatbotPipeline(rag)  # exposes: intent_clf, ner_extractor, session, handle_message()
        return rag, bot
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {e}")
        return None, None


# --------------------------
# Bootstrap (Search)
# --------------------------
@st.cache_resource(show_spinner=True)
def bootstrap_azure_components():
    """
    Initialize hybrid retrieval components.
    NOTE: We are NOT downloading any NER from Azure. NER comes from ChatbotPipeline (HF).
    """
    try:
        cfg = get_blob_settings()
        conn = cfg["connection_string"]

        with st.spinner("Downloading Matching/Ranking model..."):
            download_prefix_flat(conn, cfg["ml_container"], cfg["mr_prefix"], local_mr_dir())
        with st.spinner("Downloading pet CSV..."):
            smart_download_single_blob(conn, cfg["pets_container"], cfg["pets_csv_blob"], local_pets_csv_path())

        # Load MR student + FAISS
        student, doc_ids, doc_vecs = load_mr_model(local_mr_dir())
        faiss_index = load_faiss_index(local_mr_dir(), dim=doc_vecs.shape[1])

        # Load pets
        dfp = pd.read_csv(local_pets_csv_path())
        # Normalize some columns to lower for matching
        for c in ("animal","gender","state","breed","size","fur_length","condition","color"):
            if c in dfp.columns:
                dfp[c] = dfp[c].astype(str).fillna("").str.strip()

        # BM25 (on description-ish column)
        text_col = "doc" if "doc" in dfp.columns else ("description_clean" if "description_clean" in dfp.columns else "description")
        docs_raw = {int(i): only_text(str(t)) for i, t in zip(dfp.index, dfp[text_col].fillna("").tolist())}
        bm25 = BM25().fit(docs_raw)

        # Breed catalog + animal mapping
        breed_catalog = sorted(set([b for b in dfp.get("breed", pd.Series([], dtype=str)).astype(str).str.lower().tolist() if b]))
        breed_to_animal = {}
        if "breed" in dfp.columns and "animal" in dfp.columns:
            tmp = (dfp[["breed","animal"]]
                   .dropna()
                   .groupby("breed")["animal"].agg(lambda s: s.value_counts().idxmax()))
            breed_to_animal = tmp.to_dict()

        return {
            "cfg": cfg, "student": student, "doc_ids": doc_ids, "doc_vecs": doc_vecs,
            "faiss_index": faiss_index, "dfp": dfp, "bm25": bm25,
            "breed_catalog": breed_catalog, "breed_to_animal": breed_to_animal
        }
    except Exception as e:
        st.error(f"Failed to initialize search components: {e}")
        return None


# --------------------------
# Facet helpers (from bot.session entities)
# --------------------------
def _entities_to_facets(ents: Dict[str, str]) -> Dict[str, Any]:
    # Map ChatbotPipeline entity keys to our dataframe columns/filters
    # Supported keys from pipeline: PET_TYPE, STATE, BREED, COLOR, SIZE, GENDER, AGE, FURLENGTH
    facets = {}
    if "PET_TYPE" in ents:
        # Normalize to title-cased dataset values: Dog / Cat
        pt = ents["PET_TYPE"].strip().lower()
        facets["animal"] = "Dog" if pt == "dog" else ("Cat" if pt == "cat" else None)
    if "STATE" in ents:
        facets["state"] = ents["STATE"].title()
    if "BREED" in ents:
        facets["breed"] = ents["BREED"]
    if "COLOR" in ents:
        facets["color"] = ents["COLOR"]
    if "SIZE" in ents:
        facets["size"] = ents["SIZE"].title()
    if "GENDER" in ents:
        g = ents["GENDER"].strip().lower()
        facets["gender"] = "Male" if g.startswith("m") else ("Female" if g.startswith("f") else ents["GENDER"])
    if "FURLENGTH" in ents:
        facets["fur_length"] = ents["FURLENGTH"].title()
    # AGE: keep as free text hint (we don't have full age groups here; can parse simple hints)
    if "AGE" in ents:
        facets["age_hint"] = ents["AGE"].lower()
    return facets


def _matches_all_facets(row: pd.Series, facets: Dict[str, Any]) -> bool:
    ok = True
    if "animal" in facets and facets["animal"]:
        ok &= str(row.get("animal","")).strip().lower() == facets["animal"].strip().lower()
    if "state" in facets and facets["state"]:
        ok &= str(row.get("state","")).strip().lower() == facets["state"].strip().lower()
    if "gender" in facets and facets["gender"]:
        ok &= str(row.get("gender","")).strip().lower() == facets["gender"].strip().lower()
    if "size" in facets and facets["size"]:
        ok &= str(row.get("size","")).strip().lower() == facets["size"].strip().lower()
    if "fur_length" in facets and facets["fur_length"]:
        ok &= str(row.get("fur_length","")).strip().lower() == facets["fur_length"].strip().lower()
    if "color" in facets and facets["color"]:
        ok &= bool(re.search(rf"\b{re.escape(facets['color'])}\b", str(row.get("color","")), flags=re.IGNORECASE))
    if "breed" in facets and facets["breed"]:
        ok &= bool(re.search(rf"\b{re.escape(facets['breed'])}\b", str(row.get("breed","")), flags=re.IGNORECASE))
    # AGE hint: very light check for words like puppy/kitten/young/adult/senior in description/condition
    if "age_hint" in facets and facets["age_hint"]:
        hint = facets["age_hint"]
        desc = f"{row.get('description_clean','')} {row.get('condition','')}".lower()
        if any(tok in hint for tok in ["puppy","kitten"]):
            ok &= bool(re.search(r"\b(puppy|kitten)\b", desc))
        # otherwise, keep non-blocking
    return bool(ok)


# --------------------------
# Hybrid pet search
# --------------------------
def hybrid_pet_search(query: str,
                      env: Dict[str, Any],
                      facets: Dict[str, Any],
                      topk_cards: int = 6) -> Tuple[pd.DataFrame, Dict[str, Any], np.ndarray]:
    """
    Run BM25 + Embeddings hybrid search, then enforce facets as hard filters,
    then return up to 6 cards. Also compute highlight mask (green if exact match).
    """
    student = env["student"]; doc_ids = env["doc_ids"]; doc_vecs = env["doc_vecs"]
    faiss_index = env["faiss_index"]; dfp = env["dfp"]; bm25 = env["bm25"]

    # Start with a shallow copy we can filter
    df_filter = dfp.copy()

    # Hard filters for provided facets
    if facets.get("animal"):
        df_filter = df_filter[df_filter["animal"].str.lower() == facets["animal"].strip().lower()]
    if facets.get("state"):
        df_filter = df_filter[df_filter["state"].str.lower() == facets["state"].strip().lower()]
    if facets.get("gender"):
        df_filter = df_filter[df_filter["gender"].str.lower() == facets["gender"].strip().lower()]
    if facets.get("size"):
        df_filter = df_filter[df_filter["size"].str.lower() == facets["size"].strip().lower()]
    if facets.get("fur_length"):
        df_filter = df_filter[df_filter["fur_length"].str.lower() == facets["fur_length"].strip().lower()]
    if facets.get("color"):
        df_filter = df_filter[df_filter["color"].str.contains(rf"\b{re.escape(facets['color'])}\b", case=False, na=False)]
    if facets.get("breed"):
        # fuzzy contains (token-based)
        pat = rf"\b{re.escape(facets['breed'])}\b"
        df_filter = df_filter[df_filter["breed"].str.contains(pat, case=False, na=False)]

    if df_filter.empty:
        return pd.DataFrame(), facets, np.array([], dtype=bool)

    # Hybrid retrieval: boosted free-text query + facets stitched in
    facet_bits = []
    for k in ["animal","breed","gender","color","size","fur_length","state"]:
        if facets.get(k): facet_bits.append(str(facets[k]))
    boost_q = (query or "").strip()
    if facet_bits:
        boost_q = (boost_q + " " + " ".join(facet_bits)).strip()

    # BM25
    lex_all = bm25.search(only_text(boost_q), topk=2000)
    slex = {int(idx): float(s) for idx, s in lex_all if idx in df_filter.index}

    # Embeddings
    emb_all = emb_search(boost_q, student, doc_ids, doc_vecs, pool_topn=200, faiss_index=faiss_index)
    semb = {int(pid): float(s) for pid, s in emb_all if pid in df_filter.index}

    # Min-max per channel
    def _mm(d):
        if not d: return {}
        vals = np.fromiter(d.values(), dtype=float)
        lo, hi = float(vals.min()), float(vals.max())
        den = (hi - lo) or 1.0
        return {k: (v - lo) / den for k, v in d.items()}

    nlex, nemb = _mm(slex), _mm(semb)
    combo = {idx: 0.1*nlex.get(idx, 0.0) + 0.9*nemb.get(idx, 0.0) for idx in set(nlex) | set(nemb)}

    if not combo:
        return pd.DataFrame(), facets, np.array([], dtype=bool)

    # sort by combo score
    ranked = sorted(combo.items(), key=lambda x: x[1], reverse=True)
    top_idx = [i for i, _ in ranked[:max(200, topk_cards)] if i in df_filter.index]

    if not top_idx:
        return pd.DataFrame(), facets, np.array([], dtype=bool)

    # Build display df (limit to 6)
    display_cols = [
        "name","animal","breed","gender","state","color",
        "size","fur_length","condition","age_months",
        "description_clean","url","photo_links","video_links"
    ]
    display_cols = [c for c in display_cols if c in df_filter.columns]
    chosen = top_idx[:topk_cards]
    res_df = df_filter.loc[chosen, display_cols].copy().reset_index(drop=True)

    # Highlight mask: meets ALL provided facets
    mask = res_df.apply(lambda r: _matches_all_facets(r, facets), axis=1).to_numpy(dtype=bool)
    return res_df, facets, mask


# --------------------------
# Card rendering (fixed-size photo + green highlight)
# --------------------------
def render_pet_card(row: pd.Series, highlight: bool = False):
    name = str(row.get("name") or "Pet")
    url  = str(row.get("url") or "")
    animal = (row.get("animal") or "").title()
    breed  = str(row.get("breed") or "—")
    gender = (row.get("gender") or "—").title()
    state  = (row.get("state") or "—").title()
    color  = str(row.get("color") or "—")
    age_mo = row.get("age_months")
    age_txt = _age_text_from_months(age_mo)
    size = str(row.get("size") or "—").title()
    fur  = str(row.get("fur_length") or "—").title()
    cond = str(row.get("condition") or "—").title()

    vacc = _badge_bool(row.get("vaccinated"), "vaccinated")
    dewm = _badge_bool(row.get("dewormed"), "dewormed")
    neut = _badge_bool(row.get("neutered"), "neutered")
    spay = _badge_bool(row.get("spayed"), "spayed")

    img_url = _first_photo_url_from_row(row)

    bg = "#E8F7E1" if highlight else "#FFFFFF"
    border_left = "5px solid #22c55e" if highlight else "5px solid #ff6b9d"

    st.markdown(f"""
<div style="background:{bg}; border-radius:15px; padding:14px; margin:8px 0; border-left:{border_left}; box-shadow:0 4px 15px rgba(0,0,0,0.08);">
  <div style="font-size:1.1rem; font-weight:800; color:#2563EB; margin-bottom:6px;">
    {"<a href='"+url+"' target='_blank' style='text-decoration:none; color:#2563EB;'>🔗 " + name + "</a>" if url else name}
  </div>

  <div style="height:220px;width:100%;border-radius:10px;background:#F3F4F6;display:flex;align-items:center;justify-content:center;margin:8px 0 10px;overflow:hidden;border:1px solid #E5E7EB;">
    { (f"<img src='{img_url}' alt='photo' loading='lazy' style='max-width:100%;max-height:100%;width:auto;height:auto;object-fit:contain;object-position:center;display:block;' />") if img_url else "<div style='color:#6B7280;'>No photo available</div>" }
  </div>

  <div style="color:#374151;margin-bottom:6px;">
    <strong>{animal}</strong> • <strong>Breed:</strong> {breed} • <strong>Gender:</strong> {gender} • <strong>Age:</strong> {age_txt} • <strong>State:</strong> {state}
  </div>
  <div style="color:#374151;margin-bottom:8px;">
    <strong>Color:</strong> {color} • <strong>Size:</strong> {size} • <strong>Fur:</strong> {fur} • <strong>Condition:</strong> {cond}
  </div>
  <div style="color:#111827;">{" | ".join([vacc, dewm, neut, spay])}</div>

  <!-- Collapsible description -->
  {"".join([
      "<div style='margin-top:10px;border-top:1px solid #E5E7EB;padding-top:8px;'>",
      "<details style='cursor:pointer;'><summary style='color:#374151;font-weight:600;list-style:none;display:inline-block;'>▶ Show description</summary>",
      f"<div style='margin-top:8px;color:#374151;line-height:1.55;'>{(str(row.get('description_clean') or '').replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')).replace('\\n','<br />')}</div>",
      "</details></div>"
  ]) if str(row.get("description_clean") or "").strip() else "" }
</div>
""", unsafe_allow_html=True)


def render_grid(df: pd.DataFrame, mask: np.ndarray, max_cols: int = 3):
    if df is None or df.empty:
        st.warning("No results to show.")
        return
    rows = [r for _, r in df.iterrows()]
    flags = mask.tolist()
    n = len(rows)
    cols = max(1, min(max_cols, n))
    for i in range(0, n, cols):
        cset = st.columns(cols, gap="medium")
        for j, col in enumerate(cset):
            idx = i + j
            if idx >= n: break
            with col:
                render_pet_card(rows[idx], highlight=bool(flags[idx]))


# --------------------------
# UI (pink theme)
# --------------------------
PINK_CSS = """
<style>
.stApp {
    background: linear-gradient(135deg, #ffb6c1 0%, #ffc0cb 50%, #ffd1dc 100%);
    background-attachment: fixed;
}
.main-header {
    text-align: center; padding: 2rem 0;
    background: linear-gradient(45deg, #ff6b9d, #ff8fab);
    border-radius: 20px; margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(255, 107, 157, 0.3);
}
.main-header h1 { color: white; font-size: 3rem; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }
.main-header p  { color: white; font-size: 1.2rem; margin: .5rem 0 0 0; opacity: .9; }
.status-bar {
    background: linear-gradient(135deg, #ff6b9d, #ff8fab);
    color: white; padding: 1rem 2rem; margin: -1rem -1rem 2rem -1rem;
    border-radius: 0 0 20px 20px; box-shadow: 0 4px 20px rgba(255, 107, 157, 0.3);
}
.status-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem; align-items: center;
}
.status-item { background: rgba(255,255,255,0.2); padding:.8rem; border-radius:10px; text-align:center; backdrop-filter: blur(10px); border:1px solid rgba(255,255,255,0.3); }
.status-item.success { background: rgba(76,175,80,0.3); border-color:rgba(76,175,80,0.5); }
.status-item.warning { background: rgba(255,152,0,0.3); border-color:rgba(255,152,0,0.5); }
.status-item.error   { background: rgba(244,67,54,0.3); border-color:rgba(244,67,54,0.5); }
.status-icon { font-size:1.5rem; margin-bottom:.5rem; display:block; }
.status-text { font-weight:600; margin-bottom:.3rem; }
.status-detail { font-size:.9rem; opacity:.9; }
</style>
"""


# --------------------------
# Main App
# --------------------------
def main():
    st.markdown(PINK_CSS, unsafe_allow_html=True)
    st.markdown("""
    <div class="main-header">
        <h1>🐾 Pawfect Match</h1>
        <p>Your Intelligent Pet Assistant - Ask about pet care or find your pawfect pet</p>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("🚀 Initializing systems..."):
        rag, bot = bootstrap_rag_system()
        env = bootstrap_azure_components()

    # Status header
    rag_ok = rag is not None and bot is not None
    env_ok = env is not None and env.get("dfp") is not None
    overall_ok = rag_ok

    def badge(status): return "success" if status else "error"
    def icon(status): return "✅" if status else "❌"

    st.markdown(f"""
    <div class="status-bar">
        <div class="status-grid">
            <div class="status-item {badge(overall_ok)}">
                <span class="status-icon">{icon(overall_ok)}</span>
                <div class="status-text">App Status</div>
                <div class="status-detail">{'All Systems Ready' if overall_ok else 'Issues Detected'}</div>
            </div>
            <div class="status-item {badge(rag_ok)}">
                <span class="status-icon">{icon(rag_ok)}</span>
                <div class="status-text">RAG / Chatbot</div>
                <div class="status-detail">{'Online' if rag_ok else 'Unavailable'}</div>
            </div>
            <div class="status-item {badge(env_ok)}">
                <span class="status-icon">{icon(env_ok)}</span>
                <div class="status-text">Pet Search</div>
                <div class="status-detail">{f"{len(env['dfp'])} pets available" if env_ok else 'Unavailable'}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Show history
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # Example chips
    st.markdown("### 💡 Try asking me:")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🐕 What should I feed my puppy?", use_container_width=True):
            st.session_state.example_prompt = "What should I feed my puppy?"
    with c2:
        if st.button("🏠 Find my pawfect golden retriever", use_container_width=True):
            st.session_state.example_prompt = "I want to adopt a golden retriever in Selangor"
    with c3:
        if st.button("🏥 My cat is sick, what should I do?", use_container_width=True):
            st.session_state.example_prompt = "My cat is sick, what should I do?"

    prompt = st.chat_input("Ask me anything about pets...")

    if hasattr(st.session_state, "example_prompt"):
        prompt = st.session_state.example_prompt
        delattr(st.session_state, "example_prompt")

    if not prompt:
        return

    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- Core routing via YOUR intent model inside ChatbotPipeline ---
    # We first let the pipeline process the input; it:
    #  - runs IntentClassifier
    #  - runs HF NER
    #  - manages entities in bot.session
    #  - returns a conversational message (may include "Got it! Searching ...")
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                reply = bot.handle_message(prompt)  # uses intent model + HF NER  :contentReference[oaicite:4]{index=4}
            except Exception as e:
                reply = f"(Chat pipeline error: {e})"

            # Always show pipeline's conversational reply
            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})

            # If the pipeline decided this is pet search AND it says "Searching...", we run our hybrid search
            intent_now = bot.session.get("intent")
            if env_ok and intent_now == "find_pet" and re.search(r"\bSearching for\b", reply, flags=re.IGNORECASE):
                # Build facets from pipeline session entities
                ents = bot.session.get("entities", {})
                facets = _entities_to_facets(ents)

                # Show "facets used" banner
                parts = []
                for k in ["animal","breed","gender","color","size","fur_length","state"]:
                    if facets.get(k):
                        label = k.replace("_", " ").title()
                        parts.append(f"<span style='background:#fff;border:1px solid #eee;border-radius:8px;padding:4px 8px;margin:2px;display:inline-block;'><strong>{label}:</strong> {facets[k]}</span>")
                if parts:
                    st.markdown(
                        "<div style='margin:10px 0 4px;color:#374151;'>Facets used:</div>"
                        + "<div style='margin-bottom:10px;'>" + " ".join(parts) + "</div>",
                        unsafe_allow_html=True
                    )

                # Hybrid search (limit to 6 cards)
                results_df, used_facets, highlight_mask = hybrid_pet_search(
                    query=prompt,
                    env=env,
                    facets=facets,
                    topk_cards=6
                )

                if results_df is None or results_df.empty:
                    st.info("No pets found. Try adding/adjusting breed, color, or removing constraints.")
                else:
                    render_grid(results_df, highlight_mask, max_cols=3)

            # For pet care, the pipeline already called RAG internally and returned the text answer


if __name__ == "__main__":
    main()
