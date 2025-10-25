# -*- coding: utf-8 -*-
"""
Unified PetBot App — Single Chat Interface with Intent Classification
➡️ Dialogue flow → Intent classification →
   • Adoption intent  → NER + hybrid search → card grid
   • Pet care intent → RAG answer (ProposedRAGManager)

What's NEW
----------
- STRONG hard filters: animal/breed/gender/state (state auto-relaxes only if <6)
- Soft preferences parsed and used for BUCKET-FIRST priority:
  • Age groups (puppy/kitten, young, adult, senior) — supports "<1", "less than 1 year",
    ">1 year", "over 3 years", and exact ages ("2 years") → mapped to groups
  • Vaccinated / dewormed / neutered / spayed / healthy
  • Low adoption fee / fee cap
- Cards turn light-green only if they meet ALL requested strict + soft requirements
- Sidebar removed; top cards fixed to 6; “New search / Clear history” button
- “Searching…” line built from current facets only (no leakage)
- Summary banner uses **exact in-state (strict+soft) matches**; adds conversational recap
- UI update:
  • Name on top, bigger, blue, with 🔗 for clickable URL
  • Collapsible Description box at bottom (from description_clean)
  • Fixed-size photo area; images shrink to fit (no cropping)
"""
import streamlit as st
st.set_page_config(page_title="Unified PetBot", layout="wide")

import os, time, ast, re, json, html
import sys
from typing import List, Dict, Any, Tuple, Optional, Set

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import pandas as pd
import numpy as np

# ==== RAG & Chatbot (dialogue + intent + NER pipe) ====
from rag_system.proposed_rag_system import ProposedRAGManager
from chatbot_flow.chatbot_pipeline import ChatbotPipeline

# ==== Retrieval stack ====
from pet_retrieval.config import get_blob_settings, local_mr_dir, local_pets_csv_path
from pet_retrieval.azure_io import download_prefix_flat, smart_download_single_blob
from pet_retrieval.models import load_mr_model, load_faiss_index
from pet_retrieval.retrieval import (
    only_text, BM25,
    parse_facets_from_text, entity_spans_to_facets, sanitize_facets_ner_light,
    make_boosted_query, emb_search
)

# Optional fuzzy breed mapping
try:
    from rapidfuzz import process, fuzz
    _HAS_FUZZ = True
except Exception:
    _HAS_FUZZ = False
    process = fuzz = None

# -------------------------------------------
# STRICT SETTINGS (no sidebar; we set defaults here)
# -------------------------------------------
STRICT_DEFAULTS = {
    "strict_mode": True,
    "relax_state_if_needed": False,
    "relax_animal_if_needed": False,
    "relax_breed_if_needed": False,
    "relax_colors_if_needed": False,
    "allow_breed_contains": True,  # allow "poodle" to match "terrier + poodle"
}

# Fixed UI/config
EMB_POOL = 200
LEX_POOL = 2000
HYBRID_W = {"lex": 0.1, "emb": 0.9}
AUTO_RELAX_MIN_RESULTS = 6     # if in-state is too few, add cross-state
TOPK_CARDS = 6                 # fixed
GRID_COLS = 3                  # visual grid columns

COLOR_MAP = {"golden": "yellow", "gold": "yellow", "cream": "white", "ginger": "orange"}

# ---- Age groups (months) ----
AGE_GROUPS = {
    "puppy/kitten": (0, 12),          # [0, 12)
    "young": (12, 36),                 # [12, 36)
    "adult": (36, 84),                 # [36, 84)
    "senior": (84, None),              # [84, ∞)
}
AGE_GROUP_KEYS = list(AGE_GROUPS.keys())

# -------------------------------------------
# Helpers
# -------------------------------------------
def normalize_color(c: str) -> str:
    c = (c or "").strip().lower()
    return COLOR_MAP.get(c, c)

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

def parse_colors_cell(x) -> List[str]:
    return [normalize_color(str(t)) for t in _safe_list_from_cell(x) if str(t).strip()]

# --- NER span resolution (keep longest; prefer BREED over COLOR on conflicts) ---
def resolve_overlaps_longest(spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def lab(z: Dict[str, Any]) -> str:
        L = str(z.get("entity_group") or z.get("label") or "").upper()
        L = re.sub(r"^[BI]-", "", L)
        return L
    spans = sorted(spans or [], key=lambda s: (int(s.get("start", 0)), -(int(s.get("end", 0)) - int(s.get("start", 0)))))
    kept: List[Dict[str, Any]] = []
    for s in spans:
        s_start, s_end = int(s.get("start", 0)), int(s.get("end", 0))
        s_lab = lab(s); s_len = s_end - s_start
        drop = False
        for t in list(kept):
            t_start, t_end = int(t.get("start", 0)), int(t.get("end", 0))
            t_lab = lab(t); t_len = t_end - t_start
            overlaps = not (s_end <= t_start or s_start >= t_end)
            if overlaps:
                if (s_lab == "COLOR" and t_lab == "BREED") or (t_len >= s_len):
                    drop = True; break
        if not drop:
            kept = [t for t in kept if not (int(t.get("start", 0)) >= s_start and int(t.get("end", 0)) <= s_end)]
            kept.append(s)
    return kept

def canonicalize_gender(g: str) -> str:
    g = (g or "").strip().lower()
    if g in {"m","male","boy"}: return "male"
    if g in {"f","female","girl"}: return "female"
    return ""

def _age_text_yr_mo(age_months) -> str:
    try:
        m = int(round(float(age_months)))
        if m < 12: return f"{m} mo (puppy/kitten)"
        y, r = divmod(m, 12)
        return f"{y} yr" if r == 0 else f"{y} yr {r} mo"
    except Exception:
        return "—"

# --- Health flags ---
TRUE_STR    = {"true","yes","y","1","full","fully","done","complete","completed"}
FALSE_STR   = {"false","no","n","0","none"}
UNKNOWN_STR = {"unknown","unsure","not sure","n/a","na","nil","-"}

def _coerce_bool(x):
    if x is None: return None
    if isinstance(x, bool): return x
    if isinstance(x, (int, float)):
        if int(x) == 1: return True
        if int(x) == 0: return False
    s = str(x).strip().lower()
    if not s or s == "nan": return None
    if s in TRUE_STR: return True
    if s in FALSE_STR: return False
    if s in UNKNOWN_STR: return None
    if re.search(r"\b(unvaccinated|not\s+vaccinated|no\s+vaccine)\b", s): return False
    if re.search(r"\b(vaccinated|fully\s+vaccinated|vaccination\s+done)\b", s): return True
    if re.search(r"\b(not\s+dewormed|no\s+deworm)\b", s): return False
    if re.search(r"\b(dewormed|de-wormed)\b", s): return True
    if re.search(r"\b(intact|not\s+neuter(?:ed)?|not\s+spay(?:ed)?)\b", s): return False
    if re.search(r"\b(neuter(?:ed)?|castrat(?:e|ed|ion)|fixed|sterilis(?:e|ed|ation)|spay(?:ed)?)\b", s): return True
    return None

def _pick_bool(row: dict, cols: list[str]) -> Optional[bool]:
    for c in cols:
        if c in row and str(row.get(c, "")).strip() != "":
            v = _coerce_bool(row[c])
            if v is not None: return v
    return None

def _extract_health_from_text(row: dict) -> tuple[Optional[bool], Optional[bool], Optional[bool], Optional[bool]]:
    text_cols = ["health", "medical", "condition", "notes", "description_clean", "description"]
    text = " ".join(str(row.get(c, "")) for c in text_cols if c in row).lower()
    v = _coerce_bool("vaccinated" if re.search(r"\b(fully\s+)?vaccinated\b", text) else
                     ("not vaccinated" if re.search(r"\bunvaccinated|not\s+vaccinated|no\s+vaccine\b", text) else None))
    d = _coerce_bool("dewormed" if re.search(r"\bde[-\s]?wormed\b", text) else
                     ("not dewormed" if re.search(r"\bnot\s+dewormed|no\s+deworm\b", text) else None))
    neut_true = bool(re.search(r"\b(neuter(?:ed)?|castrat(?:e|ed|ion)|fixed|sterilis(?:e|ed|ation))\b", text))
    spay_true = bool(re.search(r"\bspay(?:ed)?\b", text))
    n = True if neut_true else None
    s = True if spay_true else None
    return v, d, n, s

def _resolve_health_flags(row: pd.Series) -> tuple[Optional[bool], Optional[bool], Optional[bool], Optional[bool]]:
    r = row.to_dict()
    v = _pick_bool(r, ["vaccinated", "is_vaccinated", "vaccination", "vaccinations", "vaccine", "vacc_status"])
    d = _pick_bool(r, ["dewormed", "is_dewormed", "deworming", "wormed"])
    n = _pick_bool(r, ["neutered", "is_neutered", "neuter", "fixed", "castrated"])
    s = _pick_bool(r, ["spayed", "is_spayed", "spay", "sterilized", "sterilised"])
    tv, td, tn, ts = _extract_health_from_text(r)
    if v is None: v = tv
    if d is None: d = td
    if n is None: n = tn
    if s is None: s = ts
    gender = str(r.get("gender", "")).strip().lower()
    if s is None and (gender in {"female","f"}) and (n is True): s = True
    if n is None and (gender in {"male","m"}) and (s is True): n = True
    return v, d, n, s

def _badge(val: Optional[bool], label: str) -> str:
    if val is True:  return f"✅ {label}"
    if val is False: return f"❌ {label}"
    return f"➖ {label}"

# ---------- Cards UI (FULL highlight via single HTML block) ----------
def _safe_colors_text(colors_val) -> str:
    txt = ", ".join(colors_val) if isinstance(colors_val, (list, tuple)) else str(colors_val or "").strip()
    if txt.lower() in {"unknown","nan","none",""}: return "—"
    return txt

def _first_photo_url(row) -> Optional[str]:
    photos = row.get("photo_links")
    if not isinstance(photos, list):
        photos = _safe_list_from_cell(photos)
    if photos:
        url = str(photos[0]).strip().strip('"').strip("'")
        return url if url else None
    return None

def _escape_desc(text: str) -> str:
    if not text:
        return ""
    return html.escape(str(text)).replace("\n", "<br />")

def render_pet_card(row: pd.Series, highlight: bool = False):
    """Render the entire card (image + text) inside a single HTML block so background covers all."""
    pid = int(row.get("pet_id")) if "pet_id" in row else int(row.name)
    name = str(row.get("name") or f"Pet {pid}")
    url  = str(row.get("url") or "")
    animal = (row.get("animal") or "").title()
    breed  = str(row.get("breed") or "—")
    gender = (row.get("gender") or "—").title()
    state  = (row.get("state") or "—").title()
    color  = str(row.get("color") or "—")
    colors_canon = row.get("colors_canonical")
    colors_txt = _safe_colors_text(colors_canon if isinstance(colors_canon, list) else color)
    age_mo = row.get("age_months"); age_yrs_txt = _age_text_yr_mo(age_mo)
    size = str(row.get("size") or "—").title()
    fur  = str(row.get("fur_length") or "—").title()
    cond = str(row.get("condition") or "—").title()
    desc = str(row.get("description_clean") or "").strip()

    v, d, n, s = _resolve_health_flags(row)
    img_url = _first_photo_url(row)

    bg = "#E8F7E1" if highlight else "#FFFFFF"

    badges = " | ".join([
        _badge(v, "vaccinated"),
        _badge(d, "dewormed"),
        _badge(n, "neutered"),
        _badge(s, "spayed"),
    ])

    # Title (blue, large, with 🔗 if clickable) — placed at the TOP
    if url:
        title_html = f'<a href="{url}" target="_blank" style="text-decoration:none;color:#2563EB;"><span style="font-weight:800;">🔗 {html.escape(name)}</span></a>'
    else:
        title_html = f'<span style="color:#1F2937;font-weight:800;">{html.escape(name)}</span>'

    # Fixed-size image area: 220px tall, shrink-to-fit (no cropping)
    if img_url:
        img_html = f'''
<div style="height:220px;width:100%;border-radius:10px;background:#F3F4F6;display:flex;align-items:center;justify-content:center;margin-top:8px;margin-bottom:10px;overflow:hidden;border:1px solid #E5E7EB;">
  <img src="{img_url}" alt="photo" loading="lazy"
       style="max-width:100%;max-height:100%;width:auto;height:auto;object-fit:contain;object-position:center;display:block;" />
</div>'''
    else:
        img_html = '''
<div style="height:220px;width:100%;border-radius:10px;background:#F9FAFB;display:flex;align-items:center;justify-content:center;margin-top:8px;margin-bottom:10px;overflow:hidden;border:1px dashed #D1D5DB;color:#6B7280;">
  No photo available
</div>'''

    # Collapsible description (hidden first)
    desc_html = _escape_desc(desc)
    details_html = ""
    if desc_html:
        details_html = f"""
<div style="margin-top:10px;border-top:1px solid #E5E7EB;padding-top:8px;">
  <details style="cursor:pointer;">
    <summary style="color:#374151;font-weight:600;list-style:none;display:inline-block;">
      ▶ Show description
    </summary>
    <div style="margin-top:8px;color:#374151;line-height:1.55;">
      {desc_html}
    </div>
  </details>
</div>
"""

    # Compose single card HTML so the background applies to everything
    html_card = f"""
<div style="
  background:{bg};
  border:1px solid #E5E7EB;
  border-radius:14px;
  padding:14px;
  box-shadow: 0 1px 2px rgba(0,0,0,0.04);
">
  <div style="font-size:1.25rem;line-height:1.2;margin-bottom:2px;">{title_html}</div>
  {img_html}
  <div style="color:#374151;margin-bottom:6px;">
    <strong>{animal}</strong> • <strong>Breed:</strong> {html.escape(breed)} • <strong>Gender:</strong> {html.escape(gender)} •
    <strong>Age:</strong> {html.escape(age_yrs_txt)} • <strong>State:</strong> {html.escape(state)}
  </div>
  <div style="color:#374151;margin-bottom:6px;">
    <strong>Color(s):</strong> {html.escape(colors_txt)} • <strong>Size:</strong> {html.escape(size)} • <strong>Fur:</strong> {html.escape(fur)} • <strong>Condition:</strong> {html.escape(cond)}
  </div>
  <div style="color:#111827;">{badges}</div>
  {details_html}
</div>
"""
    st.markdown(html_card, unsafe_allow_html=True)

def render_results_grid(res_df: pd.DataFrame, meet_all_mask: pd.Series, max_cols: int = GRID_COLS):
    if res_df is None or res_df.empty:
        st.warning("No results to show.")
        return
    rows = [r for _, r in res_df.iterrows()]
    flags = [bool(x) for x in meet_all_mask.tolist()]
    n = len(rows)
    col_count = max(1, min(max_cols, n))
    for i in range(0, n, col_count):
        cols = st.columns(col_count, gap="medium")
        for j, col in enumerate(cols):
            idx = i + j
            if idx >= n: continue
            with col:
                render_pet_card(rows[idx], highlight=flags[idx])

# -------------------------------------------
# State detection
# -------------------------------------------
MALAYSIA_STATES = {
    "johor","kedah","kelantan","malacca","melaka","negeri sembilan","pahang",
    "penang","pulau pinang","perak","perlis","sabah","sarawak","selangor",
    "terengganu","kuala lumpur","labuan","putrajaya"
}
STATE_ALIASES = {"kl": "kuala lumpur", "pulau pinang": "penang", "melaka": "malacca", "kuala lumpur": "kuala lumpur"}

def detect_state(text: str) -> Optional[str]:
    t = (text or "").lower()
    for s in sorted(MALAYSIA_STATES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(s)}\b", t): return STATE_ALIASES.get(s, s)
    for k, v in STATE_ALIASES.items():
        if re.search(rf"\b{re.escape(k)}\b", t): return v
    return None

# -------------------------------------------
# Breed mapping
# -------------------------------------------
BREED_ALIASES = {
    "husky": ["siberian husky", "alaskan husky"],
    "gsd": ["german shepherd", "german shepherd dog"],
    "gr": ["golden retriever"],
    "grd": ["golden retriever"],
}

def map_breed_to_catalog(breed_text: Optional[str], catalog_breeds: List[str], *, strict: bool, min_score: int = 87) -> Optional[str]:
    if not breed_text: return None
    b = str(breed_text).strip().lower().rstrip(",.;:")
    if not b: return None
    if b in catalog_breeds: return b
    if b in BREED_ALIASES:
        for cand in BREED_ALIASES[b]:
            c = cand.lower()
            if c in catalog_breeds: return c
    if not _HAS_FUZZ or not catalog_breeds:
        return None if strict else b
    thresh = 95 if strict else min_score
    cand, score, _ = process.extractOne(b, catalog_breeds, scorer=fuzz.token_sort_ratio)
    if score >= thresh and cand in catalog_breeds: return cand
    return None if strict else b

# -------------------------------------------
# Age group parsing (from free text)
# -------------------------------------------
def _val_to_months(val: float, unit: str) -> float:
    unit = unit.lower()
    if unit.startswith("y"):   # year
        return 12.0 * val
    return val  # months

def _groups_for_threshold(op: str, months: float) -> Set[str]:
    groups = set()
    for g, (lo, hi) in AGE_GROUPS.items():
        lo_m = float(lo if lo is not None else -1e9)
        hi_m = float(hi if hi is not None else 1e9)
        if op in ("<", "lt", "less"):
            if lo_m < months: groups.add(g)
        elif op in ("<=", "le"):
            if lo_m <= months: groups.add(g)
        elif op in (">", "gt", "more"):
            if hi is None or hi_m > months: groups.add(g)
        elif op in (">=", "ge", "atleast"):
            if hi is None or hi_m >= months: groups.add(g)
    if op in ("<","lt","less") and months <= 12: return {"puppy/kitten"}
    if op in (">",">=","gt","ge","more","atleast") and 12 <= months < 36: return {"young","adult","senior"}
    if op in (">",">=","gt","ge","more","atleast") and 36 <= months < 84: return {"adult","senior"}
    if op in (">",">=","gt","ge","more","atleast") and months >= 84: return {"senior"}
    if op in ("<","<=","lt","le","less") and months == 84: return {"puppy/kitten","young","adult"}
    return groups if groups else set(AGE_GROUP_KEYS)

def _group_for_exact_months(months: float) -> str:
    for g, (lo, hi) in AGE_GROUPS.items():
        lo_m = lo if lo is not None else -1e9
        hi_m = hi if hi is not None else 1e9
        if months >= lo_m and months < hi_m:
            return g
    return "adult"

def parse_age_group_prefs(text: str) -> Set[str]:
    t = only_text(text or "").lower()
    prefs: Set[str] = set()
    if re.search(r"\b(puppy|kitten|puppies|kittens)\b", t): prefs.add("puppy/kitten")
    if re.search(r"\byoung\b", t): prefs.add("young")
    if re.search(r"\badult\b", t): prefs.add("adult")
    if re.search(r"\bsenior\b", t): prefs.add("senior")
    comp_patterns = [
        r"(<=|>=|<|>)\s*(\d+(?:\.\d+)?)\s*(years?|yrs?|y|months?|mos?)",
        r"\b(less\s+than|under)\s*(\d+(?:\.\d+)?)\s*(years?|yrs?|y|months?|mos?)",
        r"\b(more\s+than|over|at\s+least)\s*(\d+(?:\.\d+)?)\s*(years?|yrs?|y|months?|mos?)",
    ]
    for pat in comp_patterns:
        for m in re.finditer(pat, t):
            op_raw = m.group(1)
            val = float(m.group(2))
            unit = m.group(3)
            months = _val_to_months(val, unit)
            op = op_raw.strip()
            if op in {"less than","under"}: op = "less"
            if op in {"more than","over"}: op = "more"
            if op in {"at least"}: op = "atleast"
            prefs |= _groups_for_threshold(op, months)
    if not prefs:
        m_exact = re.search(r"\b(\d+(?:\.\d+)?)\s*(years?|yrs?|y|months?|mos?)\b", t)
        if m_exact:
            months = _val_to_months(float(m_exact.group(1)), m_exact.group(2))
            prefs.add(_group_for_exact_months(months))
    return prefs

# -------------------------------------------
# Soft prefs parser (non-age + age groups)
# -------------------------------------------
def parse_soft_prefs(text: str) -> Dict[str, Any]:
    t = only_text(text or "").lower()
    prefer_vaccinated = bool(re.search(r"\b(vaccinated|fully\s+vaccinated)\b", t))
    prefer_dewormed   = bool(re.search(r"\b(de[-\s]?wormed)\b", t))
    prefer_neutered   = bool(re.search(r"\b(neuter(?:ed)?|fixed|castrat(?:e|ed|ion))\b", t))
    prefer_spayed     = bool(re.search(r"\bspay(?:ed)?\b", t))
    prefer_healthy    = bool(re.search(r"\b(healthy|good\s+condition|good\s+health)\b", t))
    prefer_low_fee    = bool(re.search(r"\b(low|cheap|budget|afford|under\s*\d+|<\s*\d+)\b.*\b(fee|adoption)\b", t)) \
                        or bool(re.search(r"\b(adoption\s+fee)\b.*\b(low|cheap|budget|under|<)\b", t))
    fee_cap = None
    m_fee = re.search(r"(?:fee|adoption\s+fee)\s*(?:under|<|<=)?\s*(\d{2,5})", t)
    if m_fee:
        try: fee_cap = float(m_fee.group(1))
        except Exception: fee_cap = None
    age_groups_pref = parse_age_group_prefs(t)
    return {
        "prefer_vaccinated": prefer_vaccinated,
        "prefer_dewormed": prefer_dewormed,
        "prefer_neutered": prefer_neutered,
        "prefer_spayed": prefer_spayed,
        "prefer_healthy": prefer_healthy,
        "prefer_low_fee": prefer_low_fee,
        "fee_cap": fee_cap,
        "age_groups_pref": sorted(age_groups_pref) if age_groups_pref else [],
    }

# -------------------------------------------
# BOOTSTRAP
# -------------------------------------------
@st.cache_resource(show_spinner=True)
def bootstrap_all():
    rag = ProposedRAGManager()
    documents_dir = os.path.join(project_root, "documents")
    if os.path.exists(documents_dir): rag.add_directory(documents_dir)
    bot = ChatbotPipeline(rag)

    ner_pipe = None
    for attr in ("ner_extractor", "entity_extractor", "ner"):
        if hasattr(bot, attr) and hasattr(getattr(bot, attr), "ner_pipe"):
            ner_pipe = getattr(getattr(bot, attr), "ner_pipe"); break
    if ner_pipe is None and hasattr(bot, "ner_pipe"):
        ner_pipe = getattr(bot, "ner_pipe")
    if ner_pipe is None:
        raise AttributeError("ChatbotPipeline does not expose a NER pipeline.")

    cfg = get_blob_settings()
    conn = cfg["connection_string"]
    with st.spinner("Downloading Matching/Ranking model (flat folder)..."):
        download_prefix_flat(conn, cfg["ml_container"], cfg["mr_prefix"], local_mr_dir())
    with st.spinner("Downloading pet CSV..."):
        smart_download_single_blob(conn, cfg["pets_container"], cfg["pets_csv_blob"], local_pets_csv_path())

    student, doc_ids, doc_vecs = load_mr_model(local_mr_dir())
    faiss_index = load_faiss_index(local_mr_dir(), dim=doc_vecs.shape[1])

    dfp = pd.read_csv(local_pets_csv_path())

    for c in ("animal", "gender", "state", "breed", "size", "fur_length", "condition"):
        if c in dfp.columns:
            dfp[c] = dfp[c].astype(str).fillna("").str.strip().str.lower()

    if "colors_canonical" in dfp.columns:
        dfp["colors_canonical"] = dfp["colors_canonical"].apply(parse_colors_cell)
    for media_col in ["photo_links", "video_links"]:
        if media_col in dfp.columns:
            dfp[media_col] = dfp[media_col].apply(_safe_list_from_cell)

    breed_to_animal = {}
    if "breed" in dfp.columns and "animal" in dfp.columns:
        tmp = (dfp[["breed","animal"]]
               .dropna()
               .groupby("breed")["animal"].agg(lambda s: s.value_counts().idxmax()))
        breed_to_animal = tmp.to_dict()

    text_col = "doc" if "doc" in dfp.columns else ("description_clean" if "description_clean" in dfp.columns else "description")
    docs_raw = {int(i): only_text(str(t)) for i, t in zip(dfp.index, dfp[text_col].fillna("").tolist())}
    bm25 = BM25().fit(docs_raw)

    breed_catalog = sorted(set([b for b in dfp.get("breed", pd.Series([], dtype=str)).astype(str).str.lower().tolist() if b]))

    return {
        "rag": rag, "bot": bot, "ner": ner_pipe,
        "student": student, "doc_ids": doc_ids, "doc_vecs": doc_vecs, "faiss_index": faiss_index,
        "dfp": dfp, "bm25": bm25, "breed_catalog": breed_catalog, "breed_to_animal": breed_to_animal, "cfg": cfg,
    }

ENV = bootstrap_all()
rag = ENV["rag"]; bot = ENV["bot"]; ner = ENV["ner"]
student = ENV["student"]; doc_ids = ENV["doc_ids"]; doc_vecs = ENV["doc_vecs"]
faiss_index = ENV["faiss_index"]; dfp = ENV["dfp"]; bm25 = ENV["bm25"]
breed_catalog = ENV["breed_catalog"]; breed_to_animal = ENV["breed_to_animal"]

# -------------------------------------------
# PET SEARCH PIPELINE
# -------------------------------------------
def expand_kid_friendly(q: str) -> str:
    t = q.lower()
    if "kid" in t or "child" in t or "family" in t:
        extras = " kid-friendly child-friendly family-friendly gentle with kids good with children good with kids family dog"
        return q + " " + extras
    return q

def _minmax(d: Dict[int, float]) -> Dict[int, float]:
    if not d: return {}
    vals = np.fromiter(d.values(), dtype=float)
    lo, hi = float(vals.min()), float(vals.max())
    den = (hi - lo) or 1.0
    return {k: (v - lo) / den for k, v in d.items()}

def _in_age_group(age_months: Optional[float], group: str) -> bool:
    if age_months is None: return False
    try: m = float(age_months)
    except Exception: return False
    lo, hi = AGE_GROUPS[group]
    lo_m = lo if lo is not None else -1e9
    hi_m = hi if hi is not None else 1e9
    return (m >= lo_m) and (m < hi_m)

def _age_groups_match(row: pd.Series, groups: List[str]) -> bool:
    if not groups: return False
    age_mo = row.get("age_months")
    return any(_in_age_group(age_mo, g) for g in groups)

def _health_fee_priority(row: pd.Series, soft_prefs: Dict[str, Any]) -> Dict[str, int]:
    flags = {"vaccinated": 0, "dewormed": 0, "neutered": 0, "spayed": 0, "healthy": 0, "fee_ok": 0}
    v, d, n, s = _resolve_health_flags(row)
    if soft_prefs.get("prefer_vaccinated") and v is True: flags["vaccinated"] = 1
    if soft_prefs.get("prefer_dewormed")   and d is True: flags["dewormed"]   = 1
    if soft_prefs.get("prefer_neutered")   and n is True: flags["neutered"]   = 1
    if soft_prefs.get("prefer_spayed")     and s is True: flags["spayed"]     = 1
    if soft_prefs.get("prefer_healthy"):
        cond = str(row.get("condition","")).strip().lower()
        if cond in {"healthy","good"}: flags["healthy"] = 1
    if soft_prefs.get("prefer_low_fee") or (soft_prefs.get("fee_cap") is not None):
        cap = soft_prefs.get("fee_cap") or 300.0
        try:
            fee = float(row.get("adoption_fee"))
            if fee <= cap: flags["fee_ok"] = 1
        except Exception:
            pass
    return flags

def _feature_score(row: pd.Series, facets: Dict[str, Any], soft_prefs: Dict[str, Any]) -> float:
    bonus = 0.0
    v, d, n, s = _resolve_health_flags(row)
    w_base, w_strong = 0.05, 0.08

    if v is True: bonus += w_strong if soft_prefs.get("prefer_vaccinated") else w_base
    if d is True: bonus += w_strong if soft_prefs.get("prefer_dewormed") else w_base
    if n is True: bonus += w_strong if soft_prefs.get("prefer_neutered") else w_base
    if s is True: bonus += w_strong if soft_prefs.get("prefer_spayed") else w_base

    fee = row.get("adoption_fee")
    try:
        fee = float(fee)
        cap = soft_prefs.get("fee_cap") or (300.0 if soft_prefs.get("prefer_low_fee") else 400.0)
        if fee <= cap:
            w_fee = 0.08 if soft_prefs.get("prefer_low_fee") else 0.05
            bonus += w_fee * (1.0 - max(0.0, min(1.0, fee / cap)))
    except Exception:
        pass

    groups = soft_prefs.get("age_groups_pref") or []
    if groups and _age_groups_match(row, groups):
        bonus += 0.05

    want_size = str(facets.get("size") or "").lower().strip()
    want_fur  = str(facets.get("fur_length") or "").lower().strip()
    if want_size and str(row.get("size","")).lower().strip() == want_size: bonus += 0.03
    if want_fur  and str(row.get("fur_length","")).lower().strip() == want_fur: bonus += 0.03

    if str(row.get("condition","")).strip().lower() in {"healthy","good"}:
        bonus += 0.05 if soft_prefs.get("prefer_healthy") else 0.03

    return max(0.0, min(0.35, bonus))

def adoption_search(query: str, ui: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any], List[Tuple[int, float]], pd.Series]:
    q = query or ""
    prev = st.session_state.get("last_facets", {}) or {}

    raw_spans = ner([q[:300]])[0] if q else []
    spans = resolve_overlaps_longest(raw_spans)
    mf = entity_spans_to_facets(spans)
    rf = parse_facets_from_text(q)

    def safe_merge_val(mv, rv):
        if mv is None or (isinstance(mv, (list, tuple)) and len(mv) == 0):
            return rv
        return mv

    facets = {
        "animal":     safe_merge_val(mf.get("animal"),     rf.get("animal")),
        "breed":      safe_merge_val(mf.get("breed"),      rf.get("breed")),
        "gender":     safe_merge_val(mf.get("gender"),     rf.get("gender")),
        "colors_any": safe_merge_val(mf.get("colors_any"), rf.get("colors_any")),
        "state":      safe_merge_val(mf.get("state"),      rf.get("state")),
    }

    if facets.get("colors_any"):
        facets["colors_any"] = sorted({normalize_color(c) for c in facets["colors_any"] if c})

    mapped_breed = None
    if facets.get("breed"):
        mapped_breed = map_breed_to_catalog(facets["breed"], breed_catalog, strict=True)
        facets["breed"] = mapped_breed

    if not facets.get("animal") and mapped_breed:
        inferred = breed_to_animal.get(mapped_breed)
        if inferred: facets["animal"] = inferred

    if not facets.get("state"):
        guessed_state = detect_state(q)
        if guessed_state: facets["state"] = guessed_state

    facets = sanitize_facets_ner_light(facets)

    soft_prefs = parse_soft_prefs(q)

    animal_changed = bool(facets.get("animal") and prev.get("animal") and facets["animal"] != prev["animal"])
    breed_changed  = bool(facets.get("breed")  and prev.get("breed")  and facets["breed"]  != prev["breed"])
    if not (animal_changed or breed_changed):
        for key in ["animal", "breed", "state", "gender", "colors_any"]:
            if not facets.get(key) and prev.get(key):
                facets[key] = prev[key]

    used = dict(facets)
    if soft_prefs.get("age_groups_pref"):
        used["age_groups_pref"] = list(soft_prefs["age_groups_pref"])
    used["soft_preferences"] = {
        k: v for k, v in soft_prefs.items()
        if k in {"prefer_vaccinated","prefer_dewormed","prefer_neutered","prefer_spayed","prefer_healthy","prefer_low_fee","fee_cap"}
        and v not in (False, None)
    }
    st.session_state["last_facets"] = {k: v for k, v in facets.items() if v}

    # -------- HARD PREFILTERS --------
    df_to_filter = dfp.copy()

    # ANIMAL hard filter
    if facets.get("animal") and "animal" in df_to_filter.columns:
        animal_val = str(facets["animal"]).strip().lower()
        df_to_filter["animal_clean"] = df_to_filter["animal"].astype(str).str.strip().str.lower()
        df_to_filter = df_to_filter[df_to_filter["animal_clean"] == animal_val]
        if len(df_to_filter) == 0:
            used["animal_no_hits"] = True
            return pd.DataFrame(), used, [], pd.Series(dtype=bool)

    # BREED hard (contains or equality)
    if facets.get("breed") and "breed" in df_to_filter.columns:
        breed_val = str(facets["breed"]).strip().lower()
        df_to_filter["breed_clean"] = df_to_filter["breed"].astype(str).str.strip().str.lower()
        if ui.get("allow_breed_contains", True):
            pattern = rf"\b{re.escape(breed_val)}\b"
            df_to_filter = df_to_filter[df_to_filter["breed_clean"].str.contains(pattern, case=False, na=False)]
        else:
            df_to_filter = df_to_filter[df_to_filter["breed_clean"] == breed_val]
        if len(df_to_filter) == 0:
            used["breed_no_hits"] = True
            return pd.DataFrame(), used, [], pd.Series(dtype=bool)

    # GENDER hard filter
    if facets.get("gender") and "gender" in df_to_filter.columns:
        wanted_gender = canonicalize_gender(facets.get("gender") or "")
        if wanted_gender:
            df_to_filter["gender_clean"] = df_to_filter["gender"].astype(str).str.strip().str.lower()
            df_to_filter = df_to_filter[df_to_filter["gender_clean"] == wanted_gender]
            if len(df_to_filter) == 0:
                used["gender_no_hits"] = True
                return pd.DataFrame(), used, [], pd.Series(dtype=bool)

    # STATE strict then optional relax
    df_strict = df_to_filter
    relaxed_df = pd.DataFrame()
    if facets.get("state") and "state" in df_to_filter.columns:
        user_state = str(facets["state"]).strip().lower()
        df_to_filter["state_clean"] = df_to_filter["state"].astype(str).str.strip().str.lower()
        df_strict = df_to_filter[df_to_filter["state_clean"] == user_state]
        if len(df_strict) == 0:
            used["state_no_hits"] = True
            return pd.DataFrame(), used, [], pd.Series(dtype=bool)
        if len(df_strict) < AUTO_RELAX_MIN_RESULTS:
            st.info(
                f"Only {len[df_strict]} pet(s) found in {user_state.title()}. "
                f"Showing similar pets from other states as well."
            )
            # Build relaxed set but keep animal/breed/gender hard constraints
            df_relaxed = dfp.copy()
            df_relaxed["animal_clean"] = df_relaxed["animal"].astype(str).str.strip().str.lower()
            df_relaxed["breed_clean"]  = df_relaxed["breed"].astype(str).str.strip().str.lower()
            df_relaxed["gender_clean"] = df_relaxed["gender"].astype(str).str.strip().str.lower()
            if facets.get("animal"):
                a = str(facets["animal"]).lower()
                df_relaxed = df_relaxed[df_relaxed["animal_clean"] == a]
            if facets.get("breed"):
                b = str(facets["breed"]).lower()
                if ui.get("allow_breed_contains", True):
                    pattern = rf"\b{re.escape(b)}\b"
                    df_relaxed = df_relaxed[df_relaxed["breed_clean"].str.contains(pattern, case=False, na=False)]
                else:
                    df_relaxed = df_relaxed[df_relaxed["breed_clean"] == b]
            if facets.get("gender"):
                g = canonicalize_gender(facets.get("gender") or "")
                if g:
                    df_relaxed = df_relaxed[df_relaxed["gender_clean"] == g]

            relaxed_df = df_relaxed[~df_relaxed.index.isin(df_strict.index)]
            used["state_relaxed_auto"] = True

    # Combine strict + relaxed
    df_to_filter = pd.concat([df_strict, relaxed_df]).drop_duplicates(
        subset=["pet_id"] if "pet_id" in dfp.columns else None
    )

    if "breed" in df_to_filter.columns and "breed_clean" not in df_to_filter.columns:
        df_to_filter["breed_clean"] = df_to_filter["breed"].astype(str).str.strip().str.lower()

    # ---- Breed summary (optional text) ----
    total_candidates = len(df_to_filter)
    summary_text = None
    if facets.get("breed") and "breed_clean" in df_to_filter.columns:
        breed_val = str(facets["breed"]).lower()
        pattern = rf"\b{re.escape(breed_val)}\b"
        mask_has   = df_to_filter["breed_clean"].str.contains(pattern, case=False, na=False)
        mask_exact = df_to_filter["breed_clean"].str.fullmatch(breed_val, case=False)
        mask_mix   = (df_to_filter["breed_clean"].str.contains(r"\+", na=False) |
                      df_to_filter["breed_clean"].str.contains(r"\bmix\b", case=False, na=False))
        breed_exact_count = (mask_has & mask_exact).sum()
        breed_mix_count   = (mask_has & mask_mix & ~mask_exact).sum()
    elif total_candidates > 0:
        summary_text = f"I found {total_candidates} pets that might interest you."
    used["breed_summary"] = {"total_candidates": int(total_candidates), "summary_text": summary_text}

    # ---- Colors strict (optional) ----
    if facets.get("colors_any"):
        want_colors = sorted(set(facets["colors_any"]))
        if "colors_canonical" not in df_to_filter.columns and "color" in df_to_filter.columns:
            df_to_filter = df_to_filter.copy()
            df_to_filter["colors_canonical"] = df_to_filter["color"].apply(
                lambda s: [normalize_color(t) for t in str(s or "").split(",") if t.strip()]
            )
        def _color_ok(lst):
            return isinstance(lst, list) and any(c in lst for c in want_colors)
        df_color = df_to_filter[df_to_filter["colors_canonical"].apply(_color_ok)]
        if len(df_color) == 0:
            used["colors_any_no_hits"] = True
        else:
            df_to_filter = df_color

    # === Hybrid retrieval & scoring ===
    boost_q = expand_kid_friendly(make_boosted_query(q, facets))
    lex_all = bm25.search(boost_q, topk=LEX_POOL)
    s_lex = {int(idx): float(s) for idx, s in lex_all if idx in df_to_filter.index}
    emb_all = emb_search(boost_q, student, doc_ids, doc_vecs, pool_topn=EMB_POOL, faiss_index=faiss_index)
    emb_idx = {int(pid): float(s) for pid, s in emb_all if pid in df_to_filter.index}
    nlex, nemb = _minmax(s_lex), _minmax(emb_idx)
    wl, we = HYBRID_W["lex"], HYBRID_W["emb"]
    combo = {idx: wl * nlex.get(idx, 0.0) + we * nemb.get(idx, 0.0) for idx in set(nlex) | set(nemb)}

    # --- Soft feature rerank + bucket flags ---
    feat_scores: Dict[int, float] = {}
    age_priority: Dict[int, int] = {}
    vacc_priority: Dict[int, int] = {}
    deworm_priority: Dict[int, int] = {}
    neuter_priority: Dict[int, int] = {}
    spay_priority: Dict[int, int] = {}
    healthy_priority: Dict[int, int] = {}
    fee_priority: Dict[int, int] = {}
    meets_all_soft: Dict[int, int] = {}

    req_groups = soft_prefs.get("age_groups_pref") or []

    for idx in df_to_filter.index:
        row = dfp.loc[idx]
        feat_scores[int(idx)] = _feature_score(row, facets, soft_prefs)

        age_ok = 1 if (req_groups and _age_groups_match(row, req_groups)) else (0 if req_groups else 0)
        age_priority[int(idx)] = age_ok

        flags = _health_fee_priority(row, soft_prefs)
        vacc_priority[int(idx)]   = flags["vaccinated"]
        deworm_priority[int(idx)] = flags["dewormed"]
        neuter_priority[int(idx)] = flags["neutered"]
        spay_priority[int(idx)]   = flags["spayed"]
        healthy_priority[int(idx)] = flags["healthy"]
        fee_priority[int(idx)]     = flags["fee_ok"]

        ok = True
        if req_groups: ok = ok and (age_ok == 1)
        if soft_prefs.get("prefer_vaccinated"): ok = ok and (flags["vaccinated"] == 1)
        if soft_prefs.get("prefer_dewormed"):   ok = ok and (flags["dewormed"] == 1)
        if soft_prefs.get("prefer_neutered"):   ok = ok and (flags["neutered"] == 1)
        if soft_prefs.get("prefer_spayed"):     ok = ok and (flags["spayed"] == 1)
        if soft_prefs.get("prefer_healthy"):    ok = ok and (flags["healthy"] == 1)
        if soft_prefs.get("prefer_low_fee") or (soft_prefs.get("fee_cap") is not None):
            ok = ok and (flags["fee_ok"] == 1)
        meets_all_soft[int(idx)] = 1 if ok else 0

    if feat_scores:
        vals = np.array(list(feat_scores.values()), dtype=float)
        lo, hi = float(vals.min()), float(vals.max())
        denom = (hi - lo) or 1.0
        feat_norm = {k: (v - lo) / denom for k, v in feat_scores.items()}
    else:
        feat_norm = {}

    alpha, beta = 0.85, 0.15
    combo = {idx: alpha*combo.get(idx, 0.0) + beta*feat_norm.get(idx, 0.0) for idx in combo}

    def _prio_tuple(i: int):
        return (
            meets_all_soft.get(i, 0),
            age_priority.get(i, 0),
            vacc_priority.get(i, 0),
            deworm_priority.get(i, 0),
            neuter_priority.get(i, 0),
            spay_priority.get(i, 0),
            healthy_priority.get(i, 0),
            fee_priority.get(i, 0),
            combo.get(i, 0.0),
        )

    pool = max(EMB_POOL, TOPK_CARDS)
    hits = sorted([(i, combo[i]) for i in combo.keys()], key=lambda x: _prio_tuple(int(x[0])), reverse=True)[:pool]

    if not hits:
        return pd.DataFrame(), used, [], pd.Series(dtype=bool)

    chosen_idx = [int(i) for i, _ in hits[:TOPK_CARDS] if i in df_to_filter.index]
    if not chosen_idx:
        return pd.DataFrame(), used, hits, pd.Series(dtype=bool)

    display_cols = [
        "name", "animal", "breed", "gender", "state", "color", "colors_canonical",
        "size", "fur_length", "condition", "age_months", "description_clean",
        "url", "photo_links", "video_links"
    ]
    display_cols = [c for c in display_cols if c in dfp.columns]
    res_df = df_to_filter.loc[chosen_idx, display_cols].copy().reset_index(names=["df_index"])
    res_df["score"] = [float(s) for _, s in hits[:len(res_df)]]

    if "state_relaxed_auto" in used and facets.get("state"):
        res_df["source"] = res_df["state"].apply(
            lambda s: "Strict" if s.strip().lower() == str(facets["state"]).strip().lower() else "Relaxed"
        )
        res_df.sort_values(
            by=["source", "score"],
            ascending=[True, False],
            inplace=True,
            key=lambda col: col.map({"Strict": 0, "Relaxed": 1}).fillna(1),
        )

    # --- Highlight mask: MUST meet ALL soft prefs AND ALL strict requirements ---
    requested_state   = (facets.get("state") or "").strip().lower()
    requested_animal  = (facets.get("animal") or "").strip().lower()
    requested_breed   = (facets.get("breed") or "").strip().lower()
    requested_gender  = canonicalize_gender(facets.get("gender") or "")
    requested_colors  = set(facets.get("colors_any") or [])

    def _strict_ok(row: pd.Series) -> bool:
        ok = True
        if requested_state:
            ok = ok and (str(row.get("state","")).strip().lower() == requested_state)
        if requested_animal:
            ok = ok and (str(row.get("animal","")).strip().lower() == requested_animal)
        if requested_breed:
            btxt = str(row.get("breed","")).strip().lower()
            pattern = rf"\b{re.escape(requested_breed)}\b"
            ok = ok and bool(re.search(pattern, btxt))
        if requested_gender:
            ok = ok and (canonicalize_gender(str(row.get("gender",""))) == requested_gender)
        if requested_colors:
            lst = row.get("colors_canonical")
            ok = ok and isinstance(lst, list) and any(c in lst for c in requested_colors)
        return ok

    # Highlight only if strict+soft are satisfied for the SHOWN rows
    meet_all_mask = res_df.apply(
        lambda r: bool(meets_all_soft.get(int(r["df_index"]), 0)) and _strict_ok(r),
        axis=1
    )

    # --- NEW exact-match totals on STRICT IN-STATE pool (not just shown rows)
    strict_idx_set = set(df_strict.index)
    strict_exact_total = sum(1 for i in strict_idx_set if meets_all_soft.get(int(i), 0) == 1)

    used["counts"] = {
        "strict_in_state": int(len(df_strict) if 'df_strict' in locals() else len(df_to_filter)),
        "relaxed_extra": int(len(relaxed_df)) if 'relaxed_df' in locals() else 0,
        "meets_all_strict_and_soft_shown": int(meet_all_mask.sum()),  # shown rows only
        "strict_exact_total": int(strict_exact_total),                 # in-state exact matches
        "age_group_pref": used.get("age_groups_pref", []),
    }

    return res_df, used, hits, meet_all_mask

# -------------------------------------------
# INTENT GATE
# -------------------------------------------
INTENT_ADOPTION = {"adoption", "find_pet", "pet_search", "pet_adoption"}
INTENT_PETCARE  = {"pet_care", "care", "rag", "qa"}

def classify_intent_safe(text: str) -> str:
    try:
        if hasattr(bot, "classify_intent"): return str(bot.classify_intent(text)).lower()
        if hasattr(bot, "intent_classifier") and hasattr(bot.intent_classifier, "predict"):
            return str(bot.intent_classifier.predict(text)).lower()
    except Exception:
        pass
    t = (text or "").lower()
    if any(k in t for k in ["adopt", "adoption", "puppy", "kitten", "breed", "in selangor", "in penang", "near "]):
        return "adoption"
    if any(k in t for k in ["feed", "groom", "vaccin", "train", "care", "health", "why is my"]):
        return "pet_care"
    return "adoption"

# -------------------------------------------
# MAIN APP (no sidebar; top 6 cards + Clear button)
# -------------------------------------------
st.title("🐾 Unified PetBot — Chat + Smart Search")

# Clear history / new search button
if st.button("➕ New search / Clear history", type="primary"):
    st.session_state["messages"] = []
    st.session_state["last_facets"] = {}
    st.experimental_rerun()

st.caption("Try: ‘white female **puppy** poodle in Selangor good with kids’, or ‘fee under 300 vaccinated young dog in KL’")

ui = dict(STRICT_DEFAULTS)
ui["method"] = "hybrid"
ui["use_mmr"] = False
ui["mmr_lambda"] = 0.35
ui["topk"] = TOPK_CARDS
ui["grid_cols"] = GRID_COLS

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_text = st.chat_input("Ask about pet care, or describe your ideal pet…")
if user_text:
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    # Decide intent before any reply text
    intent = classify_intent_safe(user_text)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):

            if intent in INTENT_ADOPTION:
                # Run adoption search first to get fresh facets for THIS turn
                res_df, used_facets, hits, meet_all_mask = adoption_search(user_text, ui)

                # Compose a clean “Searching…” line using ONLY current facets/soft prefs
                h = {k: v for k, v in used_facets.items() if k in {"animal","breed","state","gender","colors_any"} and v}
                age_groups = used_facets.get("age_groups_pref", [])
                parts = []
                if h.get("animal"): parts.append(h["animal"])
                if h.get("breed"): parts.append(h["breed"])
                if h.get("gender"): parts.append(h["gender"])
                if age_groups: parts.append("/".join(age_groups))
                if h.get("state"): parts.append(f"in {str(h['state']).title()}")
                search_line = "Got it! Searching" + (" for " + " ".join([p for p in parts if p]) if parts else "...")

                st.markdown(search_line)

                # Banner based on exact in-state strict+soft matches vs cards shown
                counts = used_facets.get("counts", {}) or {}
                exact_x = int(counts.get("strict_exact_total", 0))          # exact in-state matches
                res_n = int(res_df.shape[0]) if res_df is not None else 0   # cards actually shown

                if exact_x >= TOPK_CARDS:
                    st.success(
                        f"🥳 Found {exact_x} pets matching your description! "
                        f"Here are {TOPK_CARDS} fur babies waiting for a home~"
                    )
                elif res_n > exact_x:
                    st.success(
                        f"🥳 Found {exact_x} pets matching your description! "
                        f"Adding a few more similar pets waiting for a forever home 🥹"
                    )
                else:
                    st.success(f"🥳 Found {exact_x} pets matching your description!")

                # Conversational recap of the user’s facets/preferences
                soft = used_facets.get("soft_preferences", {}) or {}
                bits = []
                if h.get("gender"):
                    bits.append(str(h["gender"]))
                if h.get("breed"):
                    bits.append(str(h["breed"]))
                if h.get("animal") and not h.get("breed"):
                    bits.append(str(h["animal"]))
                where_bit = f"in {str(h['state']).title()}" if h.get("state") else None
                if where_bit:
                    bits.append(where_bit)
                extras = []
                if age_groups:
                    extras.append("/".join(age_groups) + " age group")
                if h.get("colors_any"):
                    extras.append("color(s): " + ", ".join(h["colors_any"]))
                if soft.get("fee_cap"):
                    try:
                        extras.append(f"budget under {int(float(soft['fee_cap']))}")
                    except Exception:
                        extras.append(f"budget under {soft['fee_cap']}")
                if soft.get("prefer_vaccinated"):
                    extras.append("vaccinated")
                if soft.get("prefer_dewormed"):
                    extras.append("dewormed")
                if soft.get("prefer_neutered"):
                    extras.append("neutered")
                if soft.get("prefer_spayed"):
                    extras.append("spayed")
                if soft.get("prefer_healthy"):
                    extras.append("healthy condition")

                lead = "You asked for " + " ".join(bits) if bits else "You asked for a pet"
                tail = (", " + ", ".join(extras)) if extras else ""
                st.caption(lead + tail + ".")

                # Optional conversational breed summary
                bs = used_facets.get("breed_summary", {}) or {}
                if bs.get("summary_text"):
                    st.markdown(f"🦴 {bs['summary_text']}")

                # Results
                if res_df is None or res_df.empty:
                    st.info("No results. Try relaxing filters or removing constraints.")
                else:
                    render_results_grid(res_df, meet_all_mask, max_cols=ui.get("grid_cols", GRID_COLS))

                # Store a neutral acknowledgement (not the bot pipeline text) in chat history
                st.session_state.messages.append({"role": "assistant", "content": search_line})

            else:
                # Pet-care: keep ChatbotPipeline reply behavior
                try:
                    reply = bot.handle_message(user_text)
                except Exception as e:
                    reply = f"(Chat response unavailable: {e})"
                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
