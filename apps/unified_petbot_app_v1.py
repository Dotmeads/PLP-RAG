# -*- coding: utf-8 -*-
"""
Pawfect Match - Single Chat Interface with Intent Classification
Routes strictly via your intent model:
  - 'adoption'  => Pet search (NER + hybrid retrieval + strict/soft filters)
  - 'pet_care'  => RAG Q&A (ChatbotPipeline)
No heuristic fallbacks. If the model is unavailable, show a clear error.

UI features kept:
- 6 cards maximum
- Fixed-size photos (no cropping, shrink to fit)
- Highlight in green when a pet meets ALL strict + soft requirements
- Show the facets used for search (clean "Searching…" line + recap)
"""
import streamlit as st
st.set_page_config(page_title="Pawfect Match", layout="wide")

import os, ast, re, json, html
import sys
from typing import List, Dict, Any, Tuple, Optional, Set

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import pandas as pd
import numpy as np

# ---- RAG components ----
from rag_system.proposed_rag_system import ProposedRAGManager
from chatbot_flow.chatbot_pipeline import ChatbotPipeline

# ---- Retrieval stack (no Azure NER here) ----
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
# CONFIG / CONSTANTS
# -------------------------------------------
TOPK_CARDS = 6                 # show at most 6 cards
EMB_POOL = 200
LEX_POOL = 2000
HYBRID_W = {"lex": 0.1, "emb": 0.9}
AUTO_RELAX_MIN_RESULTS = 6     # if strict in-state < 6, auto-include cross-state
GRID_COLS = 3                  # visual grid columns

INTENT_ADOPTION = "adoption"
INTENT_PETCARE  = "pet_care"
ALLOWED_INTENTS = {INTENT_ADOPTION, INTENT_PETCARE}

# -------------------------------------------
# Color normalization & whitelist
# -------------------------------------------
COLOR_WHITELIST = {
    # base hues
    "white","black","brown","gray","grey","cream","beige","tan","yellow","gold","golden",
    "orange","ginger","red","chocolate","liver","blue","silver","fawn","apricot",
    # patterns / coat terms
    "brindle","merle","sable","seal","champagne","coffee",
    "tricolor","tri-color","bicolor","bi-color",
    "calico","tortoiseshell","tortoise","point"
}
COLOR_SYNONYMS = {
    "grey": "gray",
    "gold": "yellow",
    "golden": "yellow",
    "cream": "white",
    "ginger": "orange",
    "tri-color": "tricolor",
    "bi-color": "bicolor",
    "tortoise": "tortoiseshell",
}

def normalize_color(c: str) -> str:
    c = (c or "").strip().lower()
    if not c: return ""
    c = COLOR_SYNONYMS.get(c, c)
    return c if c in COLOR_WHITELIST else ""

def _safe_list_from_cell(x):
    """Parse strings like "['a','b']" or '["a","b"]' or comma strings into list."""
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
    return [t for t in (normalize_color(str(t)) for t in _safe_list_from_cell(x)) if t]

# -------------------------------------------
# Age groups (months) & parsing
# -------------------------------------------
AGE_GROUPS = {
    "puppy/kitten": (0, 12),          # [0, 12)
    "young": (12, 36),                 # [12, 36)
    "adult": (36, 84),                 # [36, 84)
    "senior": (84, None),              # [84, ∞)
}
AGE_GROUP_KEYS = list(AGE_GROUPS.keys())

def _val_to_months(val: float, unit: str) -> float:
    return 12.0 * val if unit.lower().startswith("y") else val

def _groups_for_threshold(op: str, months: float) -> Set[str]:
    groups = set()
    for g, (lo, hi) in AGE_GROUPS.items():
        lo_m = float(lo if lo is not None else -1e9)
        hi_m = float(hi if hi is not None else 1e9)
        if op in ("<", "lt", "less"):
            if lo_m < months: groups.add(g)
        elif op in ("<=", "le"):
            if lo_m <= months: groups.add(g)
        elif op in (">", "gt", "more", "atleast", ">=","ge"):
            if hi is None or hi_m >= months: groups.add(g)
    # Heuristic tightening
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
# NER & helpers (HF pipeline from ChatbotPipeline)
# -------------------------------------------
def resolve_overlaps_longest(spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Resolve overlaps; keep longest; prefer BREED over COLOR on conflicts."""
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

# -------------------------------------------
# Health flags (parsed from columns + text)
# -------------------------------------------
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
            v = _coerce_bool(row[c]); 
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

# -------------------------------------------
# Soft preference parsing
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
# Bootstrap functions (separate)
# -------------------------------------------
@st.cache_resource(show_spinner=True)
def bootstrap_rag_system():
    """
    Initialize RAG and Chatbot pipeline, and expose the Hugging Face NER & INTENT CLASSIFIER.
    """
    rag = ProposedRAGManager()
    documents_dir = os.path.join(project_root, "documents")
    if os.path.exists(documents_dir):
        rag.add_directory(documents_dir)
    bot = ChatbotPipeline(rag)

    # Try to find HF NER inside your ChatbotPipeline
    ner_pipe = None
    for attr in ("ner_extractor", "entity_extractor", "ner"):
        if hasattr(bot, attr) and hasattr(getattr(bot, attr), "ner_pipe"):
            ner_pipe = getattr(getattr(bot, attr), "ner_pipe"); break
    if ner_pipe is None and hasattr(bot, "ner_pipe"):
        ner_pipe = getattr(bot, "ner_pipe")
    if ner_pipe is None:
        raise AttributeError("ChatbotPipeline does not expose a NER pipeline (HF).")

    # Ensure an intent classifier is available
    has_model_classifier = (
        hasattr(bot, "classify_intent") or
        (hasattr(bot, "intent_classifier") and hasattr(bot.intent_classifier, "predict"))
    )
    if not has_model_classifier:
        raise AttributeError("Intent classification model not found on ChatbotPipeline. "
                             "Expose `classify_intent(text)` or `intent_classifier.predict(text)`.")

    return rag, bot, ner_pipe

@st.cache_resource(show_spinner=True)
def bootstrap_azure_components():
    """
    Initialize matching/ranking model + FAISS + pet CSV + BM25.
    (No Azure NER download here.)
    """
    cfg = get_blob_settings()
    conn = cfg["connection_string"]

    with st.spinner("Downloading Matching/Ranking model (flat folder)..."):
        download_prefix_flat(conn, cfg["ml_container"], cfg["mr_prefix"], local_mr_dir())
    with st.spinner("Downloading pet CSV..."):
        smart_download_single_blob(conn, cfg["pets_container"], cfg["pets_csv_blob"], local_pets_csv_path())

    # Load MR & index
    student, doc_ids, doc_vecs = load_mr_model(local_mr_dir())
    faiss_index = load_faiss_index(local_mr_dir(), dim=doc_vecs.shape[1])

    # Load pets CSV and normalize
    dfp = pd.read_csv(local_pets_csv_path())
    for c in ("animal", "gender", "state", "breed", "size", "fur_length", "condition", "color"):
        if c in dfp.columns:
            dfp[c] = dfp[c].astype(str).fillna("").str.strip().str.lower()

    if "colors_canonical" in dfp.columns:
        dfp["colors_canonical"] = dfp["colors_canonical"].apply(parse_colors_cell)
    else:
        if "color" in dfp.columns:
            dfp["colors_canonical"] = dfp["color"].apply(
                lambda s: [normalize_color(t) for t in str(s or "").split(",") if t.strip()]
            )

    for media_col in ["photo_links", "video_links"]:
        if media_col in dfp.columns:
            dfp[media_col] = dfp[media_col].apply(_safe_list_from_cell)

    # Breed mappings
    breed_to_animal = {}
    if "breed" in dfp.columns and "animal" in dfp.columns:
        tmp = (dfp[["breed","animal"]]
               .dropna()
               .groupby("breed")["animal"].agg(lambda s: s.value_counts().idxmax()))
        breed_to_animal = tmp.to_dict()

    # BM25 on text
    text_col = "doc" if "doc" in dfp.columns else ("description_clean" if "description_clean" in dfp.columns else "description")
    docs_raw = {int(i): only_text(str(t)) for i, t in zip(dfp.index, dfp[text_col].fillna("").tolist())}
    bm25 = BM25().fit(docs_raw)

    breed_catalog = sorted(set([b for b in dfp.get("breed", pd.Series([], dtype=str)).astype(str).str.lower().tolist() if b]))

    return {
        "cfg": cfg,
        "student": student, "doc_ids": doc_ids, "doc_vecs": doc_vecs, "faiss_index": faiss_index,
        "dfp": dfp, "bm25": bm25, "breed_catalog": breed_catalog, "breed_to_animal": breed_to_animal
    }

# -------------------------------------------
# Breed & state helpers
# -------------------------------------------
BREED_ALIASES = {
    "husky": ["siberian husky", "alaskan husky"],
    "gsd": ["german shepherd", "german shepherd dog"],
    "gr": ["golden retriever"],
    "grd": ["golden retriever"],
}
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
# Retrieval helpers
# -------------------------------------------
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

# -------------------------------------------
# Intent (STRICTLY via your model)
# -------------------------------------------
def classify_intent_via_model(bot: ChatbotPipeline, text: str) -> str:
    """
    Always use your model. No heuristic fallbacks.
    Must return 'adoption' or 'pet_care'.
    """
    # Prefer classify_intent(text)
    if hasattr(bot, "classify_intent"):
        try:
            pred = str(bot.classify_intent(text)).lower().strip()
        except Exception as e:
            raise RuntimeError(f"Intent classifier error: {e}")
    elif hasattr(bot, "intent_classifier") and hasattr(bot.intent_classifier, "predict"):
        try:
            pred = str(bot.intent_classifier.predict(text)).lower().strip()
        except Exception as e:
            raise RuntimeError(f"Intent classifier error: {e}")
    else:
        raise RuntimeError("Intent classification model not available on ChatbotPipeline.")
    # Map unknown labels to pet_care (still honoring model output by passing through)
    return pred if pred in ALLOWED_INTENTS else INTENT_PETCARE

# -------------------------------------------
# Adoption search (NER + strict filters + hybrid + soft re-rank)
# -------------------------------------------
def expand_kid_friendly(q: str) -> str:
    t = q.lower()
    if "kid" in t or "child" in t or "family" in t:
        extras = " kid-friendly child-friendly family-friendly gentle with kids good with children good with kids family dog"
        return q + " " + extras
    return q

def adoption_search(query: str, ner, azure_env) -> Tuple[pd.DataFrame, Dict[str, Any], List[Tuple[int, float]], pd.Series]:
    """Returns (res_df, used_facets, hits(scores), meet_all_mask)."""
    student = azure_env["student"]; doc_ids = azure_env["doc_ids"]; doc_vecs = azure_env["doc_vecs"]
    faiss_index = azure_env["faiss_index"]; dfp = azure_env["dfp"]; bm25 = azure_env["bm25"]
    breed_catalog = azure_env["breed_catalog"]; breed_to_animal = azure_env["breed_to_animal"]

    q = query or ""
    prev = st.session_state.get("last_facets", {}) or {}

    # NER & facet parsing via HF pipeline
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

    # Normalize colors
    if facets.get("colors_any"):
        cleaned = []
        for c in facets["colors_any"]:
            cc = normalize_color(c)
            if cc: cleaned.append(cc)
        if cleaned:
            facets["colors_any"] = sorted(set(cleaned))
        else:
            facets.pop("colors_any", None)

    # Breed mapping / animal inference
    mapped_breed = None
    if facets.get("breed"):
        mapped_breed = map_breed_to_catalog(facets["breed"], breed_catalog, strict=True)
        facets["breed"] = mapped_breed
    if not facets.get("animal") and mapped_breed:
        inferred = breed_to_animal.get(mapped_breed)
        if inferred: facets["animal"] = inferred

    # State detection fallback (kept, but not intent)
    if not facets.get("state"):
        guessed_state = detect_state(q)
        if guessed_state: facets["state"] = guessed_state

    # Final sanitize
    facets = sanitize_facets_ner_light(facets)

    # Soft preferences
    soft_prefs = parse_soft_prefs(q)

    # Persist previously chosen hard facets if user didn't change breed/animal this turn
    animal_changed = bool(facets.get("animal") and prev.get("animal") and facets["animal"] != prev["animal"])
    breed_changed  = bool(facets.get("breed")  and prev.get("breed")  and facets["breed"]  != prev["breed"])
    if not (animal_changed or breed_changed):
        for key in ["animal", "breed", "state", "gender", "colors_any"]:
            if not facets.get(key) and prev.get(key):
                facets[key] = prev[key]

    # Record used facets for UI
    used = dict(facets)
    if soft_prefs.get("age_groups_pref"):
        used["age_groups_pref"] = list(soft_prefs["age_groups_pref"])
    used["soft_preferences"] = {
        k: v for k, v in soft_prefs.items()
        if k in {"prefer_vaccinated","prefer_dewormed","prefer_neutered","prefer_spayed","prefer_healthy","prefer_low_fee","fee_cap"}
        and v not in (False, None)
    }
    st.session_state["last_facets"] = {k: v for k, v in facets.items() if v}

    # ---- HARD PREFILTERS ----
    df_to_filter = dfp.copy()

    # animal
    if facets.get("animal") and "animal" in df_to_filter.columns:
        animal_val = str(facets["animal"]).strip().lower()
        df_to_filter = df_to_filter[df_to_filter["animal"] == animal_val]
        if len(df_to_filter) == 0:
            used["animal_no_hits"] = True
            return pd.DataFrame(), used, [], pd.Series(dtype=bool)

    # breed (contains)
    if facets.get("breed") and "breed" in df_to_filter.columns:
        breed_val = str(facets["breed"]).strip().lower()
        pattern = rf"\b{re.escape(breed_val)}\b"
        df_to_filter = df_to_filter[df_to_filter["breed"].str.contains(pattern, case=False, na=False)]
        if len(df_to_filter) == 0:
            used["breed_no_hits"] = True
            return pd.DataFrame(), used, [], pd.Series(dtype=bool)

    # gender
    if facets.get("gender") and "gender" in df_to_filter.columns:
        wanted_gender = canonicalize_gender(facets.get("gender") or "")
        if wanted_gender:
            df_to_filter = df_to_filter[df_to_filter["gender"] == wanted_gender]
            if len(df_to_filter) == 0:
                used["gender_no_hits"] = True
                return pd.DataFrame(), used, [], pd.Series(dtype=bool)

    # State strict + optional relax
    df_strict = df_to_filter
    relaxed_df = pd.DataFrame()
    if facets.get("state") and "state" in df_to_filter.columns:
        user_state = str(facets["state"]).strip().lower()
        df_strict = df_to_filter[df_to_filter["state"] == user_state]
        if len(df_strict) == 0:
            used["state_no_hits"] = True
            return pd.DataFrame(), used, [], pd.Series(dtype=bool)
        if len(df_strict) < AUTO_RELAX_MIN_RESULTS:
            st.info(f"Only {len(df_strict)} pet(s) found in {user_state.title()}. Showing similar pets from other states as well.")
            # relax state but keep other hard constraints above
            relaxed_df = dfp.copy()
            if facets.get("animal"):
                relaxed_df = relaxed_df[relaxed_df["animal"] == str(facets["animal"]).strip().lower()]
            if facets.get("breed"):
                pattern = rf"\b{re.escape(str(facets['breed']).strip().lower())}\b"
                relaxed_df = relaxed_df[relaxed_df["breed"].str.contains(pattern, case=False, na=False)]
            if facets.get("gender"):
                g = canonicalize_gender(facets.get("gender") or "")
                if g:
                    relaxed_df = relaxed_df[relaxed_df["gender"] == g]
            relaxed_df = relaxed_df[~relaxed_df.index.isin(df_strict.index)]
            used["state_relaxed_auto"] = True

    df_to_filter = pd.concat([df_strict, relaxed_df]).drop_duplicates(subset=["pet_id"] if "pet_id" in dfp.columns else None)

    # Colors strict (optional)
    if facets.get("colors_any"):
        want_colors = sorted(set(facets["colors_any"]))
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

    # Soft feature rerank + flags
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
        if soft_prefs.get("prefer_dewormed"):   ok = ok and (flags["dewormed"]   == 1)
        if soft_prefs.get("prefer_neutered"):   ok = ok and (flags["neutered"]   == 1)
        if soft_prefs.get("prefer_spayed"):     ok = ok and (flags["spayed"]     == 1)
        if soft_prefs.get("prefer_healthy"):    ok = ok and (flags["healthy"]    == 1)
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

    # Strict+soft highlight mask for shown rows
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

    meet_all_mask = res_df.apply(
        lambda r: bool(meets_all_soft.get(int(r["df_index"]), 0)) and _strict_ok(r),
        axis=1
    )

    # Count exact in-state strict+soft matches
    strict_idx_set = set(df_strict.index)
    strict_exact_total = sum(1 for i in strict_idx_set if meets_all_soft.get(int(i), 0) == 1)

    used["counts"] = {
        "strict_in_state": int(len(df_strict) if 'df_strict' in locals() else len(df_to_filter)),
        "relaxed_extra": int(len(relaxed_df)) if 'relaxed_df' in locals() else 0,
        "meets_all_strict_and_soft_shown": int(meet_all_mask.sum()),
        "strict_exact_total": int(strict_exact_total),
        "age_group_pref": used.get("age_groups_pref", []),
    }

    return res_df, used, hits, meet_all_mask

# -------------------------------------------
# Cards UI (fixed photo area; green highlight when full match)
# -------------------------------------------
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

def _badge(val: Optional[bool], label: str) -> str:
    if val is True:  return f"✅ {label}"
    if val is False: return f"❌ {label}"
    return f"➖ {label}"

def render_pet_card(row: pd.Series, highlight: bool = False):
    """Render full card with fixed-size photo and optional green highlight."""
    pid = int(row.get("pet_id")) if "pet_id" in row else int(row.name)
    name = str(row.get("name") or f"Pet {pid}")
    url  = str(row.get("url") or "")
    animal = (row.get("animal") or "").title()
    breed  = str(row.get("breed") or "—")
    gender = (row.get("gender") or "—").title()
    state  = (row.get("state") or "—").title()
    colors_canon = row.get("colors_canonical")
    color_txt = ", ".join(colors_canon) if isinstance(colors_canon, list) and colors_canon else str(row.get("color") or "—")
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

    # Title (blue link if URL)
    if url:
        title_html = f'<a href="{url}" target="_blank" style="text-decoration:none;color:#2563EB;"><span style="font-weight:800;">🔗 {html.escape(name)}</span></a>'
    else:
        title_html = f'<span style="color:#1F2937;font-weight:800;">{html.escape(name)}</span>'

    # Fixed-size photo area (shrink-to-fit; no cropping)
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

    # Collapsible description
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
    <strong>Color(s):</strong> {html.escape(color_txt)} • <strong>Size:</strong> {html.escape(size)} • <strong>Fur:</strong> {html.escape(fur)} • <strong>Condition:</strong> {html.escape(cond)}
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
# MAIN APP (strict model-based routing)
# -------------------------------------------
def main():
    # Styling and header
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #ffb6c1 0%, #ffc0cb 50%, #ffd1dc 100%);
        background-attachment: fixed;
    }
    .main-header { text-align: center; padding: 2rem 0; background: linear-gradient(45deg, #ff6b9d, #ff8fab);
        border-radius: 20px; margin-bottom: 2rem; box-shadow: 0 8px 32px rgba(255, 107, 157, 0.3); }
    .main-header h1 { color: white; font-size: 3rem; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }
    .main-header p { color: white; font-size: 1.2rem; margin: 0.5rem 0 0 0; opacity: 0.9; }
    .status-bar { background: linear-gradient(135deg, #ff6b9d, #ff8fab); color: white; padding: 1rem 2rem;
        margin: -1rem -1rem 2rem -1rem; border-radius: 0 0 20px 20px; box-shadow: 0 4px 20px rgba(255, 107, 157, 0.3); }
    .status-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; align-items: center; }
    .status-item { background: rgba(255, 255, 255, 0.2); padding: 0.8rem; border-radius: 10px; text-align: center;
        backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.3); }
    .status-item.success { background: rgba(76, 175, 80, 0.3); border-color: rgba(76, 175, 80, 0.5); }
    .status-item.warning { background: rgba(255, 152, 0, 0.3); border-color: rgba(255, 152, 0, 0.5); }
    .status-item.error { background: rgba(244, 67, 54, 0.3); border-color: rgba(244, 67, 54, 0.5); }
    .status-icon { font-size: 1.5rem; margin-bottom: 0.5rem; display: block; }
    .status-text { font-weight: 600; margin-bottom: 0.3rem; }
    .status-detail { font-size: 0.9rem; opacity: 0.9; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="main-header">
        <h1>🐾 Pawfect Match</h1>
        <p>Your Intelligent Pet Assistant - Ask about pet care or find your perfect pet match</p>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("🚀 Initializing Pawfect Match systems..."):
        # RAG + HF NER + INTENT MODEL
        try:
            rag, chatbot, ner = bootstrap_rag_system()
        except Exception as e:
            st.error(f"Startup error (RAG/NER/Intent): {e}")
            return

        # MR + data (no Azure NER here)
        try:
            azure_env = bootstrap_azure_components()
        except Exception as e:
            st.error(f"Startup error (Search/Data): {e}")
            return

        # Optional wiring if your bot uses these objects internally
        if chatbot is not None and isinstance(azure_env, dict):
            chatbot.azure_components = azure_env

    # Status bar
    def get_status_class(status_type): return "success" if status_type=="success" else ("warning" if status_type=="warning" else ("error" if status_type=="error" else ""))
    def get_status_icon(status_type): return {"success":"✅","warning":"⚠️","error":"❌"}.get(status_type,"ℹ️")

    rag_ok = (rag is not None and chatbot is not None and ner is not None)
    mr_ok = isinstance(azure_env, dict) and azure_env.get("student") is not None

    st.markdown(f"""
    <div class="status-bar">
        <div class="status-grid">
            <div class="status-item {get_status_class('success' if rag_ok else 'error')}">
                <span class="status-icon">{get_status_icon('success' if rag_ok else 'error')}</span>
                <div class="status-text">{'RAG + HF NER + Intent Ready' if rag_ok else 'RAG/NER/Intent Issue'}</div>
                <div class="status-detail">{'Documents, NER, and intent model loaded' if rag_ok else 'Check pipeline exposure'}</div>
            </div>
            <div class="status-item {get_status_class('success' if mr_ok else 'warning')}">
                <span class="status-icon">{get_status_icon('success' if mr_ok else 'warning')}</span>
                <div class="status-text">{'Pet Search Ready' if mr_ok else 'Pet Search Limited'}</div>
                <div class="status-detail">{(lambda n: f'{n} pets available' if n is not None else 'Dataset missing')(len(azure_env.get('dfp')) if mr_ok else None)}</div>
            </div>
            <div class="status-item {get_status_class('success' if (rag_ok and mr_ok) else 'error')}">
                <span class="status-icon">{get_status_icon('success' if (rag_ok and mr_ok) else 'error')}</span>
                <div class="status-text">{'All Systems Ready' if (rag_ok and mr_ok) else 'System Issues'}</div>
                <div class="status-detail">{'Ready to help!' if (rag_ok and mr_ok) else 'Please check configuration'}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
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
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.session_state["last_facets"] = {}
            st.rerun()
        st.markdown("---")
        st.markdown("### 🎨 Quick Tips")
        st.markdown("""
        - Ask about **pet health** for medical advice  
        - Describe your ideal pet for **adoption search**  
        - Use **specific breeds & states** for precision  
        - Add preferences like **vaccinated** or **under 300 fee**
        """)

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
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
            st.session_state.example_prompt = "white young female golden retriever in Selangor vaccinated under 300"
    with col3:
        if st.button("🏥 My cat is sick, what should I do?", use_container_width=True):
            st.session_state.example_prompt = "My cat is vomiting and not eating. What should I do?"

    prompt = st.chat_input("Ask me anything about pets...")
    if hasattr(st.session_state, 'example_prompt'):
        prompt = st.session_state.example_prompt
        delattr(st.session_state, 'example_prompt')

    if prompt:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Assistant thinking / routing
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    # 1) ALWAYS use your intent model to decide the route
                    intent = classify_intent_via_model(chatbot, prompt)

                    # 2) For pet care: just run your normal ChatbotPipeline (RAG)
                    if intent == INTENT_PETCARE:
                        reply = chatbot.handle_message(prompt)
                        st.markdown(reply)
                        st.session_state.messages.append({"role": "assistant", "content": reply})
                        return

                    # 3) For adoption: run the dedicated adoption search pipeline
                    if intent == INTENT_ADOPTION:
                        if not mr_ok:
                            msg = "Pet search is currently unavailable."
                            st.error(msg)
                            st.session_state.messages.append({"role": "assistant", "content": msg})
                            return

                        res_df, used_facets, hits, meet_all_mask = adoption_search(prompt, ner, azure_env)

                        # "Searching..." line with facets used THIS turn
                        h = {k: v for k, v in used_facets.items() if k in {"animal","breed","state","gender","colors_any"} and v}
                        age_groups = used_facets.get("age_groups_pref", [])
                        parts = []
                        if h.get("animal"): parts.append(h["animal"])
                        if h.get("breed"): parts.append(h["breed"])
                        if h.get("gender"): parts.append(h["gender"])
                        if age_groups: parts.append("/".join(age_groups))
                        if h.get("state"): parts.append(f"in {str(h['state']).title()}")
                        search_line = "Got it! Searching" + (" for " + " ".join([p for p in parts if p]) if parts else "…")
                        st.markdown(search_line)

                        counts = used_facets.get("counts", {}) or {}
                        exact_x = int(counts.get("strict_exact_total", 0))
                        shown_n = int(res_df.shape[0]) if res_df is not None else 0

                        if res_df is None or res_df.empty:
                            st.info("No results. Try relaxing filters or removing constraints.")
                        else:
                            if exact_x >= TOPK_CARDS:
                                st.success(f"🥳 Found {exact_x} pets matching your description! Showing {TOPK_CARDS}.")
                            elif shown_n > exact_x:
                                st.success(f"🥳 Found {exact_x} exact in-state matches; adding a few similar ones too.")
                            else:
                                st.success(f"🥳 Found {exact_x} pets matching your description!")

                            # Conversational recap of facets used
                            soft = used_facets.get("soft_preferences", {}) or {}
                            bits = []
                            if h.get("gender"): bits.append(str(h["gender"]))
                            if h.get("breed"): bits.append(str(h["breed"]))
                            if h.get("animal") and not h.get("breed"): bits.append(str(h["animal"]))
                            where_bit = f"in {str(h['state']).title()}" if h.get("state") else None
                            if where_bit: bits.append(where_bit)
                            extras = []
                            if age_groups: extras.append("/".join(age_groups) + " age group")
                            if h.get("colors_any"): extras.append("color(s): " + ", ".join(h["colors_any"]))
                            if soft.get("fee_cap"):
                                try: extras.append(f"budget under {int(float(soft['fee_cap']))}")
                                except Exception: extras.append(f"budget under {soft['fee_cap']}")
                            if soft.get("prefer_vaccinated"): extras.append("vaccinated")
                            if soft.get("prefer_dewormed"):   extras.append("dewormed")
                            if soft.get("prefer_neutered"):   extras.append("neutered")
                            if soft.get("prefer_spayed"):     extras.append("spayed")
                            if soft.get("prefer_healthy"):    extras.append("healthy condition")
                            lead = "You asked for " + " ".join(bits) if bits else "You asked for a pet"
                            tail = (", " + ", ".join(extras)) if extras else ""
                            st.caption(lead + tail + ".")

                            # Render up to 6 cards in a 3-col grid. Green if meet ALL strict+soft.
                            render_results_grid(res_df, meet_all_mask, max_cols=GRID_COLS)

                        # Store a brief history line (not full HTML)
                        st.session_state.messages.append({"role": "assistant", "content": search_line})
                        return

                    # If we get here, the model returned something unexpected; treat as pet_care safely.
                    fallback_msg = f"(Intent '{intent}' not recognized. Defaulting to Q&A.)"
                    st.info(fallback_msg)
                    reply = chatbot.handle_message(prompt)
                    st.markdown(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})

                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()
