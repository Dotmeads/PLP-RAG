# -*- coding: utf-8 -*-
"""
Pawfect Match — Chat + Smart Pet Search
- Hugging Face NER & Intent routed via ChatbotPipeline (no Azure NER download)
- STRONG hard filters: animal/breed/gender/state (animal & breed always strict)
- Auto-relax when <6 results, in order: state -> color -> age -> gender
- Soft preferences (bucket-first priority):
    • Age groups: puppy/kitten (0–12), young (12–36), adult (36–84), senior (84+)
      supports "<1", "less than 1 year", ">1 year", "over 3 years", exact ages ("2 years")
    • Vaccinated / dewormed / neutered / spayed / healthy
    • Low adoption fee / fee cap
- Hybrid ranking: BM25 + Embeddings with soft-feature bonus
- Exact matches (ALL strict + soft) shown first (green), followed by close matches by similarity
- Cards highlight light-green ONLY if they satisfy ALL strict + soft requirements
- Facets persist across turns, but animal/breed changes clear persisted facets
- Users can remove/clear constraints: "remove state", "remove breed", "clear all", etc.
- After removal: conversational status lines showing **count of exact matches** (hard+soft)
- If no facets remain: show random pets + invite user to enter fresh constraints
- Top "➕ New search / Clear history" button resets everything
"""

import os, re, json, ast, html
from typing import List, Dict, Any, Tuple, Optional, Set

import streamlit as st
st.set_page_config(page_title="Pawfect Match", page_icon="🐾", layout="wide")

import sys
import numpy as np
import pandas as pd

from ui_style import apply_global_ui, render_status_bar, render_loading_box

# --------------------------
# Project path & imports
# --------------------------
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# RAG + Chatbot (HF NER & intent model are inside this pipeline)
from rag_system.proposed_rag_system import ProposedRAGManager
from chatbot_flow.chatbot_pipeline import ChatbotPipeline  # exposes intent + HF NER

# Retrieval stack
from pet_retrieval.config import get_blob_settings, local_mr_dir, local_pets_csv_path
from pet_retrieval.azure_io import download_prefix_flat, smart_download_single_blob
from pet_retrieval.models import load_mr_model, load_faiss_index
from pet_retrieval.retrieval import only_text, BM25, emb_search

# Optional fuzzy (safe import)
try:
    from rapidfuzz import process, fuzz
    _HAS_FUZZ = True
except Exception:
    _HAS_FUZZ = False
    process = fuzz = None

# =========================================================
# CONSTANTS / UI CONFIG
# =========================================================
TOPK_CARDS = 6
GRID_COLS = 3
LEX_POOL = 2000
EMB_POOL = 200
HYBRID_W = {"lex": 0.1, "emb": 0.9}

# Age groups (months)
AGE_GROUPS = {
    "puppy/kitten": (0, 12),
    "young": (12, 36),
    "adult": (36, 84),
    "senior": (84, None),
}
AGE_GROUP_KEYS = list(AGE_GROUPS.keys())

# Malaysia states
MALAYSIA_STATES = {
    "johor","kedah","kelantan","malacca","melaka","negeri sembilan","pahang",
    "penang","pulau pinang","perak","perlis","sabah","sarawak","selangor",
    "terengganu","kuala lumpur","labuan","putrajaya"
}
STATE_ALIASES = {"kl":"kuala lumpur","pulau pinang":"penang","melaka":"malacca","kuala lumpur":"kuala lumpur"}

def _norm_state(s: str) -> str:
    x = (s or "").strip().lower()
    return STATE_ALIASES.get(x, x)

# Color normalization / whitelist
COLOR_WHITELIST = {
    "white","black","brown","gray","grey","cream","beige","tan","yellow","gold","golden",
    "orange","ginger","red","chocolate","liver","blue","silver","fawn","apricot",
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
    if not c:
        return ""
    c = COLOR_SYNONYMS.get(c, c)
    return c if c in COLOR_WHITELIST else ""

# =========================================================
# Helpers
# =========================================================
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
    if v in {"true","yes","y","1"}: return f"✔ {label}"
    if v in {"false","no","n","0"}: return f"✖ {label}"
    if v in {"unknown","nan",""}: return f"➖ {label}"
    return f"ℹ️ {label}: {x}"

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
    if re.search(r"\b(fully\s+)?vaccinated\b", s): return True
    if re.search(r"\b(not\s+dewormed|no\s+deworm)\b", s): return False
    if re.search(r"\bde[-\s]?wormed\b", s): return True
    if re.search(r"\b(intact|not\s+neuter(?:ed)?|not\s+spay(?:ed)?)\b", s): return False
    if re.search(r"\b(neuter(?:ed)?|castrat(?:e|ed|ion)|fixed|sterilis(?:e|ed|ation)|spay(?:ed)?)\b", s): return True
    return None

def _pick_bool(row: dict, cols: List[str]) -> Optional[bool]:
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

# ===== Age parsing to groups =====
def _to_months(val: float, unit: str) -> int:
    return int(round(val * (12 if unit.startswith("y") else 1)))

def _group_for_exact_months(months: float) -> str:
    for g, (lo, hi) in AGE_GROUPS.items():
        lo_m = lo if lo is not None else -1e9
        hi_m = hi if hi is not None else 1e9
        if months >= lo_m and months < hi_m:
            return g
    return "adult"

def parse_age_group_prefs(text: str) -> Set[str]:
    t = (text or "").lower()
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
    def add_groups_for_threshold(op: str, months: float):
        if op in ("<", "lt", "less"):
            if months <= 12: prefs.add("puppy/kitten")
            elif months <= 36: prefs.update(["puppy/kitten","young"])
            else: prefs.update(["puppy/kitten","young","adult"])
        elif op in (">", "gt", "more", "atleast", ">=", "ge"):
            if months < 12: prefs.update(["young","adult","senior"])
            elif months < 36: prefs.update(["adult","senior"])
            else: prefs.add("senior")
    for pat in comp_patterns:
        for m in re.finditer(pat, t):
            op_raw = m.group(1)
            val = float(m.group(2)); unit = m.group(3)
            months = _to_months(val, unit)
            op = op_raw.strip().lower()
            if op in {"less than","under"}: op = "less"
            if op in {"more than","over","at least"}: op = "more"
            add_groups_for_threshold(op, months)
    if not prefs:
        m = re.search(r"\b(\d+(?:\.\d+)?)\s*(years?|yrs?|y|months?|mos?)\b", t)
        if m:
            months = _to_months(float(m.group(1)), m.group(2))
            prefs.add(_group_for_exact_months(months))
    return prefs

def parse_soft_prefs_from_text(text: str) -> Dict[str, Any]:
    t = (text or "").lower()
    prefer_vaccinated = bool(re.search(r"\b(fully\s+)?vaccinated\b", t))
    prefer_dewormed   = bool(re.search(r"\bde[-\s]?wormed\b", t))
    prefer_neutered   = bool(re.search(r"\b(neuter(?:ed)?|fixed|castrat(?:e|ed|ion))\b", t))
    prefer_spayed     = bool(re.search(r"\bspay(?:ed)?\b", t))
    prefer_healthy    = bool(re.search(r"\b(healthy|good\s+health|good\s+condition)\b", t))
    fee_cap = None
    m_fee = re.search(r"(?:fee|adoption\s+fee)\s*(?:under|<|<=)?\s*(\d{2,5})", t)
    if m_fee:
        try: fee_cap = float(m_fee.group(1))
        except Exception: fee_cap = None
    age_groups = sorted(list(parse_age_group_prefs(t)))
    return {
        "prefer_vaccinated": prefer_vaccinated,
        "prefer_dewormed": prefer_dewormed,
        "prefer_neutered": prefer_neutered,
        "prefer_spayed": prefer_spayed,
        "prefer_healthy": prefer_healthy,
        "prefer_low_fee": fee_cap is not None,
        "fee_cap": fee_cap,
        "age_groups_pref": age_groups,
    }

# =========================================================
# Bootstrap RAG & Chatbot (HF NER + Intent)
# =========================================================
@st.cache_resource(show_spinner=False)
def bootstrap_rag_system():
    try:
        rag = ProposedRAGManager()
        docs_dir = os.path.join(project_root, "documents")
        if os.path.exists(docs_dir):
            rag.add_directory(docs_dir)
        bot = ChatbotPipeline(rag)  # contains IntentClassifier + HF NER
        return rag, bot
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {e}")
        return None, None

# =========================================================
# Bootstrap Search (BM25, Embeddings, FAISS, Pets CSV)
# =========================================================
@st.cache_resource(show_spinner=False)
def bootstrap_search_components():
    try:
        cfg = get_blob_settings()
        conn = cfg["connection_string"]

        with st.spinner("Downloading Matching/Ranking model..."):
            download_prefix_flat(conn, cfg["ml_container"], cfg["mr_prefix"], local_mr_dir())
        with st.spinner("Downloading pet CSV..."):
            smart_download_single_blob(conn, cfg["pets_container"], cfg["pets_csv_blob"], local_pets_csv_path())

        student, doc_ids, doc_vecs = load_mr_model(local_mr_dir())
        faiss_index = load_faiss_index(local_mr_dir(), dim=doc_vecs.shape[1])

        dfp = pd.read_csv(local_pets_csv_path())
        # Normalize key columns
        for c in ("animal","gender","state","breed","size","fur_length","condition","color","adoption_fee"):
            if c in dfp.columns:
                dfp[c] = dfp[c].astype(str).fillna("").str.strip().str.lower()

        # BM25 corpus
        text_col = "doc" if "doc" in dfp.columns else ("description_clean" if "description_clean" in dfp.columns else "description")
        docs_raw = {int(i): only_text(str(t)) for i, t in zip(dfp.index, dfp[text_col].fillna("").tolist())}
        bm25 = BM25().fit(docs_raw)

        # Breed catalog (lowercased)
        breed_catalog = sorted(set([b for b in dfp.get("breed", pd.Series([], dtype=str)).astype(str).str.lower().tolist() if b]))
        return {
            "cfg": cfg,
            "student": student, "doc_ids": doc_ids, "doc_vecs": doc_vecs,
            "faiss_index": faiss_index,
            "dfp": dfp,
            "bm25": bm25,
            "breed_catalog": breed_catalog
        }
    except Exception as e:
        st.error(f"Failed to initialize search components: {e}")
        return None

# =========================================================
# Entities → Facets (from ChatbotPipeline HF NER)
# =========================================================
def _entities_to_facets(ents: Dict[str, str], raw_query: str) -> Dict[str, Any]:
    facets: Dict[str, Any] = {}
    def _norm(x): return str(x or "").strip().lower()

    pet_type = _norm(ents.get("PET_TYPE")) if ents and ents.get("PET_TYPE") else ""
    rq = (raw_query or "").lower()
    if not pet_type:
        if re.search(r"\b(cat|kitten|kitties)\b", rq): pet_type = "cat"
        elif re.search(r"\b(dog|puppy|pup)\b", rq): pet_type = "dog"
    if pet_type in {"dog","cat"}:
        facets["animal"] = pet_type

    if ents and ents.get("STATE"):
        facets["state"] = _norm(ents["STATE"])
    if ents and ents.get("BREED"):
        facets["breed"] = _norm(ents["BREED"])
    if ents and ents.get("GENDER"):
        g = _norm(ents["GENDER"])
        if g.startswith("m"): facets["gender"] = "male"
        elif g.startswith("f"): facets["gender"] = "female"

    if ents and ents.get("COLOR"):
        cc = normalize_color(_norm(ents["COLOR"]))
        if cc:
            facets["color"] = cc
    if ents and ents.get("SIZE"):
        facets["size"] = _norm(ents["SIZE"])
    if ents and ents.get("FURLENGTH"):
        facets["fur_length"] = _norm(ents["FURLENGTH"])

    facets["soft"] = parse_soft_prefs_from_text(raw_query)
    return facets

# =========================================================
# Constraint Removal Parsing + Helper
# =========================================================
def is_constraint_removal_query(query: str) -> bool:
    t = (query or "").lower()
    return bool(re.search(r"\b(remove|clear|reset)\b", t))

def apply_constraint_removals(prev_facets: Dict[str, Any], query: str) -> Tuple[Dict[str, Any], List[str], bool, Set[str]]:
    """
    Returns: (updated_facets, removed_labels, cleared_all, removed_keys)
    """
    t = (query or "").strip().lower()
    facets = dict(prev_facets) if prev_facets else {}
    removed = []
    removed_keys: Set[str] = set()
    cleared_all = False

    if re.search(r"\b(clear|remove)\s+(all|everything|constraints|filters|facets)\b", t) or re.search(r"\breset\b", t):
        return {}, ["all constraints"], True, {"animal","breed","gender","state","color","size","fur_length","soft"}

    # Remove state by keyword
    if re.search(r"\b(remove|clear)\s+state\b", t):
        if "state" in facets:
            removed.append(f"state: {facets['state']}")
            facets.pop("state", None)
            removed_keys.add("state")

    # Remove state by value mention
    for s in MALAYSIA_STATES:
        if re.search(rf"\bremove\s+{re.escape(s)}\b", t):
            if facets.get("state") and _norm_state(facets["state"]) == _norm_state(s):
                removed.append(f"state: {facets['state']}")
                facets.pop("state", None)
                removed_keys.add("state")

    # Hard facets
    if re.search(r"\b(remove|clear)\s+animal\b", t):
        if "animal" in facets: removed.append(f"animal: {facets['animal']}"); facets.pop("animal", None); removed_keys.add("animal")
    if re.search(r"\b(remove|clear)\s+breed\b", t):
        if "breed" in facets: removed.append(f"breed: {facets['breed']}"); facets.pop("breed", None); removed_keys.add("breed")
    m_rb = re.search(r"\bremove\s+breed\s+([a-z ]+)\b", t)
    if m_rb and facets.get("breed") and facets["breed"] == m_rb.group(1).strip():
        removed.append(f"breed: {facets['breed']}"); facets.pop("breed", None); removed_keys.add("breed")

    if re.search(r"\b(remove|clear)\s+gender\b", t):
        if "gender" in facets: removed.append(f"gender: {facets['gender']}"); facets.pop("gender", None); removed_keys.add("gender")
    m_rg = re.search(r"\bremove\s+gender\s+(male|female)\b", t)
    if m_rg and facets.get("gender") and facets["gender"] == m_rg.group(1):
        removed.append(f"gender: {facets['gender']}"); facets.pop("gender", None); removed_keys.add("gender")

    if re.search(r"\b(remove|clear)\s+color\b", t):
        if "color" in facets: removed.append(f"color: {facets['color']}"); facets.pop("color", None); removed_keys.add("color")
    m_rc = re.search(r"\bremove\s+color\s+([a-z ]+)\b", t)
    if m_rc and facets.get("color"):
        col = normalize_color(m_rc.group(1))
        if col and facets["color"] == col:
            removed.append(f"color: {facets['color']}"); facets.pop("color", None); removed_keys.add("color")

    if re.search(r"\b(remove|clear)\s+size\b", t):
        if "size" in facets: removed.append(f"size: {facets['size']}"); facets.pop("size", None); removed_keys.add("size")
    if re.search(r"\b(remove|clear)\s+(fur|fur\s*length|furlength)\b", t):
        if "fur_length" in facets: removed.append(f"fur_length: {facets['fur_length']}"); facets.pop("fur_length", None); removed_keys.add("fur_length")

    # Soft prefs
    soft = dict(facets.get("soft", {}) or {})
    soft_removed = []

    if re.search(r"\b(remove|clear)\s+age\b", t):
        if soft.get("age_groups_pref"):
            soft_removed.append("age groups")
            soft["age_groups_pref"] = []
            removed_keys.add("soft")

    if re.search(r"\b(remove|clear)\s+fee\b", t):
        if soft.get("fee_cap") is not None or soft.get("prefer_low_fee"):
            soft_removed.append("fee cap/low-fee")
            soft["fee_cap"] = None
            soft["prefer_low_fee"] = False
            removed_keys.add("soft")

    for key, label in [
        ("prefer_vaccinated","vaccinated"),
        ("prefer_dewormed","dewormed"),
        ("prefer_neutered","neutered"),
        ("prefer_spayed","spayed"),
        ("prefer_healthy","healthy"),
    ]:
        if re.search(rf"\bremove\s+{label}\b", t):
            if soft.get(key):
                soft_removed.append(label)
                soft[key] = False
                removed_keys.add("soft")

    if soft_removed or "age_groups_pref" in soft or "fee_cap" in soft:
        facets["soft"] = soft
        removed.extend(soft_removed)
    elif "soft" in facets and soft == {}:
        facets.pop("soft", None)

    return facets, removed, cleared_all, removed_keys

# =========================================================
# Facet persistence helpers (clear when animal/breed changes)
# + Blocked facets persistence
# =========================================================
def get_persisted_facets() -> Dict[str, Any]:
    return st.session_state.get("last_facets", {}) or {}

def set_persisted_facets(facets: Dict[str, Any]):
    st.session_state["last_facets"] = {k: v for k, v in facets.items() if v not in (None, "", [], {}, False)}

def get_blocked_facets() -> Set[str]:
    return set(st.session_state.get("blocked_facets", set()))

def set_blocked_facets(blocked: Set[str]):
    st.session_state["blocked_facets"] = set(blocked)

def maybe_reset_persistence(new_facets: Dict[str, Any]) -> bool:
    prev = get_persisted_facets()
    new_animal = new_facets.get("animal")
    new_breed  = new_facets.get("breed")
    changed = False
    if prev.get("animal") and new_animal and prev["animal"] != new_animal:
        changed = True
    if prev.get("breed") and new_breed and prev["breed"] != new_breed:
        changed = True
    if changed:
        st.session_state["last_facets"] = {}
        st.session_state["blocked_facets"] = set()
    return changed

# ===== Unblock only if prompt explicitly mentions a facet =====
def explicit_add_keys_from_prompt(prompt: str) -> Set[str]:
    t = (prompt or "").lower()
    keys: Set[str] = set()
    # animal
    if re.search(r"\b(dog|puppy|pup)\b", t) or re.search(r"\b(cat|kitten|kitties)\b", t):
        keys.add("animal")
    # gender
    if re.search(r"\bmale\b", t) or re.search(r"\bfemale\b", t):
        keys.add("gender")
    # state names / "in <state>"
    for s in MALAYSIA_STATES:
        if re.search(rf"\b{s}\b", t):
            keys.add("state")
            break
    if re.search(r"\bin\s+(johor|kedah|kelantan|malacca|melaka|negeri sembilan|pahang|penang|pulau pinang|perak|perlis|sabah|sarawak|selangor|terengganu|kuala lumpur|labuan|putrajaya)\b", t):
        keys.add("state")
    # color
    for word in re.findall(r"[a-z]+", t):
        if normalize_color(word):
            keys.add("color")
            break
    # size / fur
    if re.search(r"\b(small|medium|large|xl)\b", t):
        keys.add("size")
    if re.search(r"\b(short|long)\s*fur\b", t) or re.search(r"\bfurlength|fur length\b", t):
        keys.add("fur_length")
    # breed heuristic
    if re.search(r"\bbreed\b", t) or re.search(r"\bpoodle|ragdoll|retriever|husky|persian|siamese|chihuahua|beagle|pug|bulldog|gsd\b", t):
        keys.add("breed")
    return keys

# =========================================================
# Strict filtering with stepwise relaxation
# =========================================================
def _apply_filters_once(dfp: pd.DataFrame,
                        facets: Dict[str, Any],
                        use_state: bool,
                        relax_state_to_cross: bool,
                        use_color: bool,
                        use_age: bool,
                        use_gender: bool) -> Tuple[pd.DataFrame, int]:
    """Return candidate pool and 'strict_in_state_count' (meaningful only when use_state is True)."""
    df = dfp.copy()

    # Always strict on animal/breed if provided
    if facets.get("animal"):
        df = df[df["animal"] == facets["animal"]]
        if df.empty: return df, 0
    if facets.get("breed"):
        pattern = rf"\b{re.escape(facets['breed'])}\b"
        df = df[df["breed"].str.contains(pattern, case=False, na=False)]
        if df.empty: return df, 0

    # Optional strict filters
    if use_gender and facets.get("gender"):
        df = df[df["gender"] == facets["gender"]]
        if df.empty: return df, 0

    strict_in_state = 0
    if use_state and facets.get("state"):
        strict_df = df[df["state"] == facets["state"]]
        strict_in_state = len(strict_df)
        if strict_df.empty:
            return strict_df, strict_in_state
        if relax_state_to_cross:
            cross_df = df[df["state"] != facets["state"]]
            df = pd.concat([strict_df, cross_df]).drop_duplicates()
        else:
            df = strict_df

    if use_color and facets.get("color"):
        # strict color contains
        df = df[df["color"].str.contains(rf"\b{re.escape(facets['color'])}\b", case=False, na=False)]
        if df.empty: return df, strict_in_state

    if use_age and facets.get("soft", {}).get("age_groups_pref"):
        groups = facets["soft"]["age_groups_pref"]
        def _in_age_group_local(age_months: Optional[float], group: str) -> bool:
            if age_months is None: return False
            try: m = float(age_months)
            except Exception: return False
            lo, hi = AGE_GROUPS[group]
            lo_m = lo if lo is not None else -1e9
            hi_m = hi if hi is not None else 1e9
            return (m >= lo_m) and (m < hi_m)
        def _age_ok(row):
            return any(_in_age_group_local(row.get("age_months"), g) for g in groups)
        df = df[df.apply(_age_ok, axis=1)]
        if df.empty: return df, strict_in_state

    return df, strict_in_state

def build_relaxed_pool(dfp: pd.DataFrame, facets: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Try progressively relaxing filters to reach TOPK_CARDS:
    state -> color -> age -> gender   (animal & breed always strict)
    Returns: (df_pool, relax_meta)
    """
    steps = [
        dict(use_state=True,  relax_state_to_cross=False, use_color=True,  use_age=True,  use_gender=True,  tag="strict_all"),
        dict(use_state=True,  relax_state_to_cross=True,  use_color=True,  use_age=True,  use_gender=True,  tag="relax_state"),
        dict(use_state=True,  relax_state_to_cross=True,  use_color=False, use_age=True,  use_gender=True,  tag="relax_color"),
        dict(use_state=True,  relax_state_to_cross=True,  use_color=False, use_age=False, use_gender=True,  tag="relax_age"),
        dict(use_state=True,  relax_state_to_cross=True,  use_color=False, use_age=False, use_gender=False, tag="relax_gender"),
        dict(use_state=False, relax_state_to_cross=False, use_color=True,  use_age=True,  use_gender=True,  tag="no_state_strict"),
        dict(use_state=False, relax_state_to_cross=False, use_color=False, use_age=True,  use_gender=True,  tag="no_state_relax_color"),
        dict(use_state=False, relax_state_to_cross=False, use_color=False, use_age=False, use_gender=True,  tag="no_state_relax_age"),
        dict(use_state=False, relax_state_to_cross=False, use_color=False, use_age=False, use_gender=False, tag="no_state_relax_gender"),
    ]
    if not facets.get("state"):
        steps = [s for s in steps if not s["use_state"]]

    chosen_df = pd.DataFrame()
    chosen_meta = {}
    for cfg in steps:
        df_try, strict_in_state = _apply_filters_once(
            dfp, facets,
            use_state=cfg["use_state"],
            relax_state_to_cross=cfg["relax_state_to_cross"],
            use_color=cfg["use_color"],
            use_age=cfg["use_age"],
            use_gender=cfg["use_gender"],
        )
        size_try = len(df_try)
        chosen_meta = dict(cfg)
        chosen_meta["strict_in_state_count"] = strict_in_state
        chosen_meta["pool_size"] = size_try
        if size_try >= TOPK_CARDS:
            chosen_df = df_try
            break
        chosen_df = df_try  # keep best-so-far
    return chosen_df, chosen_meta

# =========================================================
# Soft preferences + ranking & highlight mask
# =========================================================
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

def _health_fee_flags(row: pd.Series, soft: Dict[str, Any]) -> Dict[str, int]:
    flags = {"vaccinated":0,"dewormed":0,"neutered":0,"spayed":0,"healthy":0,"fee_ok":0}
    v, d, n, s = _resolve_health_flags(row)
    if soft.get("prefer_vaccinated") and v is True: flags["vaccinated"] = 1
    if soft.get("prefer_dewormed")   and d is True: flags["dewormed"]   = 1
    if soft.get("prefer_neutered")   and n is True: flags["neutered"]   = 1
    if soft.get("prefer_spayed")     and s is True: flags["spayed"]     = 1
    if soft.get("prefer_healthy"):
        cond = str(row.get("condition","")).strip().lower()
        if cond in {"healthy","good"}: flags["healthy"] = 1
    if soft.get("prefer_low_fee") or (soft.get("fee_cap") is not None):
        cap = soft.get("fee_cap") or 300.0
        try:
            fee = float(row.get("adoption_fee"))
            if fee <= cap: flags["fee_ok"] = 1
        except Exception:
            pass
    return flags

def _feature_bonus(row: pd.Series, facets: Dict[str, Any]) -> float:
    soft = facets.get("soft", {}) or {}
    bonus = 0.0
    v, d, n, s = _resolve_health_flags(row)
    w_base, w_strong = 0.05, 0.08
    if soft.get("prefer_vaccinated") and v is True: bonus += w_strong
    elif v is True: bonus += w_base
    if soft.get("prefer_dewormed") and d is True: bonus += w_strong
    elif d is True: bonus += w_base
    if soft.get("prefer_neutered") and n is True: bonus += w_strong
    elif n is True: bonus += w_base
    if soft.get("prefer_spayed") and s is True: bonus += w_strong
    elif s is True: bonus += w_base
    if soft.get("prefer_healthy"):
        cond = str(row.get("condition","")).strip().lower()
        if cond in {"healthy","good"}:
            bonus += 0.05
    if soft.get("fee_cap") is not None:
        cap = soft["fee_cap"]
        try:
            fee = float(row.get("adoption_fee"))
            if fee <= cap:
                bonus += 0.08 * (1.0 - max(0.0, min(1.0, fee / cap)))
        except Exception:
            pass
    groups = soft.get("age_groups_pref") or []
    if groups and _age_groups_match(row, groups): bonus += 0.05
    return max(0.0, min(0.35, bonus))

def match_all_strict(row: pd.Series, facets: Dict[str, Any]) -> bool:
    ok = True
    if facets.get("animal"):
        ok &= str(row.get("animal","")).strip().lower() == facets["animal"]
    if facets.get("breed"):
        btxt = str(row.get("breed","")).strip().lower()
        ok &= bool(re.search(rf"\b{re.escape(facets['breed'])}\b", btxt))
    if facets.get("gender"):
        ok &= str(row.get("gender","")).strip().lower() == facets["gender"]
    if facets.get("state"):
        ok &= str(row.get("state","")).strip().lower() == facets["state"]
    if facets.get("color"):
        ok &= bool(re.search(rf"\b{re.escape(facets['color'])}\b", str(row.get("color","")), flags=re.I))
    # age is soft, handled separately
    return bool(ok)

def match_all_soft(row: pd.Series, facets: Dict[str, Any]) -> bool:
    soft = facets.get("soft", {}) or {}
    soft_flags = _health_fee_flags(row, soft)
    if soft.get("age_groups_pref") and not _age_groups_match(row, soft["age_groups_pref"]):
        return False
    if soft.get("prefer_vaccinated") and soft_flags["vaccinated"] != 1: return False
    if soft.get("prefer_dewormed")   and soft_flags["dewormed"]   != 1: return False
    if soft.get("prefer_neutered")   and soft_flags["neutered"]   != 1: return False
    if soft.get("prefer_spayed")     and soft_flags["spayed"]     != 1: return False
    if soft.get("prefer_healthy")    and soft_flags["healthy"]    != 1: return False
    if (soft.get("prefer_low_fee") or soft.get("fee_cap") is not None) and soft_flags["fee_ok"] != 1: return False
    return True

# --------- exact match count (ALL hard + soft) ----------
def count_exact_matches(dfp: pd.DataFrame, facets: Dict[str, Any]) -> int:
    """Count pets that satisfy ALL hard facets (animal/breed/gender/state/color) AND ALL soft prefs."""
    if dfp is None or dfp.empty:
        return 0
    df = dfp.copy()
    # strict hard filters
    if facets.get("animal"):
        df = df[df["animal"] == facets["animal"]]
    if facets.get("breed"):
        df = df[df["breed"].str.contains(rf"\b{re.escape(facets['breed'])}\b", case=False, na=False)]
    if facets.get("gender"):
        df = df[df["gender"] == facets["gender"]]
    if facets.get("state"):
        df = df[df["state"] == facets["state"]]
    if facets.get("color"):
        df = df[df["color"].str.contains(rf"\b{re.escape(facets['color'])}\b", case=False, na=False)]
    if df.empty:
        return 0
    # soft check row-wise
    return int(df.apply(lambda r: match_all_soft(r, facets), axis=1).sum())

# =========================================================
# Ranking & Highlighting
# =========================================================
def hybrid_rank_and_highlight(query: str,
                              env: Dict[str, Any],
                              facets: Dict[str, Any],
                              df_pool: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Ranks candidates with hybrid (BM25 + embeddings) + soft bonus.
    Results that satisfy ALL strict + soft facets appear FIRST (green), then remaining by similarity/bonus.
    """
    student = env["student"]; doc_ids = env["doc_ids"]; doc_vecs = env["doc_vecs"]
    faiss_index = env["faiss_index"]; bm25 = env["bm25"]

    # boosted query with facet bits
    facet_bits = []
    for k in ["animal","breed","gender","color","size","fur_length","state"]:
        if facets.get(k): facet_bits.append(str(facets[k]))
    boost_q = (query or "").strip()
    if facet_bits:
        boost_q = (boost_q + " " + " ".join(facet_bits)).strip()

    # scores
    lex_all = bm25.search(only_text(boost_q), topk=LEX_POOL)
    slex = {int(idx): float(s) for idx, s in lex_all if idx in df_pool.index}
    emb_all = emb_search(boost_q, student, doc_ids, doc_vecs, pool_topn=EMB_POOL, faiss_index=faiss_index)
    semb = {int(pid): float(s) for pid, s in emb_all if pid in df_pool.index}

    def _mm(d):
        if not d: return {}
        vals = np.fromiter(d.values(), dtype=float)
        lo, hi = float(vals.min()), float(vals.max())
        den = (hi - lo) or 1.0
        return {k: (v - lo) / den for k, v in d.items()}
    nlex, nemb = _mm(slex), _mm(semb)
    base_combo = {idx: HYBRID_W["lex"]*nlex.get(idx, 0.0) + HYBRID_W["emb"]*nemb.get(idx, 0.0)
                  for idx in set(nlex) | set(nemb)}
    if not base_combo:
        return pd.DataFrame(), np.array([], dtype=bool)

    # Add soft-feature bonus
    combo_with_bonus: Dict[int, float] = {}
    for i in df_pool.index:
        combo_with_bonus[int(i)] = base_combo.get(int(i), 0.0) + _feature_bonus(df_pool.loc[i], facets)

    # exact matches first
    def _full_ok(i: int) -> bool:
        row = df_pool.loc[i]
        return match_all_strict(row, facets) and match_all_soft(row, facets)

    exact_ids = [i for i in df_pool.index if _full_ok(int(i))]
    rest_ids  = [i for i in df_pool.index if i not in exact_ids]

    exact_sorted = sorted(exact_ids, key=lambda i: combo_with_bonus.get(int(i), 0.0), reverse=True)
    rest_sorted  = sorted(rest_ids,  key=lambda i: combo_with_bonus.get(int(i), 0.0), reverse=True)

    chosen = (exact_sorted + rest_sorted)[:TOPK_CARDS]
    if not chosen:
        return pd.DataFrame(), np.array([], dtype=bool)

    res_df = df_pool.loc[chosen].copy().reset_index(drop=True)

    # highlight mask: True only for exact matches (strict + soft)
    mask = np.array([_full_ok(int(i)) for i in chosen], dtype=bool)

    return res_df, mask

# =========================================================
# Cards / Grid rendering
# =========================================================
def render_pet_card(row: pd.Series, highlight: bool = False):
    name = str(row.get("name") or "Pet")
    url = str(row.get("url") or "")
    animal = (row.get("animal") or "").title()
    breed = str(row.get("breed") or "—").title()
    gender = (row.get("gender") or "—").title()
    state = (row.get("state") or "—").title()
    color = str(row.get("color") or "—").title()
    age_mo = row.get("age_months")
    age_txt = _age_text_from_months(age_mo)
    size = str(row.get("size") or "—").title()
    fur = str(row.get("fur_length") or "—").title()
    cond = str(row.get("condition") or "—").title()

    v_b, d_b, n_b, s_b = (
        _badge_bool(row.get("vaccinated"), "vaccinated"),
        _badge_bool(row.get("dewormed"), "dewormed"),
        _badge_bool(row.get("neutered"), "neutered"),
        _badge_bool(row.get("spayed"), "spayed"),
    )

    img_url = _first_photo_url_from_row(row)
    desc_raw = str(row.get("description_clean") or "").strip()
    desc_safe = html.escape(desc_raw).replace("\n", "<br />") if desc_raw else ""

    # --- Visual theme ---
    bg = "#FFFFFF"
    border_left = "5px solid #ffc6bb"
    title_color = "#e85c7d"
    text_color = "#444"

    # --- Scrollable description style ---
    st.markdown("""
        <style>
        .desc-box {
            background: #fff9f8;
            border: 1px solid #ffe0d6;
            border-radius: 8px;
            padding: 8px 10px;
            max-height: 120px;          
            overflow-y: auto;         
            line-height: 1.55;
            color: #444;
        }
        </style>
    """, unsafe_allow_html=True)

    if img_url:
        img_html = (
            "<img src='" + html.escape(img_url) + "' alt='photo' loading='lazy' "
            "style='max-width:100%;max-height:100%;object-fit:contain;"
            "object-position:center;display:block;' />"
        )
    else:
        img_html = "<div style='color:#b3b3b3;'>No photo available</div>"

    # --- Card HTML ---
    st.markdown(
        "<div style='background:" + bg + "; border-radius:15px; padding:14px; margin:8px 0; "
        "border-left:" + border_left + "; box-shadow:0 4px 15px rgba(0,0,0,0.05);'>"

        "<div style='font-size:1.1rem; font-weight:800; color:" + title_color + "; margin-bottom:6px;'>"
        + (
            "<a href='" + html.escape(url) + "' target='_blank' "
            "style='text-decoration:none; color:" + title_color + ";'>• " + html.escape(name) + "</a>"
            if url else "• " + html.escape(name)
        )
        + "</div>"

        "<div style='height:220px;width:100%;border-radius:10px;background:#fff9f8;"
        "display:flex;align-items:center;justify-content:center;margin:8px 0 10px;"
        "overflow:hidden;border:1px solid #ffe0d6;'>"
        + img_html +
        "</div>"

        "<div style='color:" + text_color + ";margin-bottom:6px;'>"
        "<strong>" + animal + "</strong> • <strong>Breed:</strong> " + breed +
        " • <strong>Gender:</strong> " + gender + " • <strong>Age:</strong> " + age_txt +
        " • <strong>State:</strong> " + state + "</div>"

        "<div style='color:" + text_color + ";margin-bottom:8px;'>"
        "<strong>Color:</strong> " + color + " • <strong>Size:</strong> " + size +
        " • <strong>Fur:</strong> " + fur + " • <strong>Condition:</strong> " + cond +
        "</div>"

        "<div style='color:#222; margin-bottom:8px;'>" + " | ".join([v_b, d_b, n_b, s_b]) + "</div>"

        + (f"<div class='desc-box'>{desc_safe}</div>" if desc_safe else "")
        + "</div>",
        unsafe_allow_html=True,
    )

def render_grid(df: pd.DataFrame, mask: np.ndarray, max_cols: int = GRID_COLS):
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
                render_pet_card(rows[idx])


     # ---------------- Footer Section (Suggestions + New Search) ----------------
def render_footer():
    st.markdown("""
    <style>
        .footer-section {
            margin-top: 1.2rem;
            padding-top: 0.8rem;
            border-top: 1px solid #ffe0d6;
        }

        /* Arrange columns evenly and center-align */
        div[data-testid="stHorizontalBlock"] {
            align-items: center !important;
        }

        /* Label alignment */
        .footer-label {
            font-weight: 700;
            font-size: 1.05rem;
            color: #3b3b3b;
            display: flex;
            align-items: flex-end;
            justify-content: flex-end;
            white-space: nowrap;
            margin-right: 0.1rem;
            margin-bottom: 15px;
        }

        /* Suggestion buttons (general style) */
        .stButton > button {
            font-size: 0.95rem !important;
            padding: 0.45rem 0.8rem !important;
            border-radius: 10px !important;
            border: 1px solid #ffd6bf !important;
            background: #fff !important;
            color: #3b3b3b !important;
            height: 42px !important;
            width: 100% !important; /* Make buttons fill their container */
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            white-space: nowrap !important;
            transition: all 0.15s ease;
        }
        .stButton > button:hover {
            background: #fff3ef !important;
            transform: translateY(-1px);
        }

        /* New search button styling */
        div[data-testid="stVerticalBlock"] > div.new-search-button-container .stButton > button {
            width: 200px !important;   
            height: 40px !important;
            font-size: 0.9rem !important;
            background-color: red !important; /* Debugging: to confirm selector is working */
        }

        /* tighter gaps */
        div[data-testid="column"] {
            padding-left: 0.2rem !important;
            padding-right: 0.2rem !important;
            margin: 0 !important;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='footer-section'>", unsafe_allow_html=True)

    # --- Single functional row layout ---
    col_new, col_label, col1, col2, col3 = st.columns([0.8, 0.8, 1, 1, 1], gap="small")

    with col_new:
        st.markdown("<div class='footer-newsearch'>", unsafe_allow_html=True)
        if st.button("➕ New search / Clear history", key="new_search_clear"):
            st.session_state["__clear_all__"] = True
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with col_label:
        st.markdown("<div class='footer-label'>💡 Try asking me:</div>", unsafe_allow_html=True)

    with col1:
        if st.button("🐕 What should I feed my puppy?", key="ex1"):
            st.session_state.example_prompt = "What should I feed my puppy?"
            st.rerun()

    with col2:
        if st.button("🏠 Find a poodle in Selangor", key="ex2"):
            st.session_state.example_prompt = "Find a poodle in Selangor"
            st.rerun()

    with col3:
        if st.button("🏥 My cat is sick, what should I do?", key="ex3"):
            st.session_state.example_prompt = "My cat is sick, what should I do?"
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# Main App
# =========================================================
def main():
    apply_global_ui()

    # Initialize systems with auto-clearing loader ---
    loading_area = st.empty()
    render_loading_box(loading_area, "Initializing systems...", "🚀")

    rag, bot = bootstrap_rag_system()
    env = bootstrap_search_components()
    loading_area.empty()

    # If user requested a full clear, safely reset without breaking bot pipeline
    if st.session_state.pop("__clear_all__", False):
        st.session_state["messages"] = []
        st.session_state["last_facets"] = {}
        st.session_state["blocked_facets"] = set()
        if hasattr(st.session_state, "example_prompt"):
            delattr(st.session_state, "example_prompt")

        try:
            if bot is not None and hasattr(bot, "session"):
                bot.session.clear()
                bot.session["state"] = "greeted"
                bot.session["intent"] = None
                bot.session["entities"] = {}
        except Exception as e:
            st.warning(f"Bot reset warning: {e}")

        st.rerun()


    # --- Status bar ---
    rag_ok = rag is not None and bot is not None
    env_ok = env is not None and env.get("dfp") is not None

    pets_detail = f"{len(env['dfp'])} pets available" if env_ok else "Unavailable"
    app_status_ok = rag_ok and env_ok
    app_status_text = "All Systems Ready" if app_status_ok else "Issues Detected"
    
    render_status_bar(rag_ok, env_ok, app_status_ok, app_status_text, pets_detail)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_prompt" not in st.session_state:
        st.session_state.last_prompt = None

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask me anything about pets", key="chat_input_main")
    
    if hasattr(st.session_state, "example_prompt") and st.session_state.example_prompt:
        user_input = st.session_state.example_prompt
        delattr(st.session_state, "example_prompt")
        st.session_state.last_prompt = None
    
    if not user_input or not user_input.strip():
            render_footer()
            return
    
    if user_input == st.session_state.last_prompt:
        render_footer()
        return

    st.session_state.last_prompt = user_input
    prompt = user_input

    # Display user message
    st.markdown('<span style="display:none">.</span>', unsafe_allow_html=True)
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # ------------------------
    # Pet/RAG routing & logic
    # ------------------------
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # PRE-PARSE REMOVALS before calling the bot
            removal_intent = is_constraint_removal_query(prompt)
            prev_facets_for_removal = get_persisted_facets()
            prev_after_removal = None
            removed_labels = []
            cleared_all = False
            removed_keys = set()

            if removal_intent:
                prev_after_removal, removed_labels, cleared_all, removed_keys = apply_constraint_removals(
                    prev_facets_for_removal, prompt
                )
                # Force route to pet search when removing constraints
                if hasattr(bot, "session"):
                    bot.session["intent"] = "find_pet"

                # Persist blocked facets across turns
                blocked = get_blocked_facets()
                if cleared_all:
                    blocked = {"animal","breed","gender","state","color","size","fur_length","soft"}
                blocked |= set(removed_keys)
                set_blocked_facets(blocked)

            # Get the bot's text (for Q&A etc.)
            try:
                reply = bot.handle_message(prompt)
            except Exception as e:
                reply = f"(Chat pipeline error: {e})"

            # Routing
            intent_now = (getattr(bot, "session", None) or {}).get("intent")
            if removal_intent:
                intent_now = "find_pet"

            if not env_ok:
                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})

            # ----- PET SEARCH PATH -----
            if intent_now == "find_pet":
                ents = (bot.session or {}).get("entities", {}) or {}
                new_facets = _entities_to_facets(ents, raw_query=prompt)

                # persistence: clear if animal/breed changed
                reset_done = maybe_reset_persistence(new_facets)
                prev = {} if reset_done else get_persisted_facets()

                # If we already computed removals above, reuse them. Otherwise compute now.
                if removal_intent:
                    base_after_removal = prev_after_removal
                    removal_keys = removed_keys
                else:
                    base_after_removal, _, _, removal_keys = apply_constraint_removals(prev, prompt)

                # --- Blocked facets persist across turns
                blocked = get_blocked_facets()

                # Unblock only if the USER TEXT explicitly mentions the facet
                explicit_add_keys = explicit_add_keys_from_prompt(prompt)
                if explicit_add_keys:
                    blocked = blocked - explicit_add_keys
                    set_blocked_facets(blocked)

                # Do not reintroduce removed/blocked keys from current entities
                for k in (removal_keys | blocked):
                    new_facets.pop(k, None)

                # HARD GUARD: drop COLOR from new_facets unless explicitly present in THIS prompt
                if "color" in new_facets and "color" not in explicit_add_keys:
                    new_facets.pop("color", None)

                # ---- IMPORTANT FIX: do not carry forward COLOR unless explicitly present this turn ----
                base = {} if (removal_intent and cleared_all) else dict(base_after_removal)
                if "color" not in explicit_add_keys:
                    base.pop("color", None)

                # Merge base + new
                merged = dict(base)
                for k in ["animal","breed","gender","state","color","size","fur_length"]:
                    if new_facets.get(k):
                        merged[k] = new_facets[k]

                # merge soft prefs
                soft_prev = dict(base.get("soft", {}) or {})
                soft_new  = dict(new_facets.get("soft", {}) or {})
                for k, v in soft_new.items():
                    if v not in (None, [], {}, False, ""):
                        soft_prev[k] = v
                if any(bool(v) for v in soft_prev.values()):
                    merged["soft"] = soft_prev
                else:
                    merged.pop("soft", None)

                # Persist final facets
                set_persisted_facets(merged)
                facets = merged

                # If no facets at all → show random suggestions + prompt user again
                non_soft_keys = [k for k in facets.keys() if k != "soft"]
                if len(non_soft_keys) == 0 and not (facets.get("soft") and any(facets["soft"].values())):
                    st.info("No active filters right now. Consider taking these fur babies home 🥹 — here are some random sweethearts!")
                    df_all = env["dfp"]
                    if len(df_all) > 0:
                        random_df = df_all.sample(min(TOPK_CARDS, len(df_all)), random_state=None).reset_index(drop=True)
                        render_grid(random_df, np.array([False]*len(random_df)))
                    st.caption("Tell me what you’re looking for — species/breed, gender, age group, color, and state (e.g., **female young poodle in Selangor**).")
                    st.session_state.messages.append({"role":"assistant","content":"(random suggestions shown)"})
                    render_footer()

                # Facets banner
                def chip(label, value, accent=False):
                    color = "#eaffea" if accent else "#fff"
                    border = "#a3e6a3" if accent else "#eee"
                    return (
                        "<span style='background:"+color+";border:1px solid "+border+
                        ";border-radius:8px;padding:4px 8px;margin:2px;display:inline-block;'><strong>"+
                        html.escape(label)+":</strong> "+html.escape(str(value))+"</span>"
                    )

                chips = []
                labels = {"animal":"Animal","breed":"Breed","gender":"Gender","state":"State","color":"Color","size":"Size","fur_length":"Fur"}
                for k, lab in labels.items():
                    if facets.get(k): chips.append(chip(lab, facets[k]))

                soft = facets.get("soft", {}) or {}
                if soft.get("age_groups_pref"):
                    chips.append(chip("Age group", "/".join(soft["age_groups_pref"]), accent=True))
                if soft.get("prefer_vaccinated"): chips.append(chip("Pref", "vaccinated", accent=True))
                if soft.get("prefer_dewormed"):   chips.append(chip("Pref", "dewormed",   accent=True))
                if soft.get("prefer_neutered"):   chips.append(chip("Pref", "neutered",   accent=True))
                if soft.get("prefer_spayed"):     chips.append(chip("Pref", "spayed",     accent=True))
                if soft.get("prefer_healthy"):    chips.append(chip("Pref", "healthy",    accent=True))
                if soft.get("fee_cap") is not None:
                    val = soft["fee_cap"]
                    try: val = int(float(val))
                    except Exception: pass
                    chips.append(chip("Fee cap", f"≤ {val}", accent=True))

                if chips:
                    st.markdown(
                        "<div style='margin:10px 0 4px;color:#374151;'>Facets used (tip: if too few pets, try <em>remove state</em>):</div>"
                        "<div style='margin-bottom:10px;display:flex;flex-wrap:wrap;gap:6px;'>"
                        + "".join(chips) + "</div>",
                        unsafe_allow_html=True
                    )

                # ------- Build pool with stepwise relaxation -------
                df_pool, relax_meta = build_relaxed_pool(env["dfp"], facets)

                # ------- Compute exact-match count (ALL hard + soft) over strict hard filters -------
                exact_total = count_exact_matches(env["dfp"], facets)

                # Conversational status line using EXACT count
                status_msg = None
                if facets.get("state"):
                    if exact_total >= TOPK_CARDS:
                        status_msg = f"Found {exact_total} pets matching all your filters in {facets['state'].title()}! Here are some lovely matches."
                    elif exact_total > 0:
                        status_msg = f"Only {exact_total} pet(s) match all your filters in {facets['state'].title()}. Showing similar fur babies needing a forever home~"
                    else:
                        status_msg = f"No pets match all your filters in {facets['state'].title()}. Trying close matches for you ~"
                else:
                    if exact_total >= TOPK_CARDS:
                        status_msg = f"Found {exact_total} pets matching all your filters! Here are some lovely matches."
                    elif exact_total > 0:
                        status_msg = f"Only {exact_total} pet(s) match all your filters. Showing similar fur babies needing a forever home~"
                    else:
                        status_msg = "No pets match all your filters. Trying close matches for you ~"

                st.markdown(status_msg)

                # ------- Rank & highlight (EXACT first, then remainder by similarity) -------
                if df_pool is None or df_pool.empty:
                    st.info("No pets found. Try adjusting or removing some constraints (e.g. **remove state**).")
                else:
                    res_df, highlight_mask = hybrid_rank_and_highlight(prompt, env, facets, df_pool)
                    if res_df is None or res_df.empty:
                        st.info("No pets found. Try adjusting or removing some constraints (e.g. **remove state**).")
                    else:
                        render_grid(res_df, highlight_mask, max_cols=GRID_COLS)

                st.session_state.messages.append({"role": "assistant", "content": status_msg})

            else:
                # ----- RAG Q&A PATH -----
                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})

    render_footer()

if __name__ == "__main__":
    main()
