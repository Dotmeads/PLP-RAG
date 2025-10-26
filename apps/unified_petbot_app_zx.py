# -*- coding: utf-8 -*-
"""
Pawfect Match — Chat + Smart Pet Search
- Hugging Face NER & Intent via ChatbotPipeline (no Azure NER download)
- STRONG hard filters: animal/breed/gender/state (animal & breed always strict; state can auto-relax)
- Auto-relax when <6 results: state -> color -> age -> gender
- Soft preferences: age buckets, vaccinated/dewormed/neutered/spayed/healthy, low fee/cap
- Hybrid ranking: BM25 + Embeddings + soft-feature bonus
- Exact matches (ALL strict + soft) first (green), then close matches
- Facets persist across turns; animal/breed change clears persisted facets but **preserves blocked facets**
- Constraint removal parsing ("remove state", "clear all", etc.)
- "➕ New search / Clear history" appears **only after** a response; now does a true blank-slate reset.
- Viewport: jumps to the **last assistant reply**; on hard reset, jumps to the very top.
"""

import os, re, json, ast, html, time, random
from typing import List, Dict, Any, Tuple, Optional, Set

import streamlit as st
st.set_page_config(page_title="Pawfect Match", layout="wide")

import sys
import numpy as np
import pandas as pd
import streamlit.components.v1 as components

# --------------------------
# Project path & imports
# --------------------------
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# RAG + Chatbot (HF NER & intent model via this pipeline)
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
EMB_POOL = 1000
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
    if v in {"true","yes","y","1"}: return f"✅ {label}"
    if v in {"false","no","n","0"}: return f"❌ {label}"
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
        r"\b(less\s+than|under)\s*(\d+(?:\.\d+)?)\s*(years?|yrs?|y|months?|mos?)\b",
        r"\b(more\s+than|over|at\s+least)\s*(\d+(?:\.\d+)?)\s*(years?|yrs?|y|months?|mos?)\b",
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

        # Quiet downloads (no extra Streamlit spinners)
        download_prefix_flat(conn, cfg["ml_container"], cfg["mr_prefix"], local_mr_dir())
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
# SAFE CHATBOT SESSION
# =========================================================
def ensure_bot_session(bot: Optional[ChatbotPipeline]):
    try:
        if bot is None:
            return
        if not hasattr(bot, "session") or bot.session is None or not isinstance(bot.session, dict):
            bot.session = {}
        bot.session.setdefault("greeted", False)
        bot.session.setdefault("intent", None)
        bot.session.setdefault("entities", {})
    except Exception:
        pass

def hard_reset_bot_session(bot: Optional[ChatbotPipeline]):
    try:
        if bot is None:
            return
        bot.session = {"greeted": False, "intent": None, "entities": {}}
    except Exception:
        pass

# =========================================================
# Entities → Facets
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
    t = (query or "").strip().lower()
    facets = dict(prev_facets) if prev_facets else {}
    removed = []
    removed_keys: Set[str] = set()
    cleared_all = False

    if re.search(r"\b(clear|remove)\s+(all|everything|constraints|filters|facets)\b", t) or re.search(r"\breset\b", t):
        return {}, ["all constraints"], True, {"animal","breed","gender","state","color","size","fur_length","soft"}

    if re.search(r"\b(remove|clear)\s+state\b", t):
        if "state" in facets:
            removed.append(f"state: {facets['state']}")
            facets.pop("state", None)
            removed_keys.add("state")

    for s in MALAYSIA_STATES:
        if re.search(rf"\bremove\s+{re.escape(s)}\b", t):
            if facets.get("state") and _norm_state(facets["state"]) == _norm_state(s):
                removed.append(f"state: {facets['state']}")
                facets.pop("state", None)
                removed_keys.add("state")

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

def maybe_reset_persistence(new_facets: Dict[str, Any], bot: Optional[ChatbotPipeline] = None) -> bool:
    prev = get_persisted_facets()
    new_animal = new_facets.get("animal")
    new_breed  = new_facets.get("breed")
    changed = False
    if prev.get("animal") and new_animal and prev["animal"] != new_animal:
        changed = True
    if prev.get("breed") and new_breed and prev["breed"] != new_breed:
        changed = True

    if changed:
        blocked_keep = get_blocked_facets()
        st.session_state["last_facets"] = {}
        set_blocked_facets(blocked_keep)
        try:
            ensure_bot_session(bot)
            bot.session["entities"] = {}
        except Exception:
            pass
    return changed

# Only accept entities explicitly mentioned in this prompt (safety fuse)
def explicit_add_keys_from_prompt(prompt: str) -> Set[str]:
    t = (prompt or "").lower()
    keys: Set[str] = set()
    if re.search(r"\b(dog|puppy|pup)\b", t) or re.search(r"\b(cat|kitten|kitties)\b", t):
        keys.add("animal")
    if re.search(r"\bmale\b", t) or re.search(r"\bfemale\b", t):
        keys.add("gender")
    for s in MALAYSIA_STATES:
        if re.search(rf"\b{s}\b", t):
            keys.add("state"); break
    if re.search(r"\bin\s+(johor|kedah|kelantan|malacca|melaka|negeri sembilan|pahang|penang|pulau pinang|perak|perlis|sabah|sarawak|selangor|terengganu|kuala lumpur|labuan|putrajaya)\b", t):
        keys.add("state")
    for word in re.findall(r"[a-z]+", t):
        if normalize_color(word):
            keys.add("color"); break
    if re.search(r"\b(small|medium|large|xl)\b", t):
        keys.add("size")
    if re.search(r"\b(short|long)\s*fur\b", t) or re.search(r"\bfurlength|fur length\b", t):
        keys.add("fur_length")
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
    df = dfp.copy()

    if facets.get("animal"):
        df = df[df["animal"] == facets["animal"]]
        if df.empty: return df, 0
    if facets.get("breed"):
        pattern = rf"\b{re.escape(facets['breed'])}\b"
        df = df[df["breed"].str.contains(pattern, case=False, na=False)]
        if df.empty: return df, 0

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
        chosen_df = df_try
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

def count_exact_matches(dfp: pd.DataFrame, facets: Dict[str, Any]) -> int:
    if dfp is None or dfp.empty:
        return 0
    df = dfp.copy()
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
    return int(df.apply(lambda r: match_all_soft(r, facets), axis=1).sum())

# =========================================================
# Cards / Grid rendering
# =========================================================
def render_pet_card(row: pd.Series, highlight: bool = False):
    name = str(row.get("name") or "Pet")
    url  = str(row.get("url") or "")
    animal = (row.get("animal") or "").title()
    breed  = str(row.get("breed") or "—").title()
    gender = (row.get("gender") or "—").title()
    state  = (row.get("state") or "—").title()
    color  = str(row.get("color") or "—").title()
    age_mo = row.get("age_months")
    age_txt = _age_text_from_months(age_mo)
    size = str(row.get("size") or "—").title()
    fur  = str(row.get("fur_length") or "—").title()
    cond = str(row.get("condition") or "—").title()

    v_b, d_b, n_b, s_b = (_badge_bool(row.get("vaccinated"), "vaccinated"),
                          _badge_bool(row.get("dewormed"), "dewormed"),
                          _badge_bool(row.get("neutered"), "neutered"),
                          _badge_bool(row.get("spayed"), "spayed"))

    img_url = _first_photo_url_from_row(row)

    bg = "#E8F7E1" if highlight else "#FFFFFF"
    border_left = "5px solid #22c55e" if highlight else "5px solid #ff6b9d"

    desc_raw = str(row.get("description_clean") or "").strip()
    desc_safe = html.escape(desc_raw).replace("\n", "<br />") if desc_raw else ""

    st.markdown(
        "<div style='background:"+bg+"; border-radius:15px; padding:14px; margin:8px 0; border-left:"+border_left+"; box-shadow:0 4px 15px rgba(0,0,0,0.08);'>"
        "<div style='font-size:1.1rem; font-weight:800; color:#2563EB; margin-bottom:6px;'>"
        + ("<a href='"+html.escape(url)+"' target='_blank' style='text-decoration:none; color:#2563EB;'>🔗 "+html.escape(name)+"</a>" if url else html.escape(name)) +
        "</div>"
        "<div style='height:220px;width:100%;border-radius:10px;background:#F3F4F6;display:flex;align-items:center;justify-content:center;margin:8px 0 10px;overflow:hidden;border:1px solid #E5E7EB;'>"
        + (("<img src='"+html.escape(img_url)+"' alt='photo' loading='lazy' style='max-width:100%;max-height:100%;width:auto;height:auto;object-fit:contain;object-position:center;display:block;' />") if img_url else "<div style='color:#6B7280;'>No photo available</div>") +
        "</div>"
        "<div style='color:#374151;margin-bottom:6px;'>"
        "<strong>"+animal+"</strong> • <strong>Breed:</strong> "+breed+" • <strong>Gender:</strong> "+gender+" • <strong>Age:</strong> "+age_txt+" • <strong>State:</strong> "+state+
        "</div>"
        "<div style='color:#374151;margin-bottom:8px;'>"
        "<strong>Color:</strong> "+color+" • <strong>Size:</strong> "+size+" • <strong>Fur:</strong> "+fur+" • <strong>Condition:</strong> "+cond+
        "</div>"
        "<div style='color:#111827;'>"+ " | ".join([v_b, d_b, n_b, s_b]) +"</div>"
        + (("<div style='margin-top:10px;border-top:1px solid #E5E7EB;padding-top:8px;'><details style='cursor:pointer;'><summary style='color:#374151;font-weight:600;list-style:none;display:inline-block;'>▶ Show description</summary><div style='margin-top:8px;color:#374151;line-height:1.55;'>"+ desc_safe +"</div></details></div>") if desc_safe else "") +
        "</div>",
        unsafe_allow_html=True
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
                render_pet_card(rows[idx], highlight=bool(flags[idx]))

# =========================================================
# Pink UI CSS  (COMPACT STATUS BAR)
# =========================================================
PINK_CSS = """
<style>
.stApp { background: linear-gradient(135deg, #ffb6c1 0%, #ffc0cb 50%, #ffd1dc 100%); background-attachment: fixed; }
.main-header { text-align: center; padding: 1.25rem 0; background: linear-gradient(45deg, #ff6b9d, #ff8fab); border-radius: 16px; margin-bottom: 1rem; box-shadow: 0 6px 24px rgba(255, 107, 157, 0.28); }
.main-header h1 { color: white; font-size: 2.25rem; margin: 0; text-shadow: 1px 1px 3px rgba(0,0,0,0.25); }
.main-header p  { color: white; font-size: .95rem; margin: .35rem 0 0 0; opacity: .92; }
.status-bar { background: linear-gradient(135deg, #ff6b9d, #ff8fab);
  color: white; padding: .45rem .9rem; margin: -.5rem -.5rem 1rem -.5rem;
  border-radius: 0 0 14px 14px; box-shadow: 0 3px 14px rgba(255, 107, 157, 0.25); }
.status-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: .5rem; align-items: center; }
.status-item { background: rgba(255,255,255,0.18); padding:.45rem .55rem; border-radius:8px;
  text-align:center; backdrop-filter: blur(8px); border:1px solid rgba(255,255,255,0.28); }
.status-item.success { background: rgba(76,175,80,0.22); border-color:rgba(76,175,80,0.42); }
.status-item.error   { background: rgba(244,67,54,0.22); border-color:rgba(244,67,54,0.42); }
.status-icon { font-size:.95rem; margin-bottom:.2rem; display:block; line-height: 1; }
.status-text { font-weight:700; margin-bottom:.05rem; font-size:.85rem; line-height:1; }
.status-detail { font-size:.8rem; opacity:.95; line-height:1; }
.block-container { padding-top: 1rem; }
</style>
"""

# =========================================================
# Scroll helpers (JS)
# =========================================================
def _emit_scroll_js(anchor_id: Optional[str]):
    if not anchor_id:
        return
    if anchor_id == "__top__":
        components.html(
            """
            <script>
            try { parent.document.querySelector('html').style.scrollBehavior = 'auto'; } catch(e) {}
            function toTop(){ parent.window.scrollTo({top: 0, left: 0, behavior: "auto"}); }
            toTop(); setTimeout(toTop, 80); setTimeout(toTop, 160); setTimeout(toTop, 320);
            </script>
            """, height=0
        )
        return
    components.html(
        f"""
        <script>
        try {{ parent.document.querySelector('html').style.scrollBehavior = 'auto'; }} catch(e) {{}}
        const targetId = "{anchor_id}";
        let tries = 0;
        function jump() {{
          const root = parent.document;
          const el = root.getElementById(targetId);
          if (el) {{
            const r = el.getBoundingClientRect();
            parent.window.scrollTo({{ top: r.top + parent.window.pageYOffset - 8, left: 0, behavior: "auto" }});
            return true;
          }}
          return false;
        }}
        function loop() {{
          if (jump() || ++tries > 25) return;
          setTimeout(loop, 40);
        }}
        loop();
        </script>
        """, height=0
    )

# =========================================================
# Reset helpers
# =========================================================
def trigger_hard_reset():
    st.session_state["__pending_hard_reset__"] = True
    st.rerun()

def post_reset_scrub(bot: Optional[ChatbotPipeline]):
    """Extra safety: after caches are cleared and app reruns, ensure nothing stale remains."""
    st.session_state["last_facets"] = {}
    st.session_state["blocked_facets"] = set()
    st.session_state["messages"] = []
    try:
        ensure_bot_session(bot)
        bot.session = {"greeted": False, "intent": None, "entities": {}}
    except Exception:
        pass

# =========================================================
# Main App
# =========================================================
def main():
    # ---------- PHASE 1: perform hard reset if armed ----------
    if st.session_state.pop("__pending_hard_reset__", False):
        # Clear *both* caches so cached ChatbotPipeline/env cannot leak state
        try:
            st.cache_resource.clear()
        except Exception:
            pass
        try:
            st.cache_data.clear()
        except Exception:
            pass
        # Clear all session vars
        st.session_state.clear()
        # Mark for post-reset scrub on the next run
        st.session_state["__post_reset_scrub__"] = True
        st.session_state["__scroll_after_render__"] = "__top__"
        st.rerun()

    # Header
    st.markdown(PINK_CSS, unsafe_allow_html=True)
    st.markdown(
        "<div class='main-header'>"
        "<h1>🐾 Pawfect Match</h1>"
        "<p>Your Intelligent Pet Assistant - Ask about pet care or find your pawfect pet</p>"
        "</div>",
        unsafe_allow_html=True
    )

    # ===== Friendly Loading (first boot only) =====
    if not st.session_state.get("_boot_done", False):
        loading_ph = st.empty()
        st.markdown("""
        <style>
        .pm-loading { display:flex; flex-direction:column; align-items:center; justify-content:flex-start;
          min-height:64vh; padding-top:6vh; text-align:center; }
        .pm-loading .paw { font-size:3.6rem; animation: pm-spin 2s linear infinite; }
        .pm-loading .chat { font-size:2.2rem; margin-top:8px; animation: pm-bounce 1.5s infinite; }
        .pm-loading h2 { margin:10px 0 6px 0; color:#ff4d88; }
        .pm-loading p  { margin:4px 0; color:#444; }
        .pm-tip { margin-top:10px; font-size:.95rem; color:#333; background:rgba(255,255,255,0.85);
          border-radius:10px; padding:8px 12px; box-shadow:0 2px 8px rgba(0,0,0,0.08); display:inline-block; }
        @keyframes pm-spin { from {transform: rotate(0deg);} to {transform: rotate(360deg);} }
        @keyframes pm-bounce { 0%,100%{ transform: translateY(0);} 50%{ transform: translateY(-6px);} }
        </style>
        """, unsafe_allow_html=True)

        FUN_FACTS = [
            "Did you know? A quick 20-minute walk can boost your dog’s mood and health!",
            "Fun fact: Cats sleep for ~70% of their lives — champions of chill!",
            "Did you know? Puppies are born blind, deaf, and toothless!",
            "Fun fact: Dogs can learn 1,000+ words — brainy buddies!",
            "Did you know? A cat’s purr can help lower human stress levels!",
            "Tip: Fresh water and a steady routine keep pets happiest.",
        ]
        def render_stage(headline, subtext, fun_tip):
            loading_ph.markdown(
                f"""
                <div class='pm-loading'>
                  <div class='paw'>🐾</div>
                  <div class='chat'>💬</div>
                  <h2>{headline}</h2>
                  <p>{subtext}</p>
                  <div class='pm-tip'>{random.choice(FUN_FACTS)}</div>
                </div>
                """, unsafe_allow_html=True
            )

        render_stage("Paws-itively Preparing Your Chat!", "Fetching the best advice and available pets...", random.choice(FUN_FACTS))
        time.sleep(0.4)
        render_stage("🐶 Initializing Pet Adoption Database... 😺", "Fetching profiles of our furry friends...", random.choice(FUN_FACTS))
        env = bootstrap_search_components()
        render_stage("🩺 Gathering Pet Care Tips & Chat Intelligence...", "Training our assistant to answer your pet questions!", random.choice(FUN_FACTS))
        rag, bot = bootstrap_rag_system()
        ensure_bot_session(bot)
        time.sleep(0.6)
        loading_ph.empty()

        st.session_state["_boot_done"] = True
        st.session_state["_boot_env"] = env
        st.session_state["_boot_rag"] = rag
        st.session_state["_boot_bot"] = bot
    else:
        env = st.session_state.get("_boot_env")
        rag = st.session_state.get("_boot_rag")
        bot = st.session_state.get("_boot_bot")
        ensure_bot_session(bot)

    # ---------- PHASE 2: one-time scrub after a reset ----------
    if st.session_state.pop("__post_reset_scrub__", False):
        post_reset_scrub(st.session_state.get("_boot_bot"))

    # Initialize chat log if missing
    if "messages" not in st.session_state:
        hard_reset_bot_session(st.session_state.get("_boot_bot"))
        st.session_state.messages = []

    rag_ok = (st.session_state.get("_boot_rag") is not None) and (st.session_state.get("_boot_bot") is not None)
    env_ok = (st.session_state.get("_boot_env") is not None) and (st.session_state["_boot_env"].get("dfp") is not None)

    def badge(status): return "success" if status else "error"
    def icon(status): return "✅" if status else "❌"

    pets_detail = f"{len(env['dfp'])} pets available" if env_ok else "Unavailable"
    app_status_ok = rag_ok and env_ok
    app_status_text = "All Systems Ready" if app_status_ok else "Issues Detected"

    st.markdown(
        "<div class='status-bar'>"
        "<div class='status-grid'>"
        f"<div class='status-item {badge(app_status_ok)}'><span class='status-icon'>{icon(app_status_ok)}</span><div class='status-text'>App Status</div><div class='status-detail'>{app_status_ok and 'All good' or 'Check modules'}</div></div>"
        f"<div class='status-item {badge(rag_ok)}'><span class='status-icon'>{icon(rag_ok)}</span><div class='status-text'>RAG / Chatbot</div><div class='status-detail'>{'Online' if rag_ok else 'Unavailable'}</div></div>"
        f"<div class='status-item {badge(env_ok)}'><span class='status-icon'>{icon(env_ok)}</span><div class='status-text'>Pet Search</div><div class='status-detail'>{pets_detail}</div></div>"
        "</div></div>", unsafe_allow_html=True
    )

    # Render prior messages
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # Helper chips
    st.markdown("### 💡 Try asking me:")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🐕 What should I feed my puppy?", use_container_width=True):
            st.session_state.example_prompt = "What should I feed my puppy?"
    with c2:
        if st.button("🏠 Find a golden retriever in Selangor", use_container_width=True):
            st.session_state.example_prompt = "Find a golden retriever in Selangor"
    with c3:
        if st.button("🏥 My cat is sick, what should I do?", use_container_width=True):
            st.session_state.example_prompt = "My cat is sick, what should I do?"

    prompt = st.chat_input("Ask me anything about pets...")

    if "example_prompt" in st.session_state:
        prompt = st.session_state.example_prompt
        del st.session_state["example_prompt"]

    if not prompt:
        _emit_scroll_js(st.session_state.pop("__scroll_after_render__", None))
        return

    # ===== Record user message =====
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # ------------------------
    # Pet/RAG routing & logic
    # ------------------------
    assistant_anchor_id = None
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            bot = st.session_state.get("_boot_bot")
            env = st.session_state.get("_boot_env")
            ensure_bot_session(bot)

            # Extra fuse: if this is the first prompt after a reset or after a fresh boot, ensure no stale facets/entities
            if st.session_state.get("_just_started_turn", True):
                st.session_state["last_facets"] = {}
                st.session_state["blocked_facets"] = set()
                hard_reset_bot_session(bot)
            st.session_state["_just_started_turn"] = False

            removal_intent = is_constraint_removal_query(prompt)
            prev_facets_for_removal = get_persisted_facets()
            prev_after_removal = None
            cleared_all = False
            removed_keys: Set[str] = set()

            if removal_intent:
                prev_after_removal, _, cleared_all, removed_keys = apply_constraint_removals(prev_facets_for_removal, prompt)
                ensure_bot_session(bot)
                bot.session["intent"] = "find_pet"
                blocked = get_blocked_facets()
                if cleared_all:
                    blocked = {"animal","breed","gender","state","color","size","fur_length","soft"}
                blocked |= set(removed_keys)
                set_blocked_facets(blocked)

            try:
                reply = bot.handle_message(prompt)
            except Exception as e:
                reply = f"(Chat pipeline error: {e})"

            ensure_bot_session(bot)
            intent_now = bot.session.get("intent")
            if removal_intent:
                intent_now = "find_pet"

            if not env_ok:
                assistant_anchor_id = f"assistant_msg_{len(st.session_state.messages)}"
                st.markdown(f"<div id='{assistant_anchor_id}'></div>", unsafe_allow_html=True)
                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
                st.divider()
                if st.button("➕ New search / Clear history", type="primary", use_container_width=True, key="bottom_clear_emptyenv"):
                    trigger_hard_reset()
            else:
                # ----- PET SEARCH PATH -----
                if intent_now == "find_pet":
                    ents = (bot.session or {}).get("entities", {}) or {}
                    # Safety: only allow facets that are explicitly present in THIS prompt
                    explicit_keys = explicit_add_keys_from_prompt(prompt)
                    ents = {k:v for k,v in ents.items() if (
                        (k=="PET_TYPE" and "animal" in explicit_keys) or
                        (k=="STATE" and "state" in explicit_keys) or
                        (k=="BREED" and "breed" in explicit_keys) or
                        (k=="GENDER" and "gender" in explicit_keys) or
                        (k=="COLOR" and "color" in explicit_keys) or
                        (k=="SIZE" and "size" in explicit_keys) or
                        (k=="FURLENGTH" and "fur_length" in explicit_keys)
                    )}
                    new_facets = _entities_to_facets(ents, raw_query=prompt)

                    reset_done = maybe_reset_persistence(new_facets, bot)
                    prev = {} if reset_done else get_persisted_facets()

                    if removal_intent:
                        base_after_removal = prev_after_removal
                        removal_keys = removed_keys
                    else:
                        base_after_removal, _, _, removal_keys = apply_constraint_removals(prev, prompt)

                    blocked = get_blocked_facets()
                    if explicit_keys:
                        blocked = blocked - explicit_keys
                        set_blocked_facets(blocked)

                    for k in (removal_keys | blocked):
                        new_facets.pop(k, None)

                    # HARD GUARD on NEW entities: drop COLOR unless explicitly present this turn
                    if "color" in new_facets and "color" not in explicit_keys:
                        new_facets.pop("color", None)

                    base = {} if (removal_intent and cleared_all) else dict(base_after_removal)

                    if removal_intent:
                        if ("color" in removal_keys) or ("color" in blocked):
                            base.pop("color", None)

                    merged = dict(base)
                    for k in ["animal","breed","gender","state","color","size","fur_length"]:
                        if new_facets.get(k):
                            merged[k] = new_facets[k]

                    # --- ANIMAL ↔ BREED EXCLUSIVITY ---
                    # If user changed animal this turn, drop any breed.
                    # If user changed breed this turn, drop any animal.
                    # If both changed, prefer breed (more specific): drop animal.
                    animal_changed = ("animal" in new_facets) and ("animal" in explicit_keys)
                    breed_changed  = ("breed" in new_facets) and ("breed" in explicit_keys)

                    if breed_changed:
                        # Prefer breed; remove animal to avoid "animal=cat, breed"
                        merged.pop("animal", None)
                    elif animal_changed:
                        # Animal changed without breed change => remove breed
                        merged.pop("breed", None)
                    # ----------------------------------

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

                    non_soft_keys = [k for k in facets.keys() if k != "soft"]
                    assistant_anchor_id = f"assistant_msg_{len(st.session_state.messages)}"
                    st.markdown(f"<div id='{assistant_anchor_id}'></div>", unsafe_allow_html=True)

                    if len(non_soft_keys) == 0 and not (facets.get("soft") and any(facets["soft"].values())):
                        st.info("No active filters right now. Here are some random sweethearts!")
                        df_all = env["dfp"]
                        if len(df_all) > 0:
                            random_df = df_all.sample(min(TOPK_CARDS, len(df_all)), random_state=None).reset_index(drop=True)
                            render_grid(random_df, np.array([False]*len(random_df)))
                        st.caption("Tell me what you’re looking for — e.g., **female young poodle in Selangor**.")
                        st.session_state.messages.append({"role":"assistant","content":"(random suggestions shown)"})
                        st.divider()
                        if st.button("➕ New search / Clear history", type="primary", use_container_width=True, key="bottom_clear_random"):
                            trigger_hard_reset()
                    else:
                        # Facet chips
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
                        if soft.get("age_groups_pref"): chips.append(chip("Age group", "/".join(soft["age_groups_pref"]), accent=True))
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
                                "<div style='margin:10px 0 4px;color:#374151;'>Facets used (tip: try to remove a filter by typing <em>remove state</em> or <em>remove gender</em> if too few results):</div>"
                                "<div style='margin-bottom:10px;display:flex;flex-wrap:wrap;gap:6px;'>"
                                + "".join(chips) + "</div>", unsafe_allow_html=True
                            )

                        # Build pool + exact match count
                        df_pool, _ = build_relaxed_pool(env["dfp"], facets)
                        exact_total = count_exact_matches(env["dfp"], facets)

                        if facets.get("state"):
                            if exact_total >= TOPK_CARDS:
                                status_msg = f"Found {exact_total} pets matching all your filters in {facets['state'].title()}! Here are some lovely matches."
                            elif exact_total > 0:
                                status_msg = f"Only {exact_total} pet(s) match all your filters in {facets['state'].title()}. Showing similar fur babies~"
                            else:
                                status_msg = f"No pets match all your filters in {facets['state'].title()}. Trying close matches for you ~"
                        else:
                            if exact_total >= TOPK_CARDS:
                                status_msg = f"Found {exact_total} pets matching all your filters! Here are some lovely matches."
                            elif exact_total > 0:
                                status_msg = f"Only {exact_total} pet(s) match all your filters. Showing similar fur babies~"
                            else:
                                status_msg = "No pets match all your filters. Trying close matches for you ~"
                        st.markdown(status_msg)

                        if df_pool is None or df_pool.empty:
                            st.info("No pets found. Try adjusting or removing some constraints (e.g. **remove state**).")
                        else:
                            res_df, highlight_mask = hybrid_rank_and_highlight(prompt, env, facets, df_pool)
                            if res_df is None or res_df.empty:
                                st.info("No pets found. Try adjusting or removing some constraints (e.g. **remove state**).")
                            else:
                                render_grid(res_df, highlight_mask, max_cols=GRID_COLS)

                        st.session_state.messages.append({"role": "assistant", "content": status_msg})
                        st.divider()
                        if st.button("➕ New search / Clear history", type="primary", use_container_width=True, key="bottom_clear_pet"):
                            trigger_hard_reset()

                else:
                    # ----- RAG Q&A PATH -----
                    assistant_anchor_id = f"assistant_msg_{len(st.session_state.messages)}"
                    st.markdown(f"<div id='{assistant_anchor_id}'></div>", unsafe_allow_html=True)
                    st.markdown(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                    st.divider()
                    if st.button("➕ New search / Clear history", type="primary", use_container_width=True, key="bottom_clear_rag"):
                        trigger_hard_reset()

    if assistant_anchor_id:
        st.session_state["__scroll_after_render__"] = assistant_anchor_id
    _emit_scroll_js(st.session_state.pop("__scroll_after_render__", None))

# =========================================================
# Ranking & Highlighting
# =========================================================
def hybrid_rank_and_highlight(query: str, env: Dict[str, Any], facets: Dict[str, Any], df_pool: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    student = env["student"]; doc_ids = env["doc_ids"]; doc_vecs = env["doc_vecs"]
    faiss_index = env["faiss_index"]; bm25 = env["bm25"]

    facet_bits = []
    for k in ["animal","breed","gender","color","size","fur_length","state"]:
        if facets.get(k): facet_bits.append(str(facets[k]))
    boost_q = (query or "").strip()
    if facet_bits:
        boost_q = (boost_q + " " + " ".join(facet_bits)).strip()

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

    combo_with_bonus: Dict[int, float] = {}
    for i in df_pool.index:
        combo_with_bonus[int(i)] = base_combo.get(int(i), 0.0) + _feature_bonus(df_pool.loc[i], facets)

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
    mask = np.array([_full_ok(int(i)) for i in chosen], dtype=bool)
    return res_df, mask

if __name__ == "__main__":
    main()
