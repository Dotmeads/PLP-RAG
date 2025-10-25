"""
Enhanced Pet Retrieval System
Incorporates robust filtering and data handling from friend's code
"""

import re, math, time, ast, json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional

# ---------- Text Cleaning ----------
def only_text(s: str) -> str:
    """Clean text by removing URLs and special characters"""
    s = re.sub(r"http\S+", " ", str(s or ""))
    s = re.sub(r"[^a-zA-Z0-9\s/,+()\-]", " ", s)
    return re.sub(r"\s+", " ", s).strip().lower()

# ---------- Enhanced BM25 ----------
class EnhancedBM25:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.docs = []
        self.ids = []
        self.df = {}
        self.idf = {}
        self.doc_lens = None
        self.avgdl = 0.0

    def _tok(self, text):
        """Tokenize text"""
        return re.findall(r"[a-z0-9]+", text.lower())

    def fit(self, doc_map: Dict[int, str]):
        """Fit BM25 on document map"""
        self.ids = np.array(list(doc_map.keys()), dtype=int)
        self.docs = [self._tok(doc_map[i]) for i in self.ids]
        N = len(self.docs)
        self.doc_lens = np.array([len(d) for d in self.docs], dtype=float)
        self.avgdl = float(np.mean(self.doc_lens)) if N else 0.0
        
        from collections import defaultdict
        df = defaultdict(int)
        for d in self.docs:
            for t in set(d):
                df[t] += 1
        self.df = dict(df)
        self.idf = {t: math.log(1 + (N - df_t + 0.5)/(df_t + 0.5)) for t, df_t in self.df.items()}
        return self

    def search(self, query: str, topk=2000):
        """Search with BM25 scoring"""
        q = self._tok(query)
        scores = np.zeros(len(self.docs), dtype=float)
        for i, d in enumerate(self.docs):
            from collections import Counter
            tf = Counter(d)
            s = 0.0
            for t in q:
                if t not in self.idf:
                    continue
                idf = self.idf[t]
                f = tf.get(t, 0)
                denom = f + self.k1*(1 - self.b + self.b*len(d)/(self.avgdl + 1e-9))
                s += idf * (f*(self.k1+1))/(denom + 1e-9)
            scores[i] = s
        
        if topk < len(scores):
            idx = np.argpartition(-scores, topk)[:topk]
            idx = idx[np.argsort(-scores[idx])]
        else:
            idx = np.argsort(-scores)[::-1]
        return [(int(self.ids[i]), float(scores[i])) for i in idx[:topk]]

# ---------- Enhanced Facet Parsing ----------
def parse_facets_from_text(user_text: str) -> Dict[str, Any]:
    """Parse facets from user text using regex patterns"""
    t = only_text(user_text)
    
    # Animal detection
    animal = None
    if re.search(r"\b(dog|puppy|canine)\b", t):
        animal = "dog"
    if re.search(r"\b(cat|kitten|feline)\b", t):
        animal = "cat"
    
    # Gender detection
    gender = None
    if re.search(r"\bmale\b", t):
        gender = "male"
    elif re.search(r"\bfemale\b", t):
        gender = "female"
    
    # State detection
    state = None
    states = ["selangor", "kuala lumpur", "putrajaya", "johor", "penang", "pahang", 
              "sabah", "sarawak", "perak", "kedah", "kelantan", "melaka", 
              "negeri sembilan", "perlis", "terengganu", "labuan"]
    for s in states:
        if re.search(rf"\b{s}\b", t):
            state = s
            break
    
    # Color detection
    COLORS = ["black", "white", "brown", "beige", "tan", "golden", "cream", "gray", "grey", 
              "silver", "orange", "ginger", "brindle", "calico", "tortoiseshell", 
              "tortie", "tabby", "yellow"]
    found = [c for c in COLORS if re.search(rf"\b{re.escape(c)}\b", t)]
    colors_any = sorted(set(["gray" if c == "grey" else c for c in found])) if found else None
    
    return {
        "animal": animal, 
        "gender": gender, 
        "colors_any": colors_any, 
        "breed": None, 
        "state": state
    }

def entity_spans_to_facets(spans: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Convert NER entity spans to facets"""
    facets = {"animal": None, "breed": None, "gender": None, "colors_any": [], "state": None}
    for sp in (spans or []):
        label = re.sub(r"^[BI]-", "", str(sp.get("entity_group") or sp.get("label") or "")).upper()
        text = only_text(sp.get("word") or sp.get("entity") or "")
        if not label or not text:
            continue
        
        if label == "ANIMAL" and text in {"dog", "cat"}:
            facets["animal"] = text
        elif label == "BREED":
            facets["breed"] = text
        elif label == "GENDER" and text in {"male", "female"}:
            facets["gender"] = text
        elif label == "COLOR":
            facets["colors_any"].append(text)
        elif label in {"STATE", "LOCATION", "CITY"}:
            facets["state"] = text
    
    facets["colors_any"] = sorted(set(facets["colors_any"])) or None
    return facets

def sanitize_facets_ner_light(f: dict) -> dict:
    """Sanitize facets from NER"""
    out = {k: f.get(k) for k in ["animal", "breed", "gender", "colors_any", "state"]}
    if out.get("colors_any"):
        cols = []
        for c in out["colors_any"]:
            cols.append("gray" if c == "grey" else c)
        out["colors_any"] = sorted(set(cols)) or None
    return out

# ---------- Robust Data Parsing ----------
def safe_list_from_cell(x) -> List[str]:
    """Safely parse list data from various formats"""
    if x is None:
        return []
    if isinstance(x, float) and np.isnan(x):
        return []
    if isinstance(x, list):
        return [str(t).strip().lower() for t in x if str(t).strip()]
    
    s = str(x).strip()
    if not s:
        return []
    
    if s.startswith("[") and s.endswith("]"):
        try:
            obj = json.loads(s)
            if isinstance(obj, list):
                return [str(t).strip().lower() for t in obj if str(t).strip()]
        except Exception:
            pass
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, list):
                return [str(t).strip().lower() for t in obj if str(t).strip()]
        except Exception:
            return []
    
    if "," in s:
        return [t.strip().lower() for t in s.split(",") if t.strip()]
    
    return [s.lower()] if s else []

# ---------- Enhanced Filtering ----------
def filter_once(df: pd.DataFrame, facets: Dict[str, Any]) -> pd.DataFrame:
    """Apply filters to dataframe based on facets"""
    out = df.copy()
    
    # Animal filter (most important)
    if "animal" in out.columns and facets.get("animal"):
        out = out[out["animal"].astype(str).str.lower() == facets["animal"]]
    
    # Gender filter
    if "gender" in out.columns and facets.get("gender") in {"male", "female"}:
        out = out[out["gender"].astype(str).str.lower() == facets["gender"]]
    
    # State filter (case-insensitive)
    if "state" in out.columns and facets.get("state"):
        st = facets["state"].lower()
        out = out[out["state"].astype(str).str.lower().str.contains(re.escape(st), na=False)]
    
    # Colors filter (canonical list in column)
    if "colors_canonical" in out.columns and facets.get("colors_any"):
        want = set([c.lower() for c in facets["colors_any"]])
        def _hit(xs):
            xs_list = safe_list_from_cell(xs)
            return any(c in xs_list for c in want)
        out = out[out["colors_canonical"].apply(_hit)]
    
    # Breed filter (soft substring)
    if "breed" in out.columns and facets.get("breed"):
        b = str(facets["breed"]).lower()
        out = out[out["breed"].astype(str).str.lower().str.contains(re.escape(b), na=False)]
    
    return out.reset_index(drop=True)

def filter_with_relaxation(df: pd.DataFrame, facets: Dict[str, Any], 
                          order: List[str], min_floor: int = 300) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Apply filters with progressive relaxation if results are too few"""
    used = dict(facets or {})
    res = filter_once(df, used)
    
    # Progressive relaxation
    for key in order:
        if len(res) >= min_floor:
            break
        if used.get(key) not in (None, [], ""):
            used[key] = None
            res = filter_once(df, used)
    
    # Final fallback - keep only animal if present
    if len(res) == 0:
        keep = {"animal": used.get("animal"), "breed": None, "gender": None, 
                "colors_any": None, "state": None}
        used = keep
        res = filter_once(df, used)
        if len(res) == 0:
            res = df
    
    return res, used

def make_boosted_query(user_text: str, used: dict) -> str:
    """Create boosted query with facet terms"""
    facet_terms = []
    for key in ["animal", "breed", "gender", "state"]:
        if used.get(key):
            facet_terms.append(str(used[key]).lower())
    if used.get("colors_any"):
        facet_terms.extend([c.lower() for c in used["colors_any"]])
    return only_text(user_text + " " + " ".join(facet_terms))

# ---------- Enhanced Embedding Search ----------
def ann_search_faiss(faiss_index, q_vec: np.ndarray, pool_topn: int) -> List[Tuple[int, float]]:
    """Search using FAISS index"""
    D, I = faiss_index.search(q_vec.reshape(1, -1).astype("float32"), pool_topn)
    hits = []
    for idx, score in zip(I[0], D[0]):
        hits.append((int(idx), float(score)))
    return hits

def emb_search(q: str, student, doc_ids: np.ndarray, doc_vecs: np.ndarray, 
               pool_topn: int = 200, faiss_index=None) -> List[Tuple[int, float]]:
    """Enhanced embedding search with FAISS support"""
    qv = student.encode([q], convert_to_numpy=True, normalize_embeddings=True)[0].astype("float32")
    
    if faiss_index is not None:
        hits = ann_search_faiss(faiss_index, qv, pool_topn)
        return [(int(doc_ids[i]), float(s)) for (i, s) in hits]
    
    # Brute force fallback
    sims = doc_vecs @ qv
    if pool_topn < sims.shape[0]:
        idx = np.argpartition(-sims, pool_topn)[:pool_topn]
        idx = idx[np.argsort(-sims[idx])]
    else:
        idx = np.argsort(-sims)[::-1]
    return [(int(doc_ids[i]), float(sims[i])) for i in idx[:pool_topn]]

# ---------- MMR Re-ranking ----------
def mmr_rerank(candidates: List[Tuple[int, float]], doc_embs: np.ndarray, 
               doc_ids: np.ndarray, q_vec: np.ndarray, k: int = 10, 
               lambda_mult: float = 0.7) -> List[Tuple[int, float]]:
    """Maximal Marginal Relevance re-ranking"""
    if not candidates:
        return []
    
    # Map pet ID to index
    idx_map = {int(pid): i for i, pid in enumerate(doc_ids)}
    
    sel = []
    cand = candidates.copy()
    picked = set()
    
    while cand and len(sel) < k:
        best_pid, best_score = None, -1e9
        for pid, s in cand:
            i = idx_map.get(int(pid))
            if i is None:
                continue
            
            # Calculate max similarity to already selected
            rep = 0.0
            for p_sel, _ in sel:
                j = idx_map.get(int(p_sel))
                if j is None:
                    continue
                rep = max(rep, float(doc_embs[i] @ doc_embs[j]))
            
            # MMR score
            mmr_score = lambda_mult * s - (1 - lambda_mult) * rep
            if mmr_score > best_score:
                best_score = mmr_score
                best_pid = pid
        
        if best_pid is None:
            break
        
        # Move best to selected
        best_item = next((x for x in cand if x[0] == best_pid), None)
        if best_item:
            sel.append(best_item)
            picked.add(best_pid)
            cand = [x for x in cand if x[0] != best_pid]
        else:
            break
    
    return sel
