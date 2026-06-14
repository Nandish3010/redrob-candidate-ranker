"""
scorer.py: transparent, rule-based relevance scorer for the Redrob Senior AI Engineer JD.

This is the v0 baseline (the rule-engine spine from CONTEXT.md Section 5). It implements
relevance_core, the behavioral availability multiplier, and the conjunctive honeypot gate.
The capped embedding/similarity term (jd_similarity) is Phase 2; until build_artifacts.py
lands, its 0.12 weight is folded into career_evidence (career weight 0.42 here).

Design goals (Stage-4/5 defendability): every candidate's final_score decomposes into named
components, and the reasoning string is generated from the SAME components, so it can never
hallucinate and is always rank-consistent. CPU only, standard library only, no network.
"""

from __future__ import annotations

import json
import math
import os
import re
from datetime import date

# Anchor "today" for recency math. See CONTEXT.md Section 9. Update if you re-run much later.
REFERENCE_DATE = date(2026, 6, 14)

# --- Skill vocabularies (normalized: lowercased, non-alphanumerics -> single space) ---
CORE_IR_SKILLS = {
    "embeddings", "embedding", "sentence transformers", "retrieval", "information retrieval",
    "semantic search", "vector search", "vector database", "vector databases", "ranking",
    "learning to rank", "ltr", "recommendation", "recommender", "recommendation systems",
    "recommender systems", "bm25", "hybrid search", "faiss", "pinecone", "weaviate", "qdrant",
    "milvus", "opensearch", "elasticsearch", "nlp", "natural language processing", "rag",
    "retrieval augmented generation",
}
SUPPORTING_SKILLS = {
    "python", "pytorch", "tensorflow", "transformers", "fine tuning llms", "fine tuning",
    "lora", "qlora", "peft", "xgboost", "lightgbm", "hugging face", "huggingface", "llm",
    "llms", "langchain", "mlops", "feature store", "airflow", "spark",
}
EVAL_TERMS = ("ndcg", "mrr", "mean average precision", "map@", "recall@", "a/b test",
              "ab test", "offline evaluation", "ranking metric", "precision@")
IR_DESC_TERMS = ("ranking", "retrieval", "recommend", "search", "embedding", "vector",
                 "relevance", "semantic", "personaliz", "information retrieval",
                 "learning to rank", "ndcg")

# --- Title taxonomy ---
STRONG_TITLE_TERMS = ("machine learning", "ml engineer", "ai engineer", "applied scientist",
                      "applied ml", "research engineer", "data scientist", "search engineer",
                      "ranking", "recommendation", "relevance", "nlp", "ai research",
                      "ml scientist")
MEDIUM_TITLE_TERMS = ("software engineer", "backend engineer", "full stack", "data engineer",
                      "analytics engineer", "cloud engineer", "platform engineer", "devops",
                      "mlops")
# Titles that name the actual JD work (ranking/search/recommendation/relevance).
RANKING_TITLE_TERMS = ("search engineer", "search", "ranking", "recommendation", "recommender",
                       "relevance", "recsys", "discovery")

# --- Industry taxonomy ---
PRODUCT_INDUSTRY_TERMS = ("software", "internet", "fintech", "financial technology",
                          "e commerce", "ecommerce", "food delivery", "artificial intelligence",
                          "machine learning", "saas", "technology", "product", "gaming",
                          "edtech", "ed tech", "healthtech", "health tech", "mobility",
                          "social", "consumer", "marketplace", "streaming", "cloud")
SERVICES_INDUSTRY_TERMS = ("it services", "consulting", "staffing", "outsourcing", "bpo",
                           "information technology and services")

INDIA_PREF_CITIES = ("pune", "noida")
INDIA_TIER1_CITIES = ("bangalore", "bengaluru", "hyderabad", "mumbai", "delhi", "gurgaon",
                      "gurugram", "ncr", "noida", "pune", "chennai", "kolkata", "ahmedabad",
                      "gurugram")

PROFICIENCY_BASE = {"beginner": 0.25, "intermediate": 0.5, "advanced": 0.8, "expert": 1.0}
EDU_TIER_SCORE = {"tier_1": 1.0, "tier_2": 0.7, "tier_3": 0.4, "tier_4": 0.2, "unknown": 0.3}

# relevance_core weights (sum to 1.0). The career/similarity split depends on whether the
# Phase-2 similarity model is loaded:
#   - model present  -> career 0.30 + jd_similarity 0.12  (hybrid; the documented target)
#   - model absent    -> career 0.42, jd_similarity 0.00   (v0 spine; weight folded into career)
# The other five weights are identical in both modes, so relevance_core always sums to 1.0.
W_CAREER_HYBRID = 0.30
W_CAREER_SPINE = 0.42
W_SIM = 0.12
W_SKILL = 0.18
W_EXP = 0.12
W_PYEVAL = 0.12
W_LOC = 0.12
W_EDU = 0.04

# --- Phase 2: capped TF-IDF JD-similarity term (stdlib only, no model download) ---------------
# A small "model" (IDF table + four L2-normalized JD seed vectors + a data-derived calibration
# anchor) is precomputed offline by build_artifacts.py and committed to artifacts/sim_model.json.
# This is aggregate term statistics plus hand-authored JD vectors, NOT candidate data. The same
# file is loaded by rank.py (the timed step) and the sandbox, so the semantic layer can never
# drift between them. If the file is absent, the scorer transparently falls back to the v0 spine.
_SIM_MODEL = None          # {"idf": {term: idf}, "seeds": [{term: weight}, ...], "anchor": float}
_SIM_MODEL_TRIED = False
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "with", "at", "by", "from",
    "as", "is", "are", "was", "were", "be", "been", "being", "this", "that", "it", "its", "our",
    "we", "i", "you", "they", "their", "his", "her", "he", "she", "them", "us", "my", "me",
    "also", "into", "over", "across", "using", "used", "use", "via", "per", "such", "which",
    "who", "whom", "where", "when", "while", "than", "then", "there", "here", "but", "not", "no",
    "yes", "can", "will", "would", "should", "could", "may", "might", "have", "has", "had", "do",
    "does", "did", "done", "etc", "team", "work", "working", "worked", "various", "including",
}


def _tokens(text: str) -> list:
    """Unigrams + adjacent bigrams over normalized text, with light stopword removal.
    Bigrams capture JD phrases ('learning rank', 'vector search') that unigrams lose."""
    words = [w for w in norm(text).split() if len(w) > 1 and w not in _STOPWORDS]
    toks = list(words)
    toks += [f"{words[i]} {words[i + 1]}" for i in range(len(words) - 1)]
    return toks


def _candidate_doc(cand: dict) -> str:
    """Semantic surface for similarity. Deliberately EXCLUDES the raw skills array so that
    stuffing the skills list cannot move the similarity term (CONTEXT.md Section 6)."""
    profile = cand.get("profile", {}) or {}
    career = cand.get("career_history", []) or []
    parts = [profile.get("current_title", ""), profile.get("headline", ""),
             profile.get("summary", ""), profile.get("current_industry", "")]
    for r in career:
        parts.append(r.get("title", ""))
        parts.append(r.get("description", ""))
    return " ".join(p for p in parts if p)


def load_sim_model(path: str = None):
    """Load the Phase-2 similarity model once. Returns the model dict or None if unavailable.
    Default path is artifacts/sim_model.json at the repo root (one level up from src/)."""
    global _SIM_MODEL, _SIM_MODEL_TRIED
    if _SIM_MODEL_TRIED and path is None:
        return _SIM_MODEL
    if path is None:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(root, "artifacts", "sim_model.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            m = json.load(f)
        if m.get("idf") and m.get("seeds") and m.get("anchor"):
            _SIM_MODEL = m
    except Exception:
        _SIM_MODEL = None
    _SIM_MODEL_TRIED = True
    return _SIM_MODEL


def _jd_similarity(cand: dict, model: dict) -> float:
    """Cosine of the candidate's TF-IDF document vector against the closest JD seed vector,
    rescaled to [0, 1] by the model's data-derived anchor (a high pool percentile of the same
    cosine). Capped at 1.0. Out-of-vocabulary terms are dropped (standard TF-IDF pruning)."""
    idf = model["idf"]
    tf = {}
    for t in _tokens(_candidate_doc(cand)):
        if t in idf:
            tf[t] = tf.get(t, 0) + 1
    if not tf:
        return 0.0
    vec = {t: (1.0 + math.log(c)) * idf[t] for t, c in tf.items()}
    nrm = math.sqrt(sum(v * v for v in vec.values()))
    if nrm == 0.0:
        return 0.0
    best = 0.0
    for seed in model["seeds"]:  # seed vectors are stored already L2-normalized
        dot = sum(w * seed.get(t, 0.0) for t, w in vec.items())
        if dot > best:
            best = dot
    return max(0.0, min(1.0, (best / nrm) / model["anchor"]))


def norm(s) -> str:
    if not s:
        return ""
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def _title_score(title: str) -> float:
    t = norm(title)
    if not t:
        return 0.08
    if any(term in t for term in STRONG_TITLE_TERMS):
        return 1.0
    if any(term in t for term in MEDIUM_TITLE_TERMS):
        return 0.5
    return 0.08


def _industry_score(industry: str) -> float:
    i = norm(industry)
    if not i:
        return 0.3
    if any(term in i for term in SERVICES_INDUSTRY_TERMS):
        return 0.0
    if any(term in i for term in PRODUCT_INDUSTRY_TERMS):
        return 1.0
    return 0.3  # off-domain but not services (e.g. "Paper Products", "Manufacturing")


def _months_since(date_str: str) -> float:
    try:
        y, m, d = (int(x) for x in str(date_str).split("-")[:3])
        return max(0.0, (REFERENCE_DATE - date(y, m, d)).days / 30.44)
    except Exception:
        return 99.0  # unparseable -> treat as long inactive


def _experience_band(yoe: float) -> float:
    # JD wants 5 to 9 years, ideally 6 to 8. Below-band juniors are penalized hard because
    # the top-100 audit showed many 3.5 to 5yr ML/DS profiles ranked too high.
    if 6.0 <= yoe <= 8.0:
        return 1.0
    edge = 6.0 if yoe < 6.0 else 8.0
    d = abs(yoe - edge)
    val = math.exp(-(d * d) / (2 * 2.2 * 2.2))
    if yoe < 4.0:
        val *= 0.35
    elif yoe < 5.0:
        val *= 0.55
    return max(0.0, min(1.0, val))


def _honeypot_multiplier(cand: dict) -> float:
    """Return a score multiplier in (0, 1] reflecting internal-impossibility evidence.

    A skill cannot be used longer than the candidate's whole career, so any skill whose duration
    exceeds the career length is a self-contradiction. The COUNT of such skills is the honeypot
    signal: genuine seniors have at most one or two from data noise; planted honeypots have
    several (at any magnitude, just-above or far-above). We apply a GRADUATED down-weight rather
    than a binary gate, which is robust to the irreducible ambiguity (LLM judges disagree run to
    run on borderline cases; see CONTEXT.md decisions log). `basis` is the tighter of stated
    experience and summed worked time, so an inflated years_of_experience cannot hide a short
    actual career."""
    profile = cand.get("profile", {}) or {}
    yoe = float(profile.get("years_of_experience", 0) or 0)
    career = cand.get("career_history", []) or []
    skills = cand.get("skills", []) or []

    sum_m = sum(int(r.get("duration_months", 0) or 0) for r in career)
    basis = min(yoe * 12, sum_m) if sum_m > 0 else yoe * 12
    over = sum(1 for s in skills if int(s.get("duration_months", 0) or 0) > basis)
    if over >= 4:
        m = 0.05
    elif over == 3:
        m = 0.45
    elif over == 2:
        m = 0.80
    else:
        m = 1.0

    # Organizer-stated impossibilities are hard signals (force to the bottom).
    zero_dur_senior = sum(
        1 for s in skills
        if s.get("proficiency") in ("advanced", "expert") and int(s.get("duration_months", 0) or 0) == 0
    )
    if zero_dur_senior >= 3:
        m = min(m, 0.05)  # "expert in many skills with 0 months used"
    if any(int(r.get("duration_months", 0) or 0) > yoe * 12 + 12 for r in career):
        m = min(m, 0.05)  # a single role longer than the whole career
    for e in cand.get("education", []) or []:
        try:
            span = int(e.get("end_year")) - int(e.get("start_year"))
            if span <= 0 or span > 12:
                m = min(m, 0.05)
        except Exception:
            pass
    return m


def _behavioral_multiplier(sig: dict) -> float:
    rr = sig.get("recruiter_response_rate")
    rr = 0.5 if rr is None else float(rr)
    resp = 0.75 + 0.30 * max(0.0, min(1.0, rr))

    months = _months_since(sig.get("last_active_date", ""))
    if months <= 1:
        rec = 1.0
    elif months <= 3:
        rec = 0.92
    elif months <= 6:
        rec = 0.80
    elif months <= 9:
        rec = 0.68
    else:
        rec = 0.58

    otw = 1.03 if sig.get("open_to_work_flag") else 0.97

    icr = sig.get("interview_completion_rate")
    icr = 0.7 if icr is None else float(icr)
    icr_f = 0.92 + 0.13 * max(0.0, min(1.0, icr))

    notice = int(sig.get("notice_period_days", 60) or 60)
    if notice <= 30:
        nf = 1.02
    elif notice <= 60:
        nf = 1.0
    elif notice <= 90:
        nf = 0.97
    else:
        nf = 0.93

    return max(0.55, min(1.05, resp * rec * otw * icr_f * nf))


def score_candidate(cand: dict) -> dict:
    """Return {'candidate_id', 'score', 'rctx'} where rctx holds the few facts the
    reasoning template needs (so we never store all 100k full records)."""
    profile = cand.get("profile", {}) or {}
    sig = cand.get("redrob_signals", {}) or {}
    career = cand.get("career_history", []) or []
    skills = cand.get("skills", []) or []
    assessments = sig.get("skill_assessment_scores", {}) or {}
    yoe = float(profile.get("years_of_experience", 0) or 0)

    # --- career_evidence ---
    cur_title = _title_score(profile.get("current_title", ""))
    hist_title = max([_title_score(r.get("title", "")) for r in career], default=0.0)
    title_match = max(cur_title, 0.85 * hist_title)

    cur_ind = _industry_score(profile.get("current_industry", ""))
    hist_ind = max([_industry_score(r.get("industry", "")) for r in career], default=0.0)
    product_company = max(cur_ind, 0.8 * hist_ind)
    # Label the industry that actually EARNED the product signal, so reasoning never prints a
    # contradiction like "product-company background (IT Services)" (IT Services scores 0.0).
    if cur_ind >= 0.8:
        product_industry_label = profile.get("current_industry", "")
    elif hist_ind >= 1.0:
        product_industry_label = next(
            (r.get("industry", "") for r in career if _industry_score(r.get("industry", "")) >= 1.0),
            "")
    else:
        product_industry_label = ""

    # Require actual ranking/search/recommendation WORK, not just an ML title + skill list.
    # A ranking/search/recsys job TITLE, or role DESCRIPTIONS that describe that work, count.
    # CV / forecasting / fraud / classification profiles get little here even with an ML title.
    all_titles = norm(profile.get("current_title", "")) + " " + \
        " ".join(norm(r.get("title", "")) for r in career)
    ranking_title = 1.0 if any(t in all_titles for t in RANKING_TITLE_TERMS) else 0.0
    ir_roles = sum(
        1 for r in career if any(term in norm(r.get("description", "")) for term in IR_DESC_TERMS)
    )
    ir_work_evidence = max(ranking_title, min(1.0, ir_roles / 2.0))
    career_evidence = 0.36 * title_match + 0.20 * product_company + 0.44 * ir_work_evidence

    # --- skill_trust (coupled to career_evidence to collapse keyword stuffers) ---
    coupling = min(1.0, 0.4 + career_evidence)
    skill_credits = []  # (credit, display_name)
    for s in skills:
        n = norm(s.get("name", ""))
        if n in CORE_IR_SKILLS:
            w = 1.0
        elif n in SUPPORTING_SKILLS:
            w = 0.5
        else:
            continue
        base = PROFICIENCY_BASE.get(s.get("proficiency"), 0.25)
        trust = min(1.0, int(s.get("endorsements", 0) or 0) / 8.0) * \
            min(1.0, int(s.get("duration_months", 0) or 0) / 12.0)
        a = assessments.get(s.get("name"))
        assess_factor = 1.5 if (a is not None and a >= 60) else 1.0
        credit = w * base * trust * assess_factor * coupling
        if credit > 0:
            skill_credits.append((credit, s.get("name", "")))
    skill_trust = min(1.0, sum(c for c, _ in skill_credits) / 3.0)
    top_skills = [name for _, name in sorted(skill_credits, reverse=True)[:3]]

    # --- python + evaluation-framework signal ---
    has_python = any(norm(s.get("name", "")) == "python" and int(s.get("endorsements", 0) or 0) >= 0
                     for s in skills)
    blob = " ".join(norm(r.get("description", "")) for r in career) + " " + \
        " ".join(norm(s.get("name", "")) for s in skills)
    has_eval = any(term in blob for term in EVAL_TERMS)
    python_eval = 0.5 * (1.0 if has_python else 0.0) + 0.5 * (1.0 if has_eval else 0.0)

    experience_band = _experience_band(yoe)

    # --- location / relocation ---
    loc = norm(profile.get("location", ""))
    country = norm(profile.get("country", ""))
    willing = bool(sig.get("willing_to_relocate"))
    is_india = country == "india" or any(c in loc for c in INDIA_TIER1_CITIES)
    is_pref = any(c in loc for c in INDIA_PREF_CITIES)
    if is_pref:
        location = 1.0
    elif is_india:
        location = 0.85
    elif willing:
        location = 0.6
    else:
        location = 0.15

    best_edu = max([EDU_TIER_SCORE.get(e.get("tier"), 0.3) for e in cand.get("education", []) or []],
                   default=0.3)

    # --- jd_similarity (Phase 2, capped) ---
    # Active only when the precomputed similarity model is loaded. The 0.12 weight is taken from
    # career_evidence (0.42 -> 0.30) so the total is unchanged and the term can never by itself
    # resurrect a keyword stuffer (its surface excludes the raw skills array).
    model = load_sim_model()
    if model is not None:
        jd_similarity = _jd_similarity(cand, model)
        w_career, sim_term = W_CAREER_HYBRID, W_SIM * jd_similarity
    else:
        jd_similarity = 0.0
        w_career, sim_term = W_CAREER_SPINE, 0.0

    relevance_core = (w_career * career_evidence + sim_term + W_SKILL * skill_trust
                      + W_EXP * experience_band + W_PYEVAL * python_eval + W_LOC * location
                      + W_EDU * best_edu)

    behavioral = _behavioral_multiplier(sig)
    hp_mult = _honeypot_multiplier(cand)
    # No upper clip. relevance_core is in [0, 1] and behavioral is in [0.55, 1.05], so an ideal,
    # immediately-available top candidate can edge just above 1.0. Clipping to 1.0 (the old v0
    # behavior) flattened the very strongest profiles into ties, which is exactly what NDCG@10
    # penalizes; the validator imposes no [0, 1] bound, only non-increasing scores. Keep the
    # lower clamp so a score is never negative.
    final = max(0.0, relevance_core * behavioral) * hp_mult

    rr_val = sig.get("recruiter_response_rate")
    rctx = {
        "title": profile.get("current_title", "Candidate"),
        "yoe": yoe,
        "industry": product_industry_label,
        "top_skills": top_skills,
        "n_trusted": len(skill_credits),
        "rr": None if rr_val is None else float(rr_val),
        "months_inactive": _months_since(sig.get("last_active_date", "")),
        "willing": willing,
        "is_india": is_india,
        "is_pref": is_pref,
        "notice": int(sig.get("notice_period_days", 0) or 0),
        "open_to_work": bool(sig.get("open_to_work_flag")),
        "product": product_company >= 0.8,
        "honeypot": hp_mult < 0.3,
        "jd_similarity": round(jd_similarity, 3),
    }
    return {"candidate_id": cand.get("candidate_id"), "score": round(final, 6), "rctx": rctx}


def make_reasoning(rctx: dict) -> str:
    """Build a 1-2 sentence, fact-grounded, rank-consistent reasoning string from rctx.
    Every clause cites a real field; clause selection varies by which facts are present."""
    yoe = rctx["yoe"]
    sk = rctx["top_skills"]
    skill_phrase = f" ({', '.join(sk)})" if sk else ""
    primary = f"{rctx['title']} with {yoe:.1f} yrs; {rctx['n_trusted']} trusted AI/IR skills{skill_phrase}"
    if rctx["product"] and rctx["industry"]:
        primary += f"; product-company background ({rctx['industry']})"

    concerns = []
    rr = rctx["rr"]
    if rr is not None and rr < 0.2:
        concerns.append(f"low recruiter response {rr:.2f}")
    if rctx["months_inactive"] >= 5:
        concerns.append(f"inactive ~{rctx['months_inactive']:.0f} mo")
    if not rctx["is_india"] and not rctx["willing"]:
        concerns.append("onsite-only abroad")
    if rctx["notice"] and rctx["notice"] > 90:
        concerns.append(f"{rctx['notice']}d notice")

    positives = []
    if rctx["is_pref"]:
        positives.append("Pune/Noida-based")
    elif rctx["is_india"]:
        positives.append("India-based")
    elif rctx["willing"]:
        positives.append("willing to relocate")
    if rctx["months_inactive"] <= 1:
        positives.append("active this month")
    if rr is not None and rr >= 0.6:
        positives.append("responsive to recruiters")
    if rctx["open_to_work"]:
        positives.append("open to work")
    if rctx.get("jd_similarity", 0.0) >= 0.7:
        positives.append("strong semantic match to the JD")

    if concerns:
        tail = "Concerns: " + ", ".join(concerns) + "."
    elif positives:
        joined = ", ".join(positives)
        tail = joined[:1].upper() + joined[1:] + "."  # capitalize first char only
    else:
        tail = "Available, moderate engagement signals."
    return f"{primary}. {tail}"
