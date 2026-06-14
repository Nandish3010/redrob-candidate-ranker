#!/usr/bin/env python3
"""Build a small DETERMINISTIC gold set of clear candidate archetypes for offline regression.

We cannot see the hidden ground truth, so we hand-define archetypes by explicit, human-defensible
rules over RAW candidate fields (title, industry, experience, skill endorsement/duration, internal
impossibility). These rules are intentionally INDEPENDENT of scorer.py's weighting, so using the
set to check the scorer is not circular. We scan the pool, pick the first few candidates matching
each archetype, and freeze {candidate_id -> tier, archetype, why} into eval/gold_labels.json.

Relevance convention (matches the challenge: tier_3+ is "relevant" for P@10):
  5 = textbook fit (ranking/search/recsys at a product company, in band, trusted IR skills)
  4 = strong AI/ML engineer at a product company, in band, with real IR work in descriptions
  3 = relevant but with one clear gap (just out of the experience band)
  2 = ML/DS in the wrong sub-domain (CV/forecasting/etc.) or a below-band junior
  1 = services-only career (consulting/IT-services) with no product work
  0 = keyword stuffer, honeypot (impossible profile), or clearly non-technical role

Run:  python eval/build_gold_set.py --candidates <path>   (writes eval/gold_labels.json)
"""
import argparse
import gzip
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "gold_labels.json")

RANKING_TITLE = ("search", "ranking", "recommendation", "recommender", "relevance", "recsys", "discovery")
ML_TITLE = ("machine learning", "ml engineer", "ai engineer", "applied scientist", "data scientist",
            "nlp", "research engineer", "ml scientist", "applied ml")
NONTECH_TITLE = ("marketing", "sales", "account manager", "human resource", "hr ", "recruiter",
                 "operations manager", "finance", "customer success", "business development",
                 "content", "social media", "product manager", "project manager")
PRODUCT_IND = ("software", "internet", "fintech", "e commerce", "ecommerce", "saas", "technology",
               "artificial intelligence", "machine learning", "edtech", "healthtech", "gaming",
               "consumer", "marketplace", "streaming", "mobility", "social")
SERVICES_IND = ("it services", "consulting", "staffing", "outsourcing", "bpo",
                "information technology and services")
CORE_IR = {"embeddings", "embedding", "retrieval", "information retrieval", "semantic search",
           "vector search", "vector database", "ranking", "learning to rank", "recommendation",
           "recommender", "recommendation systems", "bm25", "faiss", "pinecone", "weaviate",
           "qdrant", "elasticsearch", "opensearch", "rag", "retrieval augmented generation"}
IR_DESC = ("ranking", "retrieval", "recommend", "search", "embedding", "vector", "relevance",
           "semantic", "personaliz", "learning to rank", "ndcg")
CV_FC = ("computer vision", "image", "forecast", "fraud", "speech", "object detection",
         "segmentation", "time series", "anomaly")


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s or "").lower()).strip()


def open_pool(path):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8")


def features(c):
    p = c.get("profile", {}) or {}
    career = c.get("career_history", []) or []
    skills = c.get("skills", []) or []
    yoe = float(p.get("years_of_experience", 0) or 0)
    titles = norm(p.get("current_title", "")) + " " + " ".join(norm(r.get("title", "")) for r in career)
    inds = norm(p.get("current_industry", "")) + " " + " ".join(norm(r.get("industry", "")) for r in career)
    descs = " ".join(norm(r.get("description", "")) for r in career)
    sum_m = sum(int(r.get("duration_months", 0) or 0) for r in career)
    trusted_ir = sum(1 for s in skills if norm(s.get("name", "")) in CORE_IR
                     and int(s.get("endorsements", 0) or 0) >= 5
                     and int(s.get("duration_months", 0) or 0) >= 12)
    over = sum(1 for s in skills if int(s.get("duration_months", 0) or 0) > (sum_m if sum_m else yoe * 12))
    return {
        "yoe": yoe, "titles": titles, "inds": inds, "descs": descs,
        "ranking_title": any(t in titles for t in RANKING_TITLE),
        "ml_title": any(t in titles for t in ML_TITLE),
        "nontech_title": any(t in titles for t in NONTECH_TITLE),
        "product": any(t in inds for t in PRODUCT_IND),
        "services_only": any(t in inds for t in SERVICES_IND) and not any(t in inds for t in PRODUCT_IND),
        "ir_desc": any(t in descs for t in IR_DESC),
        "cv_fc": any(t in descs for t in CV_FC) or any(t in titles for t in CV_FC),
        "trusted_ir": trusted_ir,
        "over": over,
        "n_core_ir_skills": sum(1 for s in skills if norm(s.get("name", "")) in CORE_IR),
        "in_band": 5.0 <= yoe <= 9.0,
        "near_band": 3.5 <= yoe < 5.0 or 9.0 < yoe <= 11.5,
    }


def classify(f):
    """Return (tier, archetype) for clear cases only, else (None, None)."""
    # tier 0 first (impossible / stuffer / non-technical), highest confidence
    if f["over"] >= 8:
        return 0, "honeypot (>=8 skills exceed career length)"
    if f["nontech_title"] and not f["ir_desc"] and not f["ml_title"] and f["n_core_ir_skills"] >= 3:
        return 0, "keyword stuffer (non-technical title, stuffed AI skills, no IR work)"
    if f["nontech_title"] and not f["ml_title"] and not f["ir_desc"] and f["n_core_ir_skills"] == 0:
        return 0, "clearly non-technical role"
    # tier 5: textbook
    if (f["ranking_title"] and f["product"] and f["in_band"] and f["trusted_ir"] >= 2
            and f["ir_desc"] and f["over"] < 2):
        return 5, "ranking/search/recsys engineer at product company, in band, trusted IR skills"
    # tier 4: strong AI/ML at product company with real IR work, and NOT a CV/forecasting role
    # (a pure computer-vision/forecasting engineer is a weak fit for a ranking/retrieval JD).
    if (f["ml_title"] and f["product"] and f["in_band"] and f["ir_desc"] and not f["cv_fc"]
            and f["trusted_ir"] >= 1 and f["over"] < 2):
        return 4, "strong AI/ML engineer at product company, in band, real IR work (not CV/forecasting)"
    # tier 3: relevant but just out of band
    if (f["ml_title"] and f["product"] and f["ir_desc"] and not f["cv_fc"] and f["near_band"] and f["over"] < 2):
        return 3, "relevant AI/ML + IR work but just outside the experience band"
    # tier 2: ML in the wrong sub-domain (CV/forecasting), or no real IR work
    if (f["ml_title"] and (f["cv_fc"] or not f["ir_desc"]) and not f["services_only"] and f["over"] < 2):
        return 2, "ML/DS in a non-IR sub-domain (CV/forecasting) or without real IR work"
    # tier 1: services-only career
    if (f["services_only"] and (f["ml_title"] or f["n_core_ir_skills"] > 0) and not f["ir_desc"]
            and f["over"] < 2):
        return 1, "services-only career (consulting / IT services), no product IR work"
    return None, None


QUOTA = {5: 6, 4: 6, 3: 4, 2: 6, 1: 5, 0: 8}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    picked = {t: [] for t in QUOTA}
    with open_pool(args.candidates) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            tier, arche = classify(features(c))
            if tier is None or len(picked[tier]) >= QUOTA[tier]:
                continue
            picked[tier].append({"candidate_id": c["candidate_id"], "tier": tier,
                                 "archetype": arche, "title": c.get("profile", {}).get("current_title", "")})
            if all(len(picked[t]) >= QUOTA[t] for t in QUOTA):
                break

    labels = {}
    for t in QUOTA:
        for row in picked[t]:
            labels[row["candidate_id"]] = {"tier": row["tier"], "archetype": row["archetype"],
                                           "title": row["title"]}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)
    print(f"Wrote {args.out}: {len(labels)} labeled candidates")
    for t in sorted(QUOTA, reverse=True):
        print(f"  tier {t}: {len(picked[t])}/{QUOTA[t]}")


if __name__ == "__main__":
    main()
