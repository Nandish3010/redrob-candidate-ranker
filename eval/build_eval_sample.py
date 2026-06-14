#!/usr/bin/env python3
"""
build_eval_sample.py: two jobs.

1) KEYWORD-STUFFER AUDIT: scan the whole pool for profiles whose current title is non-technical
   but whose skills list is packed with AI/IR keywords (e.g. "Marketing Manager" listing RAG,
   Pinecone, embeddings). Report where our ranker places them, to prove the trust-gating works.

2) BUILD A BLIND EVAL SAMPLE for the offline NDCG/MAP/P@10 proxy: our top 50, a seeded random
   sample from OUTSIDE our top 100 (to catch candidates we under-ranked), and known stuffers.
   Writes compact profiles split into batch files for parallel LLM labeling, plus a meta file
   mapping each candidate to its bucket and our rank/score (the labeler never sees bucket/rank).

Outputs go to eval/_work/ (gitignored). Deterministic (fixed seed).
"""

import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
from scorer import (score_candidate, norm, _title_score,  # noqa: E402
                    CORE_IR_SKILLS, SUPPORTING_SKILLS)

WORK = os.path.join(HERE, "_work")
NUM_BATCHES = 10
N_TOP = 50          # our top-50, to measure top-precision
N_RANDOM = 90       # random outside-top-100, to catch misses (recall)
N_STUFFER = 20      # planted keyword-stuffers, to confirm they stay down
SEED = 20260614


def is_ai_skill(name: str) -> bool:
    n = norm(name)
    return n in CORE_IR_SKILLS or n in SUPPORTING_SKILLS


def compact(cand: dict) -> dict:
    p = cand.get("profile", {}) or {}
    sig = cand.get("redrob_signals", {}) or {}
    return {
        "candidate_id": cand.get("candidate_id"),
        "current_title": p.get("current_title"),
        "current_industry": p.get("current_industry"),
        "years_of_experience": p.get("years_of_experience"),
        "headline": p.get("headline"),
        "summary": (p.get("summary") or "")[:600],
        "career_history": [
            {"title": r.get("title"), "industry": r.get("industry"),
             "company_size": r.get("company_size"), "duration_months": r.get("duration_months"),
             "description": (r.get("description") or "")[:240]}
            for r in (cand.get("career_history") or [])[:4]
        ],
        "education": [
            {"degree": e.get("degree"), "field_of_study": e.get("field_of_study"),
             "tier": e.get("tier")} for e in (cand.get("education") or [])[:2]
        ],
        "skills": [
            {"name": s.get("name"), "proficiency": s.get("proficiency"),
             "endorsements": s.get("endorsements"), "duration_months": s.get("duration_months")}
            for s in (cand.get("skills") or [])
        ],
        "signals": {
            "location": p.get("location"), "country": p.get("country"),
            "last_active_date": sig.get("last_active_date"),
            "recruiter_response_rate": sig.get("recruiter_response_rate"),
            "open_to_work_flag": sig.get("open_to_work_flag"),
            "willing_to_relocate": sig.get("willing_to_relocate"),
            "github_activity_score": sig.get("github_activity_score"),
            "interview_completion_rate": sig.get("interview_completion_rate"),
            "notice_period_days": sig.get("notice_period_days"),
        },
    }


def main():
    global WORK, SEED, N_TOP, N_RANDOM, N_STUFFER
    ap = argparse.ArgumentParser(description="Build a blind eval sample for the offline proxy.")
    ap.add_argument("candidates")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--work", default=WORK)
    ap.add_argument("--top", type=int, default=N_TOP)
    ap.add_argument("--random", type=int, default=N_RANDOM)
    ap.add_argument("--stuffer", type=int, default=N_STUFFER)
    args = ap.parse_args()
    WORK, SEED = args.work, args.seed
    N_TOP, N_RANDOM, N_STUFFER = args.top, args.random, args.stuffer
    path = args.candidates
    if not os.path.exists(path):
        sys.exit("Usage: python eval/build_eval_sample.py <candidates.jsonl[.gz]> "
                 "[--seed N --work DIR --top N --random N --stuffer N]")
    os.makedirs(WORK, exist_ok=True)

    import gzip
    opener = gzip.open if path.endswith(".gz") else open

    scored = []                 # (score, cid)
    profiles = {}               # cid -> compact profile
    stuffers = []               # (n_ai_skills, cid) for non-tech-title + many AI skills
    n = 0
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cand = json.loads(line)
            r = score_candidate(cand)
            cid = r["candidate_id"]
            scored.append((r["score"], cid))
            profiles[cid] = compact(cand)
            title_strength = max(
                _title_score(cand.get("profile", {}).get("current_title", "")),
                max([_title_score(x.get("title", "")) for x in (cand.get("career_history") or [])],
                    default=0.0),
            )
            n_ai = sum(1 for s in (cand.get("skills") or []) if is_ai_skill(s.get("name", "")))
            # non-technical title (current AND best historical title both weak) but >=5 AI skills
            cur_weak = _title_score(cand.get("profile", {}).get("current_title", "")) <= 0.1
            if cur_weak and title_strength <= 0.1 and n_ai >= 5:
                stuffers.append((n_ai, cid, r["score"]))
            n += 1

    scored.sort(key=lambda x: (-x[0], x[1]))
    rank_of = {cid: i + 1 for i, (_, cid) in enumerate(scored)}
    score_of = {cid: s for s, cid in scored}
    top100 = set(cid for _, cid in scored[:100])

    # ---- Stuffer report ----
    stuffers.sort(reverse=True)  # most AI skills first
    stuffer_ranks = [rank_of[c] for _, c, _ in stuffers]
    n_total = len(scored)
    report = {
        "n_stuffers_found": len(stuffers),
        "definition": "current_title and best career title both non-technical, but >=5 AI/IR skills listed",
        "in_top_100": sum(1 for rk in stuffer_ranks if rk <= 100),
        "in_top_1000": sum(1 for rk in stuffer_ranks if rk <= 1000),
        "best_rank_any_stuffer": min(stuffer_ranks) if stuffer_ranks else None,
        "median_rank": sorted(stuffer_ranks)[len(stuffer_ranks) // 2] if stuffer_ranks else None,
        "pool_size": n_total,
        "examples": [
            {"candidate_id": c, "n_ai_skills": k, "our_score": round(sc, 4),
             "our_rank": rank_of[c], "percentile": round(100 * rank_of[c] / n_total, 1),
             "current_title": profiles[c]["current_title"],
             "top_ai_skills": [s["name"] for s in profiles[c]["skills"] if is_ai_skill(s["name"])][:6]}
            for k, c, sc in stuffers[:8]
        ],
    }
    with open(os.path.join(WORK, "stuffer_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # ---- Build blind eval sample ----
    rng = random.Random(SEED)
    top_ids = [cid for _, cid in scored[:N_TOP]]
    outside = [cid for _, cid in scored[100:]]
    rng.shuffle(outside)
    random_ids = outside[:N_RANDOM]
    stuffer_ids = [c for _, c, _ in stuffers if c not in top100][:N_STUFFER]

    bucket = {}
    for cid in top_ids:
        bucket[cid] = "top50"
    for cid in random_ids:
        bucket.setdefault(cid, "random_outside")
    for cid in stuffer_ids:
        bucket.setdefault(cid, "stuffer")

    sample_ids = list(bucket.keys())
    rng.shuffle(sample_ids)  # shuffle so the labeler cannot infer bucket from order

    meta = {cid: {"bucket": bucket[cid], "our_rank": rank_of[cid],
                  "our_score": round(score_of[cid], 6)} for cid in sample_ids}
    with open(os.path.join(WORK, "sample_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    batches = [[] for _ in range(NUM_BATCHES)]
    for i, cid in enumerate(sample_ids):
        batches[i % NUM_BATCHES].append(profiles[cid])
    for i, b in enumerate(batches):
        with open(os.path.join(WORK, f"batch_{i:02d}.json"), "w") as f:
            json.dump(b, f, indent=2)

    print(f"Scored {n_total} candidates.")
    print(f"Stuffer report -> {WORK}/stuffer_report.json")
    print(f"  stuffers found: {report['n_stuffers_found']}, in top 100: {report['in_top_100']}, "
          f"best stuffer rank: {report['best_rank_any_stuffer']} of {n_total}")
    print(f"Eval sample: {len(sample_ids)} candidates -> {NUM_BATCHES} batch files in {WORK}/")
    print(f"  buckets: top50={len(top_ids)}, random_outside={len(random_ids)}, stuffer={len(stuffer_ids)}")


if __name__ == "__main__":
    main()
