#!/usr/bin/env python3
"""
build_artifacts.py: UNTIMED Phase-2 precompute. Builds the small TF-IDF similarity model the
ranker uses for the capped jd_similarity term, and writes it to artifacts/sim_model.json.

    python build_artifacts.py --candidates ./data/candidates.jsonl

The model is:
  - idf:    {term -> inverse document frequency} over a capped vocabulary (df-pruned, top-N).
  - seeds:  four L2-normalized TF-IDF vectors, one per JD theme (see JD_SEEDS below).
  - anchor: a high pool percentile (default 92nd) of the candidate-vs-best-seed cosine, used to
            rescale the cosine into a [0, 1] component. Data-derived, so it is NOT a hand-tuned
            knob: it just maps "as on-topic as the top ~8% of the pool" to ~1.0.

This is aggregate term statistics plus hand-authored JD vectors. It contains NO candidate data,
so artifacts/sim_model.json is committed (a model file, not data). CPU only, stdlib only.
The candidate document surface deliberately excludes the raw skills array (see scorer._candidate_doc),
so keyword stuffing cannot inflate similarity. Pre-computation may exceed 5 minutes; only rank.py
is on the 5-minute clock, and it just loads this file.
"""

import argparse
import gzip
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from scorer import _tokens, _candidate_doc  # noqa: E402  (shared so build and score tokenize identically)

# JD themes (Senior AI Engineer at a talent-intelligence startup building ranking/retrieval).
# These are the queries the candidate document is matched against; phrased in plain JD language.
JD_SEEDS = [
    "ranking relevance search retrieval recommendation systems personalization discovery matching",
    "embeddings vector search semantic retrieval rag information retrieval similarity nearest neighbor",
    "learning to rank ndcg mrr offline evaluation ab testing ranking metrics precision recall",
    "production machine learning python pytorch large scale systems mlops model serving deployment",
]

MIN_DF = 5            # ignore terms appearing in fewer than this many candidate docs (noise)
MAX_DF_RATIO = 0.5    # ignore terms in more than half the docs (too common to discriminate)
MAX_VOCAB = 12000     # keep the model small enough to commit (top terms by document frequency)
ANCHOR_PCTL = 92      # pool percentile of max-cosine that maps to similarity 1.0
ANCHOR_SAMPLE = 25000 # candidates sampled (first N) to estimate the anchor; untimed, so generous


def open_candidates(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def l2_normalize(vec: dict) -> dict:
    nrm = math.sqrt(sum(v * v for v in vec.values()))
    if nrm == 0.0:
        return vec
    return {k: v / nrm for k, v in vec.items()}


def candidate_vector(cand: dict, idf: dict):
    """TF-IDF vector (dict) for a candidate doc, restricted to the model vocabulary."""
    tf = {}
    for t in _tokens(_candidate_doc(cand)):
        if t in idf:
            tf[t] = tf.get(t, 0) + 1
    if not tf:
        return None
    return {t: (1.0 + math.log(c)) * idf[t] for t, c in tf.items()}


def main():
    ap = argparse.ArgumentParser(description="Build the Phase-2 TF-IDF similarity model.")
    ap.add_argument("--candidates", required=True, help="Path to candidates.jsonl or .jsonl.gz")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "artifacts", "sim_model.json"))
    args = ap.parse_args()

    if not os.path.exists(args.candidates):
        sys.exit(f"ERROR: candidates file not found: {args.candidates}")

    # Pass 1: document frequencies over candidate docs.
    df = {}
    n_docs = 0
    with open_candidates(args.candidates) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cand = json.loads(line)
            for t in set(_tokens(_candidate_doc(cand))):
                df[t] = df.get(t, 0) + 1
            n_docs += 1
            if n_docs % 20000 == 0:
                print(f"  pass1 {n_docs} docs, vocab {len(df)}", file=sys.stderr)

    # Build the pruned, capped vocabulary + IDF. Force-include every JD seed term so the seed
    # vectors are never empty even if a seed term is rare in the pool.
    max_df = int(MAX_DF_RATIO * n_docs)
    seed_terms = set(t for s in JD_SEEDS for t in _tokens(s))
    kept = [(t, c) for t, c in df.items() if (MIN_DF <= c <= max_df) or t in seed_terms]
    kept.sort(key=lambda x: -x[1])
    kept = kept[:MAX_VOCAB]
    vocab = {t for t, _ in kept}
    vocab |= (seed_terms & set(df))  # ensure seed terms present in the pool are kept
    idf = {t: math.log(n_docs / df[t]) for t in vocab if df.get(t)}

    # Build the seed vectors with the same IDF, L2-normalized.
    seeds = []
    for s in JD_SEEDS:
        tf = {}
        for t in _tokens(s):
            if t in idf:
                tf[t] = tf.get(t, 0) + 1
        vec = {t: (1.0 + math.log(c)) * idf[t] for t, c in tf.items()}
        seeds.append(l2_normalize(vec))

    # Pass 2 (sampled): estimate the calibration anchor from the pool's max-cosine distribution.
    cosines = []
    seen = 0
    with open_candidates(args.candidates) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cand = json.loads(line)
            vec = candidate_vector(cand, idf)
            if vec:
                nrm = math.sqrt(sum(v * v for v in vec.values()))
                if nrm > 0:
                    best = 0.0
                    for seed in seeds:
                        dot = sum(w * seed.get(t, 0.0) for t, w in vec.items())
                        if dot > best:
                            best = dot
                    cosines.append(best / nrm)
            seen += 1
            if seen >= ANCHOR_SAMPLE:
                break
    cosines.sort()
    if cosines:
        idx = min(len(cosines) - 1, int(ANCHOR_PCTL / 100.0 * len(cosines)))
        anchor = max(1e-6, cosines[idx])
    else:
        anchor = 1.0

    model = {
        "idf": idf,
        "seeds": seeds,
        "anchor": anchor,
        "meta": {
            "n_docs": n_docs,
            "vocab_size": len(idf),
            "anchor_pctl": ANCHOR_PCTL,
            "anchor_sample": min(seen, ANCHOR_SAMPLE),
            "jd_seeds": JD_SEEDS,
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(model, f)
    size_kb = os.path.getsize(args.out) / 1024.0
    print(f"Wrote {args.out}: vocab {len(idf)}, anchor {anchor:.4f} "
          f"(p{ANCHOR_PCTL} of {len(cosines)} sampled), {size_kb:.0f} KB.")


if __name__ == "__main__":
    main()
