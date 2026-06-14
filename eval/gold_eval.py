#!/usr/bin/env python3
"""Evaluate scorer.py against the frozen deterministic gold set (eval/gold_labels.json).

Scores every labeled candidate with the REAL scorer, ranks them, and reports NDCG@10/@50, MAP and
P@10 (relevant = tier_3+), plus the ranked list with tiers so regressions are visible at a glance.
This is a stable yardstick: a change for submission #2 should hold or improve these numbers and must
keep tier-5/4 clustered at the top and tier-0 at the bottom. It is a proxy, not the hidden truth.

Run:  python eval/gold_eval.py --candidates <path>
"""
import argparse
import gzip
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from scorer import score_candidate  # noqa: E402

LABELS = os.path.join(HERE, "gold_labels.json")


def open_pool(path):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8")


def dcg(rels):
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def ndcg_at(rels, k):
    ideal = sorted(rels, reverse=True)
    idcg = dcg(ideal[:k])
    return dcg(rels[:k]) / idcg if idcg > 0 else 0.0


def average_precision(is_rel):
    hits, ap = 0, 0.0
    for i, rel in enumerate(is_rel, start=1):
        if rel:
            hits += 1
            ap += hits / i
    total_rel = sum(is_rel)
    return ap / total_rel if total_rel else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    args = ap.parse_args()
    labels = json.load(open(LABELS, encoding="utf-8"))

    rows = []  # (score, cid, tier, title)
    found = {}
    with open_pool(args.candidates) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            cid = c.get("candidate_id")
            if cid in labels:
                r = score_candidate(c)
                found[cid] = True
                rows.append((r["score"], cid, labels[cid]["tier"], labels[cid].get("title", "")))
            if len(found) == len(labels):
                break

    missing = set(labels) - set(found)
    if missing:
        print(f"WARNING: {len(missing)} labeled ids not found in pool: {sorted(missing)[:5]}")

    rows.sort(key=lambda x: (-x[0], x[1]))  # same ordering rule as rank.py
    tiers = [t for _, _, t, _ in rows]
    is_rel = [1 if t >= 3 else 0 for t in tiers]

    ndcg10 = ndcg_at(tiers, 10)
    ndcg50 = ndcg_at(tiers, 50)
    mapv = average_precision(is_rel)
    p10 = sum(is_rel[:10]) / 10.0
    composite = 0.50 * ndcg10 + 0.30 * ndcg50 + 0.15 * mapv + 0.05 * p10

    print(f"Gold set: {len(rows)} candidates ({sum(is_rel)} relevant, tier_3+)")
    print(f"  NDCG@10 {ndcg10:.4f} | NDCG@50 {ndcg50:.4f} | MAP {mapv:.4f} | P@10 {p10:.2f}")
    print(f"  composite (0.50/0.30/0.15/0.05 proxy) = {composite:.4f}")
    print()
    print("  rank  tier  score    candidate     title")
    for i, (sc, cid, t, title) in enumerate(rows, start=1):
        flag = "  <-- tier-0 in top 10" if (t == 0 and i <= 10) else ""
        print(f"  {i:>4}   t{t}   {sc:.4f}  {cid}  {title[:30]}{flag}")

    # Regression guards: the strong, unambiguous expectations.
    top10_tiers = tiers[:10]
    bottom = tiers[-8:]
    ok = True
    if any(t == 0 for t in top10_tiers):
        print("\nCONCERN: a tier-0 archetype reached the top 10."); ok = False
    if sum(1 for t in top10_tiers if t >= 4) < 8:
        print("\nNOTE: fewer than 8 of the top 10 are tier-4/5 (check head precision)."); ok = False
    if any(t >= 4 for t in bottom):
        print("\nCONCERN: a tier-4/5 archetype sank to the bottom."); ok = False
    print("\nGold-set check:", "PASS" if ok else "REVIEW")


if __name__ == "__main__":
    main()
