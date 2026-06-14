#!/usr/bin/env python3
"""
proxy_eval.py: compute a DIRECTIONAL NDCG@10/@50, MAP, P@10 and the weighted composite for our
ranking against independent LLM gold tiers.

Honest limitations (printed at the end):
  - The "ground truth" here is an LLM's judgment, not the organizers' hidden labels.
  - Metrics are computed over the LABELED UNIVERSE (our top 50 + a random outside-top-100 sample
    + planted stuffers), not the full 100k. So NDCG here measures ordering quality within that
    universe and partially captures recall (a strong candidate we ranked low drags our NDCG down
    because it lifts the ideal ranking).

Reads eval/_work/sample_meta.json and eval/_work/gold_labels.json.
"""

import argparse
import json
import math
import os
import sys

WORK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_work")


def dcg(tiers):
    return sum((2 ** t - 1) / math.log2(i + 2) for i, t in enumerate(tiers))


def ndcg_at_k(our_tiers_in_order, all_tiers, k):
    actual = dcg(our_tiers_in_order[:k])
    ideal = dcg(sorted(all_tiers, reverse=True)[:k])
    return (actual / ideal) if ideal > 0 else 0.0


def main():
    global WORK
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=WORK)
    args = ap.parse_args()
    WORK = args.work
    meta = json.load(open(os.path.join(WORK, "sample_meta.json")))
    labels = json.load(open(os.path.join(WORK, "gold_labels.json")))

    universe = [cid for cid in meta if cid in labels]
    universe.sort(key=lambda c: meta[c]["our_rank"])  # our ranking order, restricted to labeled set
    our_tiers = [int(labels[c]["tier"]) for c in universe]
    all_tiers = list(our_tiers)

    p10 = sum(1 for t in our_tiers[:10] if t >= 3) / 10.0
    ndcg10 = ndcg_at_k(our_tiers, all_tiers, 10)
    ndcg50 = ndcg_at_k(our_tiers, all_tiers, 50)

    # MAP (relevant = tier >= 3), average precision over our restricted ranking
    R = sum(1 for t in all_tiers if t >= 3)
    hits, ap = 0, 0.0
    for i, t in enumerate(our_tiers, start=1):
        if t >= 3:
            hits += 1
            ap += hits / i
    map_score = (ap / R) if R > 0 else 0.0

    composite = 0.50 * ndcg10 + 0.30 * ndcg50 + 0.15 * map_score + 0.05 * p10

    # Bucket diagnostics
    buckets = {}
    for cid in universe:
        b = meta[cid]["bucket"]
        buckets.setdefault(b, []).append(int(labels[cid]["tier"]))

    print("=" * 64)
    print("OFFLINE PROXY SCORE (LLM gold tiers; directional, not official)")
    print("=" * 64)
    print(f"Labeled universe: {len(universe)} candidates | relevant (tier>=3): {R}")
    print(f"  NDCG@10   {ndcg10:.3f}")
    print(f"  NDCG@50   {ndcg50:.3f}")
    print(f"  MAP       {map_score:.3f}")
    print(f"  P@10      {p10:.3f}")
    print(f"  COMPOSITE {composite:.3f}   (0.50*NDCG@10 + 0.30*NDCG@50 + 0.15*MAP + 0.05*P@10)")
    print()
    print("Tier distribution by bucket (mean tier; count by tier 0..5):")
    for b in ("top50", "random_outside", "stuffer"):
        ts = buckets.get(b, [])
        if not ts:
            continue
        dist = [ts.count(i) for i in range(6)]
        print(f"  {b:16s} n={len(ts):3d}  mean={sum(ts)/len(ts):.2f}  tiers0-5={dist}")

    # Recall flags: strong candidates (tier>=4) we ranked OUTSIDE our top 100
    misses = [(cid, labels[cid]["tier"], meta[cid]["our_rank"])
              for cid in universe
              if meta[cid]["bucket"] == "random_outside" and int(labels[cid]["tier"]) >= 4]
    print()
    if misses:
        print(f"RECALL FLAG: {len(misses)} sampled candidate(s) rated tier>=4 but ranked outside our top 100:")
        for cid, t, rk in sorted(misses, key=lambda x: -x[1]):
            print(f"  {cid}  tier={t}  our_rank={rk}  reason={labels[cid].get('reason','')[:80]}")
    else:
        print("RECALL CHECK: no sampled outside-top-100 candidate was rated tier>=4 (no obvious misses).")
    print()
    print("CAVEATS: LLM labels are a proxy, not the hidden ground truth. Metrics are over the")
    print("labeled universe, not the full 100k. Treat as directional only.")


if __name__ == "__main__":
    main()
