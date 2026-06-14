#!/usr/bin/env python3
"""
self_audit.py: offline safety net to run BEFORE spending one of the 3 precious submissions.

Checks the things the official validate_submission.py does NOT (it only checks format,
uniqueness, rank coverage, score monotonicity, and the candidate_id tie-break):
  1. candidate_id MEMBERSHIP: every id in the submission exists in the candidate pool.
  2. HONEYPOT RATE in the top 100 (hard DQ if > 10%).
  3. Format sanity (100 rows, ranks 1..100, score non-increasing).
  4. Score distribution (not all-equal; reasonable spread).

Usage:
    python eval/self_audit.py --candidates ./data/candidates.jsonl --submission ./outputs/submission.csv
"""

import argparse
import csv
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from scorer import _honeypot_multiplier  # noqa: E402


def open_candidates(path):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") \
        else open(path, "r", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--submission", required=True)
    args = ap.parse_args()

    with open(args.submission, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    sub_ids = [r["candidate_id"] for r in rows]
    sub_id_set = set(sub_ids)

    # Stream the pool once; collect membership + honeypot flags for submitted ids only.
    pool_ids = set()
    honey_flags = {}
    with open_candidates(args.candidates) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cand = json.loads(line)
            cid = cand.get("candidate_id")
            pool_ids.add(cid)
            if cid in sub_id_set:
                honey_flags[cid] = _honeypot_multiplier(cand) < 0.3

    issues = []
    if len(rows) != 100:
        issues.append(f"Expected 100 rows, found {len(rows)}.")

    missing = [cid for cid in sub_ids if cid not in pool_ids]
    if missing:
        issues.append(f"{len(missing)} candidate_id(s) NOT in the pool, e.g. {missing[:5]}.")

    ranks = [int(r["rank"]) for r in rows]
    if sorted(ranks) != list(range(1, len(rows) + 1)):
        issues.append("Ranks are not exactly 1..N each once.")

    scores = [float(r["score"]) for r in rows]
    if any(scores[i] < scores[i + 1] for i in range(len(scores) - 1)):
        issues.append("Scores are not non-increasing by rank.")
    if len(set(scores)) == 1:
        issues.append("All scores are identical (model is not differentiating).")

    n_honey = sum(1 for cid in sub_ids if honey_flags.get(cid))
    honey_rate = 100.0 * n_honey / max(1, len(sub_ids))
    if honey_rate > 10.0:
        issues.append(f"Honeypot rate {honey_rate:.1f}% exceeds the 10% DQ threshold.")

    print(f"Pool size: {len(pool_ids)} | submitted: {len(sub_ids)} | unique: {len(sub_id_set)}")
    print(f"Honeypots in top 100: {n_honey} ({honey_rate:.1f}%)")
    print(f"Score range: {min(scores):.4f} .. {max(scores):.4f} | distinct: {len(set(scores))}")
    if issues:
        print("\nSELF-AUDIT FAILED:")
        for i in issues:
            print(f"  - {i}")
        sys.exit(1)
    print("\nSelf-audit passed. Safe to validate with the official validator and submit.")


if __name__ == "__main__":
    main()
