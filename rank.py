#!/usr/bin/env python3
"""
rank.py: TIMED ranking step. Reads the candidate pool and writes the top-100 submission CSV.

Single reproduce command (Stage-3):
    python rank.py --candidates ./data/candidates.jsonl --out ./outputs/submission.csv

Reads both candidates.jsonl and candidates.jsonl.gz (detects the .gz suffix), mirroring the
official validator. CPU only, no network. v0 baseline streams the pool once, scores every
candidate with src/scorer.py, and writes the top 100 ranked best-first with the spec columns:
candidate_id,rank,score,reasoning. Equal scores are ordered by candidate_id ascending so the
official validator's tie-break rule passes.
"""

import argparse
import csv
import gzip
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from scorer import score_candidate, make_reasoning  # noqa: E402

TOP_K = 100


def open_candidates(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Rank candidates for the Redrob JD.")
    ap.add_argument("--candidates", required=True, help="Path to candidates.jsonl or .jsonl.gz")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--topk", type=int, default=TOP_K)
    args = ap.parse_args()

    if not os.path.exists(args.candidates):
        sys.exit(f"ERROR: candidates file not found: {args.candidates}\n"
                 f"Place the organizer-provided file there or pass --candidates (see README).")

    t0 = time.time()
    scored = []          # (score, candidate_id)
    rctx_by_id = {}      # candidate_id -> compact reasoning context
    n = 0
    skipped = 0          # rows we could not parse or score (never abort the whole run for one bad row)
    with open_candidates(args.candidates) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            # Defensive: a single off-schema or unparseable row must not abort a 100k-row run
            # (Stage-3 grades on a file we have never seen). Skip it and keep going.
            try:
                cand = json.loads(line)
                r = score_candidate(cand)
                cid = r["candidate_id"]
                if not cid:
                    raise ValueError("missing candidate_id")
                scored.append((r["score"], cid))
                rctx_by_id[cid] = r["rctx"]
                n += 1
            except Exception as e:
                skipped += 1
                if skipped <= 10:
                    print(f"  WARNING: skipped line {line_no}: {type(e).__name__}: {e}", file=sys.stderr)
                continue
            if n % 20000 == 0:
                print(f"  scored {n} candidates ({time.time() - t0:.1f}s)", file=sys.stderr)
    if skipped:
        print(f"  Skipped {skipped} unparseable/invalid row(s) total.", file=sys.stderr)

    # Sort best-first; equal scores -> candidate_id ascending (validator tie-break rule).
    scored.sort(key=lambda x: (-x[0], x[1]))
    top = scored[:args.topk]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (score, cid) in enumerate(top, start=1):
            w.writerow([cid, rank, f"{score:.6f}", make_reasoning(rctx_by_id[cid])])

    n_honey = sum(1 for _, cid in top if rctx_by_id[cid]["honeypot"])
    print(f"Ranked {n} candidates in {time.time() - t0:.1f}s. Wrote top {len(top)} to {args.out}.")
    print(f"Honeypots in top {len(top)}: {n_honey} "
          f"({100.0 * n_honey / max(1, len(top)):.1f}%; must stay under 10%).")


if __name__ == "__main__":
    main()
