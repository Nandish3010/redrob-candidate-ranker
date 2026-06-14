#!/usr/bin/env python3
"""
audit_top100.py: two jobs on the ACTUAL submitted top 100.

1) IMPOSSIBILITY EXPOSURE: how many of our top-100 trip the organizers' stated honeypot
   signatures (A: 3+ expert/advanced skills with 0 months; B: a role longer than the whole
   career; C: summed tenure far exceeding stated experience) or the skill-vs-career heuristic.
   This is the only number that matters for the >10%-honeypots-in-top-100 disqualification.

2) DUMP the 100 compact profiles into batch files for a blind LLM quality audit.

Usage: python eval/audit_top100.py <candidates.jsonl[.gz]> [--submission outputs/submission.csv]
"""

import argparse
import csv
import gzip
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_eval_sample import compact  # noqa: E402

WORK = os.path.join(HERE, "_work_top100")
NUM_BATCHES = 10


def opener(path):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8")


def signatures(cand):
    p = cand.get("profile", {}) or {}
    yoe = p.get("years_of_experience", 0) or 0
    career = cand.get("career_history", []) or []
    skills = cand.get("skills", []) or []
    A = sum(1 for s in skills if s.get("proficiency") in ("advanced", "expert") and (s.get("duration_months", 0) or 0) == 0)
    B = any((r.get("duration_months", 0) or 0) > yoe * 12 + 12 for r in career)
    C = sum((r.get("duration_months", 0) or 0) for r in career) > yoe * 12 + 36
    over = sum(1 for s in skills if (s.get("duration_months", 0) or 0) > yoe * 12 + 6)
    return {"A_zero_dur_experts": A, "B_role_gt_career": B, "C_tenure_gt_exp": C, "skills_over_career": over}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidates")
    ap.add_argument("--submission", default=os.path.join(os.path.dirname(HERE), "outputs", "submission.csv"))
    args = ap.parse_args()
    os.makedirs(WORK, exist_ok=True)

    with open(args.submission, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rank_of = {r["candidate_id"]: int(r["rank"]) for r in rows}
    ids = set(rank_of)

    prof, sigs = {}, {}
    with opener(args.candidates) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            cid = c.get("candidate_id")
            if cid in ids:
                prof[cid] = compact(c)
                sigs[cid] = signatures(c)

    flagged_any = [cid for cid in ids if sigs[cid]["A_zero_dur_experts"] >= 3 or sigs[cid]["B_role_gt_career"]
                   or sigs[cid]["C_tenure_gt_exp"]]
    flagged_two = [cid for cid in ids if sum([sigs[cid]["A_zero_dur_experts"] >= 3, sigs[cid]["B_role_gt_career"],
                                              sigs[cid]["C_tenure_gt_exp"]]) >= 2]
    over4 = [cid for cid in ids if sigs[cid]["skills_over_career"] >= 4]
    over8 = [cid for cid in ids if sigs[cid]["skills_over_career"] >= 8]

    print("=" * 60)
    print("TOP-100 IMPOSSIBILITY EXPOSURE (DQ if >10 honeypots in top 100)")
    print("=" * 60)
    print(f"  trip >=1 organizer signature (A>=3 / B / C): {len(flagged_any)}")
    print(f"  trip >=2 organizer signatures (conjunctive): {len(flagged_two)}")
    print(f"  >=4 skills over career (our gate threshold):  {len(over4)}")
    print(f"  >=8 skills over career (extreme):             {len(over8)}")
    for cid in sorted(flagged_any, key=lambda c: rank_of[c]):
        print(f"    rank {rank_of[cid]:3d} {cid}  {sigs[cid]}")

    # dump batches in rank order for the LLM audit
    ordered = sorted(ids, key=lambda c: rank_of[c])
    json.dump({cid: rank_of[cid] for cid in ordered}, open(os.path.join(WORK, "rank_meta.json"), "w"), indent=2)
    batches = [[] for _ in range(NUM_BATCHES)]
    for i, cid in enumerate(ordered):
        batches[i % NUM_BATCHES].append(prof[cid])
    for i, b in enumerate(batches):
        json.dump(b, open(os.path.join(WORK, f"batch_{i:02d}.json"), "w"), indent=2)
    print(f"\nDumped {len(ordered)} top-100 profiles to {NUM_BATCHES} batch files in {WORK}/ for LLM audit.")


if __name__ == "__main__":
    main()
