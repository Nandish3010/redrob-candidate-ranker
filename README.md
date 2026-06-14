# Redrob Candidate Ranker

Ranks the top 100 best-fit candidates for the Redrob "Senior AI Engineer" job description
out of a 100,000-candidate pool, for the H2S Data & AI Challenge.

The system is a transparent rule-engine scorer with a capped embedding/similarity recall
layer. Final scores decompose into named components, so every rank is explainable. See
`CONTEXT.md` for the full design, constraints, and decisions.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Place the organizer-provided candidates file at ./data/ (gitignored), or pass --candidates.
# rank.py reads both candidates.jsonl and candidates.jsonl.gz.
python rank.py --candidates ./data/candidates.jsonl --out ./outputs/submission.csv

# Validate the output against the official spec before submitting:
python "/path/to/bundle/validate_submission.py" ./outputs/submission.csv

# Run local safety checks (id-membership, honeypot rate, format, score distribution):
python eval/self_audit.py --candidates ./data/candidates.jsonl --submission ./outputs/submission.csv
```

## Single reproduce command (Stage-3)

```
python rank.py --candidates ./data/candidates.jsonl --out ./outputs/<participant_id>.csv
```

Runs on CPU only, with no network access, within 5 minutes, on a clean checkout where the
candidates file has been placed at the given path. Pre-computation (`build_artifacts.py`,
Phase 2) runs offline and is not part of the timed step.

## Data

The 100k pool is **not** in this repo (it is gitignored: 465 MB raw / 52 MB gzipped, and
git-lfs is not used). The organizer supplies it at Stage 3; for local runs, place
`candidates.jsonl` or `candidates.jsonl.gz` under `./data/`.

## Layout

See the file map in `CONTEXT.md` Section 11.

## Status

v0 baseline: the rule spine produces a valid submission. Phase 2 (embedding recall layer)
and the offline NDCG proxy are the next steps. See `CONTEXT.md` Section 14.
