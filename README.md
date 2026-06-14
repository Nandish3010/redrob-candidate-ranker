# Redrob Candidate Ranker (Team Shortlist)

Ranks the top 100 best-fit candidates for the Redrob "Senior AI Engineer" job description out of
a 100,000-candidate pool, for the H2S "India runs on AI" Data & AI Challenge.

The system is a **transparent rule engine** with a **capped semantic-similarity layer**. Every
rank decomposes into named numbers, so each decision is fully explainable and defendable. The
ranking step uses **only the Python standard library** (no GPU, no network, no model download),
runs in about a minute on CPU, and keeps planted "honeypot" profiles out of the top 100.

## How it works

For each candidate the final score is:

```
final_score = relevance_core x behavioral_availability x honeypot_factor
```

- **relevance_core** is a weighted sum of named signals (weights sum to 1.0):
  `career_evidence 0.30`, `jd_similarity 0.12`, `skill_trust 0.18`, `experience_band 0.12`,
  `python_eval 0.12`, `location 0.12`, `education 0.04`.
  - `career_evidence` rewards actual ranking/search/recommendation work (role titles plus IR terms
    in role descriptions) at product (not pure-services) companies.
  - `jd_similarity` is a capped TF-IDF cosine between the candidate's text (title, headline,
    summary, role titles and descriptions, but **never the raw skills array**) and four JD theme
    vectors. Weighted low so it cannot override hard evidence or resurrect a keyword stuffer.
  - `skill_trust` credits JD-relevant skills only when endorsed, used over time, or assessment
    verified, and is coupled to career evidence so buzzwords on a non-IR career collapse.
- **behavioral_availability** (clipped to `[0.85, 1.05]`) is a secondary nudge from recency,
  recruiter response, notice period and open-to-work status. It separates otherwise-equal
  candidates without overriding a genuine relevance gap.
- **honeypot_factor** down-weights internally impossible profiles (e.g. many skills whose duration
  exceeds the whole career), keeping them out of the top 100.

**Trap handling.** Keyword stuffers earn nothing without corroborating career evidence; strong
plain-language profiles still surface via descriptions and semantic similarity; "behavioral twins"
are separated by availability and a deterministic tie-break; impossible profiles are gated.

**Explainability.** Each candidate's `reasoning` string is generated deterministically from the
same feature values that produced its rank (see `make_reasoning` in `src/scorer.py`). No LLM runs
at rank time, so nothing can be hallucinated, and honest concerns are surfaced alongside strengths.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # the ranking step itself needs no third-party packages

# Place the organizer-provided candidates file at ./data/ (gitignored), or pass --candidates.
# rank.py reads both candidates.jsonl and candidates.jsonl.gz.
python rank.py --candidates ./data/candidates.jsonl --out ./outputs/Shortlist.csv

# Validate against the official spec, and run local safety checks:
python /path/to/bundle/validate_submission.py ./outputs/Shortlist.csv
python eval/self_audit.py --candidates ./data/candidates.jsonl --submission ./outputs/Shortlist.csv
```

## Single reproduce command (Stage-3)

```
python rank.py --candidates ./data/candidates.jsonl --out ./outputs/Shortlist.csv
```

CPU only, no network, well under 5 minutes (about 55s on 100k), verified on a clean checkout to
produce a byte-identical CSV. The semantic model (`artifacts/sim_model.json`, 152 KB) is committed,
so no pre-computation is required to reproduce; rebuild it if desired with
`python build_artifacts.py --candidates ./data/candidates.jsonl`.

## Repository layout

```
rank.py                 Timed ranking step: candidates -> top-100 CSV (stdlib only)
build_artifacts.py      Offline precompute: builds artifacts/sim_model.json (TF-IDF model)
src/scorer.py           The scorer + deterministic reasoning generator
artifacts/sim_model.json  Committed semantic model (IDF + JD seed vectors; aggregate stats, no PII)
eval/
  self_audit.py         Honeypot rate, id-membership, format, score-distribution checks
  build_gold_set.py     Builds a deterministic gold set of clear archetypes
  gold_eval.py          NDCG@10/@50, MAP, P@10 of the scorer over the gold set (offline proxy)
sandbox/                Streamlit demo that imports the same scorer (cannot drift from rank.py)
deck/                   Pitch deck (.pptx source, build script, exported PDF)
submission_metadata.yaml  Portal metadata mirror
```

## Data

The 100k pool is **not** in this repo (gitignored: 465 MB raw / 52 MB gzipped). The organizer
supplies it at Stage 3; for local runs, place `candidates.jsonl` or `candidates.jsonl.gz` under
`./data/`.

## Offline evaluation

The ground truth is hidden and there is no leaderboard, so quality is checked two ways:
`eval/self_audit.py` (hard gates: honeypot rate, format, id-membership) and `eval/gold_eval.py`
(a deterministic gold set of hand-defined archetypes scored for NDCG@10/@50, MAP and P@10 as a
stable regression yardstick). The gold set is used to catch regressions, not to tune weights.
