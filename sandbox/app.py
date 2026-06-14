"""
Redrob Candidate Ranker - Sandbox (Streamlit).

A hosted demo that runs the EXACT src/scorer.py ranking module on a <=100-candidate sample,
so a reviewer can see, for every candidate, the final score, the signals that produced it, and
the generated reasoning. Same code path as rank.py: CPU only, no network, no LLM at rank time.

Run locally:   streamlit run sandbox/app.py
Deploy:        see sandbox/README.md (Streamlit Community Cloud or HuggingFace Spaces).
"""

import json
import os
import sys

import streamlit as st

# Import the real scorer module so the demo cannot drift from the submission.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from scorer import score_candidate, make_reasoning  # noqa: E402

MAX_CANDIDATES = 100
SAMPLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_candidates.json")

st.set_page_config(page_title="Redrob Candidate Ranker", layout="wide")
st.title("Redrob Candidate Ranker - Sandbox")
st.caption(
    "Ranks candidates for the Redrob Senior AI Engineer JD using the exact src/scorer.py module "
    "(the same code that produces the submission). CPU only, no network, no LLM at rank time."
)

with st.expander("What this ranks for (the job description, in brief)"):
    st.markdown(
        "- Production embeddings / retrieval, vector databases, ranking-evaluation frameworks\n"
        "- Shipped a ranking, search, or recommendation system at a PRODUCT company\n"
        "- Roughly 6 to 8 years, in or willing to relocate to Pune / Noida, active and reachable\n"
        "- Down-weights: keyword stuffers, computer-vision / forecasting look-alikes, inactive "
        "profiles, and internally-impossible 'honeypot' profiles."
    )


def load_candidates(raw_bytes):
    """Accept a JSON array, a single JSON object, or JSONL."""
    text = raw_bytes.decode("utf-8")
    try:
        data = json.loads(text.strip())
        if isinstance(data, dict):
            return [data]
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    out = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


source = st.radio("Candidate source", ["Bundled sample (50 candidates)", "Upload your own JSON (<=100)"])

candidates = None
if source.startswith("Bundled"):
    with open(SAMPLE_PATH, encoding="utf-8") as f:
        candidates = json.load(f)
else:
    upload = st.file_uploader("Upload a JSON array (or JSONL) of candidate records", type=["json", "jsonl", "txt"])
    if upload is not None:
        try:
            candidates = load_candidates(upload.read())
        except Exception as e:  # noqa: BLE001 - surface any parse error to the user
            st.error(f"Could not parse the file: {e}")

if candidates:
    if len(candidates) > MAX_CANDIDATES:
        st.warning(f"Got {len(candidates)} candidates; ranking the first {MAX_CANDIDATES} (sandbox cap).")
        candidates = candidates[:MAX_CANDIDATES]

    scored, skipped = [], 0
    for c in candidates:
        try:
            scored.append(score_candidate(c))
        except Exception:  # noqa: BLE001 - one malformed record should not break the demo
            skipped += 1
    # Same ordering as rank.py: best-first, ties broken by candidate_id ascending.
    scored.sort(key=lambda r: (-r["score"], r["candidate_id"] or ""))
    if skipped:
        st.warning(f"Skipped {skipped} malformed record(s).")

    rows = []
    for rank, r in enumerate(scored, start=1):
        ctx = r["rctx"]
        rows.append({
            "rank": rank,
            "candidate_id": r["candidate_id"],
            "title": ctx["title"],
            "yrs": round(ctx["yoe"], 1),
            "score": round(r["score"], 4),
            "honeypot": "yes" if ctx["honeypot"] else "",
            "reasoning": make_reasoning(ctx),
        })

    st.subheader(f"Ranked {len(rows)} candidates")
    st.dataframe(rows, use_container_width=True, hide_index=True)

    st.subheader("Why each candidate ranks where it does")
    for row, r in zip(rows, scored):
        ctx = r["rctx"]
        with st.expander(f"#{row['rank']}   {row['candidate_id']}   {row['title']} (score {row['score']})"):
            st.write(f"**Reasoning:** {row['reasoning']}")
            st.json({
                "years_experience": ctx["yoe"],
                "trusted_AI_IR_skills": ctx["top_skills"],
                "n_trusted_skills": ctx["n_trusted"],
                "product_company_background": ctx["product"],
                "current_industry": ctx["industry"],
                "recruiter_response_rate": ctx["rr"],
                "months_since_active": round(ctx["months_inactive"], 1),
                "in_india_or_tier1_city": ctx["is_india"],
                "pune_or_noida": ctx["is_pref"],
                "open_to_work": ctx["open_to_work"],
                "willing_to_relocate": ctx["willing"],
                "honeypot_flag": ctx["honeypot"],
            })
else:
    st.info("Pick the bundled sample or upload a JSON file to see the ranking.")
