#!/usr/bin/env python3
"""Populate the Redrob idea-submission template with the Shortlist deck content.

Reads the blank template from the parent folder, fills each slide's guidance box with the
real answer, draws a simple architecture diagram on slide 7, and writes deck/Shortlist_Redrob.pptx.
Local tooling only (not part of the ranking step). Run: python deck/build_deck.py
"""
import os
from pptx import Presentation
from pptx.util import Pt, Emu, Inches
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "..", "..", "Idea Submission Template Redrob.pptx")
OUT = os.path.join(HERE, "Shortlist_Redrob.pptx")

INK = RGBColor(0x22, 0x26, 0x2B)
ACCENT = RGBColor(0x1F, 0x6F, 0xEB)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
MUTE = RGBColor(0x55, 0x5B, 0x62)


def set_bullets(shape, lines, size=12, lead_size=None):
    """Replace a shape's text with bullet lines. lines: list of (text, level) or str."""
    tf = shape.text_frame
    tf.word_wrap = True
    tf.clear()
    for i, item in enumerate(lines):
        text, level = (item if isinstance(item, tuple) else (item, 0))
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.level = level
        p.space_after = Pt(4)
        run = p.add_run()
        run.text = text
        run.font.size = Pt(lead_size if (i == 0 and lead_size) else size)
        run.font.color.rgb = INK
        run.font.bold = (i == 0 and lead_size is not None)


def guidance_shape(slide):
    """The grey question box is the text shape with the longest text on the slide."""
    cand = [s for s in slide.shapes if s.has_text_frame and s.text_frame.text.strip()]
    return max(cand, key=lambda s: len(s.text_frame.text)) if cand else None


def add_box(slide, x, y, w, h, text, fill, line=ACCENT, fg=WHITE, size=11, bold=True):
    sp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sp.fill.solid(); sp.fill.fore_color.rgb = fill
    sp.line.color.rgb = line; sp.line.width = Pt(1)
    tf = sp.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_top = Pt(2); tf.margin_bottom = Pt(2)
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = text; r.font.size = Pt(size); r.font.bold = bold; r.font.color.rgb = fg
    return sp


def add_arrow(slide, x, y, w, h):
    sp = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x), Inches(y), Inches(w), Inches(h))
    sp.fill.solid(); sp.fill.fore_color.rgb = MUTE; sp.line.fill.background()
    return sp


prs = Presentation(TEMPLATE)
S = prs.slides

# --- Slide 1: title ---
for sh in S[0].shapes:
    if not sh.has_text_frame:
        continue
    t = sh.text_frame.text.strip()
    if t.startswith("Team Name"):
        sh.text_frame.paragraphs[0].runs and None
        sh.text_frame.text = "Team Name :  Shortlist"
    elif t.startswith("Problem Statement"):
        sh.text_frame.text = ("Problem Statement :  Intelligent Candidate Discovery & Ranking. "
                              "From 100,000 candidates, surface the top 100 best-fit profiles for one "
                              "Senior AI Engineer role, ranked by genuine fit rather than keyword overlap.")
    elif t.startswith("Team Leader"):
        sh.text_frame.text = "Team Leader Name :  Nandish S  (solo)"

# --- Slide 2: Solution Overview & differentiation ---
set_bullets(guidance_shape(S[1]), [
    ("A transparent rule-engine ranker with a capped semantic-similarity layer.", 0),
    "final_score = relevance_core x behavioral_availability x honeypot_factor. Every rank decomposes into named, auditable numbers.",
    "What differentiates it from traditional keyword matching:",
    ("Keyword-proof: AI skills earn credit only via a trust gate (endorsements + duration + Redrob assessment) AND corroborating career evidence, so stuffers never rank.", 1),
    ("Beyond text matching: a capped TF-IDF JD-similarity term adds semantic recall, weighted so it can never override hard evidence.", 1),
    ("Trap-aware: detects keyword stuffers, behavioral twins, and the planted impossible 'honeypot' profiles.", 1),
    ("Fully explainable: reasoning is generated deterministically from the same features that drive the score. No LLM at rank time, no hallucination.", 1),
    ("Constraint-safe: CPU only, no network, ~55s on 100k candidates.", 1),
], size=11.5, lead_size=14)

# --- Slide 3: JD Understanding & Candidate Evaluation ---
set_bullets(guidance_shape(S[2]), [
    ("Key requirements extracted from the JD:", 0),
    "5 to 9 yrs experience (ideal 6 to 8); production ranking / search / recommendation / retrieval work; Python + ML (PyTorch, embeddings, RAG, vector search); offline evaluation (NDCG, A/B testing); product-company (not pure services) background; India-based or willing to relocate.",
    ("Signals that most determine relevance:", 0),
    "Actual IR/ranking WORK evidence (role titles + descriptions), trust-gated skills, experience in band, product-vs-services company, and availability/engagement signals.",
    ("How we evaluate beyond keyword matching:", 0),
    "Skills count only when career history corroborates them; semantic similarity surfaces strong plain-language profiles that never write 'RAG'; suspicious profiles are caught by internal contradictions, not by what they claim.",
], size=11.5, lead_size=13)

# --- Slide 4: Ranking Methodology ---
set_bullets(guidance_shape(S[3]), [
    ("Retrieve:", 0),
    "Stream all 100,000 candidates with no lossy pre-filter, so recall stays 100% (the pool is small enough to score in full within budget).",
    ("Score (relevance_core, weights sum to 1.0):", 0),
    "career_evidence 0.30, jd_similarity 0.12 (capped TF-IDF cosine to four JD-theme vectors), skill_trust 0.18, experience_band 0.12, python_eval 0.12, location 0.12, education 0.04.",
    ("Adjust:", 0),
    "behavioral_availability multiplier (0.55 to 1.05) from recency, recruiter response, notice period, open-to-work; honeypot_factor down-weights internally impossible profiles.",
    ("Rank:", 0),
    "Sort by score, deterministic tie-break by candidate_id, write the top 100. Heuristics are combined arithmetically so every decision is traceable.",
], size=11.5, lead_size=13)

# --- Slide 5: Explainability & Data Validation ---
set_bullets(guidance_shape(S[4]), [
    ("Explaining ranking decisions:", 0),
    "Each candidate gets a 1 to 2 sentence reasoning string built deterministically from the exact feature values that produced its rank, so the explanation can never disagree with the score.",
    ("Preventing hallucination / unsupported claims:", 0),
    "Every clause cites a real field (title, years, trusted skills, product industry, notice period). No LLM is called at rank time (network is off), so nothing can be invented. Honest concerns are surfaced too (low response rate, inactivity, onsite-only abroad, long notice).",
    ("Handling incomplete or suspicious profiles:", 0),
    "Sentinels treated as missing not zero (-1 GitHub/offer history); schema-tolerant parsing; a conjunctive honeypot gate (>= 2 independent impossibility signals) keeps planted impossible profiles out of the top 100; format, ID-membership and honeypot self-audits run before every submission.",
], size=11, lead_size=13)

# --- Slide 6: End-to-End Workflow ---
set_bullets(guidance_shape(S[5]), [
    ("Complete flow from JD to ranked output:", 0),
    "1. Offline (untimed) build_artifacts.py: stream the pool, build a TF-IDF IDF table + four JD seed vectors + a data-derived calibration anchor, save sim_model.json (152 KB, committed).",
    "2. Timed (< 5 min) rank.py: stream candidates, score each (relevance_core + behavioral + honeypot), sort with the candidate_id tie-break, write the top-100 CSV with reasoning.",
    "3. Validate: official validate_submission.py plus a self-audit (honeypot rate, ID membership, format, score distribution).",
    "4. The hosted sandbox imports the SAME scorer module, so the live demo can never drift from what the grader reproduces.",
], size=12, lead_size=13)

# --- Slide 7: System Architecture (diagram) ---
y = 1.7
add_box(S[6], 0.5, y, 1.7, 0.8, "candidates.jsonl(.gz)\n100k profiles", RGBColor(0xEC,0xF1,0xF8), fg=INK, bold=True, size=10)
add_arrow(S[6], 2.25, y+0.28, 0.35, 0.24)
add_box(S[6], 2.65, y, 1.7, 0.8, "Streaming reader\n+ feature extraction", RGBColor(0xEC,0xF1,0xF8), fg=INK, size=10)
add_arrow(S[6], 4.4, y+0.28, 0.35, 0.24)
add_box(S[6], 4.8, y, 2.2, 0.8, "Scorer\nrelevance_core (+ capped\njd_similarity)", ACCENT, size=10)
add_arrow(S[6], 7.05, y+0.28, 0.35, 0.24)
add_box(S[6], 7.45, y, 2.05, 0.8, "x behavioral\nx honeypot factor", ACCENT, size=10)
# second row
y2 = 3.0
add_box(S[6], 4.8, y2, 2.2, 0.75, "Sort + candidate_id\ntie-break -> top 100", RGBColor(0xEC,0xF1,0xF8), fg=INK, size=10)
add_arrow(S[6], 7.05, y2+0.26, 0.35, 0.24)
add_box(S[6], 7.45, y2, 2.05, 0.75, "submission.csv\n+ reasoning", RGBColor(0xDF,0xF0,0xE2), fg=INK, size=10)
# feeders
add_box(S[6], 0.5, y2, 2.2, 0.75, "build_artifacts.py\n-> sim_model.json (offline)", RGBColor(0xFB,0xF1,0xDC), fg=INK, size=10)
add_box(S[6], 0.5, 4.0, 2.2, 0.7, "Sandbox (Streamlit)\nimports same scorer", RGBColor(0xF3,0xE9,0xF7), fg=INK, size=10)
note = S[6].shapes.add_textbox(Inches(3.0), Inches(4.05), Inches(6.4), Inches(0.7))
np = note.text_frame.paragraphs[0]; r = np.add_run()
r.text = "Stdlib-only ranking step. No GPU, no network, no model download. The committed 152 KB model is the only precomputed artifact."
r.font.size = Pt(10); r.font.italic = True; r.font.color.rgb = MUTE

# --- Slide 8: Results & Performance ---
set_bullets(guidance_shape(S[7]), [
    ("Ranking-quality evidence:", 0),
    "Top 10 are all in-band IR/ML engineers at product companies (Search, Recommendation, NLP, ML, AI Research; 6 to 9 yrs). Strong plain-language profiles surface; keyword stuffers sink (best stuffer reaches only rank 628).",
    "Keyword stuffers: 5,588 detected in the pool, 0 in the top 100.",
    "Honeypots: 0% in the top 100 (the disqualification line is 10%).",
    "Recall: 100% on hand-checked golden archetypes.",
    ("Runtime & compute constraints (all comfortably met):", 0),
    "~55s on 100k (limit 5 min); well under 16 GB RAM; CPU only; no network; ~150 KB precomputed model (limit 5 GB disk). Reproduces from a single command on a clean checkout.",
], size=11.5, lead_size=13)

# --- Slide 9: Technologies Used ---
set_bullets(guidance_shape(S[8]), [
    ("Python 3 standard library only for the ranking step (json, gzip, csv, re, math).", 0),
    "Zero ML dependencies, so the timed step is trivially reproducible in the grader's Docker, fast, and free of version/download risk.",
    ("Custom stdlib TF-IDF for the semantic layer (no scikit-learn, no model download).", 0),
    "Chosen because the dataset's role descriptions are templated, so heavyweight embeddings add cost and reduce explainability without adding signal. A judge-panel comparison scored the transparent rule engine above an embeddings hybrid.",
    ("Streamlit", 0),
    "for the hosted sandbox demo (not used by the ranking step).",
    ("Git", 0),
    "for authentic, incremental version history.",
], size=11.5, lead_size=12.5)

# --- Slide 10: Submission Assets (guidance box is short, target it explicitly) ---
asset_box = next(s for s in S[9].shapes
                 if s.has_text_frame and "github video" in s.text_frame.text.lower())
set_bullets(asset_box, [
    ("GitHub repository:", 0),
    "https://github.com/Nandish3010/redrob-candidate-ranker",
    ("Hosted sandbox:", 0),
    "https://redrob-candidate-ranker-ns.streamlit.app/",
    ("Ranked output:", 0),
    "outputs/submission.csv (top 100, candidate_id / rank / score / reasoning).",
    ("Single reproduce command:", 0),
    "python rank.py --candidates ./data/candidates.jsonl --out ./outputs/submission.csv",
], size=12.5, lead_size=13)

# --- Slide 11: Closing ---
tb = S[10].shapes.add_textbox(Inches(1.0), Inches(2.0), Inches(8.0), Inches(1.8))
tf = tb.text_frame; tf.word_wrap = True
p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
r = p.add_run(); r.text = "Shortlist"; r.font.size = Pt(40); r.font.bold = True; r.font.color.rgb = ACCENT
p2 = tf.add_paragraph(); p2.alignment = PP_ALIGN.CENTER
r2 = p2.add_run()
r2.text = "Rank like a great recruiter: transparent, trap-aware, reproducible."
r2.font.size = Pt(16); r2.font.color.rgb = INK
p3 = tf.add_paragraph(); p3.alignment = PP_ALIGN.CENTER
r3 = p3.add_run(); r3.text = "Thank you."; r3.font.size = Pt(14); r3.font.color.rgb = MUTE

prs.save(OUT)
print("Wrote", OUT)
