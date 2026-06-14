# Sandbox: hosted demo of the Redrob ranker

A Streamlit app that runs the exact `src/scorer.py` ranking module on a small candidate sample
(the bundled 50, or your own JSON upload, capped at 100). For each candidate it shows the final
score, the signals that produced it, and the generated reasoning. CPU only, no network, same code
path as `rank.py`, so the demo can never drift from the submission.

This satisfies the challenge's Stage-1 sandbox requirement: a hosted environment where the ranker
runs end to end on a <=100-candidate sample.

## Run locally

```bash
pip install -r sandbox/requirements.txt
streamlit run sandbox/app.py
```

## Deploy (pick one)

### Streamlit Community Cloud (easiest, repo is already on GitHub)
1. Go to https://share.streamlit.io and sign in with GitHub.
2. New app, repository `Nandish3010/redrob-candidate-ranker`, branch `main`,
   main file path `sandbox/app.py`.
3. Deploy. Streamlit Cloud installs `streamlit` from the repo-root `requirements.txt`.
4. Put the resulting public URL into `submission_metadata.yaml` as `sandbox_link`.

### HuggingFace Spaces (Streamlit SDK)
1. Create a new Space, SDK = Streamlit.
2. Add `app.py`, `requirements.txt`, `sample_candidates.json` from this folder, and a copy of
   `../src/scorer.py` (or add this repo as a git submodule / clone step) so the import resolves.
3. The Space builds and serves automatically. Use its URL as `sandbox_link`.

## Notes
- `sample_candidates.json` is the organizer-provided 50-candidate sample, used as pre-loaded demo
  data so a reviewer can run the ranker without uploading anything.
- The app imports `score_candidate` and `make_reasoning` from `src/scorer.py` directly; there is no
  second copy of the scoring logic to keep in sync.
