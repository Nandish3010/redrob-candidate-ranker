#!/usr/bin/env python3
"""
build_artifacts.py: UNTIMED offline precompute for the Phase-2 embedding/similarity layer.

STATUS: stub. Not yet implemented. See CONTEXT.md Section 5 (pipeline) and Section 14 (next steps).

Planned output (written to ./artifacts/, which is gitignored):
  - doc_vectors.npy : TF-IDF + TruncatedSVD (or bge-small/e5-small CPU embeddings) over
                      title + headline + summary + trusted-skill text ONLY (never the raw
                      skills array, so keyword stuffers cannot buy a dense/lexical win).
  - bm25_index      : BM25 over candidate titles + headlines for lexical recall.
  - jd_seeds.npy    : 4 JD sub-query seed vectors (retrieval / ranking / recommendation /
                      production-ML-at-product-company).
  - meta.json       : model id, seeds, artifact hashes for deterministic Stage-3 reproduction.

This step MAY exceed 5 minutes (it runs offline). rank.py then loads these artifacts and the
ranking step itself stays inside the 5-minute budget. Until this lands, rank.py runs the v0
rule spine and the jd_similarity weight is folded into career_evidence (see src/scorer.py).
"""

import sys


def main():
    sys.exit("build_artifacts.py is a Phase-2 stub. See CONTEXT.md Section 14 before implementing.")


if __name__ == "__main__":
    main()
