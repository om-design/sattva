"""
testing/sattva_emergent_diagnostics.py

Emergent diagnostics for Baby SATTVA:

1. Polygon family ReSi-style diagnostic
   - Families: triangle, square, rectangle, pentagon, hexagon,
               complex_polygon, dodeca_like
   - Mean Precision@k / Recall@k
   - Per-family Precision@k / Recall@k
   - Nearest-neighbor dumps for a few seeds per family

2. Tiny text-alignment probe (very exploratory)
   - Encodes a few WORD:: tags
   - Checks cosine similarity vs polygon encodings

Run from baby_sattva root (with venv active):

  python testing/sattva_emergent_diagnostics.py
"""

import os
import sys
import math
import random
from collections import defaultdict
from typing import List, Dict, Any, Tuple

import numpy as np

# Ensure baby_sattva root (where container.py lives) is on sys.path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # testing/
ROOT_DIR = os.path.dirname(ROOT_DIR)                   # baby_sattva/
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from container import SattvaContainer


# ============================================================
# LOAD TRAINED CONTAINER
# ============================================================

def load_container() -> SattvaContainer:
    snapshot_path = os.path.join("artifacts", "snapshots", "baby_sattva.pkl")
    if not os.path.exists(snapshot_path):
        raise FileNotFoundError(f"Snapshot not found at {snapshot_path}")

    import pickle
    with open(snapshot_path, "rb") as f:
        container: SattvaContainer = pickle.load(f)
    return container


# ============================================================
# POLYGON FAMILIES
# ============================================================

FAMILIES = [
    "triangle",
    "square",
    "rectangle",
    "pentagon",
    "hexagon",
    "complex_polygon",
    "dodeca_like",
]

FAMILY_TO_POLY2D = {
    "triangle": "POLY2D::triangle",
    "square": "POLY2D::square",
    "rectangle": "POLY2D::rectangle",
    "pentagon": "POLY2D::pentagon",
    "hexagon": "POLY2D::hexagon",
    "complex_polygon": "POLY2D::pentagon",   # reuse tag, differ by params
    "dodeca_like": "POLY2D::pentagon",
}


def sample_polygon_spec(family: str) -> Dict[str, Any]:
    if family == "triangle":
        num_sides = 3
        irregularity = random.uniform(0.0, 0.15)
    elif family == "square":
        num_sides = 4
        irregularity = random.uniform(0.0, 0.10)
    elif family == "rectangle":
        num_sides = 4
        irregularity = random.uniform(0.05, 0.25)
    elif family == "pentagon":
        num_sides = 5
        irregularity = random.uniform(0.0, 0.20)
    elif family == "hexagon":
        num_sides = 6
        irregularity = random.uniform(0.0, 0.25)
    elif family == "complex_polygon":
        num_sides = random.randint(7, 20)
        irregularity = random.uniform(0.3, 0.8)
    elif family == "dodeca_like":
        num_sides = random.randint(11, 13)
        irregularity = random.uniform(0.1, 0.4)
    else:
        raise ValueError(f"Unknown family: {family}")

    return {
        "family": family,
        "poly_tag": FAMILY_TO_POLY2D[family],
        "num_sides": num_sides,
        "radius": random.uniform(0.8, 1.2),
        "irregularity": irregularity,
        "jitter": random.uniform(0.0, 0.1),
    }


def build_dataset(
    samples_per_family: int = 40,
    seed: int = 42
) -> List[Dict[str, Any]]:
    random.seed(seed)
    all_samples = []
    for fam in FAMILIES:
        for _ in range(samples_per_family):
            all_samples.append(sample_polygon_spec(fam))
    return all_samples


# ============================================================
# ENCODING & RESONANCE
# ============================================================

def encode_shape(container: SattvaContainer, shape_spec: Dict[str, Any]) -> np.ndarray:
    space = container.embedding
    base_ids = ["OBJ2D::poly"]
    instr_ids = [shape_spec["poly_tag"]]
    v = space.encode_program(base_ids, instr_ids)
    n = np.linalg.norm(v)
    if n > 0:
        v = v / n
    return v


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def resonance_score(
    container: SattvaContainer,
    encoded_a: np.ndarray,
    encoded_b: np.ndarray
) -> float:
    # For now, just cosine; swap in GA-specific similarity later if desired.
    return cosine(encoded_a, encoded_b)


# ============================================================
# METRICS
# ============================================================

def precision_recall_at_k(
    ranked_labels: List[str],
    target_family: str,
    k: int
) -> Tuple[float, float]:
    top_k = ranked_labels[:k]
    num_relevant = sum(1 for lab in ranked_labels if lab == target_family)
    true_positives_k = sum(1 for lab in top_k if lab == target_family)
    precision_k = true_positives_k / max(k, 1)
    recall_k = true_positives_k / max(num_relevant, 1)
    return precision_k, recall_k


def mean_precision_recall_at_k(
    all_ranked_labels: Dict[str, List[str]],
    k: int
) -> Tuple[float, float]:
    precisions, recalls = [], []
    for qid, ranked_labels in all_ranked_labels.items():
        target_family = qid.split("::")[0]
        p, r = precision_recall_at_k(ranked_labels, target_family, k)
        precisions.append(p)
        recalls.append(r)
    return sum(precisions) / len(precisions), sum(recalls) / len(recalls)


def per_family_precision_recall(
    all_ranked_labels: Dict[str, List[str]],
    k: int
) -> Dict[str, Tuple[float, float]]:
    per_fam: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for qid, ranked_labels in all_ranked_labels.items():
        fam = qid.split("::")[0]
        p, r = precision_recall_at_k(ranked_labels, fam, k)
        per_fam[fam].append((p, r))
    out: Dict[str, Tuple[float, float]] = {}
    for fam, prs in per_fam.items():
        ps = [x[0] for x in prs]
        rs = [x[1] for x in prs]
        out[fam] = (sum(ps)/len(ps), sum(rs)/len(rs))
    return out


# ============================================================
# NEAREST-NEIGHBOR DUMPS
# ============================================================

def dump_neighbors(
    encoded_samples: List[Tuple[int, Dict[str, Any], np.ndarray]],
    container: SattvaContainer,
    seeds_per_family: int = 2,
    top_n: int = 8
) -> None:
    print("\n--- Nearest neighbors per family (cosine similarity) ---")
    indices_by_family = defaultdict(list)
    for idx, spec, _ in encoded_samples:
        indices_by_family[spec["family"]].append(idx)

    for fam in FAMILIES:
        candidates = indices_by_family[fam]
        if not candidates:
            continue
        chosen = random.sample(candidates, min(seeds_per_family, len(candidates)))
        for seed_idx in chosen:
            _, seed_spec, seed_vec = encoded_samples[seed_idx]
            scores = []
            for idx, spec, vec in encoded_samples:
                s = resonance_score(container, seed_vec, vec)
                scores.append((s, spec["family"], idx))
            scores.sort(key=lambda x: x[0], reverse=True)
            print(f"\nSeed idx={seed_idx}, family={seed_spec['family']}")
            for s, f, i in scores[:top_n]:
                print(f"  -> idx={i:3d}, fam={f:15s}, sim={s: .3f}")


# ============================================================
# TINY TEXT PROBE
# ============================================================

def text_probe(container: SattvaContainer,
               encoded_samples: List[Tuple[int, Dict[str, Any], np.ndarray]]
               ) -> None:
    """
    Very small exploratory probe:
    - Encodes a few WORD:: tags
    - Checks similarity to polygon encodings
    """
    space = container.embedding

    def encode_word_tag(tag: str) -> np.ndarray:
        base_ids = ["WORD::token"]
        instr_ids = [tag]
        v = space.encode_program(base_ids, instr_ids)
        n = np.linalg.norm(v)
        if n > 0:
            v = v / n
        return v

    word_tags = [
        "WORD::triangle",
        "WORD::square",
        "WORD::shape",
        "WORD::many_sided",
    ]

    word_vecs = {tag: encode_word_tag(tag) for tag in word_tags}

    # Compute average polygon encoding per family
    fam_vecs: Dict[str, List[np.ndarray]] = defaultdict(list)
    for _, spec, vec in encoded_samples:
        fam_vecs[spec["family"]].append(vec)
    fam_centroids: Dict[str, np.ndarray] = {}
    for fam, vecs in fam_vecs.items():
        fam_centroids[fam] = np.mean(np.stack(vecs, axis=0), axis=0)

    print("\n--- Tiny text probe (cosine between WORD:: tags and family centroids) ---")
    for tag, wv in word_vecs.items():
        print(f"\n{tag}:")
        for fam in FAMILIES:
            fv = fam_centroids[fam]
            sim = cosine(wv, fv)
            print(f"  vs {fam:15s}: {sim: .3f}")


# ============================================================
# MAIN
# ============================================================

def run_all(
    samples_per_family: int = 40,
    prototypes_per_family: int = 3,
    ks: List[int] = [5, 10, 20]
):
    print("Loading SATTVA container ...")
    container = load_container()

    print("Building synthetic polygon dataset...")
    dataset = build_dataset(samples_per_family=samples_per_family)
    print(f"Total samples: {len(dataset)}")

    print("Encoding all samples...")
    encoded_samples: List[Tuple[int, Dict[str, Any], np.ndarray]] = []
    for idx, spec in enumerate(dataset):
        v = encode_shape(container, spec)
        encoded_samples.append((idx, spec, v))

    # Build indices for queries
    indices_by_family = defaultdict(list)
    for idx, spec, _ in encoded_samples:
        indices_by_family[spec["family"]].append(idx)

    # Choose prototypes per family
    query_prototypes = []
    for fam in FAMILIES:
        candidates = indices_by_family[fam]
        if not candidates:
            continue
        chosen = random.sample(candidates, min(prototypes_per_family, len(candidates)))
        for i, idx in enumerate(chosen):
            query_id = f"{fam}::{i}"
            query_prototypes.append((query_id, idx))

    print(f"Total query prototypes: {len(query_prototypes)}")

    # Ranking per query
    all_ranked_labels: Dict[str, List[str]] = {}
    for query_id, q_idx in query_prototypes:
        _, _, q_vec = encoded_samples[q_idx]
        scores = []
        for _, spec, vec in encoded_samples:
            s = resonance_score(container, q_vec, vec)
            scores.append((s, spec["family"]))
        scores.sort(key=lambda x: x[0], reverse=True)
        ranked_labels = [fam for _, fam in scores]
        all_ranked_labels[query_id] = ranked_labels

    # Global metrics
    print("\n=== ReSi-style SATTVA Polygon Diagnostic ===")
    print(f"Families: {FAMILIES}")
    print(f"Samples per family: {samples_per_family}")
    print(f"Prototypes per family: {prototypes_per_family}\n")

    for k in ks:
        mp, mr = mean_precision_recall_at_k(all_ranked_labels, k)
        print(f"Top-{k}: mean_precision={mp:.3f}, mean_recall={mr:.3f}")

    # Per-family metrics at a chosen k (say k=10)
    k_pf = 10
    pf_stats = per_family_precision_recall(all_ranked_labels, k=k_pf)
    print(f"\n--- Per-family Precision/Recall at k={k_pf} ---")
    for fam in FAMILIES:
        if fam in pf_stats:
            p, r = pf_stats[fam]
            print(f"{fam:15s}: precision={p:.3f}, recall={r:.3f}")
        else:
            print(f"{fam:15s}: (no queries)")

    # Nearest neighbors
    dump_neighbors(encoded_samples, container, seeds_per_family=2, top_n=8)

    # Tiny text probe
    text_probe(container, encoded_samples)


if __name__ == "__main__":
    run_all()
