from __future__ import annotations

import random
import numpy as np

from vis_sattva.engine.config import FieldConfig
from vis_sattva.engine.field_core import FieldState
from vis_sattva.engine.graph_encoder import (
    random_ngon,
    encode_stick_graph_to_field_input,
)

# Family label -> number of sides
FAMILIES = {
    "tri": 3,
    "quad": 4,
    "penta": 5,
}

FAMILY_LIST = list(FAMILIES.keys())
FAMILY_INDEX = {fam: i for i, fam in enumerate(FAMILY_LIST)}


def generate_dataset_with_input_noise(
    samples_per_family,
    rng,
    input_noise=0.0,
    n_units=256,
):
    """
    Generate (label, input_vec) pairs for each n-gon family.
    input_noise is Gaussian noise added directly in the encoded input vector.
    """
    data = []

    for fam, n_sides in FAMILIES.items():
        for _ in range(samples_per_family):
            graph = random_ngon(n_sides=n_sides, rng=rng)
            inp = encode_stick_graph_to_field_input(
                graph=graph,
                n_units=n_units,
                n_angle_bins=16,
                n_radius_bins=16,
            )
            if input_noise > 0.0:
                noise = rng.normal(scale=input_noise, size=inp.shape)
                inp = inp + noise
            data.append((fam, inp))

    random.shuffle(data)
    return data


def run_field_on_inputs(data, n_units, n_steps, cfg, rng):
    """
    Run the field for n_steps on each input vector, record final activations.
    """
    field = FieldState.init_random(n_units=n_units, rng=rng)
    results = []

    for label, inp in data:
        field.a[:] = 0.0
        for _ in range(n_steps):
            field.step(input_vec=inp, cfg=cfg)
        results.append((label, field.a.copy()))

    return results


def nearest_neighbor_stats(reps):
    """
    reps: list of (label, rep_vector)
    Returns per-family 1-NN accuracy using cosine similarity,
    plus a confusion matrix (true x predicted).
    """
    labels = [lab for lab, _ in reps]
    vecs = np.stack([v for _, v in reps], axis=0)
    n = len(labels)

    # cosine similarity
    sims = vecs @ vecs.T
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    sims = sims / (norms @ norms.T + 1e-9)

    # don't match a vector to itself
    np.fill_diagonal(sims, -1.0)

    correct = {fam: 0 for fam in FAMILIES.keys()}
    total = {fam: 0 for fam in FAMILIES.keys()}

    # confusion matrix: rows = true, cols = predicted
    C = np.zeros((len(FAMILY_LIST), len(FAMILY_LIST)), dtype=np.int64)

    for i in range(n):
        fam_i = labels[i]
        total[fam_i] += 1
        j = int(np.argmax(sims[i]))
        fam_j = labels[j]
        if fam_i == fam_j:
            correct[fam_i] += 1
        ti = FAMILY_INDEX[fam_i]
        pj = FAMILY_INDEX[fam_j]
        C[ti, pj] += 1

    acc = {}
    for fam in FAMILIES.keys():
        if total[fam] > 0:
            acc[fam] = correct[fam] / float(total[fam])
        else:
            acc[fam] = float("nan")

    return acc, C


def print_confusion_matrix(C):
    """
    Print confusion matrix with rows normalized per true family.
    """
    print("  Confusion (rows=true, cols=pred):")
    header = "       " + " ".join(f"{fam:>7s}" for fam in FAMILY_LIST)
    print(header)
    row_sums = C.sum(axis=1, keepdims=True).astype(float)

    for i, fam in enumerate(FAMILY_LIST):
        row = C[i].astype(float)
        if row_sums[i, 0] > 0:
            row = row / row_sums[i, 0]
        else:
            row[:] = 0.0
        vals = " ".join(f"{v:7.3f}" for v in row)
        print(f"  {fam:5s} {vals}")


def run_curriculum():
    rng = np.random.default_rng(1234)

    total_samples = 10000  # per difficulty
    n_units = 256
    n_steps = 5

    # difficulty levels: input_noise in encoded vector space
    difficulties = [0.0, 0.05, 0.10, 0.20, 0.30]

    samples_per_family = total_samples // len(FAMILIES)

    cfg = FieldConfig()

    print("=== Curriculum diagnostics over difficulty levels ===")
    print(f"Total samples per difficulty: {total_samples}")
    print(f"Samples per family: {samples_per_family}\n")

    for noise in difficulties:
        print(f"--- difficulty: input_noise={noise:.2f} ---")
        data = generate_dataset_with_input_noise(
            samples_per_family=samples_per_family,
            rng=rng,
            input_noise=noise,
            n_units=n_units,
        )
        reps = run_field_on_inputs(
            data=data,
            n_units=n_units,
            n_steps=n_steps,
            cfg=cfg,
            rng=rng,
        )
        acc, C = nearest_neighbor_stats(reps)
        for fam, a in acc.items():
            print(f"  {fam:5s}: {a:.3f}")
        print_confusion_matrix(C)
        print("")


def main():
    run_curriculum()


if __name__ == "__main__":
    main()
