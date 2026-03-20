from __future__ import annotations

import random
import numpy as np

from vis_sattva.engine.config import FieldConfig
from vis_sattva.engine.field_core import FieldState
from vis_sattva.engine.graph_encoder import (
    random_ngon,
    encode_stick_graph_to_field_input,
)

FAMILIES = {
    "tri": 3,
    "quad": 4,
    "penta": 5,
}


def generate_dataset(samples_per_family, rng):
    data = []
    n_units = 256  # field dimensionality for this test

    for fam, n_sides in FAMILIES.items():
        for _ in range(samples_per_family):
            graph = random_ngon(n_sides=n_sides, rng=rng)
            inp = encode_stick_graph_to_field_input(
                graph=graph,
                n_units=n_units,
                n_angle_bins=16,
                n_radius_bins=16,
            )
            data.append((fam, inp))

    random.shuffle(data)
    return data


def run_field_on_inputs(data, n_units, n_steps, cfg, rng):
    field = FieldState.init_random(n_units=n_units, rng=rng)
    results = []

    for label, inp in data:
        field.a[:] = 0.0
        for _ in range(n_steps):
            field.step(input_vec=inp, cfg=cfg)
        results.append((label, field.a.copy()))

    return results


def nearest_neighbor_accuracy(reps):
    labels = [lab for lab, _ in reps]
    vecs = np.stack([v for _, v in reps], axis=0)
    n = len(labels)

    sims = vecs @ vecs.T
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    sims = sims / (norms @ norms.T + 1e-9)

    np.fill_diagonal(sims, -1.0)

    correct = {fam: 0 for fam in FAMILIES.keys()}
    total = {fam: 0 for fam in FAMILIES.keys()}

    for i in range(n):
        fam_i = labels[i]
        total[fam_i] += 1
        j = int(np.argmax(sims[i]))
        fam_j = labels[j]
        if fam_i == fam_j:
            correct[fam_i] += 1

    acc = {}
    for fam in FAMILIES.keys():
        acc[fam] = correct[fam] / max(1, total[fam])
    return acc


def main():
    rng = np.random.default_rng(1234)

    samples_per_family = 50
    n_units = 256
    n_steps = 5

    print("Building dataset...")
    data = generate_dataset(samples_per_family=samples_per_family, rng=rng)

    cfg = FieldConfig()
    print("Running field on inputs...")
    reps = run_field_on_inputs(
        data=data,
        n_units=n_units,
        n_steps=n_steps,
        cfg=cfg,
        rng=rng,
    )

    print("\n=== Nearest-neighbor family accuracy ===")
    acc = nearest_neighbor_accuracy(reps)
    for fam, a in acc.items():
        print(f"{fam:5s}: {a:.3f}")


if __name__ == "__main__":
    main()
