# vis_sattva/engine/graph_encoder.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np


@dataclass
class Node2D:
    """A node (junction / endpoint) in continuous 2D space."""
    x: float
    y: float


@dataclass
class Stick:
    """
    A stick (line segment) between two nodes.
    We store indices into a node list rather than coordinates directly.
    """
    i: int  # index of first node
    j: int  # index of second node


@dataclass
class StickGraph:
    """
    A simple stick-and-node graph in 2D.

    For vis_sattva v1 we only need:
    - nodes: list of Node2D
    - sticks: list of Stick
    """
    nodes: List[Node2D]
    sticks: List[Stick]


def regular_ngon(
    n_sides: int,
    radius: float = 1.0,
    center: Tuple[float, float] = (0.0, 0.0),
    phase: float = 0.0,
) -> StickGraph:
    """
    Generate a regular n-gon as a stick graph in continuous 2D.

    Nodes are placed on a circle; sticks connect consecutive nodes in a cycle.
    """
    cx, cy = center
    angles = np.linspace(0.0, 2.0 * np.pi, num=n_sides, endpoint=False) + phase
    nodes = [
        Node2D(
            x=float(cx + radius * np.cos(theta)),
            y=float(cy + radius * np.sin(theta)),
        )
        for theta in angles
    ]

    sticks: List[Stick] = []
    for i in range(n_sides):
        j = (i + 1) % n_sides
        sticks.append(Stick(i=i, j=j))

    return StickGraph(nodes=nodes, sticks=sticks)


def random_ngon(
    n_sides: int,
    rng: Optional[np.random.Generator] = None,
    radius_range: Tuple[float, float] = (0.8, 1.2),
    jitter: float = 0.05,
) -> StickGraph:
    """
    Generate a slightly perturbed n-gon (no tokens, pure geometry).

    - radius is sampled from radius_range
    - center is jittered slightly around (0,0)
    - overall phase (rotation) is random
    """
    if rng is None:
        rng = np.random.default_rng()

    radius = float(rng.uniform(radius_range[0], radius_range[1]))
    cx = float(rng.uniform(-jitter, jitter))
    cy = float(rng.uniform(-jitter, jitter))
    phase = float(rng.uniform(0.0, 2.0 * np.pi))

    return regular_ngon(
        n_sides=n_sides,
        radius=radius,
        center=(cx, cy),
        phase=phase,
    )


def encode_stick_graph_to_field_input(
    graph: StickGraph,
    n_units: int,
    n_angle_bins: int = 16,
    n_radius_bins: int = 8,
) -> np.ndarray:
    """
    Very simple first encoder:
    - We create orientation/radius "receptors" arranged in a fixed bank.
    - For each stick, we compute its angle and average distance from origin.
    - We increment the corresponding receptor bin.

    The result is a fixed-length vector of size n_units (padded or truncated),
    which can be fed as input_vec into FieldState.step().
    """
    # Compute features for each stick
    angles = []
    radii = []

    for stick in graph.sticks:
        n1 = graph.nodes[stick.i]
        n2 = graph.nodes[stick.j]
        dx = n2.x - n1.x
        dy = n2.y - n1.y
        angle = np.arctan2(dy, dx)  # [-pi, pi]
        angles.append(angle)

        # Approximate radius as mean distance of endpoints from origin
        r1 = np.hypot(n1.x, n1.y)
        r2 = np.hypot(n2.x, n2.y)
        radii.append(0.5 * (r1 + r2))

    if not angles:
        return np.zeros(n_units, dtype=np.float32)

    angles = np.array(angles, dtype=np.float32)
    radii = np.array(radii, dtype=np.float32)

    # Normalize to bins
    # Angle in [0, 2pi)
    angles = (angles + 2.0 * np.pi) % (2.0 * np.pi)
    angle_bins = np.floor(angles / (2.0 * np.pi) * n_angle_bins).astype(int)
    angle_bins = np.clip(angle_bins, 0, n_angle_bins - 1)

    # Radius bins based on simple heuristic range [0, 2]
    radii_clipped = np.clip(radii, 0.0, 2.0)
    radius_bins = np.floor(radii_clipped / 2.0 * n_radius_bins).astype(int)
    radius_bins = np.clip(radius_bins, 0, n_radius_bins - 1)

    # Receptor index = angle_bin * n_radius_bins + radius_bin
    n_receptors = n_angle_bins * n_radius_bins
    receptor_vec = np.zeros(n_receptors, dtype=np.float32)
    for a_bin, r_bin in zip(angle_bins, radius_bins):
        idx = a_bin * n_radius_bins + r_bin
        receptor_vec[idx] += 1.0

    # Normalize receptor activations
    if receptor_vec.sum() > 0.0:
        receptor_vec /= receptor_vec.sum()

    # Match requested n_units by padding or truncating
    if n_units <= n_receptors:
        return receptor_vec[:n_units]
    else:
        out = np.zeros(n_units, dtype=np.float32)
        out[:n_receptors] = receptor_vec
        return out
