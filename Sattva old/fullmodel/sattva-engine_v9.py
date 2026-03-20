"""
SATTVA Engine v9 — GA-inspired vector implementation with usage-based stability

Semantic Attractor Training of Transforming Vector Associations

This version adds:
- Per-primitive usage tracking (access_count, usage_ema).
- A sigmoid-based stability_factor(p) in [0, 1] from usage_ema.
- Plasticity scaling by (1 - stability_factor) for bandwidth updates,
  consolidation drift, and decomposition/pruning decisions.
"""

from __future__ import annotations

import math
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional

import numpy as np


# ─────────────────────────────────────────────────────────────
# Basic geometry utilities
# ─────────────────────────────────────────────────────────────

def l2_distance_sq(a: np.ndarray, b: np.ndarray) -> float:
    """Squared Euclidean distance between two embeddings."""
    return float(np.sum((a - b) ** 2))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in embedding space."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ─────────────────────────────────────────────────────────────
# GA-inspired program embedding (no external GA library)
# ─────────────────────────────────────────────────────────────

@dataclass
class ProgramEmbedding:
    """
    Lightweight GA-inspired encoder.

    We conceptually view a complex concept as:

        PROGRAM = BASE ⊗ INSTRUCTION

    but we implement it in R^d by reserving subspaces:

        embedding = [ base_part || instr_part ]

    where:
      - base_part encodes the reusable "shape"/meaning-loop.
      - instr_part encodes a transform or role (e.g. rotation,
        context, structural role).

    This keeps the SATTVA engine operating purely on vectors
    while allowing you to reason in terms of "primitive +
    instruction" when designing or interpreting experiments.
    """

    dim_base: int
    dim_instr: int
    rng: np.random.Generator

    def __post_init__(self) -> None:
        self.dim = self.dim_base + self.dim_instr
        # We use deterministic hashes to map symbolic ids
        # to stable random vectors within each subspace.
        self._base_cache: Dict[str, np.ndarray] = {}
        self._instr_cache: Dict[str, np.ndarray] = {}

    # --- internal helpers -------------------------------------------------

    def _hash_to_rng(self, key: str) -> np.random.Generator:
        # Simple deterministic seeding from string name
        seed = abs(hash(key)) % (2**32)
        return np.random.default_rng(seed)

    def _unit_vector(self, dim: int, key: str) -> np.ndarray:
        rng = self._hash_to_rng(key)
        v = rng.standard_normal(dim)
        n = np.linalg.norm(v)
        if n == 0:
            return v
        return v / n

    # --- base / instruction encoding -------------------------------------

    def base_vector(self, base_id: str) -> np.ndarray:
        """
        Get or create a stable unit vector for a base primitive.
        """
        if base_id not in self._base_cache:
            self._base_cache[base_id] = self._unit_vector(self.dim_base, f"BASE::{base_id}")
        return self._base_cache[base_id]

    def instr_vector(self, instr_id: str) -> np.ndarray:
        """
        Get or create a stable unit vector for an instruction/role.
        """
        if instr_id not in self._instr_cache:
            self._instr_cache[instr_id] = self._unit_vector(self.dim_instr, f"INSTR::{instr_id}")
        return self._instr_cache[instr_id]

    def encode_program(
        self,
        base_ids: Iterable[str],
        instr_ids: Iterable[str],
        base_weights: Optional[Iterable[float]] = None,
        instr_weights: Optional[Iterable[float]] = None,
    ) -> np.ndarray:
        """
        Encode a "program" composed of one or more bases and one
        or more instructions:

            embedding = concat(weighted_sum(bases), weighted_sum(instrs))

        This is deliberately simple and linear; SATTVA's dynamics
        (crystallisation/decomposition) operate on top of this,
        discovering when a composite is better represented via
        shared subparts.
        """
        base_ids = list(base_ids)
        instr_ids = list(instr_ids)

        if not base_ids:
            raise ValueError("encode_program requires at least one base_id")
        if base_weights is None:
            base_weights = [1.0] * len(base_ids)
        if instr_weights is None:
            instr_weights = [1.0] * len(instr_ids) if instr_ids else []

        base_weights = list(base_weights)
        instr_weights = list(instr_weights)

        # Weighted sum in each subspace
        b_vecs = [w * self.base_vector(bid) for bid, w in zip(base_ids, base_weights)]
        base_part = np.sum(b_vecs, axis=0)
        # Normalize to unit length if nonzero
        nb = np.linalg.norm(base_part)
        if nb > 0:
            base_part = base_part / nb

        if instr_ids:
            i_vecs = [w * self.instr_vector(iid) for iid, w in zip(instr_ids, instr_weights)]
            instr_part = np.sum(i_vecs, axis=0)
            ni = np.linalg.norm(instr_part)
            if ni > 0:
                instr_part = instr_part / ni
        else:
            instr_part = np.zeros(self.dim_instr, dtype=float)

        return np.concatenate([base_part, instr_part])

    def decompose_embedding(
        self, embedding: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split an engine embedding back into (base_part, instr_part).
        Purely structural; does not attempt symbolic identification.
        """
        if embedding.shape[0] != self.dim:
            raise ValueError("Embedding dimension mismatch in decompose_embedding")
        base_part = embedding[: self.dim_base]
        instr_part = embedding[self.dim_base :]
        return base_part, instr_part


# ─────────────────────────────────────────────────────────────
# SATTVA primitives and engine
# ─────────────────────────────────────────────────────────────

class Primitive:
    """
    A primitive (or composite) semantic well in geometric space.

    Attributes
    ----------
    id : str
        Unique identifier.
    embedding : np.ndarray
        Position/orientation in semantic geometry.
    energy : float
        Current activation energy (decays over time).
    bandwidth : float
        Effective myelination; higher bandwidth lowers activation
        threshold and increases gain.
    components : list[str]
        Child primitive ids if this is a composite.
    parents : list[str]
        Parent composite ids that use this as a component.
    predicted_strength : float
        How strongly this primitive was predicted (top–down)
        on the current step. Reset each step.
    last_novelty : float
        Novelty (burst) from last activation in [0, 1].
    access_count : int
        How many times this primitive has been meaningfully accessed.
    usage_ema : float
        Exponential moving average of recent access (0..1).
    """

    def __init__(self, embedding: np.ndarray, complexity: float = 1.0) -> None:
        self.id = str(uuid.uuid4())
        self.embedding = embedding.astype(float)

        # Energy dynamics
        self.energy = 0.0
        self.decay_rate = 0.001 / complexity

        # Bandwidth (myelination)
        self.bandwidth = 1.0
        self.bandwidth_cap = 10.0
        self.bandwidth_decay = 0.0005

        # Structural relations
        self.components: List[str] = []  # child primitive ids (if composite)
        self.parents: List[str] = []     # parent composite ids

        # Predictive state
        self.predicted_strength = 0.0

        # Novelty signal from last activation (formerly 'surprise')
        self.last_novelty = 0.0

        # Usage / stability tracking
        self.access_count: int = 0
        self.usage_ema: float = 0.0

    def inject_energy(self, amount: float) -> None:
        self.energy += amount

    def decay(self) -> None:
        """Decay energy and bandwidth; reset predictive state."""
        self.energy -= self.decay_rate * self.energy
        self.bandwidth -= self.bandwidth_decay * self.bandwidth
        if self.bandwidth < 1.0:
            self.bandwidth = 1.0
        self.predicted_strength = 0.0

    def reinforce_bandwidth(self, amount: float) -> None:
        """Strengthen myelination up to a cap."""
        self.bandwidth = min(self.bandwidth + amount, self.bandwidth_cap)


class Engine:
    """
    Geometric semantic engine with self-organizing wells.

    Parameters
    ----------
    dim : int
        Embedding dimensionality.
    base_activation_threshold : float
        Minimum cosine similarity required for a primitive with
        bandwidth=1 to activate.
    prediction_suppression : float
        How strongly predicted components are suppressed.
    composite_fire_threshold : float
        Min cosine similarity for a composite to predict components.
    novelty_baseline : float
        Baseline novelty for lateral pressure.
    tension_window : int
        History length for tension stats.
    epiphany_k_sigma : float
        Outlier threshold (in std devs) for high-tension gating.
    tension_min_history : int
        Min steps before trusting tension stats.
    usage_beta : float
        EMA update rate for usage_ema.
    stability_center : float
        Center of sigmoid for stability_factor (in usage_ema space).
    stability_sharpness : float
        Steepness of sigmoid for stability_factor.
    """

    def __init__(
        self,
        dim: int = 8,
        base_activation_threshold: float = 0.1,
        prediction_suppression: float = 0.7,
        composite_fire_threshold: float = 0.5,
        novelty_baseline: float = 0.1,
        tension_window: int = 100,
        epiphany_k_sigma: float = 2.0,
        tension_min_history: int = 30,
        usage_beta: float = 0.01,
        stability_center: float = 0.5,
        stability_sharpness: float = 6.0,
    ) -> None:
        self.dim = dim
        self.primitives: Dict[str, Primitive] = {}

        # Activation / prediction
        self.base_activation_threshold = base_activation_threshold
        self.prediction_suppression = prediction_suppression
        self.composite_fire_threshold = composite_fire_threshold

        # Coherence / crystallisation
        self.coherence_threshold = 0.8
        self.crystallisation_margin = 0.05
        self.temperature = 1.0

        # Consolidation
        self.consolidation_interval = 50
        self.step_count = 0

        # Novelty state
        self._last_novelty: Dict[str, float] = {}

        # Tension / epiphany dynamics
        self.novelty_baseline = novelty_baseline
        self.tension_window = tension_window
        self.epiphany_k_sigma = epiphany_k_sigma
        self.tension_min_history = tension_min_history

        self._tension_history: deque[float] = deque(maxlen=tension_window)
        self._tension_count: int = 0
        self._tension_mean: float = 0.0
        self._tension_m2: float = 0.0  # for online variance
        self._last_tension: float = 0.0

        # Usage / stability parameters
        self.usage_beta = usage_beta
        self.stability_center = stability_center
        self.stability_sharpness = stability_sharpness

    # ── Creation ─────────────────────────────────────────────

    def create_primitive(self, embedding: np.ndarray, complexity: float = 1.0) -> str:
        if embedding.shape[0] != self.dim:
            raise ValueError(f"Embedding dimension {embedding.shape[0]} != engine dim {self.dim}")
        p = Primitive(embedding, complexity)
        self.primitives[p.id] = p
        return p.id

    # ── Stability factor ────────────────────────────────────

    def stability_factor(self, p: Primitive) -> float:
        """
        Sigmoid-based stability in [0, 1) from usage_ema.

        Low usage_ema  -> stability ~ 0 (high plasticity).
        High usage_ema -> stability ~ 1 (low plasticity).
        """
        x = float(p.usage_ema)  # 0..1
        k = self.stability_sharpness
        c = self.stability_center
        return 1.0 / (1.0 + math.exp(-k * (x - c)))

    # ── Activation (two-pass, bandwidth-gated) ───────────────

    def activate_input(self, input_vector: np.ndarray, magnitude: float = 1.0) -> Dict[str, float]:
        """
        Two-pass activation with prediction and novelty.

        Returns
        -------
        dict {primitive_id: novelty_score}
        novelty_score in [0, 1]. 0 = fully predicted. 1 = full burst.
        """
        if input_vector.shape[0] != self.dim:
            raise ValueError(f"Input dimension {input_vector.shape[0]} != engine dim {self.dim}")

        # Pass 1: raw similarity and top–down prediction
        raw_sim: Dict[str, float] = {}
        for pid, p in self.primitives.items():
            raw_sim[pid] = cosine_similarity(p.embedding, input_vector)

        for pid, p in self.primitives.items():
            if not p.components:
                continue
            if raw_sim[pid] < self.composite_fire_threshold:
                continue
            pred_strength = (raw_sim[pid] - self.composite_fire_threshold) / (
                1.0 - self.composite_fire_threshold + 1e-9
            )
            pred_strength = min(max(pred_strength, 0.0), 1.0)
            for cid in p.components:
                if cid in self.primitives:
                    child = self.primitives[cid]
                    child.predicted_strength = min(
                        child.predicted_strength + pred_strength, 1.0
                    )

        # Pass 2: thresholded activation, novelty, energy, usage
        novelty_scores: Dict[str, float] = {}
        beta = self.usage_beta

        for pid, p in self.primitives.items():
            sim = raw_sim[pid]
            eff_threshold = self.base_activation_threshold / (p.bandwidth ** 0.5)
            if sim <= eff_threshold:
                novelty_scores[pid] = 0.0
                p.last_novelty = 0.0
                # Decay usage_ema slightly when not accessed
                p.usage_ema = (1.0 - beta) * p.usage_ema
                continue

            bw_gain = p.bandwidth / p.bandwidth_cap  # in (0, 1]
            suppression = self.prediction_suppression * p.predicted_strength
            energy_factor = max(0.0, 1.0 - suppression)
            novelty = sim * (1.0 - p.predicted_strength)

            # Usage update: this primitive was meaningfully accessed
            p.access_count += 1
            p.usage_ema = (1.0 - beta) * p.usage_ema + beta * 1.0

            # Stability and plasticity scale
            stab = self.stability_factor(p)
            plastic = 1.0 - stab

            # Energy is stateful; we do not scale it by plasticity
            p.inject_energy(magnitude * sim * bw_gain * energy_factor)

            # Bandwidth reinforcement scaled by plasticity:
            # high-usage wells become hard to further myelinate/reshape
            p.reinforce_bandwidth(0.1 * sim * energy_factor * plastic)

            novelty_scores[pid] = novelty
            p.last_novelty = novelty

        self._last_novelty = novelty_scores

        # Update tension statistics for epiphany gating
        mean_n = self.mean_novelty()
        lateral = self._lateral_pressure()
        tension = mean_n * lateral
        self._last_tension = tension
        self._update_tension_stats(tension)

        return novelty_scores

    # ── Novelty / triage ─────────────────────────────────────

    def mean_novelty(self) -> float:
        active = [n for n in self._last_novelty.values() if n > 0.0]
        if not active:
            return 0.0
        return float(np.mean(active))

    def anomaly_score(self) -> float:
        """Alias for mean_novelty() for backwards compatibility."""
        return self.mean_novelty()

    def triage_score(self) -> float:
        active_novelties = [n for n in self._last_novelty.values() if n > 0.0]
        n_activated = len(active_novelties)
        total_primitives = len(self.primitives)
        if total_primitives == 0:
            return 1.0

        recognition = n_activated / total_primitives if total_primitives > 0 else 0.0
        recognition = min(recognition * 3.0, 1.0)  # ~30% activation = full recognition

        novelty = self.mean_novelty()
        raw = (1.0 - recognition) * 0.6 + novelty * 0.4
        return float(min(max(raw, 0.0), 1.0))

    # ── Epiphany (candidates + rare events) ─────────────────

    def _ancestors(self, pid: str, visited: Optional[set[str]] = None) -> set[str]:
        if visited is None:
            visited = set()
        p = self.primitives.get(pid)
        if p is None:
            return visited
        for parent_id in p.parents:
            if parent_id not in visited:
                visited.add(parent_id)
                self._ancestors(parent_id, visited)
        return visited

    def epiphany_candidates(self, novelty_threshold: float = 0.3):
        bursting = {
            pid: n for pid, n in self._last_novelty.items()
            if n >= novelty_threshold
        }
        if len(bursting) < 2:
            return []

        ancestor_votes: defaultdict[str, float] = defaultdict(float)
        ancestor_contributors: defaultdict[str, set[str]] = defaultdict(set)
        ancestor_depth: Dict[str, int] = {}

        for pid, novelty in bursting.items():
            p = self.primitives.get(pid)
            if p is None:
                continue

            # Immediate parents
            for parent_id in p.parents:
                if parent_id in self.primitives:
                    ancestor_votes[parent_id] += novelty
                    ancestor_contributors[parent_id].add(pid)
                    ancestor_depth[parent_id] = min(ancestor_depth.get(parent_id, 1), 1)

            # Higher ancestors
            all_ancestors = self._ancestors(pid)
            direct_parents = set(p.parents)
            for anc_id in all_ancestors - direct_parents:
                if anc_id in self.primitives:
                    ancestor_votes[anc_id] += novelty * 0.5
                    ancestor_contributors[anc_id].add(pid)
                    ancestor_depth[anc_id] = min(ancestor_depth.get(anc_id, 99), 2)

        results = [
            (
                cid,
                list(ancestor_contributors[cid]),
                ancestor_votes[cid],
                ancestor_depth.get(cid, 1),
            )
            for cid in ancestor_votes
            if len(ancestor_contributors[cid]) >= 2
        ]
        results.sort(key=lambda x: -x[2])
        return results

    def epiphany_check(self, novelty_threshold: float = 0.3):
        """
        High-level epiphany event; rare by construction.

        Only returns candidates when current tension is a statistical
        outlier relative to recent history (mean + k * std). Otherwise
        returns an empty list.
        """
        if self._tension_count < self.tension_min_history:
            return []

        std = self._tension_std()
        if std == 0.0:
            return []

        threshold = self._tension_mean + self.epiphany_k_sigma * std
        if self._last_tension <= threshold:
            return []

        return self.epiphany_candidates(novelty_threshold=novelty_threshold)

    # ── Routing / coherence / plasticity ────────────────────

    def routing_cost(self, pid_from: str, pid_to: str) -> float:
        p1 = self.primitives[pid_from]
        p2 = self.primitives[pid_to]
        return (1.0 / p2.bandwidth) + l2_distance_sq(p1.embedding, p2.embedding)

    def transition_probability(self, pid_from: str, pid_to: str) -> float:
        p2 = self.primitives[pid_to]
        d2 = l2_distance_sq(self.primitives[pid_from].embedding, p2.embedding)
        bw_factor = p2.bandwidth / p2.bandwidth_cap
        return float(math.exp(-d2 / max(self.temperature, 1e-9)) * bw_factor)

    def compute_coherence_matrix(self):
        ids = list(self.primitives.keys())
        n = len(ids)
        M = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                pi = self.primitives[ids[i]]
                pj = self.primitives[ids[j]]
                sim = cosine_similarity(pi.embedding, pj.embedding)
                M[i, j] = sim * (pi.energy + pj.energy) / 2.0
        return M, ids

    def attempt_crystallisation(self) -> None:
        if len(self.primitives) < 2:
            return
        M, ids = self.compute_coherence_matrix()
        eigenvalues, eigenvectors = np.linalg.eigh(M)
        dominant_index = int(np.argmax(eigenvalues))
        if eigenvalues[dominant_index] < self.coherence_threshold:
            return
        dominant_vector = eigenvectors[:, dominant_index]
        participating = [
            ids[i] for i, val in enumerate(dominant_vector) if abs(val) > 0.3
        ]
        if len(participating) < 2:
            return
        centroid = np.mean(
            [self.primitives[pid].embedding for pid in participating],
            axis=0,
        )
        cost_before = sum(
            self.routing_cost(i, j)
            for i in participating for j in participating if i != j
        )
        new_id = self.create_primitive(centroid, complexity=len(participating))
        cost_after = sum(self.routing_cost(pid, new_id) for pid in participating)
        if cost_after + self.crystallisation_margin < cost_before:
            self.primitives[new_id].components = list(participating)
            for pid in participating:
                if new_id not in self.primitives[pid].parents:
                    self.primitives[pid].parents.append(new_id)
        else:
            del self.primitives[new_id]

    def attempt_decomposition(self) -> None:
        """
        Remove composites whose routing benefit has vanished.

        Decomposition is made conservative for high-stability composites:
        they are only removed if the direct-cost advantage is large
        enough relative to their stability.
        """
        to_remove: List[str] = []
        for pid, p in list(self.primitives.items()):
            if not p.components:
                continue
            live_components = [c for c in p.components if c in self.primitives]
            if len(live_components) < 2:
                to_remove.append(pid)
                continue

            composite_cost = sum(
                self.routing_cost(pid, c) for c in live_components
            )
            direct_cost = sum(
                self.routing_cost(live_components[i], live_components[j])
                for i in range(len(live_components))
                for j in range(len(live_components))
                if i != j
            )

            # Stability-based margin: high-stability composites require
            # a larger direct-cost advantage to be decomposed.
            stab = self.stability_factor(p)
            margin = self.crystallisation_margin * (1.0 + 4.0 * stab)

            if direct_cost + margin < composite_cost:
                to_remove.append(pid)

        for pid in to_remove:
            p = self.primitives.get(pid)
            if p is None:
                continue
            live = [c for c in p.components if c in self.primitives]
            for cid in live:
                child = self.primitives[cid]
                if pid in child.parents:
                    child.parents.remove(pid)
            del self.primitives[pid]

    def consolidate(self) -> None:
        """
        Slowly align embeddings along the dominant variance direction.

        Embedding drift is scaled by (1 - stability_factor); high-usage
        wells barely move, low-usage wells can realign.
        """
        if len(self.primitives) < 2:
            return
        embeddings = np.array([p.embedding for p in self.primitives.values()])
        cov = np.cov(embeddings.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        dominant_vec = eigenvectors[:, int(np.argmax(eigenvalues))]
        for p in self.primitives.values():
            alignment = cosine_similarity(p.embedding, dominant_vec)
            stab = self.stability_factor(p)
            plastic = 1.0 - stab
            p.embedding += 0.01 * plastic * alignment * dominant_vec

    def step(self) -> None:
        self.step_count += 1
        for p in self.primitives.values():
            p.decay()
        self.attempt_crystallisation()
        self.attempt_decomposition()
        if self.step_count % self.consolidation_interval == 0:
            self.consolidate()

    # ── Tension / lateral pressure internals ─────────────────

    def _lateral_pressure(self) -> float:
        total = len(self.primitives)
        if total == 0:
            return 0.0
        high_ids = [
            pid for pid, n in self._last_novelty.items()
            if n >= self.novelty_baseline
        ]
        if not high_ids:
            return 0.0
        frac_active = len(high_ids) / total
        ancestor_union: set[str] = set()
        for pid in high_ids:
            ancestor_union |= self._ancestors(pid)
        diversity = len(ancestor_union) / max(1, total)
        lateral = frac_active * (0.5 + 0.5 * diversity)
        return float(min(max(lateral, 0.0), 1.0))

    def _update_tension_stats(self, tension: float) -> None:
        self._tension_history.append(tension)
        self._tension_count += 1
        delta = tension - self._tension_mean
        self._tension_mean += delta / self._tension_count
        delta2 = tension - self._tension_mean
        self._tension_m2 += delta * delta2

    def _tension_std(self) -> float:
        if self._tension_count < 2:
            return 0.0
        return math.sqrt(self._tension_m2 / (self._tension_count - 1))

    @property
    def last_tension(self) -> float:
        return self._last_tension


# ─────────────────────────────────────────────────────────────
# Simple simulation harness (unchanged except for Engine usage)
# ─────────────────────────────────────────────────────────────

def _simulate():
    """
    Toy simulation demonstrating:

    - Training a few "programs" that share sub-structure
      (common transform factors in the instr subspace).
    - Running a sequence of inputs near those programs.
    - Observing novelty, triage_score, and rare epiphany
      candidates as tension occasionally spikes.

    This is not a benchmark, just a smoke test / illustration.
    """
    rng = np.random.default_rng(42)

    # GA-inspired encoder: 4 dims for base, 4 for instr.
    space = ProgramEmbedding(dim_base=4, dim_instr=4, rng=rng)
    eng = Engine(dim=space.dim, base_activation_threshold=0.1)

    # Define some symbolic bases and instructions
    bases = ["B_core", "B_alt"]
    instrs = ["Cr_rot", "D_shift", "E_misc"]

    # Programs (names for humans only)
    # A = B_core + D_shift + E_misc (via embedding)
    # H = B_core + Cr_rot + D_shift
    # G = B_alt + Cr_rot
    program_defs = {
        "A": (["B_core"], ["D_shift", "E_misc"]),
        "H": (["B_core"], ["Cr_rot", "D_shift"]),
        "G": (["B_alt"], ["Cr_rot"]),
    }

    # Create primitives in engine
    name_to_pid: Dict[str, str] = {}
    for name, (b_ids, i_ids) in program_defs.items():
        emb = space.encode_program(b_ids, i_ids)
        pid = eng.create_primitive(emb, complexity=len(b_ids) + len(i_ids))
        name_to_pid[name] = pid

    print("Running toy stream...")
    print("-" * 60)

    stream = []
    for t in range(200):
        if t % 40 == 20:
            a_emb = eng.primitives[name_to_pid["A"]].embedding
            h_emb = eng.primitives[name_to_pid["H"]].embedding
            v = 0.5 * (a_emb + h_emb)
        else:
            prog_name = rng.choice(list(program_defs.keys()))
            v = eng.primitives[name_to_pid[prog_name]].embedding.copy()
        v += rng.standard_normal(space.dim) * 0.05
        v /= np.linalg.norm(v)
        stream.append(v)

    for t, inp in enumerate(stream):
        eng.activate_input(inp, magnitude=1.0)
        triage = eng.triage_score()
        mean_n = eng.mean_novelty()
        print(
            f"t={t:03d} "
            f"mean_novelty={mean_n:.3f} "
            f"triage={triage:.3f} "
            f"tension={eng.last_tension:.3f}"
        )

        if t % 40 == 20:
            cands = eng.epiphany_check(novelty_threshold=0.2)
            if cands:
                print("  *** Epiphany event candidates:")
                for cid, contributors, vote, depth in cands:
                    print(
                        f"    ancestor={cid[:8]} vote={vote:.3f} "
                        f"depth={depth} n_contrib={len(contributors)}"
                    )

        eng.step()


if __name__ == "__main__":
    _simulate()
