"""
SATTVA-GA v2.0
Semantic Attractor Training of Transforming Vector Associations
Multivector Geometric Algebra Formulation

Implementation Notes
--------------------
The `clifford` library requires network access to install. This file
provides a self-contained, drop-in Geometric Algebra layer (GAEngine)
that faithfully implements the algebra of G(n,0) using NumPy:
  - Multivectors with full grade decomposition
  - Geometric product, wedge product, inner product
  - Reverse (involute), scalar part, norm
  - Rotor exponential / sandwich product
  - Grade extraction

All SATTVA-GA v2.0 components (ICL, RDL, RTL, PCL, AWF, curiosity,
epiphany) are implemented above this layer in terms of the GA
primitives, exactly as specified in FORMALspecV2.md.

To swap in the real `clifford` library once network access is available,
replace GAEngine with a thin wrapper around a clifford Layout object.
"""

from __future__ import annotations
import math
import itertools
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.linalg import norm


# ============================================================
#  I.  Self-Contained Geometric Algebra Engine  G(n, 0)
# ============================================================

class MultiVector:
    """
    Sparse multivector in G(n, 0).

    Internally stored as a dict mapping basis-blade tuples to floats.
    A blade is represented as a sorted tuple of 1-based indices, e.g.
      scalar  → ()
      e1      → (1,)
      e1^e2   → (1, 2)
      e1^e2^e3→ (1, 2, 3)
    """

    def __init__(self, n: int, coeffs: Optional[Dict] = None):
        self.n = n                          # space dimension
        self._c: Dict[Tuple, float] = {}    # blade → coefficient
        if coeffs:
            for blade, val in coeffs.items():
                if abs(val) > 1e-14:
                    self._c[tuple(sorted(blade))] = float(val)

    # ---- factory helpers ----------------------------------------

    @staticmethod
    def scalar(n: int, val: float) -> "MultiVector":
        mv = MultiVector(n)
        if abs(val) > 1e-14:
            mv._c[()] = val
        return mv

    @staticmethod
    def basis_vector(n: int, idx: int) -> "MultiVector":
        """e_idx  (1-based)"""
        mv = MultiVector(n)
        mv._c[(idx,)] = 1.0
        return mv

    @staticmethod
    def from_vector(n: int, vec: np.ndarray) -> "MultiVector":
        """Build grade-1 multivector from a numpy array (length n)."""
        mv = MultiVector(n)
        for i, v in enumerate(vec):
            if abs(v) > 1e-14:
                mv._c[(i + 1,)] = float(v)
        return mv

    def to_vector(self) -> np.ndarray:
        """Extract grade-1 component as numpy array."""
        v = np.zeros(self.n)
        for blade, val in self._c.items():
            if len(blade) == 1:
                v[blade[0] - 1] = val
        return v

    # ---- grade projection ----------------------------------------

    def grade(self, k: int) -> "MultiVector":
        mv = MultiVector(self.n)
        mv._c = {b: v for b, v in self._c.items() if len(b) == k}
        return mv

    def scalar_part(self) -> float:
        return self._c.get((), 0.0)

    # ---- arithmetic ----------------------------------------------

    def __add__(self, other: "MultiVector") -> "MultiVector":
        result = MultiVector(self.n, self._c.copy())
        for blade, val in other._c.items():
            result._c[blade] = result._c.get(blade, 0.0) + val
        result._c = {b: v for b, v in result._c.items() if abs(v) > 1e-14}
        return result

    def __sub__(self, other: "MultiVector") -> "MultiVector":
        return self + (other * (-1.0))

    def __mul__(self, other) -> "MultiVector":
        if isinstance(other, (int, float)):
            mv = MultiVector(self.n)
            mv._c = {b: v * other for b, v in self._c.items()}
            return mv
        return _geometric_product(self, other)

    def __rmul__(self, scalar) -> "MultiVector":
        return self * scalar

    def __neg__(self) -> "MultiVector":
        return self * (-1.0)

    # ---- norms / similarity -------------------------------------

    def norm_sq(self) -> float:
        """⟨M M̃⟩₀  (Euclidean, all signatures +1)"""
        return (self * self.reverse()).scalar_part()

    def norm(self) -> float:
        ns = self.norm_sq()
        return math.sqrt(abs(ns))

    def normalized(self) -> "MultiVector":
        n = self.norm()
        if n < 1e-14:
            return MultiVector(self.n)
        return self * (1.0 / n)

    # ---- reverse  M̃ ---------------------------------------------

    def reverse(self) -> "MultiVector":
        """Reverse: grade-k blade picks up sign (-1)^(k(k-1)/2)."""
        mv = MultiVector(self.n)
        for blade, val in self._c.items():
            k = len(blade)
            sign = (-1) ** (k * (k - 1) // 2)
            mv._c[blade] = val * sign
        return mv

    # ---- wedge product  a ∧ b -----------------------------------

    def wedge(self, other: "MultiVector") -> "MultiVector":
        result = MultiVector(self.n)
        for b1, v1 in self._c.items():
            for b2, v2 in other._c.items():
                # blades must be disjoint for non-zero wedge
                if set(b1).isdisjoint(set(b2)):
                    merged = tuple(sorted(b1 + b2))
                    sign = _reorder_sign(b1 + b2)
                    coeff = sign * v1 * v2
                    result._c[merged] = result._c.get(merged, 0.0) + coeff
        result._c = {b: v for b, v in result._c.items() if abs(v) > 1e-14}
        return result

    # ---- scalar inner product ⟨A B̃⟩₀ ---------------------------

    def inner_scalar(self, other: "MultiVector") -> float:
        return (self * other.reverse()).scalar_part()

    def __repr__(self) -> str:
        terms = []
        for blade, val in sorted(self._c.items(), key=lambda x: len(x[0])):
            blade_str = "".join(f"e{i}" for i in blade) if blade else "1"
            terms.append(f"{val:.4f}*{blade_str}")
        return " + ".join(terms) if terms else "0"


# ---- geometric product helper -----------------------------------

def _reorder_sign(indices: Tuple[int, ...]) -> int:
    """Bubble-sort sign for reordering a tuple to sorted order."""
    lst = list(indices)
    swaps = 0
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            if lst[i] > lst[j]:
                lst[i], lst[j] = lst[j], lst[i]
                swaps += 1
    return (-1) ** swaps


def _geometric_product(A: MultiVector, B: MultiVector) -> MultiVector:
    """Full geometric product in G(n, 0):  e_i e_i = 1."""
    result = MultiVector(A.n)
    for b1, v1 in A._c.items():
        for b2, v2 in B._c.items():
            blade, sign = _blade_product(b1, b2)
            coeff = sign * v1 * v2
            if abs(coeff) > 1e-14:
                result._c[blade] = result._c.get(blade, 0.0) + coeff
    result._c = {b: v for b, v in result._c.items() if abs(v) > 1e-14}
    return result


def _blade_product(b1: Tuple, b2: Tuple) -> Tuple[Tuple, int]:
    """
    Multiply two sorted blades in G(n,0).
    Cancellation: e_i e_i = +1.
    Returns (sorted_result_blade, sign).
    """
    lst = list(b1) + list(b2)
    sign = 1
    # bubble sort, counting swaps; cancel duplicates
    i = 0
    while i < len(lst):
        j = i + 1
        while j < len(lst):
            if lst[i] == lst[j]:
                # e_k^2 = +1 → contributes +1, remove both
                lst.pop(j)
                lst.pop(i)
                # sign from moving j to i position: (j-i-1) swaps
                sign *= (-1) ** (j - i - 1)
                i = -1  # restart outer
                break
            elif lst[i] > lst[j]:
                lst[i], lst[j] = lst[j], lst[i]
                sign *= -1
            j += 1
        i += 1
    return tuple(lst), sign


def rotor_exp(bivector: MultiVector) -> MultiVector:
    """
    R = exp(B)  where B is a bivector.
    Uses series expansion: exp(B) = cos(|B|) + sin(|B|)/|B| * B
    Valid for simple bivectors in G(n,0).
    """
    n = bivector.n
    theta = bivector.norm()
    if theta < 1e-14:
        return MultiVector.scalar(n, 1.0)
    unit_B = bivector * (1.0 / theta)
    # R = cos(θ)·1 + sin(θ)·B̂
    R = MultiVector.scalar(n, math.cos(theta)) + unit_B * math.sin(theta)
    return R


def sandwich(R: MultiVector, X: MultiVector) -> MultiVector:
    """R X R̃"""
    return R * X * R.reverse()


# ============================================================
#  II.  Sensorimotor Sandbox
# ============================================================

class SensorimotorSandbox:
    """
    Physical environment: objects under gravity with bounce.
    Mass-independence of acceleration is an invariant the ICL must find.
    State vector: [y, v, mass, restitution]  (dim = 4)
    """

    def __init__(self, g: float = 9.81):
        self.g = g

    def simulate_object(self, restitution: float, mass: float,
                        steps: int = 50, dt: float = 0.05) -> np.ndarray:
        y, v = 10.0, 0.0
        states = []
        for _ in range(steps):
            v -= self.g * dt
            y += v * dt
            if y <= 0:
                y = 0.0
                v = -restitution * v
            states.append([y, v, mass, restitution])
        return np.array(states)


# ============================================================
#  III.  Invariant Primitive Layer  (ICL)
# ============================================================

class InvariantLayer:
    """
    Primitives are grade-1 multivectors (blades) in G(n, 0).

    Stability is measured as ⟨P_i^(t) P̃_i^(t-1)⟩₀ / (|P_i^t| |P_i^(t-1)|)
    across successive observation windows.
    """

    def __init__(self, n_dim: int, stability_threshold: float = 0.85):
        self.n = n_dim
        self.stability_threshold = stability_threshold
        self.primitives: List[MultiVector] = []   # grade-1 blades
        self.depths: List[float] = []
        self._history: List[List[MultiVector]] = []   # for stability tracking

    def extract_primitives(self, data: np.ndarray, top_k: int = 3):
        """Eigendecomposition of data covariance → top-k eigenvectors as blades."""
        cov = np.cov(data.T)
        vals, vecs = np.linalg.eigh(cov)
        idx = np.argsort(vals)[::-1]
        vecs = vecs[:, idx]

        self.primitives = []
        self.depths = []
        for i in range(min(top_k, vecs.shape[1])):
            vec = vecs[:, i]
            mv = MultiVector.from_vector(self.n, vec / norm(vec))
            self.primitives.append(mv)
            self.depths.append(1.0)

    def measure_stability(self, window_A: List[MultiVector],
                          window_B: List[MultiVector]) -> List[float]:
        """
        Eq. II: Stability(P_i) = (1/T) Σ_t ⟨P_i^t P̃_i^(t-1)⟩₀ / (|P^t||P^(t-1)|)
        Compares corresponding blades across two time windows.
        """
        stabilities = []
        T = min(len(window_A), len(window_B))
        for i, P in enumerate(self.primitives):
            s = 0.0
            for t in range(T):
                # In practice windows contain perturbed versions of the blade
                Pa = window_A[min(i, len(window_A) - 1)]
                Pb = window_B[min(i, len(window_B) - 1)]
                na, nb = Pa.norm(), Pb.norm()
                if na > 1e-14 and nb > 1e-14:
                    s += (Pa * Pb.reverse()).scalar_part() / (na * nb)
            stabilities.append(s / T if T else 0.0)
        return stabilities

    def project(self, X: MultiVector) -> np.ndarray:
        """c_i = ⟨X P̃_i⟩₀"""
        return np.array([X.inner_scalar(P) for P in self.primitives])

    def invariant_energy(self, X: MultiVector) -> float:
        """E^ICL = Σ c_i²"""
        return float(np.sum(self.project(X) ** 2))

    def update_depths(self, X: MultiVector, pred_error: float,
                      alpha: float = 0.001, beta: float = 0.01):
        """Eq. VII: D_i += α⟨X P̃_i⟩₀² − β·E^PCL"""
        coeffs = self.project(X)
        for i, c in enumerate(coeffs):
            self.depths[i] += alpha * c ** 2 - beta * max(0.0, pred_error - 0.01)
            self.depths[i] = max(self.depths[i], 0.0)

    def weighted_coeffs(self, X: MultiVector) -> np.ndarray:
        """c_i^weighted = D_i · ⟨X P̃_i⟩₀"""
        return np.array([d * c for d, c in
                         zip(self.depths, self.project(X))])


# ============================================================
#  IV.  Repetition Density Layer  (RDL)
# ============================================================

class RepetitionLayer:
    """
    Repetition accumulates only as grade-0 scalar curvature.
    R_k(t+1) = R_k(t) + ρ
    ∇^RDL = ∇ log(R_k + 1)
    It cannot alter blade orientations (Eq. IV / XII).
    """

    def __init__(self, n_wells: int):
        self.counts = np.zeros(n_wells)   # R_k per well
        self.rho = 1.0                    # increment

    def update(self, probs: np.ndarray):
        self.counts += self.rho * probs

    def gradient(self) -> np.ndarray:
        """Scalar gradient field ∇ log(R_k + 1)"""
        return 1.0 / (self.counts + 1.0)

    def energy(self) -> float:
        return float(np.sum(np.log(self.counts + 1.0)))


# ============================================================
#  V.  Relational Topology Layer  (RTL)
# ============================================================

class RelationalLayer:
    """
    Network topology as weighted bivector sum:
    T = Σ_{i<j} w_{ij} (e_i ∧ e_j)
    E^RTL(X) = ⟨X T̃⟩₀
    """

    def __init__(self, n_dim: int):
        self.n = n_dim
        self.T: MultiVector = MultiVector(n_dim)
        self.W: Dict[Tuple[int, int], float] = {}

    def add_edge(self, i: int, j: int, w: float = 1.0):
        """Register a relational edge between entities i and j (1-based)."""
        key = (min(i, j), max(i, j))
        self.W[key] = self.W.get(key, 0.0) + w
        self._rebuild_bivector()

    def _rebuild_bivector(self):
        self.T = MultiVector(self.n)
        for (i, j), w in self.W.items():
            ei = MultiVector.basis_vector(self.n, i)
            ej = MultiVector.basis_vector(self.n, j)
            self.T = self.T + ei.wedge(ej) * w

    def energy(self, X: MultiVector) -> float:
        """E^RTL = ⟨X T̃⟩₀"""
        return X.inner_scalar(self.T)

    def update_from_coeffs(self, coeffs: np.ndarray, threshold: float = 0.3):
        """
        Co-activate pairs of primitives when both coefficients are large.
        This grows relational topology from experience.
        """
        n = len(coeffs)
        for i in range(n):
            for j in range(i + 1, n):
                if abs(coeffs[i]) > threshold and abs(coeffs[j]) > threshold:
                    self.add_edge(i + 1, j + 1, abs(coeffs[i] * coeffs[j]))


# ============================================================
#  VI.  Predictive Consistency Layer  (PCL)  — Rotor-based
# ============================================================

class PredictiveLayer:
    """
    X_{t+1}^pred = R X_t R̃
    Rotor update: R_new = exp(-η (X_{t+1} ∧ X_t)) R
    """

    def __init__(self, n_dim: int, eta: float = 0.05):
        self.n = n_dim
        self.eta = eta
        self.R: MultiVector = MultiVector.scalar(n_dim, 1.0)   # identity rotor

    def predict(self, X: MultiVector) -> MultiVector:
        return sandwich(self.R, X)

    def update(self, X_curr: MultiVector, X_next: MultiVector):
        """
        Update rotor from prediction error bivector.
        dB = -η (X_{t+1} ∧ X_t)
        R ← exp(dB) · R
        """
        dB = (X_next.wedge(X_curr)) * (-self.eta)
        dR = rotor_exp(dB)
        self.R = dR * self.R

    def prediction_error(self, X_pred: MultiVector,
                         X_actual: MultiVector) -> float:
        """E^PCL = |X_{t+1} − X_{t+1}^pred|²"""
        diff = X_actual - X_pred
        return diff.norm_sq()


# ============================================================
#  VII.  Attractor Well Field  (AWF)
# ============================================================

class AttractorWell:
    """A single attractor well in multivector space."""

    def __init__(self, center: MultiVector, depth: float = 1.0):
        self.center = center
        self.depth = depth

    def distance_sq(self, X: MultiVector) -> float:
        """E_k(X) = ⟨(X − W_k)(X̃ − W̃_k)⟩₀"""
        diff = X - self.center
        return diff.norm_sq()

    def similarity(self, other: "AttractorWell") -> float:
        """⟨W_A W̃_B⟩₀ / (|W_A| |W_B|)"""
        na, nb = self.center.norm(), other.center.norm()
        if na < 1e-14 or nb < 1e-14:
            return 0.0
        return self.center.inner_scalar(other.center) / (na * nb)


class AttractorField:
    """
    Collection of multivector attractor wells.
    Supports assignment, depth update, and epiphany (well merge).
    """

    def __init__(self, n_dim: int, gamma: float = 0.05,
                 delta: float = 0.0001):
        self.n = n_dim
        self.gamma = gamma   # repetition deepens wells
        self.delta = delta   # shear erodes wells (kept small to avoid collapse)
        self.wells: List[AttractorWell] = []

    def initialize_from_data(self, Xs: List[MultiVector],
                             n_wells: int = 2):
        """K-means-style initialization using multivector distances."""
        if len(Xs) < n_wells:
            n_wells = len(Xs)
        # Pick seeds at random, spread by distance
        seeds = [Xs[np.random.randint(len(Xs))]]
        while len(seeds) < n_wells:
            dists = np.array([min(x.norm_sq() - s.norm_sq()
                                  for s in seeds) ** 2
                              for x in Xs])
            probs = dists / (dists.sum() + 1e-14)
            seeds.append(Xs[np.random.choice(len(Xs), p=probs)])
        self.wells = [AttractorWell(s) for s in seeds]

    def assign_probs(self, X: MultiVector) -> np.ndarray:
        """Softmax over negative squared distances."""
        energies = np.array([w.distance_sq(X) for w in self.wells])
        logits = -energies
        logits -= logits.max()
        probs = np.exp(logits)
        return probs / (probs.sum() + 1e-14)

    def update_depths(self, probs: np.ndarray, shear: float):
        """D_k(t+1) = D_k(t) + γ R_k − δ Shear"""
        for k, well in enumerate(self.wells):
            well.depth += self.gamma * probs[k] - self.delta * shear
            well.depth = max(well.depth, 0.0)

    def check_epiphany(self, invariant_layer: InvariantLayer,
                       merge_threshold: float = 0.90) -> bool:
        """
        Eq. XI: If S = ⟨W_A W̃_B⟩₀ / (|W_A||W_B|) > θ and both deep,
        create higher-grade blade P_new = Normalize(W_A ∧ W_B) and
        merge wells into one.
        """
        if len(self.wells) < 2:
            return False

        merged = False
        i = 0
        while i < len(self.wells):
            j = i + 1
            while j < len(self.wells):
                wa, wb = self.wells[i], self.wells[j]
                S = wa.similarity(wb)
                if S > merge_threshold and wa.depth > 1.0 and wb.depth > 1.0:
                    # Synthesize new higher-grade invariant primitive
                    P_new = wa.center.wedge(wb.center)
                    n = P_new.norm()
                    if n > 1e-14:
                        P_new = P_new * (1.0 / n)
                        invariant_layer.primitives.append(P_new)
                        invariant_layer.depths.append(
                            (wa.depth + wb.depth) / 2.0)

                    # Merge wells
                    new_center = wa.center * 0.5 + wb.center * 0.5
                    new_depth = max(wa.depth, wb.depth)
                    self.wells[i] = AttractorWell(new_center, new_depth)
                    self.wells.pop(j)
                    print(f"  [Epiphany] Wells {i} & {j-1} merged → "
                          f"new grade-{len(list(P_new._c.keys())[0]) if P_new._c else 0} "
                          f"primitive added. Similarity={S:.4f}")
                    merged = True
                else:
                    j += 1
            i += 1
        return merged


# ============================================================
#  VIII.  Shear and Curiosity  (Eq. IX / X)
# ============================================================

class CuriosityFunctional:
    """
    Shear(X) = |E^ICL − E^RDL| + |E^ICL − E^RTL|
    C(X)     = H(P(W_k|X)) + η·Shear(X)
    α_eff    = α (1 − σ(C))
    """

    def __init__(self, eta: float = 1.0):
        self.eta = eta

    def shear(self, e_icl: float, e_rdl: float, e_rtl: float) -> float:
        return abs(e_icl - e_rdl) + abs(e_icl - e_rtl)

    def entropy(self, probs: np.ndarray) -> float:
        p = np.clip(probs, 1e-12, 1.0)
        return float(-np.sum(p * np.log(p)))

    def curiosity(self, probs: np.ndarray, shear: float) -> float:
        return self.entropy(probs) + self.eta * shear

    def effective_alpha(self, alpha: float, C: float) -> float:
        sigma = 1.0 / (1.0 + math.exp(-C))
        return alpha * (1.0 - sigma)


# ============================================================
#  IX.  Total Field Energy  (Eq. XIII)
# ============================================================

def total_field_energy(e_icl: float, e_pcl: float, shear: float,
                       well_depth: float,
                       lam: float = 0.5, mu: float = 0.1) -> float:
    """E = E^ICL + E^PCL + λ·Shear − μ·D_Wk"""
    return e_icl + e_pcl + lam * shear - mu * well_depth


# ============================================================
#  X.  SATTVA-GA Core Engine
# ============================================================

class SattvaGA:
    """
    SATTVA-GA v2.0 — full Geometric Algebra formulation.

    Developmental sequence (Eq. XIV):
      1. Extract grade-1 blades from covariance (ICL)
      2. Train rotor predictor (PCL)
      3. Form multivector attractor wells (AWF)
      4. Online: project, compute energies, update all layers
      5. Curiosity modulates learning rate
      6. Epiphany synthesizes higher-grade primitives
    """

    def __init__(self, n_dim: int = 4, n_wells: int = 2,
                 n_primitives: int = 3):
        self.n = n_dim
        self.icl = InvariantLayer(n_dim)
        self.rdl: Optional[RepetitionLayer] = None   # init after wells
        self.rtl = RelationalLayer(n_dim)
        self.pcl = PredictiveLayer(n_dim, eta=0.03)
        self.awf = AttractorField(n_dim)
        self.curiosity_fn = CuriosityFunctional(eta=0.8)
        self.n_primitives = n_primitives
        self.n_wells = n_wells
        self._prev_X: Optional[MultiVector] = None
        self.step_count = 0

    # ---- 1. Developmental phase ----------------------------------

    def developmental_phase(self, sandbox: SensorimotorSandbox,
                            restitutions=(0.2, 0.5, 0.8),
                            masses=(1.0, 3.0, 5.0)):
        """
        Eq. XIV Stage 1: extract grade-1 blades from covariance.
        """
        print("[Dev Phase] Simulating sensorimotor data …")
        data_all = []
        for r in restitutions:
            for m in masses:
                sim = sandbox.simulate_object(r, m)
                data_all.append(sim)
        data = np.vstack(data_all)

        # Extract invariant primitives
        self.icl.extract_primitives(data, top_k=self.n_primitives)
        print(f"  → {len(self.icl.primitives)} grade-1 primitive blades extracted.")

        # Warm-up rotor on consecutive pairs
        mvs = [MultiVector.from_vector(self.n, row) for row in data]
        for t in range(len(mvs) - 1):
            self.pcl.update(mvs[t], mvs[t + 1])
        print("  → Rotor predictor trained on trajectory.")

        return data, mvs

    # ---- 2. Form attractor wells ---------------------------------

    def form_wells(self, mvs: List[MultiVector]):
        """
        Project onto ICL coefficients, then cluster in coefficient space
        using multivector distances (Eq. VIII).
        """
        # Use coefficient vectors to find initial well centers
        coeff_vecs = np.array([self.icl.project(X) for X in mvs])

        # K-means in coefficient space to seed well centers
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=self.n_wells, n_init=10, random_state=42)
        labels = km.fit_predict(coeff_vecs)

        # Well centers as multivectors reconstructed from cluster centroids
        self.awf.wells = []
        for k in range(self.n_wells):
            centroid = km.cluster_centers_[k]
            # Reconstruct as weighted sum of primitives
            center_mv = MultiVector(self.n)
            for ci, P in zip(centroid, self.icl.primitives):
                center_mv = center_mv + P * ci
            self.awf.wells.append(AttractorWell(center_mv, depth=1.0))

        self.rdl = RepetitionLayer(self.n_wells)
        print(f"  → {self.n_wells} attractor wells formed in multivector space.")
        return labels

    # ---- 3. Online processing -----------------------------------

    def process(self, X: MultiVector) -> Dict:
        """
        Single-step update:
          - Project onto ICL
          - Rotor prediction + error
          - Well assignment + energies
          - Shear + curiosity
          - Depth updates (all layers)
          - Epiphany check every 50 steps
        """
        self.step_count += 1

        # ICL projection
        coeffs = self.icl.project(X)
        weighted_coeffs = self.icl.weighted_coeffs(X)
        e_icl = float(np.sum(coeffs ** 2))

        # PCL: rotor prediction error
        X_pred = self.pcl.predict(X)
        e_pcl = self.pcl.prediction_error(X_pred, X)
        if self._prev_X is not None:
            self.pcl.update(self._prev_X, X)
        self._prev_X = X

        # RDL energy
        e_rdl = self.rdl.energy() if self.rdl else 0.0

        # RTL update + energy
        self.rtl.update_from_coeffs(coeffs, threshold=0.2)
        e_rtl = self.rtl.energy(X)

        # Well assignment
        probs = self.awf.assign_probs(X)

        # Repetition update
        if self.rdl:
            self.rdl.update(probs)

        # Shear + curiosity
        shear = self.curiosity_fn.shear(e_icl, e_rdl, e_rtl)
        C = self.curiosity_fn.curiosity(probs, shear)
        alpha_eff = self.curiosity_fn.effective_alpha(0.001, C)

        # Depth updates
        self.icl.update_depths(X, e_pcl, alpha=alpha_eff, beta=0.01)
        self.awf.update_depths(probs, shear)

        # Total energy (Eq. XIII)
        best_well_depth = max(w.depth for w in self.awf.wells)
        E_total = total_field_energy(e_icl, e_pcl, shear, best_well_depth)

        # Epiphany check
        epiphany = False
        if self.step_count % 50 == 0:
            epiphany = self.awf.check_epiphany(self.icl)

        return {
            "step": self.step_count,
            "coeffs": coeffs,
            "weighted_coeffs": weighted_coeffs,
            "e_icl": e_icl,
            "e_pcl": e_pcl,
            "e_rdl": e_rdl,
            "e_rtl": e_rtl,
            "shear": shear,
            "curiosity": C,
            "alpha_eff": alpha_eff,
            "probs": probs,
            "E_total": E_total,
            "epiphany": epiphany,
            "well_depths": [w.depth for w in self.awf.wells],
            "n_primitives": len(self.icl.primitives),
        }

    def run_online(self, mvs: List[MultiVector],
                   n_steps: int = 100) -> List[Dict]:
        """Process a sequence of multivectors, returning per-step results."""
        results = []
        indices = np.random.randint(0, len(mvs), size=n_steps)
        for idx in indices:
            res = self.process(mvs[idx])
            results.append(res)
        return results


# ============================================================
#  XI.  Main Simulation
# ============================================================

if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 60)
    print("  SATTVA-GA v2.0 — Geometric Algebra Formulation")
    print("=" * 60)

    N_DIM = 4   # state vector: [y, v, mass, restitution]

    # Instantiate engine
    engine = SattvaGA(n_dim=N_DIM, n_wells=3, n_primitives=3)
    sandbox = SensorimotorSandbox()

    # ---- Stage 1: Developmental phase --------------------------
    print("\n[Stage 1] Developmental Phase")
    data, mvs = engine.developmental_phase(sandbox)

    # ---- Stage 2: Form wells -----------------------------------
    print("\n[Stage 2] Forming Attractor Wells")
    labels = engine.form_wells(mvs)

    # ---- Stage 3: Online processing ----------------------------
    print("\n[Stage 3] Online Processing (200 steps)\n")
    results = engine.run_online(mvs, n_steps=200)

    # ---- Summary -----------------------------------------------
    print("\n" + "=" * 60)
    print("  Results Summary")
    print("=" * 60)

    e_icl_vals  = [r["e_icl"]      for r in results]
    shear_vals  = [r["shear"]      for r in results]
    cur_vals    = [r["curiosity"]  for r in results]
    e_pcl_vals  = [r["e_pcl"]     for r in results]
    E_vals      = [r["E_total"]    for r in results]

    def stats(name, vals):
        a = np.array(vals)
        print(f"  {name:<22} mean={a.mean():.4f}  std={a.std():.4f}  "
              f"min={a.min():.4f}  max={a.max():.4f}")

    stats("E^ICL",          e_icl_vals)
    stats("E^PCL",          e_pcl_vals)
    stats("Shear",          shear_vals)
    stats("Curiosity",      cur_vals)
    stats("E_total",        E_vals)

    print(f"\n  Final primitive blades : {results[-1]['n_primitives']}")
    print(f"  Final well depths      : "
          f"{[f'{d:.3f}' for d in results[-1]['well_depths']]}")

    print("\n[Stage 4] Manual Epiphany Check")
    engine.awf.check_epiphany(engine.icl, merge_threshold=0.50)
    print(f"  Total primitives after epiphany check: "
          f"{len(engine.icl.primitives)}")

    print("\n[GA Invariance Demo]")
    P0 = engine.icl.primitives[0]
    print(f"  Primitive blade (grade-1) : {P0}")
    print(f"  Blade norm                : {P0.norm():.6f}")
    rev = P0.reverse()
    print(f"  ⟨P P̃⟩₀ (should be 1)    : {P0.inner_scalar(P0):.6f}")

    if len(engine.icl.primitives) >= 2:
        P1 = engine.icl.primitives[1]
        wedge_blade = P0.wedge(P1)
        print(f"  P0 ∧ P1 (grade-2 blade)   : {wedge_blade}")

    # Rotor demo
    sample_X = mvs[0]
    X_pred = engine.pcl.predict(sample_X)
    print(f"\n  Sample X norm             : {sample_X.norm():.4f}")
    print(f"  R X R̃ norm               : {X_pred.norm():.4f}")
    print(f"  Rotor R                   : {engine.pcl.R}")

    print("\n" + "=" * 60)
    print("  SATTVA-GA v2.0 run complete.")
    print("=" * 60)
