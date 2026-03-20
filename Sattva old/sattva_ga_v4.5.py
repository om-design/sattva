"""
SATTVA-GA v4.0
Semantic Attractor Training of Transforming Vector Associations
Multivector Geometric Algebra Formulation

Builds on v3.0 (all v3 fixes retained).  New in v4:

NEW-1  Well Resonance (from SE(n) comparison analysis)
       Two attractor wells whose centers have PARTIAL geometric alignment
       (resonance_min < |cos(W_A, W_B)| < resonance_max) attract each
       other by a small weighted step every N_RESONATE processing steps.
       This is the direct GA realisation of the fMRI cross-domain resonance
       hypothesis: structures that are neither identical nor orthogonal
       influence each other's representation.

       In GA multivector space the coupling is:
           S   = inner_scalar(W_A, W_B) / (|W_A| |W_B|)   (geometric cosine)
           Δ   = strength · S · (W_B.center − W_A.center)
           W_A.center += Δ
           W_B.center −= Δ
       Condition: resonance_min < |S| < resonance_max
         |S| < resonance_min  → too orthogonal, unrelated domains, no coupling
         |S| > resonance_max  → too similar, epiphany (merge) is appropriate
         In between           → the resonance zone

NEW-2  Entropy-Based Well Splitting (from SE(n) comparison analysis)
       Each well maintains a rolling buffer of ICL coefficient vectors for
       recently-assigned states.  Dispersion is the mean squared distance
       of members from the well center in coefficient space.
       When dispersion exceeds dispersion_threshold AND the buffer has
       ≥ min_split_members, the well splits along its first principal
       coefficient direction (SVD of the centered member matrix).
       The parent's rdl.counts entry is split proportionally.
       This gives wells the ability to DIFFERENTIATE as well as consolidate
       — both halves of cognitive categorisation.

       Splitting is capped by max_wells; when the cap is reached, splits
       are suppressed and a warning is emitted.

NEW-3  v3 bipartite eigenvector matching already implemented in extract_
       primitives (greedy best-match across windows, not index matching).
       v4 makes this explicit in the docstring and diagnostic test.

Interaction order in process():
  resonance runs BEFORE epiphany/split checks so that resonance-nudged
  wells can trigger epiphany in the same periodic check.

v3 → v4 component changes:
  AttractorWell    : + _coeff_buf, receive(), dispersion(), split_direction()
  AttractorField   : + resonate(), check_splits(), max_wells, timing params
  SattvaGA.process : + resonance and split calls, extended return dict
  SattvaGA.__init__: + resonance and split parameters
  Diagnostics      : + three new tests (resonance zone, split, max_wells cap)

All v3 components (GA engine, ICL, RDL, RTL, PCL, CuriosityFunctional,
total_field_energy, four-stage develop()) are unchanged.
"""

from __future__ import annotations
import math
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.linalg import norm as np_norm
from sklearn.cluster import KMeans


# ============================================================
#  I.  Geometric Algebra Engine  G(n, 0)
#      Unchanged from v3 — full self-contained implementation.
# ============================================================

class MultiVector:
    """
    Sparse multivector in G(n, 0).

    Stored as dict: sorted blade-tuple → float coefficient.
      ()       → grade-0 scalar
      (i,)     → grade-1 basis vector e_i  (1-based)
      (i,j)    → grade-2 bivector e_i ∧ e_j
      (i,j,k)  → grade-3 trivector, etc.
    """

    def __init__(self, n: int, coeffs: Optional[Dict] = None):
        self.n = n
        self._c: Dict[Tuple, float] = {}
        if coeffs:
            for blade, val in coeffs.items():
                if abs(val) > 1e-14:
                    self._c[tuple(sorted(blade))] = float(val)

    # ── factories ────────────────────────────────────────────────

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
        """Grade-1 multivector from numpy array of length n."""
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

    # ── grade operations ──────────────────────────────────────────

    def grade(self, k: int) -> "MultiVector":
        mv = MultiVector(self.n)
        mv._c = {b: v for b, v in self._c.items() if len(b) == k}
        return mv

    def scalar_part(self) -> float:
        return self._c.get((), 0.0)

    def max_grade(self) -> int:
        if not self._c:
            return 0
        return max(len(b) for b in self._c)

    # ── arithmetic ────────────────────────────────────────────────

    def __add__(self, other: "MultiVector") -> "MultiVector":
        result = MultiVector(self.n)
        result._c = dict(self._c)
        for blade, val in other._c.items():
            result._c[blade] = result._c.get(blade, 0.0) + val
        result._c = {b: v for b, v in result._c.items() if abs(v) > 1e-14}
        return result

    def __sub__(self, other: "MultiVector") -> "MultiVector":
        return self + (other * (-1.0))

    def __mul__(self, other) -> "MultiVector":
        if isinstance(other, (int, float, np.floating)):
            f = float(other)
            mv = MultiVector(self.n)
            mv._c = {b: v * f for b, v in self._c.items()
                     if abs(v * f) > 1e-14}
            return mv
        return _geometric_product(self, other)

    def __rmul__(self, scalar) -> "MultiVector":
        return self * scalar

    def __neg__(self) -> "MultiVector":
        return self * (-1.0)

    # ── norms ─────────────────────────────────────────────────────

    def norm_sq(self) -> float:
        """⟨M M̃⟩₀  — always ≥ 0 in G(n,0)."""
        return (self * self.reverse()).scalar_part()

    def norm(self) -> float:
        return math.sqrt(max(0.0, self.norm_sq()))

    def normalized(self) -> "MultiVector":
        n = self.norm()
        return self * (1.0 / n) if n > 1e-14 else MultiVector(self.n)

    # ── reverse ───────────────────────────────────────────────────

    def reverse(self) -> "MultiVector":
        """M̃: grade-k blade acquires sign (−1)^{k(k−1)/2}."""
        mv = MultiVector(self.n)
        for blade, val in self._c.items():
            k = len(blade)
            sign = (-1) ** (k * (k - 1) // 2)
            mv._c[blade] = val * sign
        return mv

    # ── wedge product ─────────────────────────────────────────────

    def wedge(self, other: "MultiVector") -> "MultiVector":
        """Outer product A ∧ B."""
        result = MultiVector(self.n)
        for b1, v1 in self._c.items():
            for b2, v2 in other._c.items():
                if set(b1).isdisjoint(set(b2)):
                    merged = tuple(sorted(b1 + b2))
                    sign = _reorder_sign(b1 + b2)
                    coeff = sign * v1 * v2
                    result._c[merged] = result._c.get(merged, 0.0) + coeff
        result._c = {b: v for b, v in result._c.items() if abs(v) > 1e-14}
        return result

    # ── scalar inner product ⟨A B̃⟩₀ ──────────────────────────────

    def inner_scalar(self, other: "MultiVector") -> float:
        """⟨A B̃⟩₀ — scalar part of geometric product with reverse."""
        return (self * other.reverse()).scalar_part()

    def __repr__(self) -> str:
        if not self._c:
            return "0"
        terms = []
        for blade, val in sorted(self._c.items(),
                                  key=lambda x: (len(x[0]), x[0])):
            label = "".join(f"e{i}" for i in blade) if blade else "1"
            terms.append(f"{val:+.4f}*{label}")
        return " ".join(terms)


# ── GA helpers ────────────────────────────────────────────────────────────────

def _reorder_sign(indices: Tuple[int, ...]) -> int:
    """Sign from bubble-sorting indices to canonical ascending order."""
    lst = list(indices)
    swaps = 0
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            if lst[i] > lst[j]:
                lst[i], lst[j] = lst[j], lst[i]
                swaps += 1
    return (-1) ** swaps


def _blade_product(b1: Tuple, b2: Tuple) -> Tuple[Tuple, int]:
    """
    Geometric product of two sorted basis blades in G(n,0).
    e_i² = +1.  Returns (result_blade, total_sign).
    """
    lst = list(b1) + list(b2)
    sign = 1
    i = 0
    while i < len(lst):
        j = i + 1
        while j < len(lst):
            if lst[i] == lst[j]:
                sign *= (-1) ** (j - i - 1)
                lst.pop(j)
                lst.pop(i)
                i = -1
                break
            elif lst[i] > lst[j]:
                lst[i], lst[j] = lst[j], lst[i]
                sign *= -1
            j += 1
        i += 1
    return tuple(lst), sign


def _geometric_product(A: MultiVector, B: MultiVector) -> MultiVector:
    result = MultiVector(A.n)
    for b1, v1 in A._c.items():
        for b2, v2 in B._c.items():
            blade, sign = _blade_product(b1, b2)
            coeff = sign * v1 * v2
            if abs(coeff) > 1e-14:
                result._c[blade] = result._c.get(blade, 0.0) + coeff
    result._c = {b: v for b, v in result._c.items() if abs(v) > 1e-14}
    return result


def rotor_exp(bivector: MultiVector, n_terms: int = 16) -> MultiVector:
    """
    R = exp(B) via truncated power series: Σ_{k=0}^{n_terms} B^k / k!

    Valid for any multivector B; in particular for compound bivectors
    (sums of multiple e_i∧e_j planes) as produced by X_curr ∧ X_next.
    For |B| up to ~10, 16 terms give |R R̃ − 1| < 1e-8.
    """
    n = bivector.n
    result = MultiVector.scalar(n, 1.0)
    term   = MultiVector.scalar(n, 1.0)
    for k in range(1, n_terms + 1):
        term = (term * bivector) * (1.0 / k)
        result = result + term
        if term.norm() < 1e-15:
            break
    return result


def sandwich(R: MultiVector, X: MultiVector) -> MultiVector:
    """Versor sandwich product: R X R̃"""
    return R * X * R.reverse()


def rotor_normalise(R: MultiVector) -> MultiVector:
    """Re-normalise rotor so that R R̃ = 1 (counters float drift)."""
    ns = (R * R.reverse()).scalar_part()
    if ns < 1e-28:
        return MultiVector.scalar(R.n, 1.0)
    return R * (1.0 / math.sqrt(ns))


# ============================================================
#  II.  Sensorimotor Sandbox
#       Unchanged from v3.
# ============================================================

class SensorimotorSandbox:
    """
    Bouncing-ball physics. State: [y, v, mass, restitution]  (n=4)

    Key property: acceleration = −g regardless of mass.
    The mass column is constant within a trajectory and is correctly
    rejected by the ICL stability gate (near-zero cross-window variance).
    """

    def __init__(self, g: float = 9.81):
        self.g = g

    def simulate_object(self, restitution: float, mass: float,
                        steps: int = 60, dt: float = 0.05) -> np.ndarray:
        y, v = 10.0, 0.0
        states = []
        for _ in range(steps):
            v -= self.g * dt
            y += v * dt
            if y <= 0.0:
                y = 0.0
                v = -restitution * v
            states.append([y, v, mass, restitution])
        return np.array(states)

    def generate_dataset(self,
                         restitutions=(0.2, 0.5, 0.8),
                         masses=(1.0, 3.0, 5.0)) -> np.ndarray:
        return np.vstack([
            self.simulate_object(r, m)
            for r in restitutions for m in masses
        ])

    def generate_novel_dataset(self,
                               restitutions=(0.1, 0.95),
                               masses=(0.5, 10.0),
                               initial_heights=(5.0, 20.0)) -> np.ndarray:
        """
        NEW in v4: novel-domain data for demonstrating well splitting.
        Covers restitution and height regimes outside the training set.
        """
        segs = []
        for r in restitutions:
            for m in masses:
                for h0 in initial_heights:
                    y, v = h0, 0.0
                    states = []
                    for _ in range(60):
                        v -= self.g * 0.05
                        y += v * 0.05
                        if y <= 0.0:
                            y = 0.0
                            v = -r * v
                        states.append([y, v, m, r])
                    segs.append(np.array(states))
        return np.vstack(segs)


# ============================================================
#  III.  Invariant Primitive Layer  (ICL)
#        v3 FIX 4 retained: greedy bipartite matching across windows.
#        v4 note: bipartite matching is explicitly documented here.
# ============================================================

class InvariantLayer:
    """
    The ICL is the physics-grounded constraint manifold (Spec §II).

    Bipartite window matching (v3 FIX 4, made explicit in v4):
      For each candidate eigenvector in the reference window, find the
      BEST-matching direction in each subsequent window by maximising
      cosine similarity across ALL candidates (greedy assignment).
      This is correct: index-based matching fails when eigenvalues are
      close and eigenvectors permute across windows.

    Grade-2 and higher primitives are added by Stage 2 (bivector
    emergence) and by the epiphany mechanism (Spec §XI).
    """

    def __init__(self, n_dim: int,
                 stability_threshold: float = 0.90,
                 n_windows: int = 4):
        self.n = n_dim
        self.stability_threshold = stability_threshold
        self.n_windows = n_windows
        self.primitives: List[MultiVector] = []
        self.depths: List[float] = []

    def extract_primitives(self, data: np.ndarray,
                           top_k_candidates: int = 6,
                           max_primitives: int = 3) -> List[Tuple[float, int]]:
        """
        Greedy bipartite window matching + stability gate.

        For each reference candidate, track the best-matching direction
        greedily through subsequent windows.  Score = mean cosine similarity
        across the chain.  Promote only directions with score ≥ threshold.

        Returns (stability_score, candidate_index) for logging.
        """
        N = len(data)
        win = N // self.n_windows
        if win < 2:
            raise ValueError("Too few samples for the requested n_windows.")

        window_vecs: List[List[np.ndarray]] = []
        for w in range(self.n_windows):
            chunk = data[w * win: (w + 1) * win]
            cov = np.cov(chunk.T)
            _, vecs = np.linalg.eigh(cov)
            vecs = vecs[:, ::-1]            # descending eigenvalue order
            normed = []
            for i in range(min(top_k_candidates, vecs.shape[1])):
                v = vecs[:, i].copy()
                v /= np_norm(v) + 1e-14
                if v[np.argmax(np.abs(v))] < 0:
                    v = -v                  # canonical sign
                normed.append(v)
            window_vecs.append(normed)

        reference = window_vecs[0]
        scored: List[Tuple[float, np.ndarray]] = []

        for ref_vec in reference:
            sims = []
            current = ref_vec
            for w in range(1, self.n_windows):
                candidates = window_vecs[w]
                cos_sims = [abs(float(np.dot(current, c)))
                            for c in candidates]
                best = int(np.argmax(cos_sims))
                sims.append(cos_sims[best])
                current = candidates[best]   # greedy: track the chain
            scored.append((float(np.mean(sims)), ref_vec))

        scored.sort(key=lambda x: -x[0])

        self.primitives = []
        self.depths = []
        admitted = 0
        for stab, vec in scored:
            if stab >= self.stability_threshold and admitted < max_primitives:
                self.primitives.append(MultiVector.from_vector(self.n, vec))
                self.depths.append(1.0)
                admitted += 1

        if admitted == 0:
            stab, vec = scored[0]
            self.primitives.append(MultiVector.from_vector(self.n, vec))
            self.depths.append(1.0)
            print(f"  [ICL WARNING] No candidate met stability threshold "
                  f"{self.stability_threshold:.2f} (best={stab:.4f}). "
                  f"Admitting top candidate as fallback.")

        return [(s, i) for i, (s, _) in enumerate(scored)]

    def project(self, X: MultiVector) -> np.ndarray:
        """c_i = ⟨X P̃_i⟩₀  (Spec §III)"""
        return np.array([X.inner_scalar(P) for P in self.primitives])

    def invariant_energy(self, X: MultiVector) -> float:
        """E^ICL = Σ_i c_i²"""
        return float(np.sum(self.project(X) ** 2))

    def update_depths(self, X: MultiVector, pred_error: float,
                      alpha: float = 0.01, beta: float = 0.002):
        """D_i += α ⟨X P̃_i⟩₀² − β E^PCL"""
        coeffs = self.project(X)
        for i, c in enumerate(coeffs):
            self.depths[i] += alpha * c ** 2 - beta * max(0.0, pred_error)
            self.depths[i] = max(self.depths[i], 0.0)

    def weighted_coeffs(self, X: MultiVector) -> np.ndarray:
        """c_i^{weighted} = D_i · ⟨X P̃_i⟩₀"""
        return np.array([d * c for d, c in
                         zip(self.depths, self.project(X))])

    def promote_bivector(self, bv: MultiVector, depth: float = 0.5):
        """Add a higher-grade blade (Stage 2 and epiphany)."""
        self.primitives.append(bv)
        self.depths.append(depth)


# ============================================================
#  IV.  Repetition Density Layer  (RDL)
#       Grade-0 only — formal bias-agnostic guarantee.
#       Unchanged from v3.
# ============================================================

class RepetitionLayer:
    """
    R_k(t+1) = R_k(t) + ρ · P(W_k|X_t)  (Spec §IV)

    Repetition accumulates ONLY as grade-0 scalar curvature.
    Cannot influence blade orientation, bivector topology, or rotors.
    This is the formal basis of the bias-agnostic guarantee.
    """

    def __init__(self, n_wells: int, rho: float = 1.0):
        self.counts = np.zeros(n_wells)
        self.rho = rho

    def update(self, probs: np.ndarray):
        self.counts += self.rho * probs

    def gradient(self) -> np.ndarray:
        return 1.0 / (self.counts + 1.0)

    def energy(self) -> float:
        return float(np.sum(np.log(self.counts + 1.0)))

    def resize(self, n: int):
        """Grow or shrink counts array, preserving existing values."""
        old = self.counts.copy()
        self.counts = np.zeros(n)
        self.counts[:len(old)] = old

    def split_entry(self, k: int, frac_a: float) -> None:
        """
        NEW in v4: split entry k into two (k and the last entry).
        Called by AttractorField.check_splits().
        """
        count_k = self.counts[k]
        self.counts = np.append(self.counts, count_k * (1.0 - frac_a))
        self.counts[k] = count_k * frac_a

    def merge_entries(self, i: int, j: int) -> None:
        """Merge entry j into i (epiphany)."""
        self.counts[i] += self.counts[j]
        self.counts = np.delete(self.counts, j)


# ============================================================
#  V.  Relational Topology Layer  (RTL)
#      v3 FIX 7 retained: Frobenius co-activation energy.
#      Unchanged from v3.
# ============================================================

class RelationalLayer:
    """
    T = Σ_{i<j} w_{ij} (e_i ∧ e_j)  (Spec §V)

    Energy (FIX 7): E^RTL = Σ_{i<j} w_{ij} · c_i · c_j
    This is well-defined for grade-1 states (unlike the v2 formula
    ⟨X T̃⟩₀ which is zero by grade arithmetic for grade-1 X and
    grade-2 T).
    """

    def __init__(self, n_dim: int):
        self.n = n_dim
        self.W: Dict[Tuple[int, int], float] = {}

    def add_edge(self, i: int, j: int, w: float = 1.0):
        key = (min(i, j), max(i, j))
        self.W[key] = min(self.W.get(key, 0.0) + w, 1.0)

    def bivector(self) -> MultiVector:
        T = MultiVector(self.n)
        for (i, j), w in self.W.items():
            T = T + (MultiVector.basis_vector(self.n, i)
                     .wedge(MultiVector.basis_vector(self.n, j))) * w
        return T

    def energy(self, coeffs: np.ndarray) -> float:
        total = 0.0
        for (i, j), w in self.W.items():
            ci = float(coeffs[i - 1]) if i - 1 < len(coeffs) else 0.0
            cj = float(coeffs[j - 1]) if j - 1 < len(coeffs) else 0.0
            total += w * ci * cj
        return total

    def update_from_coeffs(self, coeffs: np.ndarray,
                           threshold: float = 0.25):
        if len(coeffs) == 0:
            return
        c_norm = coeffs / (np.abs(coeffs).max() + 1e-14)
        n = len(c_norm)
        for i in range(n):
            for j in range(i + 1, n):
                if abs(c_norm[i]) > threshold and abs(c_norm[j]) > threshold:
                    inc = abs(c_norm[i] * c_norm[j]) * 0.001
                    self.add_edge(i + 1, j + 1, inc)

    def stable_bivectors(self, weight_threshold: float = 0.3
                         ) -> List[MultiVector]:
        blades = []
        for (i, j), w in self.W.items():
            if w >= weight_threshold:
                bv = (MultiVector.basis_vector(self.n, i)
                      .wedge(MultiVector.basis_vector(self.n, j)))
                blades.append((w, bv.normalized()))
        blades.sort(key=lambda x: -x[0])
        return [bv for _, bv in blades]


# ============================================================
#  VI.  Predictive Consistency Layer  (PCL)
#       v3 FIX 1+2 retained.  Unchanged from v3.
# ============================================================

class PredictiveLayer:
    """
    X_{t+1}^{pred} = R X_t R̃   (Spec §VI)

    R_new = exp(−η · normalise(X_curr ∧ X_next)) · R

    FIX 1: exp via power series (compound bivectors).
    FIX 2: correct operand order + unit-bivector scaling.
    """

    def __init__(self, n_dim: int, eta: float = 0.05,
                 normalise_every: int = 20):
        self.n = n_dim
        self.eta = eta
        self.normalise_every = normalise_every
        self._update_count = 0
        self.R: MultiVector = MultiVector.scalar(n_dim, 1.0)

    def predict(self, X: MultiVector) -> MultiVector:
        return sandwich(self.R, X)

    def update(self, X_curr: MultiVector, X_next: MultiVector):
        bv = X_curr.wedge(X_next)
        bv_norm = bv.norm()
        if bv_norm < 1e-14:
            return
        bv_unit = bv * (1.0 / bv_norm)
        dB = bv_unit * (-self.eta)
        dR = rotor_exp(dB)
        self.R = dR * self.R
        self._update_count += 1
        if self._update_count % self.normalise_every == 0:
            self.R = rotor_normalise(self.R)

    def prediction_error(self, X_pred: MultiVector,
                         X_actual: MultiVector) -> float:
        return (X_actual - X_pred).norm_sq()

    def rotor_unit_check(self) -> float:
        return (self.R * self.R.reverse()).scalar_part()


# ============================================================
#  VII.  Attractor Wells
#        v4 NEW: member coefficient buffer, dispersion, split_direction.
# ============================================================

class AttractorWell:
    """
    Multivector attractor center with depth tracking.

    NEW in v4:
      _coeff_buf  — rolling deque of ICL coefficient vectors for recently
                    assigned states (maxlen = member_buffer).
      receive()   — record an assignment (hard: only the argmax well).
      dispersion() — mean squared distance of members from center in
                    coefficient space; high dispersion → candidate for split.
      split_direction() — first principal component of the centered member
                    matrix (SVD); the natural split axis.
    """

    def __init__(self, center: MultiVector, depth: float = 1.0,
                 member_buffer: int = 150):
        self.center = center
        self.depth = depth
        self._coeff_buf: deque = deque(maxlen=member_buffer)

    def receive(self, coeffs: np.ndarray) -> None:
        """Record that the state with these ICL coefficients was assigned here."""
        self._coeff_buf.append(coeffs.copy())

    def dispersion(self) -> float:
        """
        Mean squared distance of buffered members from their centroid
        in ICL coefficient space.  Returns 0 if buffer has < 2 entries.
        """
        if len(self._coeff_buf) < 2:
            return 0.0
        M = np.array(list(self._coeff_buf))
        mu = M.mean(axis=0)
        return float(np.mean(np.sum((M - mu) ** 2, axis=1)))

    def split_direction(self) -> np.ndarray:
        """
        First principal component of the centered member coefficient
        matrix.  This is the axis of greatest internal variance —
        the natural direction to split along.
        """
        if len(self._coeff_buf) < 2:
            d = np.zeros(self._coeff_buf.maxlen or 1)
            if len(d) > 0:
                d[0] = 1.0
            return d
        M = np.array(list(self._coeff_buf))
        mu = M.mean(axis=0)
        centered = M - mu
        if centered.shape[0] < 2 or centered.shape[1] < 1:
            d = np.zeros(centered.shape[1])
            if len(d) > 0:
                d[0] = 1.0
            return d
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        return Vt[0]

    def distance_sq(self, X: MultiVector) -> float:
        """E_k(X) = ⟨(X − W_k)(X̃ − W̃_k)⟩₀  (Spec §VIII)"""
        return (X - self.center).norm_sq()

    def similarity(self, other: "AttractorWell") -> float:
        """⟨W_A W̃_B⟩₀ / (|W_A| |W_B|)  (Spec §XI)"""
        na, nb = self.center.norm(), other.center.norm()
        if na < 1e-14 or nb < 1e-14:
            return 0.0
        return self.center.inner_scalar(other.center) / (na * nb)


# ============================================================
#  VIII.  Attractor Field
#         v3 fixes retained (FIX 5 depth from R_k, FIX 8 epiphany).
#         v4 NEW: resonate(), check_splits(), max_wells, reset_member_buffers().
# ============================================================

class AttractorField:
    """
    Collection of attractor wells in multivector state space.

    v4 additions:
      resonate()      — cross-well geometric coupling in the partial-alignment zone.
      check_splits()  — entropy-based differentiation (complement to epiphany merging).
      max_wells       — hard cap on total well count after splitting.
      reset_member_buffers() — called after ICL expansion so buffer lengths stay valid.
    """

    def __init__(self, n_dim: int,
                 gamma: float = 0.05,
                 delta: float = 0.0001,
                 max_wells: int = 8,
                 member_buffer: int = 150,
                 resonance_min: float = 0.10,
                 resonance_max: float = 0.80,
                 resonance_strength: float = 0.005,
                 dispersion_threshold: float = 10.0,
                 min_split_members: int = 20):
        self.n = n_dim
        self.gamma = gamma
        self.delta = delta
        self.max_wells = max_wells
        self.member_buffer = member_buffer
        self.resonance_min = resonance_min
        self.resonance_max = resonance_max
        self.resonance_strength = resonance_strength
        self.dispersion_threshold = dispersion_threshold
        self.min_split_members = min_split_members
        self.wells: List[AttractorWell] = []
        # Counters for logging
        self._resonance_events: int = 0
        self._split_events: int = 0

    def initialize_from_kmeans(self, coeff_vecs: np.ndarray,
                                n_wells: int, icl: "InvariantLayer"):
        """Seed well centers from ICL coefficient space k-means."""
        km = KMeans(n_clusters=n_wells, n_init=10, random_state=42)
        km.fit(coeff_vecs)
        self.wells = []
        for k in range(n_wells):
            centroid = km.cluster_centers_[k]
            center_mv = MultiVector(self.n)
            for ci, P in zip(centroid, icl.primitives):
                if P.max_grade() == 1:
                    center_mv = center_mv + P * float(ci)
            self.wells.append(
                AttractorWell(center_mv, depth=1.0,
                              member_buffer=self.member_buffer))

    def assign_probs(self, X: MultiVector) -> np.ndarray:
        """P(W_k|X) = softmax(−E_k(X))"""
        energies = np.array([w.distance_sq(X) for w in self.wells])
        logits   = -energies - (-energies).max()
        probs    = np.exp(logits)
        return probs / (probs.sum() + 1e-14)

    def assign_hard(self, X: MultiVector, coeffs: np.ndarray) -> int:
        """
        Hard assignment: argmax of softmax probabilities.
        Also calls receive() on the winning well.
        """
        probs = self.assign_probs(X)
        k = int(np.argmax(probs))
        self.wells[k].receive(coeffs)
        return k

    def update_depths(self, rdl_counts: np.ndarray, shear: float):
        """D_Wk += γ R_k − δ Shear  (FIX 5: uses accumulated R_k)"""
        for k, well in enumerate(self.wells):
            if k < len(rdl_counts):
                well.depth += self.gamma * rdl_counts[k] - self.delta * shear
                well.depth = max(well.depth, 0.0)

    # ── NEW-1: Well Resonance ─────────────────────────────────────
    #
    # Two wells in the partial-alignment zone (resonance_min < |S| < resonance_max)
    # attract each other by a weighted step.  This is the GA realisation of the
    # fMRI cross-domain resonance hypothesis: partial geometric overlap between
    # semantic structures causes mutual influence on representation.
    #
    # The coupling is symmetric:
    #   Δ = strength · S · (W_B.center − W_A.center)
    #   W_A.center += Δ
    #   W_B.center −= Δ
    #
    # For orthogonal wells (|S| ≈ 0): no coupling.
    # For parallel wells  (|S| ≈ 1): epiphany merge is appropriate, not resonance.
    # The intermediate zone is the resonance zone.

    def resonate(self) -> int:
        """
        NEW-1: Apply one round of inter-well geometric coupling.
        Returns the number of resonating pairs found.
        """
        n_pairs = 0
        for i in range(len(self.wells)):
            for j in range(i + 1, len(self.wells)):
                wa, wb = self.wells[i], self.wells[j]
                na, nb = wa.center.norm(), wb.center.norm()
                if na < 1e-14 or nb < 1e-14:
                    continue
                S = wa.center.inner_scalar(wb.center) / (na * nb)
                if self.resonance_min < abs(S) < self.resonance_max:
                    delta = (wb.center - wa.center) * (self.resonance_strength * S)
                    self.wells[i].center = wa.center + delta
                    self.wells[j].center = wb.center - delta
                    n_pairs += 1
        self._resonance_events += n_pairs
        return n_pairs

    # ── NEW-2: Entropy-Based Well Splitting ───────────────────────
    #
    # A well whose member-coefficient dispersion exceeds the threshold is
    # split along its principal direction (SVD of the centered member matrix).
    # The parent is replaced by two children, each with half the parent depth
    # and counts proportional to their membership fraction.
    #
    # Split is suppressed when max_wells is reached.

    def check_splits(self, icl: "InvariantLayer",
                     rdl: "RepetitionLayer") -> bool:
        """
        NEW-2: Entropy-based splitting for each overly-disperse well.
        Returns True if any split occurred.
        rdl is updated in-place (split_entry).
        """
        if len(self.wells) >= self.max_wells:
            return False

        any_split = False
        k = 0
        while k < len(self.wells):
            if len(self.wells) >= self.max_wells:
                break
            well = self.wells[k]
            n_members = len(well._coeff_buf)
            disp = well.dispersion()

            if disp < self.dispersion_threshold or n_members < self.min_split_members:
                k += 1
                continue

            # Compute split direction and partition members
            split_dir = well.split_direction()
            members   = list(well._coeff_buf)
            M         = np.array(members)
            mu        = M.mean(axis=0)
            if len(split_dir) != M.shape[1]:
                # Dimension mismatch (can happen if ICL was expanded mid-run)
                k += 1
                continue
            projections = (M - mu) @ split_dir
            mask_a = projections >= 0
            mask_b = ~mask_a

            if mask_a.sum() < 2 or mask_b.sum() < 2:
                k += 1
                continue

            # Build child centers as weighted sums of grade-1 ICL primitives
            coeffs_a    = M[mask_a]
            coeffs_b    = M[mask_b]
            center_a_mv = self._center_from_coeffs(coeffs_a.mean(axis=0), icl)
            center_b_mv = self._center_from_coeffs(coeffs_b.mean(axis=0), icl)

            depth_child = max(well.depth / 2.0, 0.5)
            frac_a      = float(mask_a.sum()) / n_members

            child_a = AttractorWell(center_a_mv, depth=depth_child,
                                    member_buffer=self.member_buffer)
            child_b = AttractorWell(center_b_mv, depth=depth_child,
                                    member_buffer=self.member_buffer)

            # Seed children's buffers with their partition
            for c in coeffs_a:
                child_a.receive(c)
            for c in coeffs_b:
                child_b.receive(c)

            # Replace parent with child_a, append child_b
            self.wells[k] = child_a
            self.wells.append(child_b)

            # Synchronise rdl.counts
            if k < len(rdl.counts):
                rdl.split_entry(k, frac_a)

            print(f"  [Split] Well {k}  disp={disp:.2f}  "
                  f"members={n_members}  "
                  f"→ children ({mask_a.sum()},{mask_b.sum()})  "
                  f"total wells: {len(self.wells)}")
            self._split_events += 1
            any_split = True
            # Don't advance k — check child_a for further splitting
        return any_split

    def _center_from_coeffs(self, coeff_vec: np.ndarray,
                             icl: "InvariantLayer") -> MultiVector:
        """Reconstruct a grade-1 multivector center from ICL coefficients."""
        center_mv = MultiVector(self.n)
        for ci, P in zip(coeff_vec, icl.primitives):
            if P.max_grade() == 1:
                center_mv = center_mv + P * float(ci)
        return center_mv

    def reset_member_buffers(self) -> None:
        """
        Clear all member buffers after ICL expansion (new primitive added).
        Buffers will be refilled with fresh, correctly-dimensioned data.
        """
        for well in self.wells:
            well._coeff_buf.clear()

    # ── Epiphany — unchanged from v3 FIX 8, syncs rdl ────────────

    def check_epiphany(self, icl: "InvariantLayer",
                       rdl: Optional["RepetitionLayer"] = None,
                       cosine_threshold: float = 0.85,
                       depth_ratio_threshold: float = 0.85) -> bool:
        """
        Dual-trigger epiphany (v3 FIX 8).
        v4 addition: synchronises rdl.counts via merge_entries() when wells merge.
        """
        if len(self.wells) < 2:
            return False
        merged_any = False
        i = 0
        while i < len(self.wells):
            j = i + 1
            while j < len(self.wells):
                wa, wb = self.wells[i], self.wells[j]
                S  = wa.similarity(wb)
                da, db = wa.depth, wb.depth

                trig_A = S > cosine_threshold and da > 1.0 and db > 1.0
                d_min  = min(da, db) + 1e-9
                d_max  = max(da, db)
                trig_B = (S > 0.5
                          and (d_min / d_max) > depth_ratio_threshold
                          and da > 2.0 and db > 2.0)

                if trig_A or trig_B:
                    reason = "geometric" if trig_A else "depth-convergence"
                    P_new  = wa.center.wedge(wb.center)
                    P_norm = P_new.norm()
                    grade_new = 0
                    if P_norm > 1e-14:
                        P_new     = P_new * (1.0 / P_norm)
                        grade_new = P_new.max_grade()
                        icl.promote_bivector(P_new, depth=(da + db) / 2.0)
                        self.reset_member_buffers()  # ICL expanded

                    new_center = wa.center * 0.5 + wb.center * 0.5
                    self.wells[i] = AttractorWell(new_center, max(da, db),
                                                  member_buffer=self.member_buffer)
                    self.wells.pop(j)

                    if rdl is not None:
                        rdl.merge_entries(i, j)

                    print(f"  [Epiphany/{reason}] wells {i}&{j} merged  "
                          f"S={S:.3f}  D=({da:.2f},{db:.2f})  "
                          f"→ grade-{grade_new} primitive  "
                          f"total wells: {len(self.wells)}  "
                          f"total ICL primitives: {len(icl.primitives)}")
                    merged_any = True
                else:
                    j += 1
            i += 1
        return merged_any


# ============================================================
#  IX.  Shear and Curiosity
#       v3 FIX 3 retained.  Unchanged from v3.
# ============================================================

class CuriosityFunctional:
    """
    Shear(X) = |E^ICL − E^RDL| + |E^ICL − E^RTL|
    C(X)     = H(P(W_k|X)) + η · Shear(X)
    α_eff    = α · (1 − σ(C/τ))

    FIX 3: temperature τ prevents sigmoid saturation at typical C magnitudes.
    """

    def __init__(self, eta: float = 0.8, tau: float = 50.0):
        self.eta = eta
        self.tau = tau

    def shear(self, e_icl: float, e_rdl: float, e_rtl: float) -> float:
        return abs(e_icl - e_rdl) + abs(e_icl - e_rtl)

    def entropy(self, probs: np.ndarray) -> float:
        p = np.clip(probs, 1e-12, 1.0)
        return float(-np.sum(p * np.log(p)))

    def curiosity(self, probs: np.ndarray, shear: float) -> float:
        return self.entropy(probs) + self.eta * shear

    def effective_alpha(self, alpha: float, C: float) -> float:
        sigma = 1.0 / (1.0 + math.exp(-C / self.tau))
        return alpha * (1.0 - sigma)


# ============================================================
#  X.  Total Field Energy  (Spec §XIII)
#      Unchanged from v3.
# ============================================================

def total_field_energy(e_icl: float, e_pcl: float, shear: float,
                       well_depth: float,
                       lam: float = 0.5, mu: float = 0.1) -> float:
    """ε(X) = E^ICL + E^PCL + λ·Shear − μ·D_Wk"""
    return e_icl + e_pcl + lam * shear - mu * well_depth


# ============================================================
#  XI.  SATTVA-GA v4 Core Engine
# ============================================================

class SattvaGA:
    """
    SATTVA-GA v4.0

    Inherits full v3 developmental sequence.  Adds:
      NEW-1  Well resonance — partial-alignment attraction between wells,
             controlled by resonance_min/max/strength and N_RESONATE cadence.
      NEW-2  Well splitting  — entropy-based differentiation,
             controlled by dispersion_threshold and N_SPLIT cadence.

    Online processing cadence (per-step):
      Every step          : soft + hard well assignment, depth update.
      Every N_RESONATE    : resonate() — partial-alignment attraction.
      Every N_SPLIT       : check_splits() then check_epiphany() —
                            differentiation before consolidation.
    """

    def __init__(self,
                 n_dim: int = 4,
                 n_wells: int = 3,
                 max_wells: int = 8,
                 n_primitives: int = 3,
                 stability_threshold: float = 0.90,
                 curiosity_tau: float = 50.0,
                 resonance_min: float = 0.10,
                 resonance_max: float = 0.80,
                 resonance_strength: float = 0.005,
                 dispersion_threshold: float = 10.0,
                 min_split_members: int = 20,
                 member_buffer: int = 150,
                 N_RESONATE: int = 10,
                 N_SPLIT: int = 50):

        self.n            = n_dim
        self.n_wells      = n_wells
        self.n_primitives = n_primitives
        self.N_RESONATE   = N_RESONATE
        self.N_SPLIT      = N_SPLIT

        self.icl = InvariantLayer(n_dim,
                                  stability_threshold=stability_threshold,
                                  n_windows=4)
        self.rdl: Optional[RepetitionLayer] = None
        self.rtl = RelationalLayer(n_dim)
        self.pcl = PredictiveLayer(n_dim, eta=0.05)
        self.awf = AttractorField(
            n_dim,
            max_wells         = max_wells,
            member_buffer     = member_buffer,
            resonance_min     = resonance_min,
            resonance_max     = resonance_max,
            resonance_strength= resonance_strength,
            dispersion_threshold = dispersion_threshold,
            min_split_members = min_split_members,
        )
        self.curiosity_fn = CuriosityFunctional(tau=curiosity_tau)

        self._prev_X: Optional[MultiVector] = None
        self.step_count: int = 0
        self._stage_log: List[str] = []

    # ── Developmental stages (unchanged from v3) ──────────────────

    def stage1_extract_primitives(self, data: np.ndarray):
        print("[Stage 1] Stability-gated grade-1 primitive extraction …")
        scored = self.icl.extract_primitives(
            data,
            top_k_candidates=min(8, data.shape[1] * 2),
            max_primitives=self.n_primitives,
        )
        n1 = sum(1 for p in self.icl.primitives if p.max_grade() == 1)
        print(f"  → {n1} grade-1 primitives admitted "
              f"(threshold={self.icl.stability_threshold:.2f})")
        for stab, idx in scored[:self.n_primitives + 3]:
            admitted = idx < n1
            print(f"     {'✓' if admitted else '✗'} "
                  f"candidate {idx}: stability={stab:.4f}")
        self._stage_log.append(f"Stage 1: {n1} grade-1 primitives")

    def stage2_bivector_emergence(self, mvs: List[MultiVector],
                                   n_passes: int = 4,
                                   weight_threshold: float = 0.2):
        print("[Stage 2] Bivector emergence from relational co-activation …")
        for _ in range(n_passes):
            for X in mvs:
                coeffs = self.icl.project(X)
                self.rtl.update_from_coeffs(coeffs, threshold=0.20)

        stable = self.rtl.stable_bivectors(weight_threshold)
        promoted = 0
        for bv in stable[:2]:
            self.icl.promote_bivector(bv.normalized(), depth=0.5)
            promoted += 1

        grades = [p.max_grade() for p in self.icl.primitives]
        print(f"  → {len(self.rtl.W)} relational edges  "
              f"max_weight={max(self.rtl.W.values(), default=0):.4f}")
        print(f"  → {promoted} grade-2 blade(s) promoted to ICL")
        print(f"  → ICL: {len(self.icl.primitives)} primitives, grades={grades}")
        self._stage_log.append(
            f"Stage 2: {promoted} grade-2 bivectors; "
            f"ICL now {len(self.icl.primitives)} primitives")

    def stage3_fit_rotor(self, mvs: List[MultiVector]):
        print("[Stage 3] Fitting rotor predictor …")
        for t in range(len(mvs) - 1):
            self.pcl.update(mvs[t], mvs[t + 1])
        unit = self.pcl.rotor_unit_check()
        print(f"  → Rotor trained on {len(mvs)-1} steps.  "
              f"⟨R R̃⟩₀ = {unit:.8f}")
        self._stage_log.append("Stage 3: rotor fitted")

    def stage4_form_wells(self, mvs: List[MultiVector]):
        print("[Stage 4] Forming attractor wells …")
        coeff_vecs = np.array([self.icl.project(X) for X in mvs])
        self.awf.initialize_from_kmeans(coeff_vecs, self.n_wells, self.icl)
        self.rdl = RepetitionLayer(self.n_wells)
        print(f"  → {self.n_wells} wells in "
              f"{len(self.icl.primitives)}-D coefficient space")
        self._stage_log.append(f"Stage 4: {self.n_wells} wells formed")

    def develop(self, sandbox: "SensorimotorSandbox"):
        data = sandbox.generate_dataset()
        mvs  = [MultiVector.from_vector(self.n, row) for row in data]
        self.stage1_extract_primitives(data)
        self.stage2_bivector_emergence(mvs)
        self.stage3_fit_rotor(mvs)
        self.stage4_form_wells(mvs)
        return data, mvs

    # ── Online processing ─────────────────────────────────────────

    def process(self, X: MultiVector) -> Dict:
        """
        Single-step update.  v4 adds resonance and split calls.

        Per-step ordering:
          1. ICL projection, PCL prediction + update
          2. RDL and RTL energy
          3. Soft well assignment → RDL update → depth update (R_k)
          4. Hard well assignment → member buffer update
          5. [every N_RESONATE] resonance
          6. [every N_SPLIT]    split-check, then epiphany-check
          7. Shear, curiosity, α_eff → ICL depth update
          8. Build return dict
        """
        self.step_count += 1

        # ── 1. ICL + PCL ─────────────────────────────────────────
        coeffs     = self.icl.project(X)
        e_icl      = float(np.sum(coeffs ** 2))
        weighted_c = self.icl.weighted_coeffs(X)

        X_pred = self.pcl.predict(X)
        e_pcl  = self.pcl.prediction_error(X_pred, X)
        if self._prev_X is not None:
            self.pcl.update(self._prev_X, X)
        self._prev_X = X

        # ── 2. RDL + RTL ─────────────────────────────────────────
        e_rdl = self.rdl.energy() if self.rdl else 0.0
        self.rtl.update_from_coeffs(coeffs, threshold=0.25)
        e_rtl = self.rtl.energy(coeffs)

        # ── 3. Soft assignment → RDL update → depth update ───────
        probs = self.awf.assign_probs(X)
        if self.rdl:
            self.rdl.update(probs)
            shear_d = self.curiosity_fn.shear(e_icl, e_rdl, e_rtl)
            self.awf.update_depths(self.rdl.counts, shear_d)

        # ── 4. Hard assignment → member buffer ───────────────────
        self.awf.assign_hard(X, coeffs)

        # ── 5. Resonance (every N_RESONATE steps) ────────────────
        n_resonating = 0
        if self.step_count % self.N_RESONATE == 0:
            n_resonating = self.awf.resonate()

        # ── 6. Split then epiphany (every N_SPLIT steps) ─────────
        split_occurred   = False
        epiphany_occurred = False
        if self.step_count % self.N_SPLIT == 0:
            split_occurred    = self.awf.check_splits(self.icl, self.rdl)
            epiphany_occurred = self.awf.check_epiphany(
                self.icl, rdl=self.rdl)

        # ── 7. Shear → curiosity → α_eff → ICL depths ────────────
        shear     = self.curiosity_fn.shear(e_icl, e_rdl, e_rtl)
        C         = self.curiosity_fn.curiosity(probs, shear)
        alpha_eff = self.curiosity_fn.effective_alpha(0.01, C)
        self.icl.update_depths(X, e_pcl, alpha=alpha_eff, beta=0.002)

        # ── 8. Return dict ────────────────────────────────────────
        best_depth = max((w.depth for w in self.awf.wells), default=0.0)
        E_total    = total_field_energy(e_icl, e_pcl, shear, best_depth)

        return {
            "step":               self.step_count,
            "coeffs":             coeffs,
            "weighted_coeffs":    weighted_c,
            "e_icl":              e_icl,
            "e_pcl":              e_pcl,
            "e_rdl":              e_rdl,
            "e_rtl":              e_rtl,
            "shear":              shear,
            "curiosity":          C,
            "alpha_eff":          alpha_eff,
            "probs":              probs,
            "E_total":            E_total,
            "n_wells":            len(self.awf.wells),
            "well_depths":        [w.depth for w in self.awf.wells],
            "well_dispersions":   [w.dispersion() for w in self.awf.wells],
            "n_primitives":       len(self.icl.primitives),
            "n_resonating_pairs": n_resonating,
            "split_occurred":     split_occurred,
            "epiphany_occurred":  epiphany_occurred,
            "rotor_unit":         self.pcl.rotor_unit_check(),
        }

    def run_online(self, mvs: List[MultiVector],
                   n_steps: int = 300) -> List[Dict]:
        idx = np.random.randint(0, len(mvs), size=n_steps)
        return [self.process(mvs[i]) for i in idx]

    def run_online_seq(self, mvs: List[MultiVector],
                       n_steps: int = 300) -> List[Dict]:
        """Sequential (non-random) online processing for novel-domain demo."""
        results = []
        for i in range(min(n_steps, len(mvs))):
            results.append(self.process(mvs[i]))
        return results


# ============================================================
#  XII.  Diagnostic test suite
# ============================================================

def run_diagnostics(engine: SattvaGA,
                    mvs: List[MultiVector]) -> Tuple[int, int]:
    """
    Verify all v3 fixes plus all v4 new mechanisms.
    Returns (passed, total).
    """
    print("\n" + "=" * 62)
    print("  SATTVA-GA v4.0 — Diagnostic Tests")
    print("=" * 62)

    passed = 0
    total  = 0

    def check(label: str, ok: bool, detail: str = ""):
        nonlocal passed, total
        total += 1
        if ok:
            passed += 1
        sym = "PASS" if ok else "FAIL"
        print(f"  [{sym}] {label}")
        if detail:
            print(f"         {detail}")

    n  = engine.n
    e1 = MultiVector.basis_vector(n, 1)
    e2 = MultiVector.basis_vector(n, 2)
    e3 = MultiVector.basis_vector(n, 3)
    e4 = MultiVector.basis_vector(n, 4)

    # ── GA engine correctness ─────────────────────────────────────

    # Compound rotor is unit
    compound = e1.wedge(e2) * 0.7 + e3.wedge(e4) * 0.4
    R_c   = rotor_exp(compound)
    err_c = abs((R_c * R_c.reverse()).scalar_part() - 1.0)
    check("GA: rotor_exp(compound_bv) is unit rotor",
          err_c < 1e-8, f"|R R̃ − 1| = {err_c:.2e}")

    # Simple rotor is unit
    simple = e1.wedge(e2) * 1.2
    R_s    = rotor_exp(simple)
    err_s  = abs((R_s * R_s.reverse()).scalar_part() - 1.0)
    check("GA: rotor_exp(simple_bv) is unit rotor",
          err_s < 1e-8, f"|R R̃ − 1| = {err_s:.2e}")

    # Sandwich preserves norm
    X_s   = mvs[0]
    X_r   = sandwich(engine.pcl.R, X_s)
    ratio = X_r.norm() / (X_s.norm() + 1e-14)
    check("GA: sandwich preserves multivector norm",
          abs(ratio - 1.0) < 1e-6, f"|R X R̃|/|X| = {ratio:.8f}")

    # Trained rotor is unit
    unit  = engine.pcl.rotor_unit_check()
    check("GA: trained rotor satisfies R R̃ = 1",
          abs(unit - 1.0) < 1e-5, f"⟨R R̃⟩₀ = {unit:.8f}")

    # ── v3 fixes ──────────────────────────────────────────────────

    # FIX 2 (rotor gradient descent on rotational data)
    np.random.seed(0)
    bv_true = e1.wedge(e2) * (-math.pi / 8)
    R_star  = rotor_exp(bv_true)
    X0_tr   = MultiVector.from_vector(n, np.array([1.0, 0.3, 0.5, 0.2]))
    traj_tr = [X0_tr]
    for _ in range(30):
        traj_tr.append(sandwich(R_star, traj_tr[-1]))
    pcl_t = PredictiveLayer(n, eta=0.01)
    np.random.seed(5)
    X0_fr   = MultiVector.from_vector(n, np.random.randn(n))
    traj_fr = [X0_fr]
    for _ in range(20):
        traj_fr.append(sandwich(R_star, traj_fr[-1]))

    def _rot_err(pcl_):
        return float(np.mean([
            (traj_fr[i+1] - pcl_.predict(traj_fr[i])).norm_sq()
            for i in range(len(traj_fr) - 1)
        ]))

    err_before_r = _rot_err(pcl_t)
    for t in range(len(traj_tr) - 1):
        pcl_t.update(traj_tr[t], traj_tr[t + 1])
    err_after_r = _rot_err(pcl_t)
    check("FIX 2: rotor gradient descent on rotational data",
          err_after_r < err_before_r,
          f"error {err_before_r:.4f} → {err_after_r:.4f}")

    # FIX 3: α_eff varies and is monotone
    cf     = engine.curiosity_fn
    alphas = [cf.effective_alpha(0.01, C) for C in [5, 25, 100, 300]]
    rng    = max(alphas) - min(alphas)
    check("FIX 3a: α_eff varies meaningfully across curiosity range",
          rng > 0.001,
          f"α_eff at C=[5,25,100,300]: {[f'{a:.5f}' for a in alphas]}")
    check("FIX 3b: α_eff is monotonically decreasing in C",
          alphas[0] > alphas[1] > alphas[2] > alphas[3])

    # FIX 4: stability gate (bipartite matching)
    grade1 = sum(1 for p in engine.icl.primitives if p.max_grade() == 1)
    check("FIX 4: stability gate admits ≤ n_primitives grade-1 blades",
          grade1 <= engine.n_primitives,
          f"admitted {grade1} / {engine.n_primitives} max")

    # FIX 5: well depths nonzero
    depths = [w.depth for w in engine.awf.wells]
    check("FIX 5: well depths nonzero after processing",
          any(d > 0.5 for d in depths),
          f"depths: {[f'{d:.3f}' for d in depths]}")

    # FIX 6: Stage 2 grade-2 primitive
    grade2 = sum(1 for p in engine.icl.primitives if p.max_grade() == 2)
    check("FIX 6: Stage 2 promoted ≥ 1 grade-2 bivector",
          grade2 >= 1, f"grade-2 primitives: {grade2}")

    # FIX 7: RTL energy nonzero and varying
    e_rtl_vals = [engine.rtl.energy(engine.icl.project(X))
                  for X in mvs[:20]]
    check("FIX 7: RTL energy nonzero and varying for grade-1 states",
          any(abs(e) > 1e-10 for e in e_rtl_vals)
          and max(e_rtl_vals) - min(e_rtl_vals) > 1e-8,
          f"E^RTL sample: {[f'{e:.3f}' for e in e_rtl_vals[:5]]}")

    # Bias guarantee: scalar scaling cannot rotate a blade
    P0        = engine.icl.primitives[0]
    P0_scaled = P0 * 100.0
    cos_val   = (P0.inner_scalar(P0_scaled)
                 / (P0.norm() * P0_scaled.norm() + 1e-14))
    check("SPEC §XII: scalar ×100 does not alter blade orientation",
          abs(cos_val - 1.0) < 1e-10,
          f"cosine(P0, 100·P0) = {cos_val:.12f}")

    # ── NEW-1: Well Resonance ─────────────────────────────────────

    # Construct a fresh field with two wells in the resonance zone and
    # verify that resonance increases their cosine similarity.
    tmp_awf = AttractorField(n, resonance_min=0.1, resonance_max=0.8,
                             resonance_strength=0.05)
    # Wells with S ≈ 0.45 (partial alignment)
    v_a = np.zeros(n); v_a[0] = 1.0
    v_b = np.zeros(n); v_b[0] = 0.6; v_b[1] = 0.8  # cos = 0.6
    wa_mv = MultiVector.from_vector(n, v_a)
    wb_mv = MultiVector.from_vector(n, v_b)
    tmp_awf.wells = [
        AttractorWell(wa_mv, depth=2.0, member_buffer=10),
        AttractorWell(wb_mv, depth=2.0, member_buffer=10),
    ]
    S_before = tmp_awf.wells[0].similarity(tmp_awf.wells[1])
    for _ in range(30):
        tmp_awf.resonate()
    S_after = tmp_awf.wells[0].similarity(tmp_awf.wells[1])
    check("NEW-1: resonance increases cosine similarity in partial-alignment zone",
          S_after > S_before,
          f"S before={S_before:.4f}  after={S_after:.4f}")

    # Verify orthogonal wells are NOT coupled (|S| < resonance_min)
    tmp_awf2 = AttractorField(n, resonance_min=0.1, resonance_max=0.8,
                              resonance_strength=0.05)
    v_orth_b = np.zeros(n); v_orth_b[1] = 1.0   # orthogonal to v_a
    wa_mv2   = MultiVector.from_vector(n, v_a)
    wb_mv2   = MultiVector.from_vector(n, v_orth_b)
    tmp_awf2.wells = [
        AttractorWell(wa_mv2, depth=2.0, member_buffer=10),
        AttractorWell(wb_mv2, depth=2.0, member_buffer=10),
    ]
    c_before = tmp_awf2.wells[0].center._c.copy()
    tmp_awf2.resonate()
    c_after = tmp_awf2.wells[0].center._c.copy()
    check("NEW-1: orthogonal wells are NOT coupled by resonance",
          c_before == c_after,
          "center unchanged after resonance on orthogonal wells")

    # Total resonance events in the main run
    check("NEW-1: resonance fired at least once during online processing",
          engine.awf._resonance_events > 0,
          f"total resonance events: {engine.awf._resonance_events}")

    # ── NEW-2: Well Splitting ─────────────────────────────────────

    # Construct a well with artificially high dispersion and verify it splits.
    tmp_icl = InvariantLayer(n, stability_threshold=0.80)
    # Use two of the real primitives as placeholders
    for p in engine.icl.primitives[:2]:
        if p.max_grade() == 1:
            tmp_icl.primitives.append(p)
            tmp_icl.depths.append(1.0)

    tmp_awf3 = AttractorField(n, max_wells=4, member_buffer=200,
                              dispersion_threshold=2.0,
                              min_split_members=10)
    center_mv = engine.icl.primitives[0]
    high_disp_well = AttractorWell(center_mv, depth=3.0, member_buffer=200)
    np.random.seed(0)
    # Inject bimodal membership
    for _ in range(50):
        c = np.zeros(len(tmp_icl.primitives))
        c[0] = 5.0 + np.random.randn() * 0.1   # cluster A
        high_disp_well.receive(c)
    for _ in range(50):
        c = np.zeros(len(tmp_icl.primitives))
        c[0] = -5.0 + np.random.randn() * 0.1  # cluster B
        high_disp_well.receive(c)
    tmp_awf3.wells = [high_disp_well]

    tmp_rdl = RepetitionLayer(1)
    tmp_rdl.counts[0] = 10.0
    disp_before = high_disp_well.dispersion()
    tmp_awf3.check_splits(tmp_icl, tmp_rdl)
    n_wells_after = len(tmp_awf3.wells)
    check("NEW-2: bimodal well with high dispersion splits into 2",
          n_wells_after == 2,
          f"dispersion={disp_before:.2f}  wells before=1  after={n_wells_after}")

    # rdl.counts synchronised after split
    check("NEW-2: rdl.counts length matches wells after split",
          len(tmp_rdl.counts) == n_wells_after,
          f"rdl.counts length={len(tmp_rdl.counts)}  "
          f"wells={n_wells_after}")

    # max_wells cap prevents unbounded growth
    tmp_awf4 = AttractorField(n, max_wells=2, member_buffer=200,
                              dispersion_threshold=2.0,
                              min_split_members=10)
    for _ in range(2):
        w = AttractorWell(center_mv, depth=3.0, member_buffer=200)
        for _ in range(50):
            c = np.array([5.0 + np.random.randn() * 0.1, 0.0, 0.0, 0.0])
            w.receive(c)
        for _ in range(50):
            c = np.array([-5.0 + np.random.randn() * 0.1, 0.0, 0.0, 0.0])
            w.receive(c)
        tmp_awf4.wells.append(w)
    rdl4 = RepetitionLayer(2); rdl4.counts[:] = 10.0
    tmp_awf4.check_splits(tmp_icl, rdl4)
    check("NEW-2: max_wells cap prevents unbounded splitting",
          len(tmp_awf4.wells) <= 2,
          f"wells after attempted split with cap=2: {len(tmp_awf4.wells)}")

    print(f"\n  {passed}/{total} tests passed.")
    return passed, total


# ============================================================
#  XIII.  Main
# ============================================================

if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 62)
    print("  SATTVA-GA v4.0")
    print("  + Well Resonance (fMRI cross-domain coupling)")
    print("  + Entropy-Based Well Splitting (differentiation)")
    print("=" * 62)

    sandbox = SensorimotorSandbox()
    engine  = SattvaGA(
        n_dim              = 4,
        n_wells            = 3,
        max_wells          = 8,
        n_primitives       = 3,
        stability_threshold= 0.88,
        curiosity_tau      = 50.0,
        resonance_min      = 0.10,
        resonance_max      = 0.80,
        resonance_strength = 0.005,
        dispersion_threshold = 10.0,
        min_split_members  = 20,
        member_buffer      = 150,
        N_RESONATE         = 10,
        N_SPLIT            = 50,
    )

    # ── Developmental phases 1-4 ──────────────────────────────────
    print("\n── Developmental Sequence ──────────────────────────────────")
    data, mvs = engine.develop(sandbox)

    # ── Phase A: online run on training-domain data ───────────────
    print("\n── Phase A: Online Processing — Training Domain (400 steps) ─")
    results_a = engine.run_online(mvs, n_steps=400)

    # ── Phase B: novel-domain data (tests splitting) ──────────────
    print("\n── Phase B: Novel-Domain Data (wider restitution + height) ─")
    novel_data = sandbox.generate_novel_dataset()
    novel_mvs  = [MultiVector.from_vector(engine.n, row)
                  for row in novel_data]
    results_b  = engine.run_online_seq(novel_mvs, n_steps=len(novel_mvs))

    results = results_a + results_b

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  Results Summary")
    print("=" * 62)

    def stats(name: str, key: str, fmt: str = "10.4f"):
        vals = np.array([r[key] for r in results
                         if isinstance(r.get(key), (int, float, np.floating))])
        if len(vals) == 0:
            return
        print(f"  {name:<26} "
              f"mean={vals.mean():{fmt}}  "
              f"std={vals.std():9.4f}  "
              f"min={vals.min():9.4f}  "
              f"max={vals.max():9.4f}")

    stats("E^ICL",              "e_icl")
    stats("E^PCL",              "e_pcl")
    stats("E^RTL",              "e_rtl")
    stats("Shear",              "shear")
    stats("Curiosity",          "curiosity")
    stats("α_eff",              "alpha_eff")
    stats("E_total",            "E_total")
    stats("Well count",         "n_wells")
    stats("Resonating pairs",   "n_resonating_pairs")

    total_splits    = sum(1 for r in results if r.get("split_occurred"))
    total_epiphanies = sum(1 for r in results if r.get("epiphany_occurred"))
    print(f"\n  Split checks that produced splits : {total_splits}")
    print(f"  Epiphany checks that merged wells : {total_epiphanies}")
    print(f"  Total resonance pair-events       : {engine.awf._resonance_events}")
    print(f"  Total split events                : {engine.awf._split_events}")

    final = results[-1]
    print(f"\n  Final n_wells      : {final['n_wells']}")
    print(f"  Final n_primitives : {final['n_primitives']}")
    print(f"  Primitive grades   : "
          f"{[p.max_grade() for p in engine.icl.primitives]}")
    print(f"  Final well depths  : "
          f"{[f'{d:.3f}' for d in final['well_depths']]}")
    print(f"  Final dispersions  : "
          f"{[f'{d:.2f}' for d in final['well_dispersions']]}")
    print(f"  Rotor unit check   : {final['rotor_unit']:.8f}")

    # ── Trace: n_wells over time ──────────────────────────────────
    print("\n── Well count trajectory ───────────────────────────────────")
    steps      = [r["step"]    for r in results]
    n_wells_t  = [r["n_wells"] for r in results]
    changes    = [(s, k) for s, k in zip(steps, n_wells_t)
                  if s == 1 or k != n_wells_t[max(0, steps.index(s)-1)]]
    for s, k in changes[:20]:
        print(f"  step {s:>4}: n_wells = {k}")
    if len(changes) > 20:
        print(f"  … ({len(changes)} total changes)")

    # ── Diagnostics ───────────────────────────────────────────────
    passed, total_tests = run_diagnostics(engine, mvs)

    print("\n── Developmental log ───────────────────────────────────────")
    for entry in engine._stage_log:
        print(f"  {entry}")

    print(f"\n  Final result: {passed}/{total_tests} diagnostic tests passed.")
    print("\n" + "=" * 62)
    print("  SATTVA-GA v4.0 complete.")
    print("=" * 62)
