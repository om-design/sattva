# SATTVA-GA v3.0

## Semantic Attractor Training of Transforming Vector Associations

### Multivector Geometric Algebra Formulation — Corrected Specification

**Changes from v2.0:** Sections II, V, VI, X, XIV, and XV are materially
revised. A new Section XVI documents the rotor predictor's scope and
limitations. All corrections correspond to verified code fixes in
`sattva_ga_v3.py`.

---

# I. Algebraic Foundation

*(Unchanged from v2.0)*

Let G(p,q) be a real geometric algebra over R^{p,q}.

For semantic purposes, begin with Euclidean signature G(n,0).

Basis vectors: {e_1, e_2, ..., e_n}

Geometric product:

    ab = a·b + a∧b

where a·b is the inner product (metric structure) and a∧b is the outer
product (subspace construction).

Every semantic object is a multivector:

    M = α + Σ_i v_i e_i + Σ_{i<j} B_{ij} (e_i∧e_j) + ...

Grades:
- Grade-0: scalar (intensity / repetition density)
- Grade-1: vector (primitive direction)
- Grade-2: bivector (relational plane)
- Grade-k: k-blade (higher-order invariant structure)

---

# II. Primitive Invariant Layer (ICL) — REVISED

Invariant primitives are stable blades admitted through a stability gate.

**Extraction protocol:**

1. Divide the training data into T non-overlapping time windows.
2. Compute the top-k eigenvectors of the covariance matrix in each window.
3. For each candidate direction from window 1, find its closest match in
   each subsequent window and record the cosine similarity.
4. The stability score is the mean cross-window cosine similarity:

       Stability(P_i) = (1/T-1) Σ_{t=2}^{T} |P_i^{(t)} · P_i^{(1)}| / (|P_i^{(t)}| |P_i^{(1)}|)

5. **Only candidates with Stability(P_i) > θ_stable are admitted to the
   invariant layer.** Transient directions (e.g. columns that are constant
   within trajectories) are correctly rejected.

*v2.0 used the v1 formula ⟨P_i^t P̃_i^{t-1}⟩_0 / (|P_i^t||P_i^{t-1}|)
across pre-computed window lists, but the stability gate was never enforced
in the extraction code — all top-k candidates were admitted unconditionally.
v3.0 enforces the gate as part of extraction.*

Higher-grade blades (grade-2 and above) are promoted to the ICL by:
- Stage 2 bivector emergence (Section XIV)
- Epiphany well-merge synthesis (Section XI)

---

# III. Semantic State as Multivector Field

*(Unchanged from v2.0)*

At time t: X_t ∈ G(n,0)

Projection onto invariant primitive (applies to all grades):

    c_i = ⟨X_t P̃_i⟩_0

Invariant energy:

    E^ICL(X_t) = Σ_i c_i²

---

# IV. Repetition Density as Scalar Curvature

*(Unchanged from v2.0)*

Repetition accumulates only as grade-0 scalar curvature:

    R_k(t+1) = R_k(t) + ρ

Repetition gradient is a scalar field:

    ∇^RDL = ∇ log(R_k + 1) = 1/(R_k + 1)

Repetition contributes only to the grade-0 component. It cannot alter
higher-grade invariant blades. This is a structural, not incidental,
property enforced by grade arithmetic.

---

# V. Relational Topology as Bivector Structure — REVISED

Network topology accumulates as a weighted bivector sum:

    T = Σ_{i<j} w_{ij} (e_i ∧ e_j)

Edge weights grow from co-activation and are capped at 1.0 per edge:

    w_{ij}(t+1) = min(w_{ij}(t) + δ · ĉ_i · ĉ_j, 1.0)

where ĉ_i = c_i / max_k(|c_k|) are normalised projection coefficients
and δ is a small learning rate.

**RTL Energy — CORRECTED:**

v2.0 defined E^RTL = ⟨X_t T̃⟩_0. This quantity is zero by grade arithmetic
for pure grade-1 states X_t and grade-2 bivector T, because the geometric
product of grade-1 and grade-2 elements contains only grades 1 and 3 — never
grade-0. The v2 formula therefore produced no meaningful signal.

The correct formulation is the Frobenius co-activation resonance:

    E^RTL(X_t) = Σ_{i<j} w_{ij} · c_i · c_j

This measures how much the current activation pattern resonates with the
established relational topology. It is always well-defined for grade-1
states and grows when two related primitives are simultaneously active.

The bivector T is used for topological visualisation and as the parent
structure for epiphany synthesis; it is not used directly in the energy
computation.

---

# VI. Predictive Consistency in GA — REVISED

Prediction operator as rotor R (sandwich product):

    X_{t+1}^{pred} = R X_t R̃

Prediction error:

    E^PCL = |X_{t+1} - X_{t+1}^{pred}|²

**Rotor learning rule — CORRECTED:**

    R_new = exp(−η · B̂_{curr→next}) · R

where B̂_{curr→next} is the UNIT bivector of the wedge product:

    B̂_{curr→next} = (X_t ∧ X_{t+1}) / |X_t ∧ X_{t+1}|

and η is the rotation-angle step size (in radians).

**Two corrections from v2.0:**

1. Operand order: v2 computed X_{t+1} ∧ X_t (reversed), which equals
   −(X_t ∧ X_{t+1}), producing a double-negation with the −η factor and
   reversing the learning direction. Corrected to X_t ∧ X_{t+1}.

2. Normalisation: grade-1 state vectors in typical domains have norm >> 1.
   The raw wedge X_t ∧ X_{t+1} therefore has large norm, making −η·(raw wedge)
   a large bivector that causes wild over-rotation per step. Using the unit
   bivector direction means η is literally the rotation angle per update step,
   which is the geometrically correct interpretation of a learning rate.

**Rotor exponential — CORRECTED:**

The v2 formula R = cos(|B|) + sin(|B|)/|B| · B is only valid for SIMPLE
bivectors (a single e_i∧e_j plane). The wedge of two grade-1 vectors in
G(n,0) with n≥3 produces a compound bivector — a sum of multiple basis
planes. For compound bivectors, the correct exponential is the power series:

    exp(B) = Σ_{k=0}^{∞} B^k / k!

which converges for all finite B and produces a genuine unit rotor R R̃ = 1.
Truncating at 16 terms gives |R R̃ − 1| < 10^{-8} for |B| ≤ 10.

---

# VII. Primitive Depth Evolution

*(Unchanged from v2.0)*

Each primitive P_i has depth D_i:

    D_i(t+1) = D_i(t) + α_eff · ⟨X_t P̃_i⟩_0² − β · E^PCL

where α_eff is the curiosity-modulated effective learning rate (Section X).

Depth affects influence weight in projections:

    c_i^{weighted} = D_i · ⟨X_t P̃_i⟩_0

Deep primitives dominate geometric curvature.

---

# VIII. Attractor Wells in Multivector Space — PARTIALLY REVISED

Each well W_k is a multivector center: W_k ∈ G(n,0)

Distance metric (unchanged):

    E_k(X) = ⟨(X − W_k)(X̃ − W̃_k)⟩_0

Assignment probability (unchanged):

    P(W_k|X) = exp(−E_k(X)) / Σ_j exp(−E_j(X))

**Well depth evolution — CORRECTED:**

    D_Wk(t+1) = D_Wk(t) + γ R_k − δ Shear

where R_k is the accumulated repetition count from Section IV.

v2.0 used the momentary assignment probability P(W_k|X) in place of R_k.
These are related but distinct: R_k grows monotonically with visitation
history whereas P(W_k|X) ∈ [0,1] reflects only the current state. Using
R_k ensures that wells which were heavily visited in the past retain deep
consolidation, matching the spec's intent that repetition history deepens
semantic attractors.

---

# IX. Field Shear (Cross-Layer Tension)

*(Unchanged from v2.0)*

Shear defined as multigrade conflict:

    Shear(X) = |E^ICL − E^RDL| + |E^ICL − E^RTL|

High repetition scalar with low invariant projection → shear.
This drives curiosity.

---

# X. Curiosity Functional — REVISED

    C(X) = H(P(W_k|X)) + η · Shear(X)

where entropy:

    H = −Σ_k P(W_k|X) log P(W_k|X)

**Effective learning rate — CORRECTED:**

    α_eff = α (1 − σ(C/τ))

where σ is the logistic sigmoid and τ is a temperature parameter.

v2.0 defined α_eff = α (1 − σ(C)) without temperature scaling. Curiosity
values in typical environments range from 5 to 300+, at which scale σ(C)
saturates to ≈ 1.0, making α_eff ≈ 0 always and freezing depth dynamics.
Introducing τ ≈ 50 (calibrated to the energy scale of the environment)
keeps σ(C/τ) in the range [0.3, 0.95], preserving the intended relationship:

    High curiosity → α_eff → 0  (slow consolidation, increased exploration)
    Low curiosity  → α_eff → α  (fast consolidation)

τ is an environment-specific hyperparameter and should be re-tuned when
SATTVA-GA is applied to a new domain.

---

# XI. Epiphany (Well Merge) in GA — REVISED

Two wells W_A, W_B. Define overlap:

    S = ⟨W_A W̃_B⟩_0 / (|W_A| |W_B|)

**Two epiphany triggers (either is sufficient):**

Trigger A — Geometric similarity:
    S > θ_cosine  AND  D_WA > 1  AND  D_WB > 1

Trigger B — Depth convergence:
    S > 0.5  AND  min(D_WA, D_WB) / max(D_WA, D_WB) > θ_depth
    AND  D_WA > 2  AND  D_WB > 2

On either trigger, create a higher-grade invariant primitive:

    P_new = Normalise(W_A ∧ W_B)

Add P_new to the invariant layer. Merge the two wells into one.

**Rationale for Trigger B:** The spec's original intent was that epiphany
occurs when two *concepts have become semantically unified* through
experience. Trigger A detects geometric proximity; Trigger B detects
convergent developmental history — two wells that have accumulated similar
depth through similar visitation patterns, suggesting they serve equivalent
roles in the semantic field. Both are valid signals of conceptual merger.

---

# XII. Bias-Agnostic Guarantee (GA Formal Statement)

*(Unchanged from v2.0)*

Repetition modifies only grade-0 scalar curvature (R_k counts).
Invariant constraints exist in higher-grade blades (grade ≥ 1).
Scalar fields commute with all multivectors but cannot alter their
grade structure.

Therefore repetition cannot:
- Alter blade orientation
- Change bivector topology
- Modify rotor structure

Institutional dominance (disproportionate repetition of one class of input)
can deepen one attractor well relative to others, but it cannot rotate the
invariant primitives that define the geometric meaning of each well. The
distortion is *detectable* as a shear signal and as well-depth asymmetry —
it does not corrupt the constraint manifold silently.

---

# XIII. Truth as Energy Minimum

*(Unchanged from v2.0)*

Total field energy:

    ε(X) = E^ICL + E^PCL + λ · Shear − μ · D_Wk

Stable semantic structures minimise:

    ∇ε = 0

Low internal contradiction naturally stabilises.

---

# XIV. Developmental Sequence — REVISED

The four stages are now fully implemented:

**Stage 1 — Grade-1 primitive extraction (early stage)**

Eigenvectors are extracted per data window and stability-gated. Only
directions that remain stable across time windows (Stability > θ) are
admitted to the invariant layer. This implements "grade-1 blades from
covariance" correctly: the ICL is not merely the top PCA directions of
pooled data, but the genuinely invariant directions across the developmental
experience.

**Stage 2 — Bivector emergence (middle stage)**

Multiple passes over training data accumulate RTL co-activation edge weights.
Stable bivectors (w_{ij} > θ_bv) are promoted as grade-2 primitives to the
ICL. This stage must complete before Stage 3 begins, so that relational
structure is encoded in the constraint manifold before the rotor is fitted.

This implements the spec's intended middle developmental stage: simple
geometric directions earn their relational structure through experience.
It is the mechanism most directly analogous to fMRI findings of cross-domain
neural co-activation, where primitive representations in different cortical
areas become structurally linked by co-occurrence, not by direct connection.

**Stage 3 — Rotor predictor fitting (mature stage)**

The rotor is initialised and trained on the full data trajectory using the
corrected update rule (Section VI). The rotor now operates over a richer ICL
that includes both grade-1 and grade-2 primitives.

**Stage 4 — Attractor well formation (creative stage begins)**

Wells are seeded in ICL coefficient space via K-means. Well centers are
reconstructed as multivectors in the full GA space.

**Online processing**

Per-step: project onto ICL, compute all energies, update RDL, compute
shear and curiosity, update α_eff, update ICL depths and well depths,
check for epiphany every 50 steps.

---

# XV. Implementation Path

*(Updated to reflect v3.0 status)*

The self-contained GA engine in `sattva_ga_v3.py` provides:
- MultiVector class with sparse blade representation
- Full geometric product, wedge, reverse, grade extraction
- Power-series rotor exponential (handles compound bivectors)
- Sandwich product with periodic re-normalisation
- All four layers: ICL, RDL, RTL, PCL
- AttractorField with dual-trigger epiphany
- CuriosityFunctional with temperature-scaled α_eff
- Four-stage developmental bootstrap
- 12-test diagnostic suite

When `clifford` or another GA library becomes available, the MultiVector
class and helper functions can be replaced with a thin wrapper; all
higher-level architecture remains unchanged.

For scaling to higher-dimensional semantic spaces (n > 20):
- Sparse GA representation is already implemented (no dense 2^n arrays)
- Geometric product cost is O(|blades_A| × |blades_B|), manageable for
  sparse multivectors
- Grade-k primitives with k > 2 will require pruning low-coefficient blades
  to prevent exponential expansion

---

# XVI. Rotor Predictor Scope and Limitations (NEW)

The rotor predictor encodes transformation geometry. A rotor in G(n,0) is
a norm-preserving operator: it can represent rotations and reflections but
not general affine transformations (translations, scalings, shears).

**Implication for the bouncing-ball sandbox:**

The sandbox state transition [y, v, m, r] → [y + v·dt, v − g·dt, m, r]
is an affine map, not a rotation. The height y decreases from 10 to 0 (a
change in norm), and velocity changes sign at bounce events. A rotor cannot
reduce E^PCL to near-zero for this data; it will remain elevated.

This is *correct and meaningful* behaviour: E^PCL signals that the
trajectory is not geometrically consistent with a pure rotation. In the
energy landscape (Section XIII), high E^PCL contributes to shear and
curiosity, driving the system to seek a richer representation.

**Domains suited to the rotor predictor:**
- Joint-angle kinematics (rotations in configuration space)
- Phase-space orbits (e.g. pendulum, oscillator)
- Sequential cyclic patterns (phonology, rhythm)
- Transformations between views of a 3D scene

For affine or non-linear dynamics, the rotor predictor provides a
projection of the dynamics onto its nearest rotational equivalent —
useful as one component of a multi-layer representation, but not as a
complete dynamics model.

**Compound rotors:**

After many update steps, R accumulates contributions from multiple
rotation planes and becomes a compound rotor. This is algebraically valid
in G(n,0) and the sandwich product remains norm-preserving. Periodic
re-normalisation (R R̃ = 1) prevents floating-point drift.

---
