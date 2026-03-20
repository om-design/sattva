"""
SATTVA-GA v2.0 — Adversarial Repetition / Bias Stress Test
============================================================
Scenario:
  Phase 1  (steps   0–199): Normal training — balanced sensorimotor data.
  Phase 2  (steps 200–399): BIASED repetition — adversary hammers a single
                             attractor region 10x more than others, simulating
                             institutional dominance / echo-chamber dynamics.
  Phase 3  (steps 400–599): Recovery — balanced input resumes. Does the
                             invariant geometry hold? Do curiosity and shear
                             spike to signal the distortion? Does the system
                             self-correct via epiphany?

Key GA guarantee (Spec §XII):
  Repetition can only accumulate grade-0 scalar curvature (R_k).
  It cannot alter blade orientation, bivector topology, or rotor structure.
  → Well depths may skew, but primitives must remain stable.

Outputs:
  • bias_report.txt         — full text report
  • plot_curiosity_shear.png
  • plot_well_depths.png
  • plot_epiphany_primitive.png
"""

import sys, os
sys.path.insert(0, "/mnt/user-data/outputs")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from sattva_ga_v2 import (
    SattvaGA, SensorimotorSandbox, MultiVector
)

np.random.seed(7)

# ── Simulation parameters ─────────────────────────────────────────────────────
N_DIM        = 4
N_WELLS      = 3
N_PRIMITIVES = 3
PHASE_LEN    = 200       # steps per phase
BIAS_WELL    = 0         # which well the adversary floods
BIAS_RATIO   = 10        # how many extra times that well is hit
EPIPHANY_THR = 0.70      # merge threshold during stress test

PHASE_COLORS = {
    "Normal":   "#4CAF50",
    "Biased":   "#F44336",
    "Recovery": "#2196F3",
}

# ── Bootstrap engine ──────────────────────────────────────────────────────────
print("Bootstrapping SATTVA-GA …")
sandbox = SensorimotorSandbox()
engine  = SattvaGA(n_dim=N_DIM, n_wells=N_WELLS, n_primitives=N_PRIMITIVES)

data, mvs = engine.developmental_phase(sandbox,
    restitutions=(0.2, 0.5, 0.8),
    masses=(1.0, 3.0, 5.0))
engine.form_wells(mvs)

# snapshot primitives at start
def blade_orientations(icl):
    """Return list of grade-1 component vectors for all primitives."""
    return [p.to_vector().copy() for p in icl.primitives]

init_orientations = blade_orientations(engine.icl)

# ── Helpers ───────────────────────────────────────────────────────────────────

def cosine_drift(orig_vecs, current_icl):
    """Mean cosine similarity between initial and current grade-1 blades."""
    curr = blade_orientations(current_icl)
    sims = []
    for o, c in zip(orig_vecs, curr[:len(orig_vecs)]):
        no, nc = np.linalg.norm(o), np.linalg.norm(c)
        if no > 1e-12 and nc > 1e-12:
            sims.append(abs(np.dot(o, c) / (no * nc)))
    return float(np.mean(sims)) if sims else 1.0

def pick_normal(mvs):
    return mvs[np.random.randint(len(mvs))]

def pick_biased(mvs, engine, bias_well, bias_ratio):
    """
    Return a multivector close to the biased well's center.
    With probability bias_ratio/(bias_ratio+1) we return a
    perturbed version of the well center; otherwise random.
    """
    if np.random.rand() < bias_ratio / (bias_ratio + 1):
        # Small Gaussian noise around the biased well center
        center_vec = engine.awf.wells[bias_well].center.to_vector()
        noise = np.random.randn(N_DIM) * 0.5
        noisy = center_vec + noise
        return MultiVector.from_vector(N_DIM, noisy)
    return pick_normal(mvs)

# ── Records ───────────────────────────────────────────────────────────────────
steps        = []
curiosity    = []
shear        = []
well_depths  = {k: [] for k in range(N_WELLS)}
blade_drift  = []          # cosine similarity to initial primitives
epiphany_steps = []
n_primitives = []
phase_labels = []          # "Normal" / "Biased" / "Recovery"
e_pcl_log    = []
e_icl_log    = []

epiphany_well_counts = []  # number of wells after each epiphany

# ── Phase runner ──────────────────────────────────────────────────────────────

def run_phase(label, n_steps, input_fn):
    for _ in range(n_steps):
        X = input_fn()
        res = engine.process(X)

        t = res["step"]
        steps.append(t)
        phase_labels.append(label)
        curiosity.append(res["curiosity"])
        shear.append(res["shear"])
        e_pcl_log.append(res["e_pcl"])
        e_icl_log.append(res["e_icl"])
        n_primitives.append(res["n_primitives"])

        # Well depths (pad if wells were merged/added)
        for k in range(N_WELLS):
            depths = res["well_depths"]
            well_depths[k].append(depths[k] if k < len(depths) else 0.0)

        blade_drift.append(cosine_drift(init_orientations, engine.icl))

        if res["epiphany"]:
            epiphany_steps.append(t)
            epiphany_well_counts.append(len(engine.awf.wells))
            print(f"  [Step {t:4d}] EPIPHANY — wells now: {len(engine.awf.wells)}, "
                  f"primitives: {res['n_primitives']}")

# ── Run all three phases ──────────────────────────────────────────────────────
print("\n── Phase 1: Normal (balanced) ─────────────────────────────────────────")
run_phase("Normal",   PHASE_LEN, lambda: pick_normal(mvs))

print("\n── Phase 2: Biased repetition (adversarial) ───────────────────────────")
run_phase("Biased",   PHASE_LEN,
          lambda: pick_biased(mvs, engine, BIAS_WELL, BIAS_RATIO))

print("\n── Phase 3: Recovery (balanced restored) ──────────────────────────────")
run_phase("Recovery", PHASE_LEN, lambda: pick_normal(mvs))

# Also run explicit epiphany checks at phase boundaries
print("\n── Epiphany check at end of run ───────────────────────────────────────")
engine.awf.check_epiphany(engine.icl, merge_threshold=EPIPHANY_THR)

steps_arr     = np.array(steps)
curiosity_arr = np.array(curiosity)
shear_arr     = np.array(shear)
drift_arr     = np.array(blade_drift)

phase_boundaries = [PHASE_LEN, 2 * PHASE_LEN]

# ── Plot helpers ──────────────────────────────────────────────────────────────

def add_phase_bands(ax):
    """Shade background by phase."""
    ax.axvspan(0,            PHASE_LEN,     alpha=0.07, color=PHASE_COLORS["Normal"])
    ax.axvspan(PHASE_LEN,    2*PHASE_LEN,   alpha=0.07, color=PHASE_COLORS["Biased"])
    ax.axvspan(2*PHASE_LEN,  3*PHASE_LEN,   alpha=0.07, color=PHASE_COLORS["Recovery"])
    for b in phase_boundaries:
        ax.axvline(b, color="#888", lw=1.0, ls="--", alpha=0.7)
    for es in epiphany_steps:
        ax.axvline(es, color="gold", lw=1.2, ls=":", alpha=0.9)

def phase_legend():
    return [
        mpatches.Patch(color=PHASE_COLORS["Normal"],   alpha=0.4, label="Phase 1: Normal"),
        mpatches.Patch(color=PHASE_COLORS["Biased"],   alpha=0.4, label="Phase 2: Biased"),
        mpatches.Patch(color=PHASE_COLORS["Recovery"], alpha=0.4, label="Phase 3: Recovery"),
        mpatches.Patch(color="gold",                   alpha=0.8, label="Epiphany"),
    ]

OUT = "/mnt/user-data/outputs"

# ── Plot 1: Curiosity & Shear ─────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
fig.suptitle("SATTVA-GA v2.0 — Adversarial Bias Stress Test\nCuriosity & Shear Dynamics",
             fontsize=13, fontweight="bold")

# Rolling window smoothing
def smooth(arr, w=15):
    return np.convolve(arr, np.ones(w)/w, mode="same")

ax = axes[0]
add_phase_bands(ax)
ax.plot(steps_arr, curiosity_arr, color="#aaa", lw=0.6, alpha=0.5)
ax.plot(steps_arr, smooth(curiosity_arr), color="#9C27B0", lw=2.0, label="Curiosity C(X)")
ax.set_ylabel("Curiosity C(X)", fontsize=10)
ax.legend(handles=phase_legend() + ax.get_lines()[1:2][::-1], fontsize=8, loc="upper left")
ax.set_title("Curiosity  [H(probs) + η·Shear]", fontsize=10)

ax = axes[1]
add_phase_bands(ax)
ax.plot(steps_arr, shear_arr, color="#aaa", lw=0.6, alpha=0.5)
ax.plot(steps_arr, smooth(shear_arr), color="#FF5722", lw=2.0, label="Shear")
ax.set_ylabel("Shear", fontsize=10)
ax.set_title("Field Shear  [|E^ICL − E^RDL| + |E^ICL − E^RTL|]", fontsize=10)
ax.legend(fontsize=8, loc="upper left")

ax = axes[2]
add_phase_bands(ax)
ax.plot(steps_arr, drift_arr, color="#2196F3", lw=2.0, label="Blade orientation drift")
ax.axhline(1.0, color="#4CAF50", lw=1.0, ls="--", alpha=0.7, label="Perfect stability (1.0)")
ax.set_ylim(0, 1.1)
ax.set_ylabel("Cosine similarity\nto initial blades", fontsize=10)
ax.set_xlabel("Simulation step", fontsize=10)
ax.set_title("Invariant Blade Stability  (GA §XII guarantee)", fontsize=10)
ax.legend(fontsize=8, loc="lower left")

plt.tight_layout()
plt.savefig(f"{OUT}/plot_curiosity_shear.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plot_curiosity_shear.png")

# ── Plot 2: Well Depths ───────────────────────────────────────────────────────
well_colors = ["#E91E63", "#00BCD4", "#FF9800"]
fig, ax = plt.subplots(figsize=(12, 5))
ax.set_title("SATTVA-GA v2.0 — Attractor Well Depth Dynamics\nunder Adversarial Repetition",
             fontsize=12, fontweight="bold")
add_phase_bands(ax)
for k in range(N_WELLS):
    d = np.array(well_depths[k])
    ax.plot(steps_arr, d, color=well_colors[k % len(well_colors)],
            lw=1.8, label=f"Well {k}{' ← TARGETED' if k == BIAS_WELL else ''}")

ax.set_xlabel("Simulation step", fontsize=10)
ax.set_ylabel("Well depth D_Wk", fontsize=10)
handles = phase_legend() + [l for l in ax.get_lines() if not l.get_label().startswith("_")]
ax.legend(handles=handles, fontsize=8, loc="upper left")
plt.tight_layout()
plt.savefig(f"{OUT}/plot_well_depths.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plot_well_depths.png")

# ── Plot 3: Epiphany / Primitive count ────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
fig.suptitle("SATTVA-GA v2.0 — Epiphany Events & Primitive Blade Growth",
             fontsize=12, fontweight="bold")

add_phase_bands(ax1)
ax1.plot(steps_arr, np.array(n_primitives), color="#673AB7",
         lw=2.5, drawstyle="steps-post", label="# Invariant primitives")
for es in epiphany_steps:
    ax1.axvline(es, color="gold", lw=1.5, ls=":", alpha=0.9)
ax1.set_ylabel("Primitive count", fontsize=10)
ax1.set_title("Invariant Primitive Blades  (synthesized via W_A ∧ W_B on epiphany)", fontsize=10)
ax1.legend(handles=phase_legend(), fontsize=8, loc="upper left")

add_phase_bands(ax2)
ax2.plot(steps_arr, smooth(np.array(e_pcl_log), 20),
         color="#F44336", lw=2.0, label="E^PCL (rotor prediction error, smoothed)")
ax2.plot(steps_arr, smooth(np.array(e_icl_log), 20),
         color="#009688", lw=2.0, label="E^ICL (invariant energy, smoothed)")
ax2.set_xlabel("Simulation step", fontsize=10)
ax2.set_ylabel("Energy", fontsize=10)
ax2.set_title("Prediction vs Invariant Energy  (tension drives epiphany)", fontsize=10)
ax2.legend(fontsize=8, loc="upper left")

plt.tight_layout()
plt.savefig(f"{OUT}/plot_epiphany_primitive.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plot_epiphany_primitive.png")

# ── Text Report ───────────────────────────────────────────────────────────────
def phase_mask(label):
    return np.array([l == label for l in phase_labels])

def phase_stats(arr, label):
    m = phase_mask(label)
    v = arr[m]
    return v.mean(), v.std(), v.min(), v.max()

def fmt(label, arr):
    mn, sd, lo, hi = phase_stats(arr, label)
    return f"  mean={mn:.3f}  std={sd:.3f}  min={lo:.3f}  max={hi:.3f}"

lines = []
lines.append("=" * 65)
lines.append("  SATTVA-GA v2.0 — Adversarial Repetition Bias Stress Test")
lines.append("  Simulation Report")
lines.append("=" * 65)
lines.append("")
lines.append("SCENARIO")
lines.append(f"  Phase 1  [steps   0–{PHASE_LEN-1:3d}]  Normal balanced input")
lines.append(f"  Phase 2  [steps {PHASE_LEN:3d}–{2*PHASE_LEN-1:3d}]  Adversarial: Well {BIAS_WELL} flooded {BIAS_RATIO}x")
lines.append(f"  Phase 3  [steps {2*PHASE_LEN:3d}–{3*PHASE_LEN-1:3d}]  Recovery: balanced input restored")
lines.append("")
lines.append("GA BIAS-AGNOSTIC GUARANTEE (Spec §XII)")
lines.append(f"  Initial blade orientations recorded at step 0.")
lines.append(f"  Cosine similarity to initial blades (1.0 = no drift):")
for ph in ("Normal", "Biased", "Recovery"):
    mn, sd, lo, hi = phase_stats(drift_arr, ph)
    lines.append(f"    {ph:<10}: mean={mn:.5f}  min={lo:.5f}  max={hi:.5f}")
lines.append("")
lines.append("  → Repetition cannot rotate grade-1 blades.")
lines.append("    Any drift near 1.0 confirms the GA invariance guarantee holds.")
lines.append("")
lines.append("CURIOSITY  [C = H(probs) + η·Shear]")
for ph in ("Normal", "Biased", "Recovery"):
    lines.append(f"  {ph:<10}:{fmt(ph, curiosity_arr)}")
lines.append("")
lines.append("SHEAR  [|E^ICL − E^RDL| + |E^ICL − E^RTL|]")
for ph in ("Normal", "Biased", "Recovery"):
    lines.append(f"  {ph:<10}:{fmt(ph, shear_arr)}")
lines.append("")
lines.append("WELL DEPTHS at phase ends")
for k in range(N_WELLS):
    d = np.array(well_depths[k])
    tag = " ← TARGETED" if k == BIAS_WELL else ""
    lines.append(f"  Well {k}{tag}:")
    for ph, sl in [("Normal end", slice(PHASE_LEN-10, PHASE_LEN)),
                   ("Biased end", slice(2*PHASE_LEN-10, 2*PHASE_LEN)),
                   ("Recovery end", slice(3*PHASE_LEN-10, 3*PHASE_LEN))]:
        lines.append(f"    {ph:<14}: {d[sl].mean():.4f}")
lines.append("")
lines.append("EPIPHANY EVENTS")
if epiphany_steps:
    for es, wc in zip(epiphany_steps, epiphany_well_counts):
        ph = phase_labels[steps.index(es)]
        lines.append(f"  Step {es:4d}  [{ph}]  wells remaining: {wc}")
else:
    lines.append("  None triggered during online processing.")
lines.append(f"  Total invariant primitives at end: {len(engine.icl.primitives)}")
lines.append("")
lines.append("ROTOR PREDICTION ERROR (E^PCL)")
for ph in ("Normal", "Biased", "Recovery"):
    lines.append(f"  {ph:<10}:{fmt(ph, np.array(e_pcl_log))}")
lines.append("")
lines.append("INTERPRETATION")
lines.append("  • Curiosity and shear are expected to SPIKE during Phase 2,")
lines.append("    signalling that repetition pressure is creating cross-layer tension.")
lines.append("  • Blade orientation cosine should remain ≈1.0 throughout,")
lines.append("    proving that grade-0 scalar accumulation cannot bend higher-grade")
lines.append("    invariant geometry (the core GA guarantee of §XII).")
lines.append("  • Well 0 depth should grow disproportionately in Phase 2,")
lines.append("    while curiosity flags the distortion rather than concealing it.")
lines.append("  • On recovery, shear should fall as the field re-balances.")
lines.append("  • Epiphany (if triggered) synthesises a higher-grade blade from")
lines.append("    the two most-similar wells, upgrading representational capacity.")
lines.append("")
lines.append("=" * 65)

report = "\n".join(lines)
print("\n" + report)

with open(f"{OUT}/bias_report.txt", "w") as f:
    f.write(report)
print("\nSaved bias_report.txt")
