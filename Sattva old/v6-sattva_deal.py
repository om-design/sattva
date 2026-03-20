# sattva_deal.py
#
# SATTVA-Deal v1  —  Structural Attractor Training for Typed Venture Analysis
#
# Implements the v6-SPEC exactly:
#   §2  Event representation with typed temporal event graphs
#   §3  Geometric construction: relationship vectors, motif extraction
#   §4  Primitive scoring: Lift × Stability − λ·Complexity
#   §5  Matching function: exp(−α · structural_distance)
#   §6  Interpretable escalation: every score tied to activated motifs
#
# Fixes carried forward from MVP audit:
#   FIX-1  Timing incorporated into primitive signature (was extracted, discarded)
#   FIX-2  Labels drive primitive selection via Lift, not raw frequency
#   FIX-3  Score is calibrated; escalation threshold learned from validation set
#
# No dependency on GA engine (correctly: spec §6 says explicit relational
# geometry, not multivector arithmetic).  Requires numpy, scipy only.

import numpy as np
from itertools import combinations
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import warnings


# ============================================================
#  I.  Event Representation  (Spec §2)
# ============================================================

@dataclass
class Event:
    """
    A single typed event in a deal stream.

    vector   : embedding of the event content (e.g. from an LLM encoder).
               Geometry lives in RELATIONAL constraints between vectors,
               not in the magnitude of any single vector (Spec §2).
    t        : timestamp (any consistent unit — days, unix epoch, etc.)
    etype    : semantic type label  (e.g. "intro", "pilot", "signed")
    """
    eid:    str
    vector: np.ndarray
    t:      float
    etype:  str


@dataclass
class DealStream:
    """
    A typed temporal event graph for one deal.

    label : 1 = successful deal, 0 = unsuccessful.
            Required for training; optional at inference.
    """
    did:    str
    events: List[Event]
    label:  Optional[int] = None   # 1 = success, 0 = failure

    def sorted_events(self) -> List[Event]:
        return sorted(self.events, key=lambda e: e.t)


# ============================================================
#  II.  Geometric Construction  (Spec §3)
# ============================================================

# Time-ratio bins for the normalized_time component of a signature.
# Coarse enough to generalise across deals, fine enough to discriminate
# fast closings (ratio < 0.2) from slow ones (ratio > 0.8).
_TIME_BINS = [0.0, 0.20, 0.40, 0.60, 0.80, 1.01]

def _bin_time(ratio: float) -> int:
    for i in range(len(_TIME_BINS) - 1):
        if ratio < _TIME_BINS[i + 1]:
            return i
    return len(_TIME_BINS) - 2


def _magnitude_ratio(v1: np.ndarray, v2: np.ndarray) -> int:
    """
    Relative magnitude ratio of two relationship vectors, binned into
    3 coarse categories: v1 dominates (0), comparable (1), v2 dominates (2).
    This captures whether the semantic 'distance' between events grows or
    shrinks across the motif — a structural constraint invariant to scale.
    """
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 and n2 < 1e-9:
        return 1
    ratio = n1 / (n1 + n2 + 1e-9)
    if ratio < 0.35:
        return 0   # v1 much smaller — semantic gap widening
    elif ratio > 0.65:
        return 2   # v1 much larger — semantic gap narrowing
    return 1       # comparable


def extract_motifs(stream: DealStream) -> List[Tuple]:
    """
    Extract all 3-event motifs from a deal stream.  (Spec §3)

    Each motif is a hashable tuple:
        (type_pattern, time_bin_pair, magnitude_bin)

    type_pattern   : (etype_1, etype_2, etype_3)  — ordered by time
    time_bin_pair  : (bin(dt1 / total_dt), bin(dt2 / total_dt))
                     FIX-1: timing now part of the primitive signature
    magnitude_bin  : coarse ratio of relationship-vector norms
                     captures whether semantic gap grows or shrinks

    Events are sorted by timestamp before enumeration.
    """
    evs = stream.sorted_events()
    motifs = []

    for e1, e2, e3 in combinations(evs, 3):
        dt1 = e2.t - e1.t
        dt2 = e3.t - e2.t
        total_dt = dt1 + dt2 + 1e-9

        if dt1 < 0 or dt2 < 0:
            # Reject out-of-order triplets (shouldn't occur after sorting)
            continue

        type_pattern = (e1.etype, e2.etype, e3.etype)

        time_bin_pair = (
            _bin_time(dt1 / total_dt),
            _bin_time(dt2 / total_dt),
        )

        # Relationship vectors (Spec §3 r_ij = v_j − v_i)
        r12 = e2.vector - e1.vector
        r23 = e3.vector - e2.vector
        magnitude_bin = _magnitude_ratio(r12, r23)

        motifs.append((type_pattern, time_bin_pair, magnitude_bin))

    return motifs


# ============================================================
#  III.  Primitive Learning  (Spec §4)
# ============================================================

@dataclass
class Primitive:
    """
    A structural invariant extracted from labeled deal history.

    signature   : hashable motif key (type_pattern, time_bin_pair, mag_bin)
    support     : number of deals in which this motif appeared
    lift        : P(success | motif) − P(success)  — signed discrimination
    stability   : 1 / Var(normalized_time ratios across occurrences)
                  high stability = motif appears with consistent timing
    complexity  : log2(number of distinct type values in pattern)
                  penalizes rare, over-specific motifs
    score       : Lift × Stability − λ × Complexity  (Spec §4)
    weight      : softmax-normalised score across the primitive bank
                  used in the matching sum (Spec §5)
    """
    signature:  Tuple
    support:    int     = 0
    lift:       float   = 0.0
    stability:  float   = 0.0
    complexity: float   = 0.0
    score:      float   = 0.0
    weight:     float   = 0.0


class MotifLearner:
    """
    Learns the primitive bank from labeled deal streams.  (Spec §4)

    FIX-2: Labels are required.  Frequency threshold alone (MVP) retains
    failure-predictive patterns.  Only motifs with positive Lift are
    candidates for the primitive bank.

    Training procedure:
      1. Extract motifs from all deals.
      2. For each motif, count occurrences in successful vs all deals.
      3. Compute Lift, Stability, Complexity.
      4. Retain motifs with Lift > 0 and support ≥ min_support.
      5. Score each retained primitive; weight by softmax of score.
    """

    def __init__(self, min_support: int = 3,
                 min_lift: float = 0.05,
                 lambda_complexity: float = 0.1,
                 max_primitives: int = 50):
        self.min_support       = min_support
        self.min_lift          = min_lift
        self.lambda_complexity = lambda_complexity
        self.max_primitives    = max_primitives
        self.primitives: List[Primitive] = []
        self._base_rate: float = 0.0

    def fit(self, streams: List[DealStream]) -> "MotifLearner":
        labeled = [s for s in streams if s.label is not None]
        if not labeled:
            raise ValueError("MotifLearner.fit() requires labeled DealStreams.")

        self._base_rate = float(np.mean([s.label for s in labeled]))

        # Per-motif accumulators
        total_count:   Dict[Tuple, int]   = defaultdict(int)
        success_count: Dict[Tuple, int]   = defaultdict(int)
        # For stability: track normalized time ratios per occurrence
        time_ratios:   Dict[Tuple, List]  = defaultdict(list)

        for stream in labeled:
            seen_in_deal = set()
            for motif in extract_motifs(stream):
                sig = motif                        # full signature
                type_pat, time_bin, mag_bin = sig

                if sig not in seen_in_deal:        # count per-deal, not per-motif
                    total_count[sig]   += 1
                    success_count[sig] += stream.label
                    seen_in_deal.add(sig)

                # Collect raw time ratios for stability (using bin index as proxy)
                time_ratios[sig].append(time_bin[0])  # first ratio is informative

        # Build candidates
        candidates: List[Primitive] = []
        for sig, n_deals in total_count.items():
            if n_deals < self.min_support:
                continue

            p_success_given_motif = success_count[sig] / n_deals
            lift      = p_success_given_motif - self._base_rate

            if lift <= self.min_lift:
                continue   # FIX-2: reject failure-predictive and neutral motifs

            # Stability: proportion of occurrences sharing the modal time bin.
            # Ranges [0, 1].  1.0 = all occurrences have identical timing profile.
            # (Reciprocal-variance would diverge to 1000+ when variance is near 0;
            # modal-agreement is bounded and interpretable.)
            ratios = time_ratios[sig]
            if len(ratios) <= 1:
                stability = 1.0
            else:
                bins, bin_counts = np.unique(ratios, return_counts=True)
                modal_freq = int(bin_counts.max())
                stability  = modal_freq / len(ratios)

            # Complexity: penalise motifs that rely on many rare type labels
            type_pat = sig[0]
            unique_types = len(set(type_pat))
            complexity = np.log2(unique_types + 1)

            prim = Primitive(
                signature  = sig,
                support    = n_deals,
                lift       = lift,
                stability  = stability,
                complexity = complexity,
                score      = lift * stability - self.lambda_complexity * complexity,
            )
            candidates.append(prim)

        if not candidates:
            warnings.warn("No primitives met the Lift and support thresholds. "
                          "Consider lowering min_lift or min_support.")
            self.primitives = []
            return self

        # Rank by score and keep top-N
        candidates.sort(key=lambda p: -p.score)
        retained = candidates[:self.max_primitives]

        # Normalise weights by softmax of score
        scores = np.array([p.score for p in retained])
        scores -= scores.max()  # numerically stable
        weights = np.exp(scores)
        weights /= weights.sum()
        for p, w in zip(retained, weights):
            p.weight = float(w)

        self.primitives = retained
        return self

    def primitive_report(self) -> str:
        bin_labels = ["very fast", "fast", "mid", "slow", "very slow"]
        lines = [f"Primitive bank: {len(self.primitives)} primitives  "
                 f"(base rate={self._base_rate:.3f})"]
        for i, p in enumerate(self.primitives[:10]):
            tp, tb, mb = p.signature
            t_desc = f"{bin_labels[tb[0]]}→{bin_labels[tb[1]]}"
            m_desc = ["narrowing", "comparable", "widening"][mb]
            human  = f"{' → '.join(tp)}  timing:{t_desc}  gap:{m_desc}"
            lines.append(
                f"  [{i:02d}] lift={p.lift:+.3f}  stab={p.stability:.2f}  "
                f"score={p.score:.3f}  w={p.weight:.4f}  n={p.support}"
            )
            lines.append(f"        {human}")
        if len(self.primitives) > 10:
            lines.append(f"  … ({len(self.primitives)} total)")
        return "\n".join(lines)


# ============================================================
#  IV.  Matching + Scoring  (Spec §5)
# ============================================================

def structural_distance(motif: Tuple, primitive: Primitive) -> float:
    """
    Distance between an observed motif and a learned primitive.  (Spec §5)

    Three components:
      type_distance    : 0 if type pattern matches exactly, 1 otherwise.
                         Type patterns must match — this is the categorical core.
      time_distance    : L1 distance between time bin pairs, normalised to [0, 1].
      magnitude_distance: 0 or 1 depending on magnitude bin match.

    Total distance is a weighted sum.  Type match is gating:
    if types differ, the primitive is irrelevant regardless of timing.
    """
    sig = primitive.signature
    type_pat_obs,  tb_obs,  mb_obs  = motif
    type_pat_prim, tb_prim, mb_prim = sig

    if type_pat_obs != type_pat_prim:
        return 1.0   # gating: wrong type pattern = maximum distance

    n_bins   = len(_TIME_BINS) - 1
    t_dist   = (abs(tb_obs[0] - tb_prim[0]) +
                abs(tb_obs[1] - tb_prim[1])) / (2 * n_bins)
    m_dist   = 0.0 if mb_obs == mb_prim else 1.0

    return 0.6 * t_dist + 0.4 * m_dist   # timing weighted more than magnitude


class StreamScorer:
    """
    Scores an incoming deal stream against the learned primitive bank.  (Spec §5)

    Score(S) = Σ_p  w_p · exp(−α · structural_distance(S, p))

    FIX-3: Score is calibrated against a validation set.
    Escalation threshold is learned, not hand-set.

    activated_motifs() provides full interpretability:
    every escalation can be explained by the specific (motif, primitive) pairs
    that contributed, satisfying Spec §6.
    """

    def __init__(self, primitives: List[Primitive], alpha: float = 5.0):
        self.primitives  = primitives
        self.alpha       = alpha
        self._threshold: Optional[float] = None

    def score(self, stream: DealStream) -> float:
        """Raw score: Σ_p w_p · exp(−α · dist(S, p))"""
        motifs = extract_motifs(stream)
        total  = 0.0
        for p in self.primitives:
            # Take the minimum distance across all observed motifs
            # (best-match semantics: does this primitive appear ANYWHERE in the deal?)
            dists = [structural_distance(m, p) for m in motifs] if motifs else [1.0]
            d_min = min(dists)
            total += p.weight * np.exp(-self.alpha * d_min)
        return total

    def activated_motifs(self, stream: DealStream,
                         top_n: int = 5) -> List[Dict]:
        """
        Return the top-N (primitive, matched_motif, contribution) triples.

        This is the interpretability layer (Spec §6):
        every escalation decision is traceable to specific activated primitives
        and the exact motif in the deal that triggered each one.
        """
        motifs = extract_motifs(stream)
        activations = []
        for p in self.primitives:
            if not motifs:
                continue
            dists        = [(structural_distance(m, p), m) for m in motifs]
            d_min, best  = min(dists, key=lambda x: x[0])
            contribution = p.weight * np.exp(-self.alpha * d_min)
            activations.append({
                "primitive":    p.signature,
                "matched_motif": best,
                "distance":     d_min,
                "contribution": contribution,
                "lift":         p.lift,
            })

        activations.sort(key=lambda x: -x["contribution"])
        return activations[:top_n]

    def calibrate(self, streams: List[DealStream],
                  target_precision: float = 0.70) -> "StreamScorer":
        """
        Learn the escalation threshold from a labeled validation set.  (Spec §5)

        Sweeps score thresholds and selects the lowest threshold that achieves
        target_precision among escalated deals.  This ensures that when the
        system fires, it is right at least target_precision of the time.

        FIX-3: replaces hand-set threshold from MVP.
        """
        labeled = [s for s in streams if s.label is not None]
        if not labeled:
            raise ValueError("calibrate() requires labeled DealStreams.")

        scores = [(self.score(s), s.label) for s in labeled]
        scores.sort(key=lambda x: x[0])

        best_threshold = scores[-1][0]  # fallback: escalate nothing

        all_scores = np.array([sc for sc, _ in scores])
        thresholds = np.unique(all_scores)

        for thresh in thresholds:
            escalated = [(sc, lbl) for sc, lbl in scores if sc >= thresh]
            if len(escalated) == 0:
                continue
            precision = np.mean([lbl for _, lbl in escalated])
            if precision >= target_precision:
                best_threshold = thresh
                break

        self._threshold = float(best_threshold)
        return self

    def escalate(self, stream: DealStream) -> bool:
        """
        Returns True if the deal should be escalated to LLM review.
        Requires prior calibrate() call.
        """
        if self._threshold is None:
            raise RuntimeError("Call calibrate() before escalate().")
        return self.score(stream) >= self._threshold

    @property
    def threshold(self) -> Optional[float]:
        return self._threshold


# ============================================================
#  V.  Full Pipeline  (convenience wrapper)
# ============================================================

class SATTVADeal:
    """
    End-to-end pipeline.

    Usage:
        model = SATTVADeal()
        model.fit(train_streams, val_streams)
        for stream in new_streams:
            if model.should_escalate(stream):
                explanation = model.explain(stream)
                send_to_llm(stream, explanation)
    """

    def __init__(self,
                 min_support:        int   = 3,
                 min_lift:           float = 0.05,
                 lambda_complexity:  float = 0.1,
                 max_primitives:     int   = 50,
                 alpha:              float = 5.0,
                 target_precision:   float = 0.70):
        self.learner = MotifLearner(
            min_support      = min_support,
            min_lift         = min_lift,
            lambda_complexity= lambda_complexity,
            max_primitives   = max_primitives,
        )
        self._alpha    = alpha
        self._prec     = target_precision
        self.scorer: Optional[StreamScorer] = None

    def fit(self, train_streams: List[DealStream],
            val_streams:   List[DealStream]) -> "SATTVADeal":
        self.learner.fit(train_streams)
        self.scorer = StreamScorer(self.learner.primitives, alpha=self._alpha)
        self.scorer.calibrate(val_streams, target_precision=self._prec)
        return self

    def should_escalate(self, stream: DealStream) -> bool:
        return self.scorer.escalate(stream)

    def score(self, stream: DealStream) -> float:
        return self.scorer.score(stream)

    def explain(self, stream: DealStream, top_n: int = 5) -> str:
        activations = self.scorer.activated_motifs(stream, top_n=top_n)
        lines = [f"Deal {stream.did}  raw_score={self.score(stream):.4f}  "
                 f"threshold={self.scorer.threshold:.4f}  "
                 f"escalate={self.should_escalate(stream)}"]
        lines.append("Activated primitives:")
        for i, a in enumerate(activations):
            lines.append(
                f"  [{i+1}] lift={a['lift']:+.3f}  "
                f"contrib={a['contribution']:.4f}  "
                f"dist={a['distance']:.3f}  "
                f"primitive={a['primitive']}  "
                f"matched={a['matched_motif']}"
            )
        return "\n".join(lines)

    def primitive_report(self) -> str:
        return self.learner.primitive_report()


# ============================================================
#  VI.  Synthetic demo with interpretable output
# ============================================================

def _make_event(eid, etype, t, vec):
    return Event(eid=eid, etype=etype, t=float(t),
                 vector=np.array(vec, dtype=float))


def _build_synthetic_data(n_success=40, n_fail=30, seed=42):
    """
    Synthetic deal streams with a planted structural signal.

    Successful deals: tend to follow intro → pilot → signed quickly,
    with pilot appearing in the first third of the deal window.
    Failed deals: tend to stall at legal or go intro → legal → stalled.
    Noise events and timing variation added throughout.
    """
    rng = np.random.default_rng(seed)
    streams = []

    SUCCESS_PATTERNS = [
        ["intro", "pilot",  "signed"],
        ["intro", "demo",   "signed"],
        ["intro", "pilot",  "demo",   "signed"],
    ]
    FAIL_PATTERNS = [
        ["intro", "legal",  "stalled"],
        ["intro", "legal",  "expired"],
        ["intro", "demo",   "legal",  "stalled"],
    ]

    def make_stream(did, pattern, label, t_total, fast_close):
        evs = []
        t = 0.0
        n = len(pattern)
        for i, etype in enumerate(pattern):
            if fast_close and label == 1:
                # success: pilot appears early (first 30% of window)
                t = t_total * (i / n) * 0.7 + rng.uniform(0, t_total * 0.05)
            else:
                t = t_total * (i / n) + rng.uniform(0, t_total * 0.1)
            vec = rng.standard_normal(8)
            if etype in ("signed", "demo"):
                vec[0] += 2.0  # slight semantic signal
            elif etype in ("stalled", "expired"):
                vec[0] -= 2.0
            evs.append(_make_event(f"{did}-e{i}", etype, t, vec))
        return DealStream(did=did, events=evs, label=label)

    for i in range(n_success):
        pat      = SUCCESS_PATTERNS[i % len(SUCCESS_PATTERNS)]
        t_total  = rng.uniform(30, 120)
        fast     = rng.random() > 0.3
        streams.append(make_stream(f"S{i:03d}", pat, 1, t_total, fast))

    for i in range(n_fail):
        pat      = FAIL_PATTERNS[i % len(FAIL_PATTERNS)]
        t_total  = rng.uniform(60, 200)
        streams.append(make_stream(f"F{i:03d}", pat, 0, t_total, False))

    rng.shuffle(streams)
    return streams


if __name__ == "__main__":
    print("=" * 62)
    print("  SATTVA-Deal v1")
    print("  Structural Attractor Training for Typed Venture Analysis")
    print("=" * 62)

    # Build synthetic dataset
    all_streams = _build_synthetic_data(n_success=40, n_fail=30, seed=42)
    n_train = int(len(all_streams) * 0.6)
    n_val   = int(len(all_streams) * 0.2)
    train_streams = all_streams[:n_train]
    val_streams   = all_streams[n_train:n_train + n_val]
    test_streams  = all_streams[n_train + n_val:]

    print(f"\nDataset: {len(all_streams)} deals  "
          f"(train={len(train_streams)}, val={len(val_streams)}, "
          f"test={len(test_streams)})")

    # Fit
    model = SATTVADeal(
        min_support       = 2,
        min_lift          = 0.05,
        lambda_complexity = 0.1,
        max_primitives    = 30,
        alpha             = 5.0,
        target_precision  = 0.65,
    )

    print("\n── Training ────────────────────────────────────────────────")
    model.fit(train_streams, val_streams)
    print(model.primitive_report())
    print(f"\n  Escalation threshold (calibrated): {model.scorer.threshold:.4f}")

    # Evaluate on test set
    print("\n── Test-Set Evaluation ─────────────────────────────────────")
    tp = fp = tn = fn = 0
    for stream in test_streams:
        esc = model.should_escalate(stream)
        lbl = stream.label
        if esc and lbl == 1:     tp += 1
        elif esc and lbl == 0:   fp += 1
        elif not esc and lbl == 0: tn += 1
        else:                    fn += 1

    n_esc = tp + fp
    if n_esc > 0:
        precision = tp / n_esc
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        print(f"  Escalated: {n_esc}/{len(test_streams)}")
        print(f"  Precision: {precision:.3f}  (target ≥ 0.65)")
        print(f"  Recall:    {recall:.3f}")
        print(f"  F1:        {f1:.3f}")
        print(f"  TP={tp} FP={fp} TN={tn} FN={fn}")
    else:
        print("  No deals escalated — lower target_precision or min_lift.")

    # Interpretable escalation examples
    print("\n── Interpretable Escalation Examples ──────────────────────")
    shown = 0
    for stream in test_streams:
        if model.should_escalate(stream) and shown < 2:
            print(f"\n{model.explain(stream, top_n=3)}")
            shown += 1

    if shown == 0:
        # Show top-scoring deal even if below threshold
        scored = sorted(test_streams, key=model.score, reverse=True)
        print("\n(No deals escalated — showing highest-scoring deal)")
        print(model.explain(scored[0], top_n=3))

    print("\n── Robustness check: failure-predictive motif rejected ─────")
    # Verify that a motif present only in failed deals gets negative Lift
    # and is NOT admitted to the primitive bank
    failure_types = {p.signature[0]
                     for p in model.learner.primitives}
    stall_patterns = {sig for sig in failure_types
                      if "stalled" in sig or "expired" in sig}
    print(f"  Failure-type patterns in bank: {len(stall_patterns)}")
    print(f"  {'PASS' if len(stall_patterns) == 0 else 'WARN'}: "
          f"primitive bank {'contains no' if len(stall_patterns)==0 else 'CONTAINS'}"
          f" stall/expire patterns")

    print("\n" + "=" * 62)
    print("  SATTVA-Deal v1 complete.")
    print("=" * 62)
