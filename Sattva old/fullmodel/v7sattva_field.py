"""
SATTVA Field Engine
--------------------

Architecture
============
The engine receives raw deal streams (typed, timed events) and discovers
which structural patterns predict outcomes. No primitives are manually
specified — they emerge from the data.

Primitive lifecycle
-------------------
  observation  →  candidate   (motif seen ≥ min_obs times)
  candidate    →  primitive   (lift stable above threshold)
  primitive    →  pruned      (lift decayed below threshold after promotion)

Modes
-----
  naive       High plasticity. Field starts empty. Candidates form freely.
              Sleep fires automatically after every naive_sleep_every deals.
              Use until the field has stabilised enough for deployment.

  functional  Committed primitives only. Incoming streams are scored against
              the field and cause slow weight drift. Sleep fires after
              functional_sleep_after steps of inactivity.

  sleep       Offline consolidation. No new input accepted.
              Replays the buffer, recomputes lift, promotes candidates,
              prunes stale primitives. Returns to the calling mode when done.

Input format
------------
A deal is a list of Event objects:
    Event(etype="intro", t=0.0, vector=np.array([...]))

The engine extracts all 3-event motifs automatically. Each motif is a
hashable key:
    (type_pattern, time_bin_pair, magnitude_bin)

This is the natural primitive vocabulary for deal streams — the engine
discovers "intro→pilot→signed (fast)" without being told to look for it.
"""

import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, field
from itertools import combinations
from typing import List, Optional, Tuple, Dict
import warnings


# ────────────────────────────────────────────────────────────
#  Event & motif extraction  (from SATTVA-Deal)
# ────────────────────────────────────────────────────────────

@dataclass
class Event:
    etype:  str
    t:      float
    vector: np.ndarray


@dataclass
class Deal:
    did:    str
    events: List[Event]
    outcome: Optional[float] = None   # 0.0–1.0; None at inference time

    def sorted_events(self) -> List[Event]:
        return sorted(self.events, key=lambda e: e.t)


_TIME_BINS = [0.0, 0.20, 0.40, 0.60, 0.80, 1.01]

def _bin_time(r: float) -> int:
    for i in range(len(_TIME_BINS) - 1):
        if r < _TIME_BINS[i + 1]:
            return i
    return len(_TIME_BINS) - 2


def _magnitude_bin(v1: np.ndarray, v2: np.ndarray) -> int:
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    r = n1 / (n1 + n2 + 1e-9)
    return 0 if r < 0.35 else (2 if r > 0.65 else 1)


def extract_motifs(deal: Deal) -> List[Tuple]:
    evs = deal.sorted_events()
    motifs = []
    for e1, e2, e3 in combinations(evs, 3):
        dt1 = e2.t - e1.t
        dt2 = e3.t - e2.t
        if dt1 < 0 or dt2 < 0:
            continue
        total = dt1 + dt2 + 1e-9
        motifs.append((
            (e1.etype, e2.etype, e3.etype),
            (_bin_time(dt1 / total), _bin_time(dt2 / total)),
            _magnitude_bin(e2.vector - e1.vector, e3.vector - e2.vector),
        ))
    return motifs


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


# ────────────────────────────────────────────────────────────
#  Candidate — unconfirmed motif being tracked
# ────────────────────────────────────────────────────────────

@dataclass
class Candidate:
    key:          Tuple
    n_seen:       int   = 0     # total deals where this motif appeared
    n_win:        int   = 0     # of those, how many were wins
    lift:         float = 0.0   # P(win|motif) - P(win)
    lift_history: list  = field(default_factory=list)

    def update(self, outcome: float):
        self.n_seen += 1
        self.n_win  += outcome   # soft: fractional wins allowed
        p_win_given = self.n_win / self.n_seen
        self.lift_history.append(p_win_given)  # base_rate subtracted at eval time
        if len(self.lift_history) > 50:
            self.lift_history.pop(0)

    def stable_lift(self, base_rate: float) -> float:
        """Mean lift over recent observations."""
        if len(self.lift_history) < 5:
            return 0.0
        return float(np.mean(self.lift_history)) - base_rate

    def lift_variance(self) -> float:
        if len(self.lift_history) < 5:
            return 1.0
        return float(np.var(self.lift_history))


# ────────────────────────────────────────────────────────────
#  Primitive — committed to the energy field
# ────────────────────────────────────────────────────────────

@dataclass
class Primitive:
    key:         Tuple
    w_core:      float = 0.0
    w_adaptive:  float = 0.0
    age:         int   = 0
    error_ema:   float = 1.0
    lift_at_promotion: float = 0.0

    @property
    def weight(self) -> float:
        return self.w_core + self.w_adaptive

    def phi(self, motifs: List[Tuple]) -> float:
        """1 if this primitive's motif appears in the deal's motifs, else 0."""
        return 1.0 if self.key in motifs else 0.0


# ────────────────────────────────────────────────────────────
#  SATTVA Field Engine
# ────────────────────────────────────────────────────────────

class SATTVAField:
    """
    Parameters
    ----------
    naive_sleep_every       Sleep after this many deals in naive mode.
    functional_sleep_after  Sleep after this many steps of inactivity in
                            functional mode.
    min_obs                 Minimum deal observations before a candidate
                            is eligible for promotion.
    min_lift                Minimum stable lift for promotion.
    max_lift_variance       Maximum lift variance (stability gate for promotion).
    prune_lift              Primitive pruned if lift falls below this.
    lr_naive                Learning rate in naive mode.
    lr_functional           Learning rate in functional mode (slow drift).
    lr_interaction          Learning rate for interaction terms.
    adaptive_decay          Per-step decay on w_adaptive.
    max_primitives          Hard cap on committed field size.
    replay_capacity         Maximum deals stored in replay buffer.
    """

    def __init__(
        self,
        naive_sleep_every:       int   = 20,
        functional_sleep_after:  int   = 50,
        min_obs:                 int   = 8,
        min_lift:                float = 0.05,
        max_lift_variance:       float = 0.05,
        prune_lift:              float = 0.0,
        lr_naive:                float = 0.05,
        lr_functional:           float = 0.005,
        lr_interaction:          float = 0.01,
        adaptive_decay:          float = 0.0002,
        max_primitives:          int   = 30,
        replay_capacity:         int   = 200,
    ):
        self.naive_sleep_every      = naive_sleep_every
        self.functional_sleep_after = functional_sleep_after
        self.min_obs                = min_obs
        self.min_lift               = min_lift
        self.max_lift_variance      = max_lift_variance
        self.prune_lift             = prune_lift
        self.lr_naive               = lr_naive
        self.lr_functional          = lr_functional
        self.lr_interaction         = lr_interaction
        self.adaptive_decay         = adaptive_decay
        self.max_primitives         = max_primitives

        # Core state
        self.mode:       str = "naive"
        self.candidates: Dict[Tuple, Candidate]  = {}
        self.primitives: Dict[Tuple, Primitive]  = {}
        self.interactions: Dict[Tuple, float]    = defaultdict(float)

        # Statistics
        self.n_outcomes:       int   = 0
        self.sum_outcomes:     float = 0.0

        # Replay buffer — stores (motifs_set, outcome)
        self.replay: deque = deque(maxlen=replay_capacity)

        # Counters
        self._deals_since_sleep:    int = 0
        self._steps_since_activity: int = 0
        self._sleep_count:          int = 0
        self._return_mode:          str = "naive"

    # ── Mode ────────────────────────────────────────────────

    def set_mode(self, mode: str):
        assert mode in ("naive", "functional"), \
            "Set 'naive' or 'functional' directly; sleep is triggered automatically."
        self.mode = mode
        self._deals_since_sleep    = 0
        self._steps_since_activity = 0

    # ── Properties ──────────────────────────────────────────

    @property
    def base_rate(self) -> float:
        return self.sum_outcomes / self.n_outcomes if self.n_outcomes > 0 else 0.5

    @property
    def field_size(self) -> int:
        return len(self.primitives)

    # ── Energy & scoring ────────────────────────────────────

    def energy(self, motifs: List[Tuple]) -> float:
        motif_set = set(motifs)
        E = sum(p.weight * p.phi(motif_set) for p in self.primitives.values())
        for (i, j), J in self.interactions.items():
            if i in self.primitives and j in self.primitives:
                E += J * self.primitives[i].phi(motif_set) * self.primitives[j].phi(motif_set)
        return E

    def score(self, deal: Deal) -> float:
        """Score a deal. Returns P(win) ∈ [0, 1]."""
        motifs = extract_motifs(deal)
        return sigmoid(-self.energy(motifs))

    # ── Observe — main entry point ───────────────────────────

    def observe(self, deal: Deal):
        """
        Process one deal.

        In naive mode:    update candidates + field, auto-sleep every N deals.
        In functional mode: score + slow field drift, auto-sleep on inactivity.
        Sleep is not called directly — it fires automatically when due.
        """
        motifs    = extract_motifs(deal)
        motif_set = set(motifs)
        outcome   = deal.outcome

        # Always buffer (for sleep), but only if labeled
        if outcome is not None:
            self.replay.append((motif_set, float(outcome)))
            self.n_outcomes   += 1
            self.sum_outcomes += float(outcome)

        if self.mode == "naive":
            self._observe_naive(motif_set, outcome)
        elif self.mode == "functional":
            self._observe_functional(motif_set, outcome)

    def _observe_naive(self, motif_set, outcome):
        if outcome is None:
            return   # naive mode requires labels

        # Update candidates
        for key in motif_set:
            if key not in self.candidates:
                self.candidates[key] = Candidate(key)
            self.candidates[key].update(outcome)

        # Update committed primitives
        self._field_update(motif_set, outcome, lr=self.lr_naive)

        self._deals_since_sleep += 1
        if self._deals_since_sleep >= self.naive_sleep_every:
            self._sleep()
            self._deals_since_sleep = 0

    def _observe_functional(self, motif_set, outcome):
        # Score is always available (even without label)
        if outcome is not None:
            self._field_update(motif_set, outcome, lr=self.lr_functional)
            self._steps_since_activity = 0
        else:
            self._steps_since_activity += 1
            if self._steps_since_activity >= self.functional_sleep_after:
                self._sleep()
                self._steps_since_activity = 0

    # ── Field weight update ──────────────────────────────────

    def _field_update(self, motif_set, outcome, lr):
        E     = self.energy(motif_set)
        p     = sigmoid(-E)
        error = outcome - p

        for prim in self.primitives.values():
            a = prim.phi(motif_set)
            prim.w_adaptive -= lr * error * a
            prim.w_adaptive *= (1.0 - self.adaptive_decay)
            prim.age        += 1
            prim.error_ema   = 0.01 * abs(error) + 0.99 * prim.error_ema

        # Outcome-conditioned interactions
        names = list(self.primitives.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                pi = self.primitives[names[i]]
                pj = self.primitives[names[j]]
                key = (names[i], names[j])
                delta = pi.phi(motif_set) * pj.phi(motif_set) * error * self.lr_interaction
                self.interactions[key] -= delta
                if abs(self.interactions[key]) > 5.0:
                    self.interactions[key] = np.sign(self.interactions[key]) * 5.0

    # ── Sleep — consolidation ────────────────────────────────

    def _sleep(self):
        """
        Offline consolidation pass over the replay buffer.

        1. Recompute lift for all candidates using buffered outcomes.
        2. Promote candidates that earned it (if field not full).
        3. Update core weights for promoted primitives via buffer replay.
        4. Prune primitives whose lift decayed below threshold.
        5. Promote w_adaptive → w_core for stable primitives.
        """
        self._sleep_count += 1

        if len(self.replay) < self.min_obs:
            return

        # ── Step 1: recompute candidate lift from buffer ─────
        buf_candidates: Dict[Tuple, Candidate] = {}
        for motif_set, outcome in self.replay:
            for key in motif_set:
                if key not in buf_candidates:
                    buf_candidates[key] = Candidate(key)
                buf_candidates[key].update(outcome)

        # ── Step 2: promote eligible candidates ──────────────
        br = self.base_rate
        for key, cand in buf_candidates.items():
            if key in self.primitives:
                continue
            if cand.n_seen < self.min_obs:
                continue
            lift = cand.stable_lift(br)
            var  = cand.lift_variance()
            eligible = (abs(lift) >= self.min_lift and var <= self.max_lift_variance)
            if eligible and len(self.primitives) < self.max_primitives:
                prim = Primitive(key=key, lift_at_promotion=lift)
                # Positive lift → negative weight (deep basin, win attractor)
                # Negative lift → positive weight (raised energy, loss repulsor)
                prim.w_core = -lift * 2.0
                self.primitives[key] = prim
                if key not in self.candidates:
                    self.candidates[key] = cand
                else:
                    self.candidates[key] = cand

        # ── Step 3: replay buffer → update w_adaptive ────────
        for motif_set, outcome in self.replay:
            self._field_update(motif_set, outcome, lr=self.lr_naive * 0.5)

        # ── Step 4: prune stale primitives ────────────────────
        # Prune if: not seen recently, OR absolute lift fell below threshold.
        # abs() is essential — repulsors have negative lift and must not be
        # pruned simply because their lift is below 0.
        to_prune = []
        for key, prim in self.primitives.items():
            if key not in buf_candidates:
                if prim.age > self.min_obs * 2:
                    to_prune.append(key)
                continue
            cand = buf_candidates[key]
            if cand.n_seen >= self.min_obs:
                current_lift = cand.stable_lift(br)
                if abs(current_lift) <= self.prune_lift and prim.age > self.min_obs * 2:
                    to_prune.append(key)

        for key in to_prune:
            del self.primitives[key]
            # Clean up interaction terms
            for k in [k for k in list(self.interactions.keys()) if key in k]:
                del self.interactions[k]

        # ── Step 5: crystallise stable weights ───────────────
        for prim in self.primitives.values():
            if prim.error_ema < 0.25 and abs(prim.w_adaptive) > 0.01:
                prim.w_core    += prim.w_adaptive * 0.5
                prim.w_adaptive *= 0.5

    # ── Summary ─────────────────────────────────────────────

    def summary(self, verbose: bool = False):
        br = self.base_rate
        print(f"Mode: {self.mode}  field: {len(self.primitives)} primitives  "
              f"candidates: {len(self.candidates)}  "
              f"sleep_count: {self._sleep_count}  "
              f"base_rate: {br:.3f}")
        if not verbose:
            return
        print("  Primitives:")
        for p in sorted(self.primitives.values(), key=lambda p: p.weight):
            type_seq = "→".join(p.key[0])
            tbin     = p.key[1]
            mbin     = ["narrowing", "comparable", "widening"][p.key[2]]
            role     = "attractor" if p.weight < 0 else "repulsor"
            print(f"    {type_seq:<28} t={tbin} gap={mbin:<10} "
                  f"w={p.weight:+.4f}  [{role}]")
        if self.interactions:
            print("  Interactions (top 5 by magnitude):")
            for (ki, kj), v in sorted(self.interactions.items(),
                                       key=lambda kv: -abs(kv[1]))[:5]:
                i_seq = "→".join(ki[0])
                j_seq = "→".join(kj[0])
                print(f"    {i_seq} × {j_seq}: {v:+.4f}")

    def top_primitives(self, n: int = 5):
        """Return the n most negative-weight (deepest basin) primitives."""
        return sorted(self.primitives.values(), key=lambda p: p.weight)[:n]


# ────────────────────────────────────────────────────────────
#  Demo
# ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    SEP = "─" * 60

    def make_event(etype, t, signal=0.0):
        v = rng.standard_normal(6)
        v[0] += signal
        return Event(etype=etype, t=float(t), vector=v)

    def make_win_deal(i, rng):
        """Winning pattern: intro → pilot → signed, fast close."""
        t0 = 0; t1 = rng.uniform(5, 20); t2 = rng.uniform(t1+5, t1+30)
        return Deal(did=f"W{i}", outcome=1.0, events=[
            make_event("intro",  t0, signal=0.0),
            make_event("pilot",  t1, signal=0.5),
            make_event("signed", t2, signal=1.0),
        ])

    def make_loss_deal(i, rng):
        """Losing pattern: intro → legal → stalled, slow."""
        t0 = 0; t1 = rng.uniform(20, 60); t2 = rng.uniform(t1+30, t1+90)
        return Deal(did=f"L{i}", outcome=0.0, events=[
            make_event("intro",   t0, signal=0.0),
            make_event("legal",   t1, signal=-0.5),
            make_event("stalled", t2, signal=-1.0),
        ])

    def make_ambiguous_deal(i, rng):
        """Has pilot but slow close — uncertain outcome."""
        t0 = 0; t1 = rng.uniform(30, 80); t2 = rng.uniform(t1+30, t1+90)
        return Deal(did=f"A{i}", outcome=float(rng.random() > 0.6), events=[
            make_event("intro",  t0),
            make_event("pilot",  t1),
            make_event("signed", t2),
        ])

    # ── Phase 1: Naive mode ────────────────────────────────
    print(SEP)
    print("Phase 1: Naive mode — engine observes labeled deals, field emerges")
    print(SEP)

    engine = SATTVAField(
        naive_sleep_every   = 15,
        min_obs             = 6,
        min_lift            = 0.08,
        max_lift_variance   = 0.08,
        lr_naive            = 0.08,
        replay_capacity     = 150,
    )

    # Mix of wins and losses — engine doesn't know the rule
    for i in range(80):
        if rng.random() < 0.5:
            engine.observe(make_win_deal(i, rng))
        else:
            engine.observe(make_loss_deal(i, rng))

    print(f"After 80 labeled deals:")
    engine.summary(verbose=True)

    # ── Phase 2: Transition to functional ─────────────────
    print(f"\n{SEP}")
    print("Phase 2: Switch to functional mode")
    print(SEP)
    engine.set_mode("functional")

    # Deals arrive without labels (inference time), occasionally with labels
    win_scores  = []
    loss_scores = []

    for i in range(40):
        if rng.random() < 0.5:
            d = make_win_deal(80 + i, rng)
            s = engine.score(d)
            win_scores.append(s)
            # Occasionally a labeled outcome arrives
            if rng.random() < 0.3:
                engine.observe(d)
        else:
            d = make_loss_deal(60 + i, rng)
            s = engine.score(d)
            loss_scores.append(s)
            if rng.random() < 0.3:
                engine.observe(d)

    print(f"Scoring 40 unlabeled deals:")
    if win_scores:
        print(f"  P(win) for actual wins:   "
              f"mean={np.mean(win_scores):.3f}  "
              f"min={np.min(win_scores):.3f}  "
              f"max={np.max(win_scores):.3f}")
    if loss_scores:
        print(f"  P(win) for actual losses: "
              f"mean={np.mean(loss_scores):.3f}  "
              f"min={np.min(loss_scores):.3f}  "
              f"max={np.max(loss_scores):.3f}")

    sep_val = (np.mean(win_scores) - np.mean(loss_scores)
               if win_scores and loss_scores else 0)
    print(f"  Separation (win_mean - loss_mean): {sep_val:+.3f}  "
          f"{'✓' if sep_val > 0.05 else '✗'}")

    # ── Phase 3: Manual sleep + ambiguous deals ────────────
    print(f"\n{SEP}")
    print("Phase 3: Manual sleep, then ambiguous deals")
    print(SEP)
    engine._sleep()   # explicit sleep (as if called by inactivity timer)
    print("After sleep:")
    engine.summary(verbose=True)

    print("\nScoring ambiguous deals (slow pilot→signed):")
    for i in range(5):
        d = make_ambiguous_deal(i, rng)
        s = engine.score(d)
        print(f"  {d.did}: P(win)={s:.3f}  "
              f"(true outcome={d.outcome:.0f})")

    # ── Phase 4: What the engine discovered ───────────────
    print(f"\n{SEP}")
    print("Phase 4: What did the engine discover?")
    print(SEP)
    print("Top primitives by basin depth (most negative weight = strongest attractor):")
    for p in engine.top_primitives(n=5):
        type_seq = "→".join(p.key[0])
        tbin     = p.key[1]
        mbin     = ["narrowing","comparable","widening"][p.key[2]]
        print(f"  {type_seq:<28} t={tbin}  gap={mbin:<10}  "
              f"w={p.weight:+.4f}  lift_at_promo={p.lift_at_promotion:+.3f}")
    print(f"\nEngine never saw 'funding' or any pre-specified feature.")
    print(f"It discovered structure from the event stream directly.")
