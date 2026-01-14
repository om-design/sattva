# GA Discrimination with Myelination

This experiment tests whether GA-SATTVA plus a myelination layer
improves noisy pattern discrimination:

- Concepts: base (B), main (B+E1), similar (B+E2), different (D).
- Task: classify noisy cues by which concept they belong to using GA
  resonance as the decision signal.
- Compare accuracy:
  - Without myelination.
  - After training with activity- and GA-resonance-dependent myelination.

All logic is built on `src/sattva/ga_sattva_core.py` and
`src/sattva/myelination_substrate.py`.
