# GA Hierarchical Learning Experiment

This experiment tests whether GA-SATTVA can support hierarchical
structure and slow emergence of a more basic primitive:

- Construct a shared base primitive (core units) and two related
  primitives built from that base plus different extras.
- Also construct an unrelated primitive with disjoint support.
- Train only on the main primitive.
- After training, test weak cues to see whether resonance patterns
  reflect the shared base and distinguish related vs unrelated
  concepts.

All logic is built on `src/sattva/ga_sattva_core.py`.
