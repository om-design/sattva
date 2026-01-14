# GA-SATTVA Part A: Shape-Quality Composites (Checkpoint)

## Primitive vocabulary

The current GA-SATTVA implementation defines a small, interpretable set of
shape-related primitives intended to approximate early visual qualities and
part-based object structure:

- **Intrinsic qualities** (8):
  - `round_curved`: high curvature, smooth contour.
  - `sharp_cornered`: abrupt curvature changes, corners.
  - `long_extent`: elongated main axis, high aspect ratio.
  - `short_extent`: compact, low aspect ratio.
  - `single_segment`: one main continuous contour segment.
  - `multi_segment`: composed of several straight/curved segments.
  - `pointed_tip`: locally extreme curvature at an endpoint.
  - `blunt_end`: rounded/flat ends, low local curvature.

- **Parts** (4):
  - `rect_part`: rectangle-like patch (straight, multi-segment, four-sided).
  - `seat_part`: flat, rectangular support area.
  - `leg_part`: elongated, narrow support element.
  - `backrest_part`: upright, elongated, attached to seat.

Each primitive is currently implemented as a GA pattern over 64 units in a
4096-unit GAUnitSet, with disjoint supports for clarity.

## Composite categories

From these primitives, the following composite category patterns are defined as
unions of their constituent supports:

- `rectangle_like`:
  - `rect_part` + `sharp_cornered` + `multi_segment` + `long_extent`.

- `chair_like`:
  - `seat_part` + `leg_part` + `backrest_part` + `rect_part` + `multi_segment`.

- `other_shape`:
  - `round_curved` + `single_segment` + `blunt_end` + a small random extra set.

- `football_like`:
  - `round_curved` + `long_extent` + `single_segment` + `pointed_tip`.

- `irregular_shape1`:
  - `rect_part` + `round_curved` + `multi_segment` + small random extra set.

- `irregular_shape2`:
  - `sharp_cornered` + `short_extent` + `multi_segment` + small random extra set.

These composites are stored as GA patterns and used as attractors in
GASATTVADynamics.

## Discrimination results (Part A)

With 4096 units, 64 units per primitive, and noisy cues (40% dropout,
Gaussian noise on activation levels), the system achieves:

- **3-way discrimination** (rectangle_like, chair_like, other_shape):
  - Accuracies near ceiling for rectangle_like and chair_like.
  - High but slightly lower accuracy for other_shape.

- **6-way discrimination** (adding football_like, irregular_shape1,
  irregular_shape2):
  - `chair_like`, `football_like`, and `irregular_shape1` remain highly
    discriminable (≈0.91–0.93 accuracy).
  - `rectangle_like` and `irregular_shape2` show moderate accuracy
    (≈0.70–0.76), reflecting overlap in footprint and shared qualities.
  - `other_shape` becomes the hardest class (≈0.46 accuracy), consistent with
    its intentionally generic, partly random definition.

Overall mean accuracy in the 6-way case is ≈0.78. This indicates that the
current GA geometry supports robust recognition for some composite objects,
while more ambiguous categories expose overlaps that are suitable targets for
future refinement rules.

## Conceptual interpretation

- The **primitive layer** acts like a simplified V1/V2 vocabulary, encoding
  curvature, extent, segmentation, and a small set of canonical parts relevant
  to rectangles and chairs.
- **Composite categories** (rectangle-like, chair-like, football-like,
  irregulars) are stable attractors over this vocabulary.
- GA-SATTVA dynamics and peak resonance perform reliable classification under
  noise, and the multi-category setup reveals where the current geometry is
  ambiguous (especially for generic "other" shapes).

## Next directions (Part A)

Short-term extensions on top of this checkpoint:

- Implement a **resonance probe** for a fixed category (e.g., `chair_like`):
  - Show many noisy chair cues.
  - Measure mean resonance to all categories and all primitives.
  - Use this as an interpretable profile of which parts/qualities the system
    recruits when recognizing a chair.

- Introduce **few-shot refinement** for selected categories (e.g.,
  `rectangle_like` or `other_shape`) to test whether simple GA-level update
  rules can sharpen ambiguous boundaries without degrading clearly separated
  categories.

- Gradually expand the primitive set with additional parts (e.g., armrests,
  tabletop parts) and qualities (e.g., tapered, curved_back) to move toward
  richer object classes while preserving the current, grounded structure.
