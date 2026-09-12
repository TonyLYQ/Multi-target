# SPRi conditional exploration — borderless density version

- Primary asset: `spri_conditional_exploration_no_border.svg`
- Source asset preserved: `spri_conditional_exploration.svg`
- Canvas: 2180 × 620 px, landscape, white background
- Format: deterministic editable SVG; no raster content
- Typeface: Arial with Helvetica and sans-serif fallbacks
- Panel order and captions are unchanged from the source figure
- SPRi icon placement remains intentionally blank at approximately `x=1495–1665`, `y=115–225`
- Probability encoding:
  - every condition uses a single fixed hue; only opacity changes with density
  - Gaussian-softened analytical supports decay continuously into low-probability exploration space without stepped color bands
  - same-hue radial density fields create unequal interior hotspots and represent nonuniform sampling probability
  - colored points represent samples whose complete circular marker intersects the main analytical support
- No solid boundary stroke is used for panels B, C, or D. The dashed curve in panel D is retained solely as the analytical reference for the pre-SPRi sparse support.
- Panel D uses the 0.50 isocontour of the same four-kernel Gaussian-mixture field used in the bordered version; Gaussian softening supplies the continuous outward decay.
- Point rendering does not use clipping paths. A point is recolored in full whenever any part of its circular marker intersects the main support (B: 83 shared points plus 2 low-probability samples; C: 3; D: 74).
- Provenance: composed deterministically as SVG from the visual direction agreed in this Codex task.
