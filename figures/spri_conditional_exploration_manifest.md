# SPRi conditional exploration figure

- Primary asset: `spri_conditional_exploration.svg`
- Canvas: 2180 × 620 px, landscape, white background
- Format: deterministic editable SVG
- Typeface: Arial with Helvetica and sans-serif fallbacks
- Panel order: pretrained prior; data-rich condition; sparse condition; SPRi-enriched condition
- Shared point cloud: all four panels reuse `#chemical-space-cloud` from the SVG `<defs>` section
- SPRi icon placement: intentionally blank region approximately `x=1495–1665`, `y=115–225`; arrow and label remain below it
- Exact captions:
  - `Broad pretrained chemical prior`
  - `Data-rich condition enables broad exploration`
  - `Sparse condition restricts local exploration`
  - `SPRi enrichment restores conditional exploration`
- Conceptual encoding: gray points are pretrained-accessible but conditionally suppressed; colored regions represent high-probability conditional generation space; the dashed contour in panel D preserves the pre-SPRi sparse support.
- Provenance: composed deterministically as SVG from the visual direction agreed in this Codex task; no generated raster content is embedded.
