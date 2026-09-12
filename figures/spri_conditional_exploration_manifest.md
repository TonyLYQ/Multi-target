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
- Conceptual encoding: gray points are pretrained-accessible but conditionally suppressed; muted purple marks the data-rich target condition in panel B; teal marks the distinct sparse target condition in panels C and D. The ligand star is intentionally omitted.
- Panel B geometry: three disconnected, non-elliptical smooth regions encode broad but multimodal exploration; only three low-probability colored samples sit immediately beyond their boundaries.
- Panel C geometry: the sparse support is deliberately compact and contains only three highlighted samples, emphasizing the extreme low-data regime.
- Panel D geometry: one connected teal support is the 0.50 isocontour of four anisotropic Gaussian kernels, joined by three lower-weight bridge kernels. Its area is approximately 56,223 SVG² versus 56,913 SVG² across panel B's three regions (about 1.2% smaller). The three added lobes extend upper-left, right, and lower-left, contrasting with panel B's left, upper-right, and lower-right arrangement. The dashed inner contour preserves the reduced pre-SPRi support from panel C.
- Panel D points: every point from the shared chemical-space cloud whose center lies inside the 0.50 support is recolored teal.
- Provenance: composed deterministically as SVG from the visual direction agreed in this Codex task; no generated raster content is embedded.
