# Report and validation plan

- Surface initially selected: MCP artifact report, technical audience, outside
  positively identified Work Mode. Changed to portable HTML after the SQL-only
  provenance gate; HTML also failed that gate (details below). Snapshot only;
  not a live monitor. No Sites publication requested.
- Required structure: title/technical summary; learning and test evidence;
  definitions moved before evidence so denominators are visible; experimental
  design; limitations; next steps and open questions merged in the last section.
- QA stance: share stage results with caveats; final method-effectiveness claim
  needs revision until comparisons, ground-truth IoU and coverage ablation finish.
- Stage report is not a final benchmark report. No p-values or confidence bounds
  are inferred from correlated within-section centres or repeated matched cells.

## Chart contracts

1. Learning curve: is each model learning directional pseudo-labels? Native
   multi-series line, epoch x validation foreground direction accuracy, method
   grouping. Up to 300 actual epoch rows; never interpolate a missing run.
   Retain training loss/accuracy, foreground balanced accuracy and epoch time.
   Full-width report chart with named legend, endpoint labels, fraction-to-percent
   formatting. Categorical shared-renderer palette, three methods, <=3 roots.
   No arbitrary per-method colors outside the native palette contract. Legend
   and endpoint labels provide non-color identification. Interpretation is
   adjacent: pseudo-label learning is not proof of better final boundaries.
2. GenePT coverage: how much selected-input RNA survives mapping across platforms?
   Native bar, section x retained UMI fraction, dataset grouping (two platforms).
   Six true section rows, not six independent datasets. Zero-baseline percentage
   axis; all-value labels; full width; explicit dataset legend and section names.
   Categorical shared palette, two roots maximum. Retain gene mapping counts,
   empty-bin counts and split sizes in the reviewed source rows.
3. Test and RNA-consistency tables: exact audit lookups, not a synthetic ranking
   plot. Tables retain status, denominators and paired eligibility/unique pairs.
   Each correlation pair uses its own common population; cross-row rankings are
   expressly disallowed. More elaborate score charts omitted until multiple
   sections have completed the same evaluation budget.

Validation: dataset hashes/UMI sums/unique occupied neighbours/split disjointness
are checked by audit.py. Direction accuracy is recomputed from confusion matrices
by collect.py. Unit tests exercise independent dense pooling gradients, reference
ring ordering, matched RNA-region counts, zero eligible pairs and IoU misses/merges.
Use the MCP artifact validator before the single visible report render. Fix
shape errors in the canonical report JSON, not by building an unrelated summary.

## Presentation blocker (observed)

MCP `validate_artifact` rejected the actual Bash/Python collection/audit commands
with `charts[0].source must include the actual SQL query text`. Explicit inline
provenance and Python language metadata did not fix it. No invalid visible render
was attempted. The required portable HTML fallback was also attempted using the
bundled `deliver_portable_artifact.mjs`; it failed at packaging with the identical
SQL-only source check. Source inspection confirms `looksLikeSqlQuery` gates the
chart provenance without honoring Python source language. No SQL was fabricated
and no renderer was patched. No rendered HTML/MCP report or chart is claimed.

The protocol, canonical proposed artifact, CSV/JSON scores and a supporting
Markdown status note remain available. Presentation is blocked, not training or
scientific evaluation. Renderer screenshots/light-dark QA are unavailable because
neither selected surface passed its source gate.
