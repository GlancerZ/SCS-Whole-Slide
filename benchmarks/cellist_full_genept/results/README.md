# Completed seven-region benchmark snapshot

These files contain aggregate outputs from `runs/Cellist_full_genept_v1` and the
subsequent fairness/context audits on 2026-09-08. No individual-cell expression
matrices, raw RNA/image data, or trained weights are included.

Start with [comparison.md](comparison.md) and read the
[fairness review](fairness_audit/review.md) before interpreting method rankings.
`validation.json` records arithmetic and completion checks; it does not certify
that all scientific confounders have been removed. Paths inside audit records
refer to the original compute workspace. Full per-cell artifacts mentioned in
the report remain local under the ignored `runs/` directory.
