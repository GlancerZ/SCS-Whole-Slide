# SCS paper-data controlled benchmark

Question: does the current wider/deeper PyTorch model, with or without online
GenePT expression pooling, improve on the SCS architecture on its published data?

This is **a controlled benchmark on paper datasets, not a claim to reproduce the
published scores exactly**. In particular the SCS control uses the validated
PyTorch topology/AdamW port, a different split/batch/precision, and the existing
Spateo compatibility implementation. The old ST19 experiments are untouched.

## Locked comparison

- Seq-scope mouse liver: all four bundled sections/tiles 2104–2107. Each entire
  section gets one model. These are four sections of one dataset, not four datasets.
- seqFISH+ NIH3T3: experiment 1, FOVs 0 and 1 selected by index before fitting.
  Sixteen and fourteen manually annotated cells respectively. This is an initial
  subset, not the paper's full 17-field, 211-cell IoU experiment.
- Methods: `scs_reference` (width 64, depth 8, original TFA-compatible AdamW),
  `raw_scale4` (width 256, depth 32, raw counts), `genept_scale4` (width 256,
  depth 32, GenePT-weighted mean over expressed mapped genes). Scaled models
  use Muon 0.002 plus auxiliary AdamW 0.0003; coordinates are divided by 30.
  GenePT inputs are scaled by sqrt(1536). No PCA, expression log transform,
  cross-spot expression pooling, or stored full spot x 1536 embeddings.
- All methods: the same per-section 2,000 Seurat-v3 HVGs (`span=1`), ordered
  50-token neighbourhoods, pseudo-labels, stratified random 80/10/10 centre
  split, seed 3812, batch 512, BF16, explicit SDPA, 100 epochs, and identical
  unweighted losses. Batch size is fixed for equal exposure/update counts,
  not tuned separately to consume every byte of VRAM.
- Validation chooses the checkpoint by foreground-only direction accuracy.
  Test labels are used only after checkpoint selection. The random test set
  estimates within-section interpolation, not unseen-section generalization.
- HVGs are chosen transductively per section, as in SCS. Neighbour contexts can
  overlap across splits. Neither is represented as an independent external test.
- Inference covers all valid RNA-bearing centres, including confident background
  centres excluded by the training cap. Upstream omits some capped background
  centres at inference; this controlled implementation applies full coverage to
  every compared model. No model gets a different set of inference centres.
- Common SCS direction-prior adjustment, two smoothing passes, foreground
  threshold 0.1, gradient-flow tracking and small-basin filtering.
- `direction_prior_only`: trained SCS foreground probabilities, uniform direction
  logits. This isolates the direction network's addition to the nucleus prior;
  it is not a completely untrained or image-only baseline.

Compared with the earlier ST19 repair pilot, this protocol deliberately replaces
the training-centre Fano gene selection with original per-section Seurat-v3 HVGs
and uses the same unweighted SCS objective for every method. It evaluates the
current architecture/encoding under a controlled paper-data protocol, not the
unchanged old ST19 checkpoint or every training detail of that four-epoch pilot.

## Evaluation and safeguards

1. Held-out direction accuracy/macro recall and foreground balanced accuracy,
   with explicit positive/negative counts and the training-majority baseline.
2. Final segmentation cell counts, median areas, RNA assignment and nucleus
   matching coverage. More/larger cells are **not** automatically better.
3. Paper-style nucleus-matched RNA consistency: compare each method's unique
   region with their intersection, requiring >=100 UMIs in **each** of the three
   regions. Both scores in a comparison use exactly the same eligible nuclei.
   All-gene sums use int64, not the upstream evaluator's overflowing int8 array.
   Repeat correlation without the input HVGs as a sensitivity analysis on the
   same eligible population. Record the number of unique matched cell pairs;
   multiple nuclei can match the same cell. No independent-cell p-values claimed.
4. seqFISH+: one-to-one maximum-total-IoU assignment against manual ROIs; missed
   GT cells count as zero. Report mean/median IoU and GT recall at IoU >=0.5.
   Overlapping annotation boundary pixels (<0.2% in these FOVs) are ignored for
   every method. Whole-frame AP/precision is unavailable because only selected
   whole cells are annotated, not every image object.

seqFISH+ limitations: its released RNA coordinates are already grouped by
manually selected cells. They are concatenated without using those labels for
training, but the input support is annotation-conditioned. Grids of four native
pixels approximate the paper's 0.4um grids; exact calibration/conversion code was
not supplied in SCS. Two DAPI planes are maximum-projected and scaled with fixed
1st/99.9th percentiles. These preprocessing choices are locked across models,
not optimized using IoU. Do not compare our numeric IoU directly with paper 0.75.

Missing GenePT symbols are zero vectors and excluded from the pooling divisor;
exact symbol and UMI coverage are saved per run. Case folding is not claimed to
be validated mouse-to-human orthology mapping.

The integrity audit found only 58.96%–64.01% count coverage on liver, versus
99.24%–99.31% on these seqFISH+ fields. An additional `raw_matched_scale4` run on
liver 2104 is therefore queued after the primary comparisons. It masks exactly
the same missing-gene count columns as GenePT, retaining raw channels, the graph,
labels and other raw-scale4 settings. This coverage ablation is separate from the
three preregistered main methods and is reported in `coverage_ablation.json`.

## Execution

All project execution is inside allocation 20327131 on rg31503; do not run these
commands on the login node. `launch_*.sh` creates compute-node tmux sessions and
keeps the parent srun step alive so Slurm does not kill detached processes.

1. `launch_prepare.sh`: sparse liver preprocessing in `.venv-scs`.
2. `launch_train.sh`: serial GPU training in `.venv-scs-torch`, all 12 liver runs.
3. `launch_evaluate.sh`: CPU postprocessing as liver checkpoints finish.
4. `prepare_seqfish.py`: second-platform preparation and independent ROI audit.
5. `launch_seqfish.sh`: waits for liver GPU completion, then trains/evaluates six
   seqFISH+ runs. This queue aborts if upstream training exits unsuccessfully.

Output root: `runs/SCS_paper_benchmark_v1`. Each section has `prepared.json`,
counts, fixed sample IDs/splits and three isolated model directories containing
configuration, history, selected/latest checkpoints, predictions, test metrics,
and segmentation. `benchmark.json` is written after that section's three model
segmentations and direction-prior control complete.

Do not interpret a directory or training log as a completed run. Check
`completed.json`, `segmentation_completed.json`, and `benchmark.json` separately.
Do not rerun preparation/training into existing output directories. Existing
checkpoints are retained; automatic training resume is not implemented.

## Primary sources

- Published SCS: https://doi.org/10.1038/s41592-023-01939-3
- Author open manuscript and Supplementary Notes 1, 3, 7, 11–13:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC10312435/ (preprint version, not the
  typeset final article). The supplementary archive is saved under the run root.
- Official code/data: https://github.com/chenhcs/SCS ; especially `seqscope.py`,
  `src/preprocessing.py`, `src/transformer.py`, `src/postprocessing.py`,
  and `evaluation.py`.
- seqFISH+ source/ROIs/DAPI: https://zenodo.org/records/2669683 ; source-provided
  MD5 checksums verified for all three downloaded archives.
- GenePT: https://github.com/yiqunchen/GenePT ; existing official embedding asset,
  with SHA-256 recorded in every new GenePT run.

## Tests

In the CPU environment: `python -m unittest benchmarks.scs_paper.test_prepare
benchmarks.scs_paper.test_evaluate -v`. In the PyTorch environment: `python -m
unittest benchmarks.scs_paper.test_train -v`. Covers exact ring order, sparse
neighbour exclusions, online pooling and gradients, metric denominators,
int64 counts, paired-region eligibility, cell matching and missed/merged IoU.
