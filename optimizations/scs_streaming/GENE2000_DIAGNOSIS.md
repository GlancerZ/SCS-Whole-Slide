# 2000-gene collapse diagnosis and repair pilot

## Scope and resource

The old 12-epoch run and checkpoints were preserved. All experiments below run
on H100 allocation **20327131**, node **rg31503**, using `.venv-scs-torch`.
The user authorized one GPU. Submitted allocation-only command:

```bash
sbatch --account=rrg-junding --nodes=1 --gpus-per-node=1 --cpus-per-task=16 --mem=120G --time=06:00:00 --mail-type=ALL --mail-user=bowen.zhao@mail.mcgill.ca --wrap="sleep 21600"
```

Project code is run using the Slurm remote-execution helper and compute-node
tmux sessions. No new full-slide training run is implied by these pilots.

## Controlled memorization experiments

Every case uses the same 64 training centres (32 foreground, 32 background;
two foreground examples per direction), 50 tokens, width 256 and depth 32.
Dropout is disabled in every case to isolate whether the network can memorize
the input. These are **training accuracies, not validation results**. One fixed
seed and 400 updates are used; they are diagnostic evidence, not an estimate of
population performance.

| Input and change from original configuration | Direction training accuracy |
| --- | ---: |
| All-gene GenePT, original Muon LR 0.02 and scales | 6.25% |
| Same input, expression representation multiplied by sqrt(1536) | 100% |
| Same input, only coordinates divided by 30 | 31.25% |
| Same input, both input rescalings | 100% |
| Same input, only Muon LR reduced to 0.002 | 96.875% |
| 2000-gene GenePT, original scales and Muon LR | 90.625% |
| 2000 raw expression channels, original scales and Muon LR | 100% |
| 2000-gene GenePT, both input rescalings | 100% |

The original random initialization had expression-token RMS **0.01626**, versus
position-token RMS **3.73781** (about **230 times** larger). Native GenePT gene
vectors have unit L2 norm. After 12 epochs, the failed checkpoint outputs nearly
constant predictions on the diagnostic batch: 62 of 64 direction predictions
are class 0, and all spots are foreground.

These paired interventions support an **input-scale / optimization interaction**
as a contributor to collapse. They do not establish that all-gene input is
intrinsically invalid, or that reducing the number of genes alone repairs
direction generalization. Lowering Muon's learning rate alone also helps this
memorization test. Source gradients were recorded, rather than assuming all
parameters receive useful updates from a finite loss alone.

## Gene selection and input contract

The 2000 symbols are selected using 50,000 randomly sampled **training centre
spots only**. Ranking uses log dispersion standardized in 20 mean-expression
quantile bins, with detection in at least 20 sampled spots. This diagnostic
ranking is not Seurat-v3 and is not an arbitrary prefix of the old 6000 genes.
Its universe is the 21,798 GenePT-mapped symbols, making the GenePT and raw-count
arms use precisely the same genes. Validation labels are never used to select
genes.

Selection metadata, IDs and symbols are saved in
`runs/ST19_shared_6000/diagnosis_genept_2000_v1/gene_selection.json`.
The source centre spots, labels and neighbourhood graph are kept fixed. Only
expression channels are filtered; no spatial spots are pooled. In the balanced
64-centre probe, 379 of 3200 tokens, including 5 centre tokens, become empty after
filtering. Empty tokens remain in place and are not silently discarded.

For GenePT, the divisor is recalculated as the number of retained, nonzero genes.
For raw counts, the sparse projection is **a sum, not a mean**: it exactly equals
`Linear(2000, 256)` applied to the raw 2000-dimensional expression vector.
No dense whole-slide expression or spot-embedding cache is created.

## Held-out-centre pilot

Both arms use the same 131,072 randomly chosen training centres and 32,768
validation centres for four epochs. Centre IDs are explicitly checked disjoint.
Overlapping same-slide spatial context is still possible: these are not
independent-region or independent-slide validation sets.

Both use the original 32-layer / 256-wide model, dropout 0.1 and head dropout
0.5, batch 512, BF16, SDPA, Muon LR 0.002 and AdamW LR 0.0003. Coordinates are
divided by 30. The GenePT arm additionally multiplies its expression projection
by sqrt(1536), retaining expression-count information and the requested pooling
formula. Both heads remain trainable.

Training samples retain their natural class distribution: 127,521 foreground
and 3,551 background. Binary loss weights are computed only from these training
counts: `N/(2*N_background)` and `N/(2*N_foreground)`. No background oversampling
or high-density-spot removal is performed. The same fixed weights are used for
validation loss. Total weighted loss is not directly comparable to the original
unweighted run; task accuracies and confusion matrices are reported separately.

The final raw-count pilot obtains **78.06% binary balanced accuracy**, with
**83.74% foreground sensitivity**, **72.38% background specificity**, and
**6.252% direction accuracy**. Thus foreground/background discrimination improves,
but direction generalization is not repaired by this short experiment.

The final GenePT pilot obtains **73.76% binary balanced accuracy**, with
**59.35% foreground sensitivity**, **88.18% background specificity**, and
**6.321% direction accuracy**. The majority direction class selected using the
training subset obtains **6.597%** on this fixed validation subset. Neither
pilot exceeds that baseline. Both four-epoch pilots completed successfully;
this is a partial repair of collapse, not a validated direction segmenter.

Run directories:

- `runs/ST19_shared_6000/pilot2000_raw_v1`
- `runs/ST19_shared_6000/pilot2000_genept_v1`

Each contains exact configuration, selected genes, sample IDs, per-epoch
train/validation metrics, and best/latest checkpoints. Selection of the best
pilot checkpoint uses direction accuracy plus binary balanced accuracy, not
misleading overall foreground-dominated binary accuracy.

## Implementation and verification

- `torch_model.py`: explicit persisted input-scale and encoding options;
  sparse raw-count linear projection; optional foreground-class loss weights.
  Default configurations retain legacy behaviour and serialized configuration.
- `torch_data.py`: optional fixed gene filtering/remapping, preserving tokens
  and labels, and recomputing offsets including empty tokens.
- `diagnose_genept.py`: paired overfit experiments and measured activation /
  gradient statistics.
- `train_genept_repair.py`: isolated reproducible 2000-gene pilot runner.
- `test_torch.py`: 16 tests pass, including sparse/dense raw-count forward and
  gradient equivalence, empty tokens, input scales, class-weight gradients and
  preservation of labels/positions during gene filtering.

Launchers are `scripts/scs_genept_diagnose_tmux.sh`,
`scripts/scs_genept_2000_diagnose_tmux.sh`, and
`scripts/scs_2000_repair_pilot_tmux.sh`. Outputs are created exclusively, so a
rerun must use a new output path rather than overwriting evidence.
