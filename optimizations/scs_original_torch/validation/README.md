# Validation artifacts

Generated checks are stored here by `bridge check-torch`, using existing pure
CPU Slurm allocation 20305804. Synthetic and real-checkpoint results are distinct.
Real sample: first 16 training points of ST19 x11_y12, 50 neighbors, 2,000 genes.
No GPU training job is replaced and no original results are overwritten.

Observed on 2026-09-05:

- `python -B -m unittest optimizations.scs_original_torch.test_original -v`:
  all 6 tests passed, including standalone inference and best-vs-last restoration.
- `synthetic.json`: forward, losses, input gradients and AdamW checks passed.
- `st19_x11_y12.json`: real checkpoint, 16 points, all checks passed. Maximum
  direction-logit error 1.0431e-7; foreground-probability error 5.9605e-8;
  loss error 2.3842e-7. Direction argmax and 0.5 foreground threshold agree.
- `st19_x11_y12_best.pt`: converted best model weights, 729,489 parameters;
  TensorFlow optimizer state is not transferred.
- Both train and standalone inference command-line help checks passed.

No full real-tile training/inference, GPU speed benchmark, CUDA/bf16 numerical
check, or new biological segmentation-quality assessment was performed here.
