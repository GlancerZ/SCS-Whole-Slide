#!/usr/bin/env bash
set -euo pipefail
while tmux -L scs-genept-diagnosis has-session -t genes2000-probe 2>/dev/null; do
    sleep 5
done
OMP_NUM_THREADS=4 python -B -m unittest optimizations.scs_streaming.test_torch -q
python -B -u -m optimizations.scs_streaming.train_genept_repair \
    --selection runs/ST19_shared_6000/diagnosis_genept_2000_v1/gene_selection.json \
    --output runs/ST19_shared_6000/pilot2000_raw_v1 --encoding raw_counts
python -B -u -m optimizations.scs_streaming.train_genept_repair \
    --selection runs/ST19_shared_6000/diagnosis_genept_2000_v1/gene_selection.json \
    --output runs/ST19_shared_6000/pilot2000_genept_v1 --encoding genept
