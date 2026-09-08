"""Apply the published Cellist NSCLC gene-count QC to ST19 raw counts.

Run on an allocated compute node after sourcing scripts/cellist_env.sh.
The primary rule is 50 <= detected genes <= 5000, not an scRNA-seq default.
"""
from pathlib import Path
import hashlib
import json
import os
import time

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "runs/ST19_cellist/whole_shared_nuclei_tissue/merged"
OUT = ROOT / "runs/ST19_cellist/qc_paper"
PAPER = "https://doi.org/10.1038/s41588-026-02610-1"
PDF = "https://wanglabtongji.github.io/resources/publications/2026_NatGenet_Cellist.pdf"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quantiles(values):
    return dict(zip(["min", "p01", "p05", "p25", "median", "p75", "p95", "p99", "max"],
                    np.quantile(values, [0, .01, .05, .25, .5, .75, .95, .99, 1]).tolist()))


def main():
    started = time.time()
    if (OUT / "completed.json").exists():
        raise RuntimeError("QC already complete; do not overwrite the completed analysis")
    OUT.mkdir(parents=True, exist_ok=True)
    source_path = SOURCE / "cell_counts.h5ad"
    source_stat = source_path.stat()
    original = json.loads((SOURCE / "completed.json").read_text())
    assert original["complete"]
    print("Reading raw cell matrix", flush=True)
    data = ad.read_h5ad(source_path)
    assert sparse.issparse(data.X)
    data.X = data.X.tocsr()
    assert np.issubdtype(data.X.dtype, np.integer)
    assert data.X.data.min() >= 0
    data.X.sum_duplicates()
    data.X.eliminate_zeros()
    genes = np.diff(data.X.indptr)
    counts = np.asarray(data.X.sum(axis=1, dtype=np.int64)).ravel()
    assert data.n_obs == original["cell_count"]
    assert data.n_vars == original["gene_count"]
    assert data.obs_names.is_unique and data.var_names.is_unique
    assert int(counts.sum()) == original["assigned_umis"]
    assert np.array_equal(genes, data.obs["n_genes"].to_numpy())
    assert np.array_equal(counts, data.obs["total_umis"].to_numpy())
    independent_stats = pd.read_csv(SOURCE / "cell_stats.csv")
    assert np.array_equal(independent_stats.cell_id.to_numpy(), data.obs.cell_id.to_numpy())
    assert np.array_equal(genes, independent_stats.n_genes.to_numpy())
    assert np.array_equal(counts, independent_stats.total_umis.to_numpy())
    low, high = genes < 50, genes > 5000
    keep = ~(low | high)
    assert np.all(keep == ((genes >= 50) & (genes <= 5000)))
    assert int(low.sum() + high.sum() + keep.sum()) == data.n_obs
    metrics = data.obs.copy()
    metrics.index.name = "obs_name"
    metrics["n_genes_by_counts"] = genes
    metrics["total_counts"] = counts
    metrics["qc_pass_paper"] = keep
    metrics["qc_reason"] = np.select([low, high], ["genes_lt_50", "genes_gt_5000"], default="pass")
    # Diagnostic only: the cited ST analysis specifies no mitochondrial cutoff.
    symbols = data.var["gene_name"].astype(str)
    mt = symbols.str.upper().str.startswith("MT-").to_numpy()
    mt_counts = np.asarray(data.X[:, mt].sum(axis=1, dtype=np.int64)).ravel()
    metrics["mitochondrial_umis_MT_prefix"] = mt_counts
    metrics["pct_counts_mt_MT_prefix"] = 100 * mt_counts / counts if mt.any() else np.nan
    metrics.to_csv(OUT / "cell_qc_metrics.csv.gz", compression="gzip")
    metrics.loc[keep, ["cell_id"]].to_csv(OUT / "retained_cells.csv", index=True)
    metrics.loc[~keep, ["cell_id", "n_genes_by_counts", "total_counts", "qc_reason"]].to_csv(
        OUT / "removed_cells.csv.gz", compression="gzip", index=True)
    pd.DataFrame({"gene_id": data.var_names[mt], "gene_name": symbols[mt].to_numpy()}).to_csv(
        OUT / "mitochondrial_gene_list.csv", index=False)
    summary = {
        "rule": "50 <= n_genes_by_counts <= 5000", "paper_doi": PAPER,
        "paper_pdf": PDF, "pdf_page_one_based": 17,
        "methods_section": "Single-cell-level analysis of Stereo-seq",
        "threshold_context": "Cellist human NSCLC Stereo-seq downstream analysis",
        "input_cells": data.n_obs, "retained_cells": int(keep.sum()),
        "removed_cells": int((~keep).sum()), "removed_percent": float(100 * (~keep).mean()),
        "removed_genes_lt_50": int(low.sum()), "removed_genes_gt_5000": int(high.sum()),
        "cells_exactly_50_genes": int((genes == 50).sum()),
        "cells_exactly_5000_genes": int((genes == 5000).sum()),
        "input_genes": data.n_vars, "output_genes": data.n_vars,
        "input_cell_assigned_umis": int(counts.sum()),
        "retained_umis": int(counts[keep].sum()), "removed_umis": int(counts[~keep].sum()),
        "retained_umi_percent_of_input_matrix": float(100 * counts[keep].sum() / counts.sum()),
        "retained_umi_percent_of_raw_slide": float(100 * counts[keep].sum() / original["source_umis"]),
        "genes_before": quantiles(genes), "genes_after": quantiles(genes[keep]),
        "umis_before": quantiles(counts), "umis_after": quantiles(counts[keep]),
        "mitochondrial_gene_count_MT_prefix": int(mt.sum()),
        "mitochondrial_filter_applied": False, "doublet_filter_applied": False,
        "notes": ["Gene-count QC only; this does not validate segmentation boundaries.",
                  "Detected genes are computed on raw integer UMIs before normalization or gene filtering.",
                  "Paper reporting summary mentions the <50 cutoff; the Methods additionally specifies >5000.",
                  "The paper excludes mixed-lineage clusters later in myeloid subtyping; this is not applied without annotation.",
                  "MT-prefix fractions are diagnostic only and depend on the supplied gene symbols.",
                  "Cell IDs, all input genes and bin1 x,y coordinates are preserved."],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    sensitivity = []
    for lower in [20, 50, 100, 200, 500]:
        selected = (genes >= lower) & (genes <= 5000)
        sensitivity.append(dict(min_genes_inclusive=lower, max_genes_inclusive=5000,
                                status="paper_primary" if lower == 50 else "sensitivity_only",
                                retained_cells=int(selected.sum()), removed_cells=int((~selected).sum()),
                                removed_percent=float(100 * (~selected).mean()),
                                retained_umi_percent=float(100 * counts[selected].sum() / counts.sum())))
    pd.DataFrame(sensitivity).to_csv(OUT / "threshold_sensitivity.csv", index=False)

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    colors = {"kept": "#287A95", "removed": "#C75343"}
    bins = np.geomspace(1, max(genes.max(), 5001), 85)
    for mask, label in [(keep, "kept"), (~keep, "removed")]:
        axes[0, 0].hist(genes[mask], bins=bins, histtype="step", color=colors[label], label=label)
    axes[0, 0].set(xscale="log", yscale="log", xlabel="Detected genes per cell", ylabel="Cells")
    for threshold in [50, 5000]:
        axes[0, 0].axvline(threshold, color="black", linestyle="--", linewidth=.8)
    axes[0, 0].legend()
    bins = np.geomspace(1, counts.max() + 1, 85)
    for mask, label in [(keep, "kept"), (~keep, "removed")]:
        axes[0, 1].hist(counts[mask], bins=bins, histtype="step", color=colors[label], label=label)
    axes[0, 1].set(xscale="log", yscale="log", xlabel="Raw UMIs per cell", ylabel="Cells")
    xy = data.obsm["spatial"] * .5
    # Conventional display: x horizontal, y down, from input x,y bin1 coordinates.
    for ax, mask, label in [(axes[1, 0], keep, "kept"), (axes[1, 1], ~keep, "removed")]:
        ax.scatter(xy[mask, 0], xy[mask, 1], s=.12, c=colors[label], alpha=.55, rasterized=True)
        ax.set(xlim=(0, original["shape"][0] * .5), ylim=(original["shape"][1] * .5, 0),
               xlabel="x (microns)", ylabel="y (microns)", title=f"{label.capitalize()}: {mask.sum():,} cells")
        ax.set_aspect("equal")
    fig.suptitle("ST19: published Cellist NSCLC QC | 50–5,000 detected genes", fontsize=14)
    fig.savefig(OUT / "qc_overview.png", dpi=180)
    fig.savefig(OUT / "qc_overview.pdf")
    plt.close(fig)

    print("Writing filtered raw-count H5AD", flush=True)
    data.obs = metrics
    filtered = data[keep].copy()
    filtered.uns["paper_qc"] = dict(rule=summary["rule"], paper_doi=PAPER, pdf_page=17,
                                   input_cells=data.n_obs, mitochondrial_filter=False,
                                   doublet_filter=False, applied_to="raw integer UMI counts")
    building = OUT / "cell_counts.paper_qc.building.h5ad"
    filtered.write_h5ad(building, compression="gzip")
    # Re-read the saved sparse matrix in chunks to verify export and UMI conservation.
    checked = ad.read_h5ad(building, backed="r")
    assert checked.shape == (int(keep.sum()), data.n_vars)
    assert checked.obs_names.equals(data.obs_names[keep])
    assert checked.var_names.equals(data.var_names)
    assert np.array_equal(checked.obsm["spatial"], data.obsm["spatial"][keep])
    exported_total = 0
    for start in range(0, checked.n_obs, 20000):
        end = min(start + 20000, checked.n_obs)
        saved = checked.X[start:end].tocsr()
        assert (saved != filtered.X[start:end]).nnz == 0
        exported_total += int(saved.sum(dtype=np.int64))
    assert exported_total == summary["retained_umis"]
    assert summary["retained_umis"] + summary["removed_umis"] == original["assigned_umis"]
    checked.file.close()
    assert source_path.stat().st_size == source_stat.st_size
    assert source_path.stat().st_mtime_ns == source_stat.st_mtime_ns
    building.replace(OUT / "cell_counts.paper_qc.h5ad")
    summary.update(complete=True, elapsed_seconds=time.time() - started,
                   source_h5ad_sha256=sha256(source_path),
                   output_h5ad_sha256=sha256(OUT / "cell_counts.paper_qc.h5ad"),
                   source_pdf_sha256=sha256(OUT / "sources/2026_NatGenet_Cellist.pdf"),
                   validation="Raw matrix matches original CSV/metadata; exported matrix re-read and exactly compared; UMI conservation passed",
                   completed_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   slurm_job_id=os.environ.get("SLURM_JOB_ID", ""))
    (OUT / "completed.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("COMPLETE: " + json.dumps({k: summary[k] for k in ["input_cells", "retained_cells", "removed_cells", "removed_percent"]}), flush=True)


if __name__ == "__main__":
    main()
