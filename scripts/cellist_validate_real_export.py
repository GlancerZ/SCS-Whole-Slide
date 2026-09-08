"""Validate the actual gene schema and shared-ID export on two finished tiles."""
import json
import anndata as ad
import numpy as np
from cellist_cpu_stage import ROOT, save_json
from cellist_merge import merge

source = ROOT / "runs/ST19_cellist/whole_shared_nuclei_tissue"
base = ROOT / "runs/ST19_cellist/validation_real_export"
base.mkdir(exist_ok=True)
manifest = json.loads((source / "manifest.json").read_text())
manifest["tiles"] = [tile for tile in manifest["tiles"] if tile["id"] in {"x10_y11", "x11_y11"}]
assert len(manifest["tiles"]) == 2
manifest["total_records"] = sum(tile["records"] for tile in manifest["tiles"])
manifest["total_umis"] = sum(tile["umis"] for tile in manifest["tiles"])
manifest["scope"] = "two-tile export validation only"
manifest["method_scope"] = "TWO-TILE EXPORT VALIDATION ONLY; shared whole-slide nucleus identities"
save_json(base / "manifest.json", manifest)
for tile in manifest["tiles"]:
    target = base / "tiles" / tile["id"]
    target.parent.mkdir(exist_ok=True)
    if not target.exists():
        target.symlink_to(source / "tiles" / tile["id"])
if not (base / "spatial_index.h5").exists():
    (base / "spatial_index.h5").symlink_to(source / "spatial_index.h5")
merge(base)
report = json.loads((base / "merged/completed.json").read_text())
adata = ad.read_h5ad(base / "merged/cell_counts.h5ad")
assert adata.n_vars == manifest["source_gene_count"]
assert adata.obs.nucleus_id.is_unique
assert adata.obs_names.is_unique and adata.var_names.is_unique
assert int(adata.X.sum(dtype=np.uint64)) == report["assigned_umis"]
assert adata.obsm["spatial"].shape == (adata.n_obs, 2)
print(f"PASS: actual two-tile export reread, {adata.n_obs:,} cells x {adata.n_vars:,} genes; UMI conservation and unique shared nucleus IDs", flush=True)
