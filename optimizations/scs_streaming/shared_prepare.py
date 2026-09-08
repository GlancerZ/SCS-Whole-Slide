"""Build a fixed train-only gene vocabulary and compact whole-slide SCS tiles."""
import argparse
import fcntl
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
from scipy import sparse, ndimage
from scipy.spatial import cKDTree

from .shared_data import fingerprint, save_json, load_schema, neighbor_table, interior_mask

ROOT = Path(__file__).resolve().parents[2]


def aggregate(records, shape, origin, bin_size, n_genes):
    ny = math.ceil(shape[1] / bin_size)
    x = (records["x"].astype(np.int64) - origin[0]) // bin_size
    y = (records["y"].astype(np.int64) - origin[1]) // bin_size
    return sparse.csr_matrix((records["count"].astype(np.int32), (x * ny + y, records["gene"])),
                             shape=(math.ceil(shape[0] / bin_size) * ny, n_genes), dtype=np.int32)


def initialize(source, output, n_genes=6000, n_neighbors=50, bin_size=3,
               hvg_bins_per_tile=512, val_fraction=0.1, seed=20260905, tile_ids=None,
               validation_tiles=None):
    """Fit Seurat-v3 HVGs on a bounded, equal-per-training-core sparse sample.

    Held-out expression is NOT used to fit the gene list. Counts remain raw,
    matching SCS; there is no separate per-tile fitted normalization.
    """
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or source in output.parents:
        raise ValueError("Use a new sibling run; never overwrite the original queue")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Initialize requires an empty output directory")
    if not (0 < val_fraction < 1 and bin_size > 0 and n_genes > 0 and hvg_bins_per_tile > 0):
        raise ValueError("Invalid feature/split parameters")
    if not 1 <= n_neighbors <= 441:
        raise ValueError("Neighbors must be in 1..441 for the original ten rings")
    manifest = json.loads((source / "manifest.json").read_text())
    all_ids = {t["id"] for t in manifest["tiles"]}
    if tile_ids is not None and (len(set(tile_ids)) != len(tile_ids) or not set(tile_ids) <= all_ids):
        raise ValueError("Unknown or duplicate tile IDs")
    tiles = [t for t in manifest["tiles"] if tile_ids is None or t["id"] in tile_ids]
    active = [t["id"] for t in tiles if t["records"] > 0]
    if len(active) < 2:
        raise ValueError("Shared training needs at least two nonempty regions including validation")
    rng = np.random.RandomState(seed)
    if validation_tiles is None:
        validation_tiles = list(rng.choice(sorted(active), max(1, min(len(active)-1,
                                 int(round(len(active)*val_fraction)))), replace=False))
    if not set(validation_tiles) <= set(active) or not 0 < len(set(validation_tiles)) < len(active):
        raise ValueError("Validation must be a proper nonempty subset of active tiles")
    validation_tiles = sorted(set(validation_tiles))
    train_ids = sorted(set(active) - set(validation_tiles))
    # A conservative core erosion includes the halo plus the RNA search radius.
    margin = max(t["halo"] for t in tiles) + bin_size * 10
    if any(min(t["width"], t["height"]) <= 2*margin for t in tiles if t["records"]):
        raise ValueError("Core too small for the train/validation isolation margin")
    # Keep the global bin grid consistent across tile/halo origins.
    if any(t["origin_x"] % bin_size or t["origin_y"] % bin_size for t in tiles):
        raise ValueError("Tile origins must align with the global bin grid")
    sampled = []
    sample_counts = {}
    with h5py.File(source / "spatial_index.h5") as index:
        if not index.attrs.get("complete", False):
            raise ValueError("Source spatial index is incomplete")
        genes = index["genes"][:]
        for tile in tiles:
            if tile["id"] not in train_ids:
                continue
            matrix = aggregate(index[tile["id"]][:], (tile["width"], tile["height"]),
                               (tile["x"], tile["y"]), bin_size, len(genes))
            rows = np.flatnonzero(np.asarray(matrix.getnnz(axis=1)).ravel())
            ny = math.ceil(tile["height"] / bin_size)
            xy = np.column_stack((rows // ny, rows % ny)) * bin_size + tile["halo"]
            rows = rows[interior_mask(xy, tile, margin)]
            take = rng.choice(rows, min(len(rows), hvg_bins_per_tile), replace=False)
            if len(take):
                sampled.append(matrix[take])
            sample_counts[tile["id"]] = len(take)
            print(f"HVG sample {tile['id']}: {len(take)} bins", flush=True)
    if not sampled:
        raise ValueError("No RNA bins remain in training core interiors")
    import anndata as ad
    import scanpy as sc
    data = ad.AnnData(sparse.vstack(sampled, format="csr"))
    del sampled
    supported = np.flatnonzero(np.asarray((data.X > 0).sum(axis=0)).ravel() >= 2)
    if len(supported) < n_genes:
        raise ValueError(f"Only {len(supported)} genes detected in >=2 sampled bins; increase "
                         "--hvg-bins-per-tile or explicitly request fewer genes")
    variable = data[:, supported].copy()
    sc.pp.highly_variable_genes(variable, n_top_genes=min(n_genes, len(supported)),
                               flavor="seurat_v3", span=1.0)
    selected = sorted(map(int, supported[np.asarray(variable.var.highly_variable)]))
    names = [bytes(g["geneID"] or g["geneName"]).decode().rstrip("\0") for g in genes]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate source gene identifiers need explicit reconciliation")
    schema = dict(version=1, source=str(source), source_manifest_sha256=fingerprint(manifest),
                  source_gene_names_sha256=fingerprint(names), seed=seed, bin_size=bin_size,
                  n_neighbors=n_neighbors, search_rings=10, core_margin=margin,
                  gene_indices=selected, genes=[names[i] for i in selected],
                  transform="raw summed bin counts; no normalization or per-tile HVG fitting",
                  gene_selection="seurat_v3 span=1; training core interior samples only",
                  hvg_bins_per_tile=hvg_bins_per_tile, hvg_sample_counts=sample_counts,
                  splits=dict(train=train_ids, validation=validation_tiles),
                  empty_tiles=[t["id"] for t in tiles if not t["records"]],
                  partial_scope=tile_ids is not None, tile_ids=[t["id"] for t in tiles])
    schema["fingerprint"] = fingerprint(schema)
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "manifest.json", dict(manifest, tiles=tiles))
    save_json(output / "schema.json", schema)
    print(f"Initialized {len(selected)} shared genes; train={len(train_ids)}, validation={len(validation_tiles)}", flush=True)
    return schema


def halo_records(source, tile):
    pieces = []
    ox, oy = tile["origin_x"], tile["origin_y"]
    width, height = tile["width"] + 2*tile["halo"], tile["height"] + 2*tile["halo"]
    with h5py.File(Path(source) / "spatial_index.h5") as index:
        for ix in range(max(0, tile["ix"]-1), tile["ix"]+2):
            for iy in range(max(0, tile["iy"]-1), tile["iy"]+2):
                key = f"x{ix:02d}_y{iy:02d}"
                if key not in index:
                    continue
                records = index[key][:]
                take = ((records["x"] >= ox) & (records["x"] < ox+width) &
                        (records["y"] >= oy) & (records["y"] < oy+height))
                pieces.append(records[take])
        return np.concatenate(pieces), len(index["genes"])


def staining(source, tile, manifest):
    """Reuse registered HE (read-only), or compute the identical HE conversion."""
    from tifffile import imread
    original = Path(source) / "tiles" / tile["id"]
    if (original / "prepared.json").exists():
        return imread(original / "hematoxylin.tif")
    import cv2
    from skimage.color import rgb2hed
    sys.path.insert(0, str(ROOT / "scripts"))
    import scs_whole_prepare as reference
    reference.RPI = Path(manifest["rpi"])
    w, h = tile["width"] + 2*tile["halo"], tile["height"] + 2*tile["halo"]
    rgb = reference.registered_rgb(tile["origin_x"], tile["origin_y"], w, h)
    hem = rgb2hed(rgb)[..., 0]
    positive = hem[hem > 0]
    if positive.size:
        low, high = np.percentile(positive, [1, 99.5])
        hem = np.clip((hem-low) / max(high-low, 1e-8), 0, 1)
    else:
        hem = np.zeros(hem.shape)
    return cv2.resize((hem*255).astype(np.uint8), (h, w), interpolation=cv2.INTER_CUBIC)


def prepare_tile(output, tile_id):
    output = Path(output).resolve()
    if tile_id not in load_schema(output)["tile_ids"]:
        raise ValueError("Unknown shared tile")
    directory = output / "tiles" / tile_id
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".prepare.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _prepare_tile(output, tile_id)


def _prepare_tile(output, tile_id):
    output = Path(output).resolve()
    schema = load_schema(output)
    manifest = json.loads((output / "manifest.json").read_text())
    source = Path(schema["source"])
    if fingerprint(json.loads((source / "manifest.json").read_text())) != schema["source_manifest_sha256"]:
        raise ValueError("Source manifest changed after feature selection")
    tile = next(t for t in manifest["tiles"] if t["id"] == tile_id)
    directory = output / "tiles" / tile_id
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / "prepared.json"
    if marker.exists():
        meta = json.loads(marker.read_text())
        if meta["fingerprint"] != schema["fingerprint"]:
            raise ValueError("Refusing stale prepared tile")
        return meta
    if not tile["records"]:
        meta = dict(state="empty", reason="zero source RNA records in core", fingerprint=schema["fingerprint"])
        save_json(marker, meta)
        save_json(directory / "completed.json", dict(state="empty", fingerprint=schema["fingerprint"]))
        return meta
    records, ng = halo_records(source, tile)
    shape = (tile["width"]+2*tile["halo"], tile["height"]+2*tile["halo"])
    bin_size = schema["bin_size"]
    grid = tuple(math.ceil(v/bin_size) for v in shape)
    expression = aggregate(records, shape, (tile["origin_x"], tile["origin_y"]), bin_size, ng)
    expression = expression[:, schema["gene_indices"]].tocsr()
    expression.eliminate_zeros()
    if expression.nnz and expression.data.max() <= 65535:
        expression = expression.astype(np.uint16)
    neighbors, occupied = neighbor_table(expression, grid, schema["n_neighbors"], schema["search_rings"])
    stain = staining(source, tile, manifest)
    if stain.shape != shape:
        raise ValueError("Registered stain and expression coordinates differ")
    import anndata as ad
    sys.path.insert(0, str(ROOT / "SCS"))
    from src import spateo_compat as st
    layers = SimpleNamespace(layers={"stain": stain})
    if len(np.unique(stain)) >= 4:
        st.cs.mask_nuclei_from_stain(layers, otsu_classes=4, otsu_index=1)
        st.cs.find_peaks_from_mask(layers, "stain", 7)
        st.cs.watershed(layers, "stain", 5, out_layer="watershed_labels")
        nuclei = layers.layers["watershed_labels"]
    else:
        nuclei = np.zeros(shape, dtype=np.int32)
    ids = np.unique(nuclei)
    ids = ids[ids > 0]
    centers = np.zeros((int(nuclei.max())+1, 2))
    if len(ids):
        centers[ids] = ndimage.center_of_mass(nuclei > 0, nuclei, ids)
    xy = np.column_stack((neighbors[:, 0] // grid[1], neighbors[:, 0] % grid[1])) * bin_size
    labels = nuclei[xy[:, 0], xy[:, 1]]
    binary = np.full(len(neighbors), -1, dtype=np.int8)
    binary[labels > 0] = 1
    distance = cKDTree(centers[ids]).query(xy)[0] if len(ids) else np.full(len(xy), np.inf)
    background = (labels == 0) & (stain[xy[:, 0], xy[:, 1]] <= 10) & (distance > 30)
    binary[background] = 0
    directions = np.zeros((len(neighbors), 2), dtype=np.float32)
    directions[labels > 0] = centers[labels[labels > 0]] - xy[labels > 0]
    eligible = interior_mask(xy, tile, schema["core_margin"]) & (binary >= 0)
    # Retain all eligible nucleus seeds and negatives; per-epoch sampling happens
    # in the trainer. Inference includes ALL supported centers, including halos.
    sparse.save_npz(directory / "expression.npz", expression)
    for key, value in dict(neighbors=neighbors, binary=binary, directions=directions, eligible=eligible).items():
        np.save(directory / (key + ".npy"), value, allow_pickle=False)
    (directory / "data").mkdir(exist_ok=True)
    # Full-resolution sparse RNA + original watershed support existing SCS postprocessing.
    x = records["x"].astype(np.int64) - tile["origin_x"]
    y = records["y"].astype(np.int64) - tile["origin_y"]
    counts = sparse.csr_matrix((records["count"].astype(np.int32), (x, y)), shape=shape)
    ad.AnnData(X=counts, layers={"stain": stain, "watershed_labels": nuclei}).write_h5ad(
        directory / "data/spots0:0:0:0.h5ad")
    meta = dict(state="prepared", fingerprint=schema["fingerprint"], grid_shape=list(grid),
                tile=tile_id, occupied_bins=occupied, supported_centers=len(neighbors),
                insufficient_neighbor_bins=occupied-len(neighbors), nuclei=len(ids),
                foreground_samples=int(np.sum(eligible & (binary == 1))),
                background_samples=int(np.sum(eligible & (binary == 0))),
                sparse_expression_bytes=int(expression.data.nbytes + expression.indices.nbytes + expression.indptr.nbytes),
                neighbor_bytes=neighbors.nbytes, expanded_float32_bytes=int(len(neighbors)*schema["n_neighbors"]*expression.shape[1]*4))
    save_json(marker, meta)
    print(json.dumps(meta), flush=True)
    return meta


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    init = sub.add_parser("init")
    init.add_argument("--source", required=True)
    init.add_argument("--output", required=True)
    init.add_argument("--n-genes", type=int, default=6000)
    init.add_argument("--n-neighbors", type=int, default=50)
    init.add_argument("--hvg-bins-per-tile", type=int, default=512)
    init.add_argument("--val-fraction", type=float, default=0.1)
    init.add_argument("--tiles", nargs="+", help="Explicit partial scope for smoke tests only")
    init.add_argument("--validation-tiles", nargs="+")
    prep = sub.add_parser("prepare")
    prep.add_argument("--output", required=True)
    prep.add_argument("--tiles", nargs="+")
    args = p.parse_args()
    if args.action == "init":
        initialize(args.source, args.output, n_genes=args.n_genes, n_neighbors=args.n_neighbors,
                   hvg_bins_per_tile=args.hvg_bins_per_tile, val_fraction=args.val_fraction,
                   tile_ids=args.tiles, validation_tiles=args.validation_tiles)
    else:
        schema = load_schema(args.output)
        for tile_id in args.tiles or schema["tile_ids"]:
            prepare_tile(args.output, tile_id)


if __name__ == "__main__":
    main()
