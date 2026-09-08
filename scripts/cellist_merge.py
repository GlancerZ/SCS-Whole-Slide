#!/usr/bin/env python3
"""Reconcile Cellist nucleus identities, own cores once, and conserve raw UMIs."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse

from cellist_cpu_stage import ROOT, save_json
from scs_whole_merge import Union, overlap_matches


def arrays(base, tile):
    with np.load(base / "tiles" / tile["id"] / "labels.npz") as f:
        return f["cells"], f["nuclei"]


def core(tile, array):
    h = tile["halo"]
    return array[h:h+tile["width"], h:h+tile["height"]]


def aggregate_core(records, labels, n_cells, n_genes):
    """Records must use local core coordinates. Zero is an unassigned spot."""
    ids = labels[records["x"], records["y"]]
    assigned = ids > 0
    matrix = sparse.coo_matrix((records["count"][assigned].astype(np.uint32),
                               (ids[assigned]-1, records["gene"][assigned])),
                              shape=(n_cells, n_genes)).tocsr()
    return ids, matrix


def merge(base):
    base = Path(base)
    manifest = json.loads((base / "manifest.json").read_text())
    tiles, unresolved = [], []
    tile_states = {}
    for tile in manifest["tiles"]:
        marker = base / "tiles" / tile["id"] / "completed.json"
        if not marker.exists():
            unresolved.append(tile["id"])
            continue
        status = json.loads(marker.read_text())
        tile_states[tile["id"]] = status["state"]
        if status["state"] in ("segmented", "no_eligible_nuclei"):
            tiles.append(tile)
        elif status["state"] != "no_expression" or tile["records"] != 0:
            unresolved.append(tile["id"])
    if unresolved:
        raise RuntimeError(f"Refusing whole-slide export: unresolved tiles {unresolved}")
    out = base / "merged"
    out.mkdir(exist_ok=True)
    union = Union()
    core_ids = {}
    lookup_tiles = {tile["id"]: tile for tile in tiles}
    shared = bool(manifest.get("shared_nuclei"))
    identities = {}
    for tile in tiles:
        labels, nuclei = arrays(base, tile)
        expected = (tile["width"]+2*tile["halo"], tile["height"]+2*tile["halo"])
        assert labels.shape == nuclei.shape == expected
        assert set(np.unique(labels)) <= set(np.unique(nuclei)) | {0}
        for ident in np.unique(nuclei):
            if ident:
                union.find((tile["id"], int(ident)))
        core_ids[tile["id"]] = np.unique(core(tile, labels))
        if shared:
            identities[tile["id"]] = np.load(base / "tiles" / tile["id"] / "nucleus_ids.npy")
    if shared:
        canonical = {}
        for tile in tiles:
            for local, global_nucleus in enumerate(identities[tile["id"]]):
                if global_nucleus:
                    key = (tile["id"], local)
                    if int(global_nucleus) in canonical:
                        assert union.join(key, canonical[int(global_nucleus)])
                    else:
                        canonical[int(global_nucleus)] = key
    matches = []
    for tile in tiles:
        if shared:
            break  # Shared atlas IDs establish exact identity; no heuristic matching.
        _, a = arrays(base, tile)
        for dx, dy in [(0, 1), (1, -1), (1, 0), (1, 1)]:
            tid = f"x{tile['ix']+dx:02d}_y{tile['iy']+dy:02d}"
            if tid not in lookup_tiles:
                continue
            other = lookup_tiles[tid]
            _, b = arrays(base, other)
            x0, y0 = max(tile["origin_x"], other["origin_x"]), max(tile["origin_y"], other["origin_y"])
            x1 = min(tile["origin_x"]+a.shape[0], other["origin_x"]+b.shape[0])
            y1 = min(tile["origin_y"]+a.shape[1], other["origin_y"]+b.shape[1])
            if x1 <= x0 or y1 <= y0:
                continue
            aa = a[x0-tile["origin_x"]:x1-tile["origin_x"], y0-tile["origin_y"]:y1-tile["origin_y"]]
            bb = b[x0-other["origin_x"]:x1-other["origin_x"], y0-other["origin_y"]:y1-other["origin_y"]]
            for ia, ib, pixels, fraction in overlap_matches(aa, bb):
                accepted = union.join((tile["id"], ia), (tid, ib))
                matches.append(dict(tile_a=tile["id"], nucleus_a=ia, tile_b=tid, nucleus_b=ib,
                                    pixels=pixels, fraction=fraction, accepted=accepted))
    roots = sorted({union.find((tid, int(ident))) for tid, ids in core_ids.items() for ident in ids if ident})
    global_id = {root: i+1 for i, root in enumerate(roots)}
    n_cells = len(roots)
    assert n_cells > 0
    spot_counts = np.zeros(n_cells+1, dtype=np.uint64)
    sum_x = np.zeros(n_cells+1, dtype=np.float64)
    sum_y = np.zeros(n_cells+1, dtype=np.float64)
    matrix_rows, matrix_cols, matrix_data = [], [], []
    total_records = total_umis = assigned_umis = observed_spots = 0
    shape = tuple(manifest["shape"])
    preview = np.zeros(tuple((v+9)//10 for v in shape), dtype=np.uint32)
    with h5py.File(base / "spatial_index.h5") as index, \
            h5py.File(out / "cell_labels.building.h5", "w") as hf, \
            h5py.File(out / "spot_to_cell.building.h5", "w") as spot_file:
        assert bool(index.attrs["complete"])
        genes = index["genes"][:]
        labels_ds = hf.create_dataset("cell_labels", shape=shape, dtype="u4", fillvalue=0,
                                      chunks=tuple(min(v, 600) for v in shape), compression="gzip", compression_opts=1)
        hf.attrs["axis_order"] = "x,y"
        hf.attrs["resolution_um"] = 0.5
        hf.attrs["representation"] = "Cellist assignments at observed RNA spots; 0 = unassigned or no RNA"
        spot_file.attrs["axis_order"] = "x,y"
        spot_file.attrs["resolution_um"] = 0.5
        for tile in tiles:
            labels, _ = arrays(base, tile)
            labels = core(tile, labels)
            lookup = np.zeros(int(labels.max())+1, dtype=np.uint32)
            for ident in core_ids[tile["id"]]:
                if ident:
                    lookup[ident] = global_id[union.find((tile["id"], int(ident)))]
            labels = lookup[labels]
            x0, y0, w, h = tile["x"], tile["y"], tile["width"], tile["height"]
            labels_ds[x0:x0+w, y0:y0+h] = labels
            preview[x0//10:(x0+w+9)//10, y0//10:(y0+h+9)//10] = labels[::10, ::10]
            xx, yy = np.nonzero(labels)
            ids = labels[xx, yy]
            spot_counts += np.bincount(ids, minlength=n_cells+1).astype(np.uint64)
            sum_x += np.bincount(ids, weights=xx+x0, minlength=n_cells+1)
            sum_y += np.bincount(ids, weights=yy+y0, minlength=n_cells+1)
            records = index[tile["id"]][:]
            assert len(records) == tile["records"]
            assert int(records["count"].sum(dtype=np.uint64)) == tile["umis"]
            assert np.all((records["x"] >= x0) & (records["x"] < x0+w) &
                          (records["y"] >= y0) & (records["y"] < y0+h))
            total_records += len(records)
            total_umis += int(records["count"].sum(dtype=np.uint64))
            records["x"] -= x0
            records["y"] -= y0
            record_ids, matrix = aggregate_core(records, labels, n_cells, len(genes))
            assigned_umis += int(records["count"][record_ids > 0].sum(dtype=np.uint64))
            coo = matrix.tocoo()
            matrix_rows.append(coo.row)
            matrix_cols.append(coo.col)
            matrix_data.append(coo.data)
            # Each RNA spot is stored once, including unassigned spots.
            keys = np.unique(records["x"].astype(np.int64)*h + records["y"])
            sx, sy = keys // h, keys % h
            observed_spots += len(keys)
            group = spot_file.create_group(tile["id"])
            group.create_dataset("x", data=(sx+x0).astype(np.uint32), compression="gzip")
            group.create_dataset("y", data=(sy+y0).astype(np.uint32), compression="gzip")
            group.create_dataset("cell_id", data=labels[sx, sy], compression="gzip")
            print(f"Aggregated {tile['id']}: {len(keys):,} spots, {len(ids):,} assigned", flush=True)
        assert total_records == manifest["total_records"]
        assert total_umis == manifest["total_umis"]
        hf.attrs["complete"] = True
        hf.attrs["cell_count"] = n_cells
        spot_file.attrs["complete"] = True
    assert np.all(spot_counts[1:] > 0)
    counts = sparse.coo_matrix((np.concatenate(matrix_data),
                               (np.concatenate(matrix_rows), np.concatenate(matrix_cols))),
                              shape=(n_cells, len(genes))).tocsr()
    counts.sum_duplicates()
    del matrix_rows, matrix_cols, matrix_data
    assert int(counts.sum(dtype=np.uint64)) == assigned_umis
    obs = pd.DataFrame(dict(cell_id=np.arange(1, n_cells+1), n_spots=spot_counts[1:],
                            centroid_x=sum_x[1:]/spot_counts[1:], centroid_y=sum_y[1:]/spot_counts[1:],
                            total_umis=np.asarray(counts.sum(axis=1)).ravel(),
                            n_genes=np.diff(counts.indptr)), index=[str(i) for i in range(1, n_cells+1)])
    if shared:
        obs["nucleus_id"] = [identities[tid][local] for tid, local in roots]
    decode = lambda value: bytes(value).decode().rstrip("\0")
    var = pd.DataFrame(dict(gene_name=[decode(g["geneName"]) for g in genes]),
                       index=[decode(g["geneID"] or g["geneName"]) for g in genes])
    assert var.index.is_unique
    import anndata as ad
    adata = ad.AnnData(X=counts, obs=obs, var=var)
    adata.obsm["spatial"] = obs[["centroid_x", "centroid_y"]].to_numpy()
    adata.uns["resolution_um"] = 0.5
    adata.uns["method"] = manifest["method_scope"]
    adata.uns["cellist_parameters"] = dict(version="1.1.1", alpha=0.8, sigma=1.0, beta=10.0,
                                            gene_use="HVG", noise_prop=0.25, cell_radius_um=15,
                                            imputation_distance_um=2.5, two_step=False,
                                            core_size_bins=manifest.get("core_size", 1200),
                                            halo_bins=manifest.get("halo", 60))
    if shared:
        nucleus_metadata = Path(manifest["shared_nuclei"]) / "completed.json"
        if nucleus_metadata.exists():
            adata.uns["nucleus_preprocessing_json"] = nucleus_metadata.read_text()
    adata.uns["segmentation_representation"] = "Cellist assignments at observed RNA spots, not continuous membrane masks"
    adata.write_h5ad(out / "cell_counts.building.h5ad", compression="gzip")
    obs.to_csv(out / "cell_stats.csv", index=False)
    save_json(out / "overlap_matches.json", matches)
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    palette = ListedColormap(["black", *list(plt.get_cmap("tab20").colors)[:19]])
    plt.imsave(out / "segmentation_overview.png", np.where(preview.T > 0, preview.T % 19 + 1, 0),
               cmap=palette, vmin=0, vmax=19)
    summary = dict(complete=True, scope=manifest.get("scope", "whole_slide"), cell_count=n_cells, gene_count=len(genes), source_records=total_records,
                   source_umis=total_umis, assigned_umis=assigned_umis,
                   assigned_umi_fraction=assigned_umis/total_umis,
                   observed_spots=observed_spots, assigned_spots=int(spot_counts.sum()),
                   assigned_spot_fraction=int(spot_counts.sum())/observed_spots,
                   tiles_total=len(manifest["tiles"]), expression_tiles_processed=len(tiles),
                   tiles_segmented=sum(state == "segmented" for state in tile_states.values()),
                   tiles_without_eligible_nuclei=sum(state == "no_eligible_nuclei" for state in tile_states.values()),
                   accepted_nucleus_overlap_matches=sum(m["accepted"] for m in matches),
                   shape=list(shape), axis_order="x,y", resolution_um=0.5,
                   identity_source="whole-slide shared nucleus atlas" if shared else "nucleus overlap heuristic",
                   limitations=[manifest["method_scope"] + "; not one monolithic official run",
                                "Observed-spot assignments are not continuous whole-cell membrane masks",
                                "No manual ground-truth validation; Cellist internal slice boundary artifacts may remain"])
    for name in ["cell_labels.h5", "spot_to_cell.h5", "cell_counts.h5ad"]:
        path = Path(name)
        (out / f"{path.stem}.building{path.suffix}").replace(out / name)
    save_json(out / "completed.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=ROOT / "runs/ST19_cellist/whole_shared_nuclei_tissue")
    merge(parser.parse_args().base)
