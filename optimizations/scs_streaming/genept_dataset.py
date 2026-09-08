"""Pack all-gene sparse whole-slide SCS inputs for learned GenePT projection."""

import argparse
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
from scipy import ndimage, sparse
from scipy.spatial import cKDTree

from .genept import (
    file_sha256,
    load_genept_embeddings,
    make_gene_lookup,
    pool_spot_embeddings,
    source_gene_symbols,
)
from .shared_data import (
    direction_class_indices,
    interior_mask,
    load_schema,
    neighbor_table,
    random_split_rows,
    save_json,
)
REPRESENTATION = (
    "all-source-gene sparse spots with unified gene-semantics embeddings and "
    "learned projection"
)


def aggregate(records, shape, origin, bin_size, n_genes):
    """Aggregate raw molecule records into the shared 3x3 spatial grid."""
    grid_y = math.ceil(shape[1] / bin_size)
    x = (records["x"].astype(np.int64) - origin[0]) // bin_size
    y = (records["y"].astype(np.int64) - origin[1]) // bin_size
    return sparse.csr_matrix(
        (records["count"].astype(np.int32), (x * grid_y + y, records["gene"])),
        shape=(math.ceil(shape[0] / bin_size) * grid_y, n_genes),
        dtype=np.int32,
    )


def halo_records(source, tile):
    """Read the exact core plus halo records used by one prepared tile."""
    pieces = []
    origin_x, origin_y = tile["origin_x"], tile["origin_y"]
    width = tile["width"] + 2 * tile["halo"]
    height = tile["height"] + 2 * tile["halo"]
    with h5py.File(Path(source) / "spatial_index.h5") as index:
        for ix in range(max(0, tile["ix"] - 1), tile["ix"] + 2):
            for iy in range(max(0, tile["iy"] - 1), tile["iy"] + 2):
                key = f"x{ix:02d}_y{iy:02d}"
                if key not in index:
                    continue
                records = index[key][:]
                take = (
                    (records["x"] >= origin_x)
                    & (records["x"] < origin_x + width)
                    & (records["y"] >= origin_y)
                    & (records["y"] < origin_y + height)
                )
                pieces.append(records[take])
        if not pieces:
            raise ValueError(f"no spatial-index chunks found for {tile['id']}")
        return np.concatenate(pieces), len(index["genes"])


def open_array(directory, name, dtype, shape):
    return np.lib.format.open_memmap(
        directory / f"{name}.npy", mode="w+", dtype=dtype, shape=shape
    )


def write_samples(arrays, start, rows, tile, local_to_global, chunk_size):
    output_row = start
    for chunk_start in range(0, len(rows), chunk_size):
        selected = rows[chunk_start : chunk_start + chunk_size]
        local_neighbors = np.asarray(tile.neighbors[selected], dtype=np.int64)
        mapped = local_to_global[local_neighbors]
        if np.any(mapped < 0):
            raise ValueError("neighbor remapping is incomplete")
        stop = output_row + len(selected)
        arrays["neighbors"][output_row:stop] = mapped.astype(np.uint32)
        x = local_neighbors // tile.grid_shape[1]
        y = local_neighbors % tile.grid_shape[1]
        relative_x = (x - x[:, :1]) * tile.bin_size
        relative_y = (y - y[:, :1]) * tile.bin_size
        limits = np.iinfo(np.int8)
        if (
            relative_x.min(initial=0) < limits.min
            or relative_x.max(initial=0) > limits.max
            or relative_y.min(initial=0) < limits.min
            or relative_y.max(initial=0) > limits.max
        ):
            raise ValueError("relative positions do not fit int8")
        arrays["positions"][output_row:stop, :, 0] = relative_x.astype(np.int8)
        arrays["positions"][output_row:stop, :, 1] = relative_y.astype(np.int8)
        foreground = np.asarray(tile.binary[selected], dtype=np.uint8)
        arrays["foreground"][output_row:stop] = foreground
        arrays["directions"][output_row:stop] = direction_class_indices(
            tile.directions[selected], foreground
        ).astype(np.uint8)
        output_row = stop
    return output_row


def build_lookup(root, embedding_path):
    schema = load_schema(root)
    symbols = source_gene_symbols(root, schema)
    embeddings, source_dim = load_genept_embeddings(embedding_path)
    gene_embeddings, source_to_gene, gene_symbols, metadata = make_gene_lookup(
        symbols, embeddings
    )
    if np.any(source_to_gene < 0):
        missing = np.unique(
            np.asarray(symbols)[np.flatnonzero(source_to_gene < 0)]
        ).tolist()
        raise ValueError(
            "gene embedding coverage is incomplete; regenerate with an NCBI-backed "
            f"embedding table containing all source symbols. missing={len(missing)} "
            f"examples={missing[:10]}"
        )
    if source_dim != metadata["gene_embedding_dimension"]:
        raise ValueError("inconsistent gene embedding source dimension")
    metadata.update(
        embedding_file=str(Path(embedding_path).resolve()),
        embedding_sha256=file_sha256(embedding_path),
        spot_formula=(
            "sum(raw_count_i * unified_gene_embedding(gene_i)) / "
            "nonzero_mapped_genes, using all source genes rather than the old "
            "6000-HVG feature list"
        ),
        model_projection=(
            "trainable Linear(gene_embedding_dim, model_width) after spot pooling"
        ),
        efficient_order=(
            "project the fixed gene table, then sparse weighted pooling; "
            "mathematically identical to pooling native spot vectors first"
        ),
        divisor="number of nonzero mapped gene symbols in that spot",
    )
    return gene_embeddings, source_to_gene, gene_symbols, metadata


def mapped_tile_expression(source, tile_metadata, source_to_gene, n_genes, bin_size):
    """Aggregate every raw source gene and merge mapped duplicate symbols."""
    records, source_gene_count = halo_records(source, tile_metadata)
    if source_gene_count != len(source_to_gene):
        raise ValueError("source gene table changed during GenePT preparation")
    shape = (
        tile_metadata["width"] + 2 * tile_metadata["halo"],
        tile_metadata["height"] + 2 * tile_metadata["halo"],
    )
    raw = aggregate(
        records,
        shape,
        (tile_metadata["origin_x"], tile_metadata["origin_y"]),
        bin_size,
        source_gene_count,
    ).tocoo()
    compact_columns = source_to_gene[raw.col]
    keep = compact_columns >= 0
    compact = sparse.csr_matrix(
        (raw.data[keep], (raw.row[keep], compact_columns[keep])),
        shape=(raw.shape[0], n_genes),
        dtype=np.int32,
    )
    compact.eliminate_zeros()
    return compact, raw.nnz


def label_candidates(root, tile_metadata, neighbors, schema):
    """Reuse the saved nucleus segmentation to label a new all-gene spot graph."""
    path = Path(root) / "tiles" / tile_metadata["id"] / "data/spots0:0:0:0.h5ad"
    with h5py.File(path) as handle:
        stain = handle["layers/stain"][:]
        nuclei = handle["layers/watershed_labels"][:]
    expected_shape = (
        tile_metadata["width"] + 2 * tile_metadata["halo"],
        tile_metadata["height"] + 2 * tile_metadata["halo"],
    )
    if stain.shape != expected_shape or nuclei.shape != expected_shape:
        raise ValueError(f"saved segmentation shape changed for {tile_metadata['id']}")
    ids = np.unique(nuclei)
    ids = ids[ids > 0]
    centers = np.zeros((int(nuclei.max()) + 1, 2))
    if len(ids):
        centers[ids] = ndimage.center_of_mass(nuclei > 0, nuclei, ids)
    grid_shape = tuple(
        int(np.ceil(value / schema["bin_size"])) for value in expected_shape
    )
    center_rows = neighbors[:, 0]
    xy = (
        np.column_stack((center_rows // grid_shape[1], center_rows % grid_shape[1]))
        * schema["bin_size"]
    )
    labels = nuclei[xy[:, 0], xy[:, 1]]
    binary = np.full(len(neighbors), -1, dtype=np.int8)
    binary[labels > 0] = 1
    distance = (
        cKDTree(centers[ids]).query(xy)[0]
        if len(ids)
        else np.full(len(xy), np.inf)
    )
    background = (
        (labels == 0)
        & (stain[xy[:, 0], xy[:, 1]] <= 10)
        & (distance > 30)
    )
    binary[background] = 0
    directions = np.zeros((len(neighbors), 2), dtype=np.float32)
    directions[labels > 0] = centers[labels[labels > 0]] - xy[labels > 0]
    eligible = interior_mask(xy, tile_metadata, schema["core_margin"]) & (binary >= 0)
    return binary, directions, eligible, grid_shape


def split_tile_rows(tile, seed):
    result = {}
    for split in ("train", "validation"):
        positive = random_split_rows(
            np.flatnonzero(tile.eligible & (tile.binary == 1)),
            split,
            0.1,
            seed,
            f"{tile.tile_id}:positive",
        )
        negative = random_split_rows(
            np.flatnonzero(tile.eligible & (tile.binary == 0)),
            split,
            0.1,
            seed,
            f"{tile.tile_id}:negative",
        )
        rows = np.concatenate((positive, negative)).astype(np.int64, copy=False)
        result[split] = (rows, [len(positive), len(negative)])
    return result


def prepare_tile(root, schema, tile_metadata, source_to_gene, n_genes):
    expression, raw_nonzero = mapped_tile_expression(
        schema["source"],
        tile_metadata,
        source_to_gene,
        n_genes,
        schema["bin_size"],
    )
    spatial_shape = (
        tile_metadata["width"] + 2 * tile_metadata["halo"],
        tile_metadata["height"] + 2 * tile_metadata["halo"],
    )
    grid_shape = tuple(
        int(np.ceil(value / schema["bin_size"])) for value in spatial_shape
    )
    neighbors, occupied = neighbor_table(
        expression,
        grid_shape,
        schema["n_neighbors"],
        schema["search_rings"],
    )
    if not len(neighbors):
        return None, raw_nonzero, occupied
    binary, directions, eligible, grid_shape = label_candidates(
        root, tile_metadata, neighbors, schema
    )
    return (
        SimpleNamespace(
            tile_id=tile_metadata["id"],
            expression=expression,
            neighbors=neighbors,
            binary=binary,
            directions=directions,
            eligible=eligible,
            grid_shape=grid_shape,
            bin_size=schema["bin_size"],
        ),
        raw_nonzero,
        occupied,
    )


def source_tiles(root, schema):
    manifest = json.loads((Path(root) / "manifest.json").read_text())
    allowed = set(schema["tile_ids"])
    return [
        tile
        for tile in manifest["tiles"]
        if tile["id"] in allowed and tile["records"]
    ]


def probe(root, embedding_path, samples_per_tile=20):
    """Validate native all-gene GenePT pooling on independent spots."""
    root = Path(root).resolve()
    schema = load_schema(root)
    gene_embeddings, source_to_gene, _, metadata = build_lookup(root, embedding_path)
    rng = np.random.RandomState(schema["seed"])
    pooled, counts = [], []
    eligible_samples = 0
    for tile_metadata in source_tiles(root, schema):
        tile, _, _ = prepare_tile(
            root, schema, tile_metadata, source_to_gene, len(gene_embeddings)
        )
        if tile is None:
            continue
        candidates = np.flatnonzero(tile.eligible)
        eligible_samples += len(candidates)
        if not len(candidates):
            continue
        chosen = rng.choice(
            candidates, min(samples_per_tile, len(candidates)), replace=False
        )
        centers = np.asarray(tile.neighbors[chosen, 0])
        embedded, nonzero = pool_spot_embeddings(
            tile.expression[centers],
            gene_embeddings,
            np.ones(len(gene_embeddings), dtype=bool),
        )
        pooled.append(embedded)
        counts.append(nonzero)
    if not pooled:
        raise ValueError("no labeled all-gene GenePT spots were found")
    pooled = np.concatenate(pooled)
    counts = np.concatenate(counts)
    norms = np.linalg.norm(pooled, axis=1)
    dimension_std = pooled.std(axis=0)
    return {
        "genept": metadata,
        "eligible_samples": eligible_samples,
        "sampled_spots": len(pooled),
        "mapped_nonzero_genes": {
            "mean": float(counts.mean()),
            "median": float(np.median(counts)),
            "minimum": int(counts.min()),
            "maximum": int(counts.max()),
            "zero_spots": int(np.sum(counts == 0)),
        },
        "native_embedding_norm": {
            "mean": float(norms.mean()),
            "std": float(norms.std()),
            "minimum": float(norms.min()),
            "maximum": float(norms.max()),
        },
        "per_dimension_std": {
            "mean": float(dimension_std.mean()),
            "minimum": float(dimension_std.min()),
            "maximum": float(dimension_std.max()),
        },
    }


def merge(
    root,
    embedding_path,
    output_name="genept_allgenes_linear_random90",
    chunk_size=8192,
):
    root = Path(root).resolve()
    schema = load_schema(root)
    destination = root / output_name
    if destination.exists():
        raise FileExistsError(f"GenePT dataset already exists: {destination}")
    temporary = root / f".{output_name}.partial.{os.getpid()}"
    temporary.mkdir()
    try:
        gene_embeddings, source_to_gene, gene_symbols, genept_metadata = build_lookup(
            root, embedding_path
        )
        np.save(
            temporary / "gene_embeddings.npy",
            gene_embeddings.astype(np.float32, copy=False),
            allow_pickle=False,
        )
        np.save(
            temporary / "source_to_gene.npy", source_to_gene, allow_pickle=False
        )
        np.save(temporary / "gene_symbols.npy", gene_symbols, allow_pickle=False)

        plans = []
        split_sizes = {"train": 0, "validation": 0}
        split_counts = {"train": {}, "validation": {}}
        expression_rows = 0
        expression_nonzero = 0
        all_source_nonzero = 0
        all_mapped_nonzero = 0
        maximum_count = 0
        occupied_spots = 0
        supported_centers = 0
        tiles = source_tiles(root, schema)
        for tile_metadata in tiles:
            tile, raw_nonzero, occupied = prepare_tile(
                root, schema, tile_metadata, source_to_gene, len(gene_embeddings)
            )
            all_source_nonzero += int(raw_nonzero)
            occupied_spots += int(occupied)
            if tile is None:
                continue
            all_mapped_nonzero += int(tile.expression.nnz)
            supported_centers += len(tile.neighbors)
            splits = split_tile_rows(tile, schema["seed"])
            all_rows = np.concatenate([splits[name][0] for name in split_sizes])
            if not len(all_rows):
                continue
            used = np.unique(np.asarray(tile.neighbors[all_rows]).reshape(-1))
            compact = tile.expression[used]
            per_split_sizes = {}
            for split in split_sizes:
                rows, counts = splits[split]
                split_sizes[split] += len(rows)
                split_counts[split][tile.tile_id] = counts
                per_split_sizes[split] = len(rows)
            plan = {
                "tile_id": tile.tile_id,
                "expression_rows": len(used),
                "expression_nonzero": int(compact.nnz),
                "split_sizes": per_split_sizes,
            }
            plans.append(plan)
            expression_rows += len(used)
            expression_nonzero += int(compact.nnz)
            if compact.nnz:
                maximum_count = max(maximum_count, int(compact.data.max()))
            print(json.dumps({"planned": tile.tile_id, **plan}), flush=True)
            del tile, compact

        if not plans or not split_sizes["train"] or not split_sizes["validation"]:
            raise ValueError("all-gene GenePT preparation produced an empty split")
        if expression_rows > np.iinfo(np.uint32).max:
            raise OverflowError("global spot row indices do not fit uint32")
        if expression_nonzero > np.iinfo(np.uint32).max:
            raise OverflowError("sparse expression offsets do not fit uint32")
        if maximum_count > np.iinfo(np.uint16).max:
            raise OverflowError("aggregated expression counts do not fit uint16")

        expression_data = open_array(
            temporary, "expression_data", np.uint16, (expression_nonzero,)
        )
        expression_indices = open_array(
            temporary, "expression_indices", np.uint16, (expression_nonzero,)
        )
        expression_indptr = open_array(
            temporary, "expression_indptr", np.uint32, (expression_rows + 1,)
        )
        split_arrays = {
            split: {
                "neighbors": open_array(
                    temporary,
                    f"{split}_neighbors",
                    np.uint32,
                    (size, schema["n_neighbors"]),
                ),
                "positions": open_array(
                    temporary,
                    f"{split}_positions",
                    np.int8,
                    (size, schema["n_neighbors"], 2),
                ),
                "directions": open_array(
                    temporary, f"{split}_directions", np.uint8, (size,)
                ),
                "foreground": open_array(
                    temporary, f"{split}_foreground", np.uint8, (size,)
                ),
            }
            for split, size in split_sizes.items()
        }
        sample_offsets = {"train": 0, "validation": 0}
        row_offset = 0
        data_offset = 0
        metadata_by_id = {tile["id"]: tile for tile in tiles}
        for plan in plans:
            tile, _, _ = prepare_tile(
                root,
                schema,
                metadata_by_id[plan["tile_id"]],
                source_to_gene,
                len(gene_embeddings),
            )
            splits = split_tile_rows(tile, schema["seed"])
            all_rows = np.concatenate([splits[name][0] for name in split_sizes])
            used = np.unique(np.asarray(tile.neighbors[all_rows]).reshape(-1))
            compact = tile.expression[used].tocsr()
            if (
                len(used) != plan["expression_rows"]
                or compact.nnz != plan["expression_nonzero"]
            ):
                raise ValueError("all-gene tile changed between preparation passes")
            data_stop = data_offset + compact.nnz
            row_stop = row_offset + len(used)
            expression_data[data_offset:data_stop] = compact.data.astype(np.uint16)
            expression_indices[data_offset:data_stop] = compact.indices.astype(
                np.uint16
            )
            expression_indptr[row_offset : row_stop + 1] = (
                compact.indptr.astype(np.uint64) + data_offset
            ).astype(np.uint32)
            local_to_global = np.full(tile.expression.shape[0], -1, dtype=np.int64)
            local_to_global[used] = np.arange(row_offset, row_stop, dtype=np.int64)
            for split in split_sizes:
                rows = splits[split][0]
                if len(rows) != plan["split_sizes"][split]:
                    raise ValueError("sample split changed between preparation passes")
                sample_offsets[split] = write_samples(
                    split_arrays[split],
                    sample_offsets[split],
                    rows,
                    tile,
                    local_to_global,
                    chunk_size,
                )
            row_offset = row_stop
            data_offset = data_stop
            print(json.dumps({"packed": tile.tile_id, "rows": len(used)}), flush=True)
            del tile, compact

        if (
            row_offset != expression_rows
            or data_offset != expression_nonzero
            or sample_offsets != split_sizes
        ):
            raise ValueError("GenePT sparse output cardinality mismatch")
        arrays = [expression_data, expression_indices, expression_indptr]
        arrays.extend(
            array for split in split_arrays.values() for array in split.values()
        )
        for array in arrays:
            array.flush()
        del arrays, split_arrays
        manifest = {
            "version": 3,
            "complete": True,
            "schema_fingerprint": schema["fingerprint"],
            "seed": schema["seed"],
            "split_mode": "random",
            "validation_fraction": 0.1,
            "representation": REPRESENTATION,
            "support_definition": (
                "spots and 50-neighbor graph rebuilt from all source genes with "
                "a unified NCBI-backed gene embedding mapping; old 6000-HVG "
                "support is not used"
            ),
            "n_neighbors": schema["n_neighbors"],
            "n_genes": len(gene_embeddings),
            "gene_embedding_dimension": genept_metadata[
                "gene_embedding_dimension"
            ],
            "source_tiles_with_rna": len(tiles),
            "labeled_tiles": len(plans),
            "occupied_spots": occupied_spots,
            "supported_centers": supported_centers,
            "expression_rows": expression_rows,
            "expression_nonzero": expression_nonzero,
            "all_source_expression_nonzero": all_source_nonzero,
            "all_mapped_expression_nonzero": all_mapped_nonzero,
            "maximum_aggregated_gene_count": maximum_count,
            "genept": genept_metadata,
            "splits": {
                split: {"samples": split_sizes[split], "counts": split_counts[split]}
                for split in split_sizes
            },
        }
        save_json(temporary / "manifest.json", manifest)
        temporary.replace(destination)
        return manifest
    except BaseException as error:
        print(json.dumps({"failed": type(error).__name__, "message": str(error)}))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("probe", "build"))
    parser.add_argument("--root", required=True)
    parser.add_argument("--gene-embeddings", required=True)
    parser.add_argument("--output-name", default="genept_allgenes_linear_random90")
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--samples-per-tile", type=int, default=20)
    args = parser.parse_args()
    if args.chunk_size < 1 or args.samples_per_tile < 1:
        parser.error("chunk sizes must be positive")
    if args.action == "probe":
        result = probe(args.root, args.gene_embeddings, args.samples_per_tile)
    else:
        result = merge(
            args.root,
            args.gene_embeddings,
            args.output_name,
            args.chunk_size,
        )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
