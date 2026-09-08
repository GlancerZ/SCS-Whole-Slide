#!/usr/bin/env python3
"""Lossless spatial index and registered, overlapping inputs for whole-slide SCS."""
import argparse
import json
import math
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
from skimage.color import rgb2hed
from tifffile import imwrite

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "runs/ST19_whole_scs"
GEF = ROOT / "ST19/visualization/visualization/A05956D4.tissue.gef"
RPI = ROOT / "ST19/visualization/visualization/A05956D4.rpi"
DTYPE = np.dtype([("x", "<u2"), ("y", "<u2"), ("gene", "<u2"), ("count", "u1")])


def save_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def build_index():
    BASE.mkdir(parents=True, exist_ok=True)
    final = BASE / "spatial_index.h5"
    manifest_path = BASE / "manifest.json"
    if final.exists() and manifest_path.exists():
        print("Verified index already exists", flush=True)
        return
    temp = BASE / "spatial_index.building.h5"
    with h5py.File(GEF) as src, h5py.File(temp, "w") as dst:
        genes = src["geneExp/bin1/gene"][:]
        expr = src["geneExp/bin1/expression"]
        nx = int(expr.attrs["maxX"][0]) + 1
        ny = int(expr.attrs["maxY"][0]) + 1
        assert nx <= 65536 and ny <= 65536 and len(genes) <= 65536
        size, halo = 1200, 60
        ncy = math.ceil(ny / size)
        ncx = math.ceil(nx / size)
        ends = genes["offset"].astype(np.int64) + genes["count"]
        assert np.all(ends[:-1] <= ends[1:]) and ends[-1] == len(expr)
        assert np.array_equal(genes["offset"].astype(np.int64), np.r_[0, ends[:-1]])
        dst.create_dataset("genes", data=genes)
        tiles = []
        datasets = []
        for ix in range(ncx):
            for iy in range(ncy):
                tile_id = f"x{ix:02d}_y{iy:02d}"
                datasets.append(dst.create_dataset(tile_id, (0,), maxshape=(None,), dtype=DTYPE,
                                                   chunks=(65536,), compression="lzf"))
                x, y = ix * size, iy * size
                tiles.append(dict(id=tile_id, ix=ix, iy=iy, x=x, y=y,
                                  width=min(size, nx-x), height=min(size, ny-y),
                                  halo=halo, origin_x=x-halo, origin_y=y-halo,
                                  records=0, umis=0))
        nrecords = 0
        numis = 0
        for start in range(0, len(expr), 2_000_000):
            raw = expr[start:start+2_000_000]
            assert np.all(raw["x"] >= 0) and np.all(raw["y"] >= 0)
            assert np.all(raw["x"] < nx) and np.all(raw["y"] < ny)
            chunk = np.empty(len(raw), dtype=DTYPE)
            for field in ("x", "y", "count"):
                chunk[field] = raw[field]
            chunk["gene"] = np.searchsorted(ends, np.arange(start, start+len(raw)), side="right")
            keys = (raw["x"] // size) * ncy + raw["y"] // size
            order = np.argsort(keys, kind="stable")
            chunk = chunk[order]
            keys = keys[order]
            bounds = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1]) + 1, len(keys)]
            for a, b in zip(bounds[:-1], bounds[1:]):
                k = int(keys[a]); ds = datasets[k]; previous = len(ds)
                ds.resize((previous + b-a,)); ds[previous:] = chunk[a:b]
                tiles[k]["records"] += int(b-a)
                tiles[k]["umis"] += int(chunk["count"][a:b].sum(dtype=np.uint64))
            nrecords += len(raw)
            numis += int(raw["count"].sum(dtype=np.uint64))
            if start % 20_000_000 == 0:
                print(f"Index {nrecords:,}/{len(expr):,} records", flush=True)
        coarse_umis = int(src["wholeExp/bin100"][:]["MIDcount"].sum(dtype=np.uint64))
        assert nrecords == len(expr) == sum(t["records"] for t in tiles)
        assert numis == sum(t["umis"] for t in tiles)
        assert numis == coarse_umis, (numis, coarse_umis)
        dst.attrs["complete"] = True
        manifest = dict(gef=str(GEF), rpi=str(RPI), shape=[nx, ny], core_size=size,
                        halo=halo, total_records=nrecords, total_umis=numis,
                        source_gene_count=len(genes), tiles=tiles)
    temp.replace(final)
    save_json(manifest_path, manifest)
    print(f"Index complete: {nrecords:,} records, {numis:,} UMIs, {len(tiles)} tiles", flush=True)


def registered_rgb(x0, y0, width, height, layer="Image"):
    assert all(v % 2 == 0 for v in (x0, y0, width, height))
    with h5py.File(RPI) as rpi:
        group = rpi[f"HE/{layer}/bin_2"]
        nx, ny = int(group.attrs["sizex"]), int(group.attrs["sizey"])
        x0, y0, width, height = x0//2, y0//2, width//2, height//2
        result = np.full((height, width, 3), 255, dtype=np.uint8)
        for tx in range(max(0, x0)//256, math.ceil(min(nx, x0+width)/256)):
            for ty in range(max(0, y0)//256, math.ceil(min(ny, y0+height)/256)):
                tile = group[str(tx)][str(ty)][:]
                xa, ya = max(x0, tx*256), max(y0, ty*256)
                xb, yb = min(x0+width, tx*256+tile.shape[1]), min(y0+height, ty*256+tile.shape[0])
                result[ya-y0:yb-y0, xa-x0:xb-x0] = tile[ya-ty*256:yb-ty*256, xa-tx*256:xb-tx*256]
        return result.transpose(1, 0, 2)


def prepare_tile(tile):
    directory = BASE / "tiles" / tile["id"]
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "prepared.json").exists():
        return
    halo = tile["halo"]
    ox, oy = tile["origin_x"], tile["origin_y"]
    width, height = tile["width"]+2*halo, tile["height"]+2*halo
    records = []
    with h5py.File(BASE / "spatial_index.h5") as idx:
        for ix in range(max(0, tile["ix"]-1), tile["ix"]+2):
            for iy in range(max(0, tile["iy"]-1), tile["iy"]+2):
                key = f"x{ix:02d}_y{iy:02d}"
                if key not in idx:
                    continue
                chunk = idx[key][:]
                take = ((chunk["x"] >= ox) & (chunk["x"] < ox+width) &
                        (chunk["y"] >= oy) & (chunk["y"] < oy+height))
                records.append(chunk[take])
        records = np.concatenate(records)
        records = records[np.argsort(records["gene"], kind="stable")]
        genes = idx["genes"][:]
    names = np.array([bytes(g["geneID"] or g["geneName"]).decode().rstrip("\0") for g in genes])
    data = pd.DataFrame(dict(geneID=names[records["gene"]], x=records["x"].astype(int)-ox,
                            y=records["y"].astype(int)-oy, MIDCounts=records["count"]))
    data.to_csv(directory / "expression.tsv", sep="\t", index=False)
    rgb = registered_rgb(ox, oy, width, height)
    hem = rgb2hed(rgb)[..., 0]
    positive = hem[hem > 0]
    if positive.size:
        low, high = np.percentile(positive, [1, 99.5])
        hem = np.clip((hem-low) / max(high-low, 1e-8), 0, 1)
    else:
        hem = np.zeros(hem.shape)
    stain = cv2.resize((hem*255).astype(np.uint8), (height, width), interpolation=cv2.INTER_CUBIC)
    imwrite(directory / "hematoxylin.tif", stain, photometric="minisblack")
    save_json(directory / "prepared.json", dict(**tile, halo_records=len(records),
              halo_umis=int(records["count"].sum(dtype=np.uint64)), image_shape=list(stain.shape)))
    print(f"Prepared {tile['id']}: {len(records):,} records, image {stain.shape}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["index", "tile"])
    parser.add_argument("--tile")
    args = parser.parse_args()
    if args.action == "index":
        build_index()
    else:
        manifest = json.loads((BASE / "manifest.json").read_text())
        prepare_tile(next(t for t in manifest["tiles"] if t["id"] == args.tile))
