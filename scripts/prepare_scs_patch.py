#!/usr/bin/env python3
"""Extract a tissue-rich, registered Stereo-seq patch for an SCS run."""

import argparse
import json
from pathlib import Path

import cv2
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d
from skimage.color import rgb2hed
from tifffile import imwrite


def choose_dense_patch(gef, patch_size, coarse_bin=100):
    level = f"wholeExp/bin{coarse_bin}"
    whole = gef[level]["MIDcount"].astype(np.float64)
    window = int(np.ceil(patch_size / coarse_bin))
    scores = convolve2d(whole, np.ones((window, window)), mode="valid")
    coarse_x, coarse_y = np.unravel_index(np.argmax(scores), scores.shape)
    return int(coarse_x * coarse_bin), int(coarse_y * coarse_bin), float(scores[coarse_x, coarse_y])


def read_rpi_window(rpi, x0, y0, patch_size, image_bin=2, tile_size=256):
    """Return RGB in SCS axes (x, y, channel), not conventional image axes."""
    if x0 % image_bin or y0 % image_bin or patch_size % image_bin:
        raise ValueError("Patch coordinates must be divisible by the RPI image bin.")
    x0b, y0b = x0 // image_bin, y0 // image_bin
    sizeb = patch_size // image_bin
    group = rpi[f"HE/Image/bin_{image_bin}"]
    conventional = np.zeros((sizeb, sizeb, 3), dtype=np.uint8)

    # RPI uses group index for the horizontal/x tile and dataset index for
    # the vertical/y tile. SCS treats x as the first matrix axis, hence the
    # final transpose.
    col_first = x0b // tile_size
    col_last = (x0b + sizeb - 1) // tile_size
    row_first = y0b // tile_size
    row_last = (y0b + sizeb - 1) // tile_size
    for col_tile in range(col_first, col_last + 1):
        for row_tile in range(row_first, row_last + 1):
            tile = group[str(col_tile)][str(row_tile)][:]
            tile_y0 = row_tile * tile_size
            tile_x0 = col_tile * tile_size
            src_y0 = max(y0b, tile_y0) - tile_y0
            src_x0 = max(x0b, tile_x0) - tile_x0
            src_y1 = min(y0b + sizeb, tile_y0 + tile.shape[0]) - tile_y0
            src_x1 = min(x0b + sizeb, tile_x0 + tile.shape[1]) - tile_x0
            dst_y0 = max(tile_y0, y0b) - y0b
            dst_x0 = max(tile_x0, x0b) - x0b
            dst_y1 = dst_y0 + (src_y1 - src_y0)
            dst_x1 = dst_x0 + (src_x1 - src_x0)
            conventional[dst_y0:dst_y1, dst_x0:dst_x1] = tile[src_y0:src_y1, src_x0:src_x1]
    return conventional.transpose(1, 0, 2)


def hematoxylin_image(rgb_scs, patch_size):
    hematoxylin = rgb2hed(rgb_scs)[..., 0]
    positive = hematoxylin[hematoxylin > 0]
    if positive.size == 0:
        raise ValueError("No positive hematoxylin signal was found in the selected image patch.")
    low, high = np.percentile(positive, (1, 99.5))
    scaled = np.clip((hematoxylin - low) / max(high - low, np.finfo(float).eps), 0, 1)
    scaled = (scaled * 255).astype(np.uint8)
    return cv2.resize(scaled, (patch_size, patch_size), interpolation=cv2.INTER_CUBIC)


def write_expression_patch(gef, output_file, x0, y0, patch_size):
    genes = gef["geneExp/bin1/gene"]
    expression = gef["geneExp/bin1/expression"]
    total_records = 0
    with output_file.open("w", encoding="utf-8", buffering=1024 * 1024) as handle:
        handle.write("geneID\tx\ty\tMIDCounts\n")
        for index, gene in enumerate(genes):
            offset = int(gene["offset"])
            count = int(gene["count"])
            records = expression[offset : offset + count]
            keep = (
                (records["x"] >= x0)
                & (records["x"] < x0 + patch_size)
                & (records["y"] >= y0)
                & (records["y"] < y0 + patch_size)
            )
            records = records[keep]
            if records.size:
                raw_gene = gene["geneID"] or gene["geneName"]
                gene_id = raw_gene.decode("utf-8").rstrip("\x00")
                lines = (
                    f"{gene_id}\t{int(record['x']) - x0}\t{int(record['y']) - y0}\t{int(record['count'])}"
                    for record in records
                )
                handle.write("\n".join(lines))
                handle.write("\n")
                total_records += int(records.size)
            if (index + 1) % 2000 == 0:
                print(f"Scanned {index + 1}/{len(genes)} genes; retained {total_records:,} records", flush=True)
    return total_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gef", required=True, type=Path)
    parser.add_argument("--rpi", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--patch-size", type=int, default=1200)
    parser.add_argument("--x", type=int)
    parser.add_argument("--y", type=int)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.gef, "r") as gef, h5py.File(args.rpi, "r") as rpi:
        if (args.x is None) != (args.y is None):
            raise ValueError("Specify both --x and --y, or neither.")
        if args.x is None:
            x0, y0, coarse_score = choose_dense_patch(gef, args.patch_size)
        else:
            x0, y0, coarse_score = args.x, args.y, None
        max_x = int(gef["geneExp/bin1/expression"].attrs["maxX"][0]) + 1
        max_y = int(gef["geneExp/bin1/expression"].attrs["maxY"][0]) + 1
        if x0 < 0 or y0 < 0 or x0 + args.patch_size > max_x or y0 + args.patch_size > max_y:
            raise ValueError("Selected patch is outside the GEF coordinate bounds.")

        print(f"Selected patch x={x0}:{x0 + args.patch_size}, y={y0}:{y0 + args.patch_size}", flush=True)
        rgb_scs = read_rpi_window(rpi, x0, y0, args.patch_size)
        stain = hematoxylin_image(rgb_scs, args.patch_size)
        rna = gef["wholeExp/bin1"][x0 : x0 + args.patch_size, y0 : y0 + args.patch_size]["MIDcount"]
        expression_file = args.output_dir / "expression.tsv"
        retained = write_expression_patch(gef, expression_file, x0, y0, args.patch_size)

    stain_file = args.output_dir / "hematoxylin_registered.tif"
    imwrite(stain_file, stain, photometric="minisblack")

    figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    axes[0].imshow(rgb_scs.transpose(1, 0, 2))
    axes[0].set_title("Registered H&E")
    axes[1].imshow(stain.T, cmap="gray")
    axes[1].set_title("Hematoxylin for SCS")
    axes[2].imshow(np.log1p(rna).T, cmap="magma")
    axes[2].set_title("log1p RNA UMIs")
    for axis in axes:
        axis.set_axis_off()
    figure.savefig(args.output_dir / "input_qa.png", dpi=150)
    plt.close(figure)

    metadata = {
        "gef": str(args.gef.resolve()),
        "rpi": str(args.rpi.resolve()),
        "x": x0,
        "y": y0,
        "patch_size": args.patch_size,
        "retained_expression_records": retained,
        "coarse_density_score": coarse_score,
        "expression_file": str(expression_file.resolve()),
        "stain_file": str(stain_file.resolve()),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
