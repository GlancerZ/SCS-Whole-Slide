#!/usr/bin/env python3
"""Apply the Cellist watershed definition once to the complete registered H&E."""
import gc
import json
import os
import resource
import time
import cv2
import h5py
import numpy as np
from scipy import ndimage as ndi
from skimage.color import rgb2hed
from skimage.feature import peak_local_max
from skimage.feature import peak as peak_module
from skimage._shared.coord import ensure_spacing
from skimage.filters import threshold_local, threshold_multiotsu
from skimage.measure import label
from skimage.morphology import remove_small_holes
from skimage.segmentation import watershed
from tifffile import imwrite
from cellist_cpu_stage import ROOT, save_json
from scs_whole_prepare import registered_rgb

BASE = ROOT / "runs/ST19_cellist/shared_nuclei_tissue"


def main():
    assert os.environ.get("SLURM_JOB_ID"), "Run in the CPU allocation"
    BASE.mkdir(parents=True, exist_ok=True)
    if (BASE / "completed.json").exists():
        print("Shared nuclei already complete", flush=True)
        return
    start = time.monotonic()
    manifest = json.loads((ROOT / "runs/ST19_whole_scs/manifest.json").read_text())
    nx, ny = manifest["shape"]
    print("Reading registered whole-slide bin2 H&E", flush=True)
    rgb = registered_rgb(0, 0, nx, ny)
    hem = rgb2hed(rgb)[..., 0].copy()
    del rgb
    gc.collect()
    with h5py.File(ROOT / "ST19/visualization/visualization/A05956D4.tissue.gef") as f:
        expression_coverage = f["wholeExp/bin100"][:]["MIDcount"] > 0
    tissue_small = np.repeat(np.repeat(expression_coverage, 50, axis=0), 50, axis=1)[:nx//2, :ny//2]
    positive = hem[tissue_small & (hem > 0)]
    low, high = np.percentile(positive, [1, 99.5])
    del positive
    hem -= low
    hem /= max(high-low, 1e-8)
    np.clip(hem, 0, 1, out=hem)
    small = (hem*255).astype(np.uint8)
    del hem
    gc.collect()
    image = cv2.resize(small, (ny, nx), interpolation=cv2.INTER_CUBIC)
    del small
    imwrite(BASE / "hematoxylin.tif", image, photometric="minisblack", bigtiff=True)
    print("Estimating a smooth spatial multi-Otsu threshold within expression-covered tissue", flush=True)
    tissue = cv2.resize(tissue_small.astype(np.uint8), (ny, nx), interpolation=cv2.INTER_NEAREST).astype(bool)
    del tissue_small
    size, halo = manifest["core_size"], manifest["halo"]
    xs, ys = np.arange(0, nx, size), np.arange(0, ny, size)
    centers_x = xs + np.minimum(size, nx-xs)/2
    centers_y = ys + np.minimum(size, ny-ys)/2
    threshold_grid = np.full((len(xs), len(ys)), np.nan)
    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            xa, xb = max(0, x-halo), min(nx, x+size+halo)
            ya, yb = max(0, y-halo), min(ny, y+size+halo)
            values = image[xa:xb, ya:yb][tissue[xa:xb, ya:yb]]
            if len(values) >= 1024 and len(np.unique(values)) >= 3:
                threshold_grid[ix, iy] = threshold_multiotsu(values, 3)[0]
    invalid = np.isnan(threshold_grid)
    assert not invalid.all()
    nearest = ndi.distance_transform_edt(invalid, return_distances=False, return_indices=True)
    threshold_grid[invalid] = threshold_grid[tuple(nearest[:, invalid])]
    np.save(BASE / "threshold_grid.npy", threshold_grid)
    print("Applying spatial multi-Otsu and 51-bin local Gaussian threshold", flush=True)
    local_threshold = threshold_local(image, block_size=51, offset=0)
    # For uint8 images and nonnegative local thresholds this is exactly the
    # conjunction in Cellist.Watershed.Thresholding; avoid redundant copies.
    mask = image > local_threshold
    # Bilinearly interpolate between true core centers, with constant edge
    # extrapolation, so segmentation input has no threshold jumps at tile seams.
    interp_y = np.stack([np.interp(np.arange(ny), centers_y, row) for row in threshold_grid])
    interp_x = np.interp(np.arange(nx), centers_x, np.arange(len(xs)))
    for x, fractional_index in enumerate(interp_x):
        left = int(fractional_index)
        right = min(left+1, len(xs)-1)
        weight = fractional_index-left
        cutoff = (1-weight)*interp_y[left] + weight*interp_y[right]
        mask[x] &= image[x] > cutoff
    # Keep a 100-bin margin around observed expression so nearby nuclei remain
    # available; avoid segmenting unmeasured black image padding.
    support = ndi.binary_dilation(expression_coverage, iterations=1)
    support = np.repeat(np.repeat(support, 100, axis=0), 100, axis=1)[:nx, :ny]
    mask &= support
    del support, tissue
    del local_threshold, image
    mask = remove_small_holes(mask)
    gc.collect()
    print("Whole-slide distance transform", flush=True)
    distance = ndi.distance_transform_edt(mask)
    print("Finding Cellist watershed markers, min_distance=6", flush=True)
    def spacing_once(coords, **kwargs):
        # The default repeatedly rebuilds a KDTree over all previously accepted
        # points in batches of 2,000. One batch has identical greedy ordering
        # and avoids quadratic repeated work on a whole slide.
        print(f"Spacing {len(coords):,} peak candidates in one batch", flush=True)
        return ensure_spacing(coords, min_split_size=None, **kwargs)
    peak_module.ensure_spacing = spacing_once
    coords = peak_local_max(distance, min_distance=6, exclude_border=False)
    peaks = np.zeros(distance.shape, dtype=bool)
    peaks[tuple(coords.T)] = True
    markers = label(peaks)
    marker_count = int(markers.max())
    del peaks, coords
    gc.collect()
    print(f"Running watershed with {marker_count:,} markers", flush=True)
    distance *= -1
    nuclei = watershed(distance, markers, mask=mask).astype(np.uint32)
    del distance, markers
    gc.collect()
    imwrite(BASE / "nuclei.building.tif", nuclei, photometric="minisblack", bigtiff=True)
    (BASE / "nuclei.building.tif").replace(BASE / "nuclei.tif")
    report = dict(complete=True, shape=[nx, ny], axis_order="x,y", resolution_um=0.5,
                  min_distance=6, normalization_low=float(low), normalization_high=float(high),
                  threshold_method="Spatial multi-Otsu on expression-covered tissue, bilinear core-center interpolation",
                  threshold_grid=threshold_grid.tolist(), nuclei=marker_count,
                  foreground_pixels=int(np.count_nonzero(mask)),
                  elapsed_seconds=time.monotonic()-start,
                  peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    save_json(BASE / "completed.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
