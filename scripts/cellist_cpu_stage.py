#!/usr/bin/env python3
"""Run official Cellist CPU stages on registered ST19 inputs.

Load the upstream numerical modules without its eager optional registration and
Cellpose imports. No image registration is estimated here. Images use x,y axes.
"""
import argparse
import json
import os
import resource
from pathlib import Path
import sys
import time
import types
import warnings

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "tools/Cellist/src/Cellist"


def save_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def load_core():
    # The official package __init__ imports Spateo registration and Cellpose even
    # for CPU watershed/seg. These unrelated imports are unnecessary for this run.
    package = types.ModuleType("Cellist")
    package.__path__ = [str(UPSTREAM)]
    sys.modules["Cellist"] = package
    import Cellist.Segmentation as seg
    import Cellist.Watershed as ws
    warnings.filterwarnings("ignore", message="Setting an item of incompatible dtype", category=FutureWarning)
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    original_figure = plt.figure

    def bounded_figure(*args, **kwargs):
        # Upstream requests 80-inch plots; retain plots at a practical size.
        if "figsize" in kwargs:
            w, h = kwargs["figsize"]
            factor = min(1, 14 / max(w, h))
            kwargs["figsize"] = (w * factor, h * factor)
        return original_figure(*args, **kwargs)

    plt.figure = bounded_figure
    return ws, seg


def inspect_inputs():
    import h5py
    import numpy as np
    from tifffile import imread
    report = dict(host=os.uname().nodename, python=sys.executable,
                  base_python=sys.base_prefix, job_id=os.environ.get("SLURM_JOB_ID"))
    with h5py.File(ROOT / "ST19/visualization/visualization/A05956D4.tissue.gef") as f:
        def plain(value):
            value = np.asarray(value).tolist()
            if isinstance(value, bytes):
                return value.decode()
            if isinstance(value, list):
                return [plain(item) for item in value]
            return value
        report["gef_attributes"] = {k: plain(v) for k, v in f.attrs.items()}
        report["expression_attributes"] = {k: np.asarray(v).tolist() for k, v in f["geneExp/bin1/expression"].attrs.items()}
        report["expression_records"] = len(f["geneExp/bin1/expression"])
    image = imread(ROOT / "prepared/ST19_dense_1200/hematoxylin_registered.tif")
    assert image.shape == (1200, 1200) and image.dtype == np.uint8
    report["pilot_image_shape"] = list(image.shape)
    report["pilot_image_percentiles"] = np.percentile(image, [0, 25, 50, 75, 100]).tolist()
    save_json(ROOT / "runs/ST19_cellist/setup/input_inspection.json", report)
    print(json.dumps(report, indent=2), flush=True)


def quality_check(args):
    import h5py
    import numpy as np
    import pandas as pd
    from scipy import sparse
    from tifffile import imread
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from skimage.segmentation import find_boundaries
    output = Path(args.output)
    candidates = list((output / "segmentation").glob("*/ST19_Cellist_segmentation.txt"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one segmentation, found {len(candidates)}")
    result_dir = candidates[0].parent
    assert (result_dir / "parameters.json").exists(), "Cellist did not finish"
    df = pd.read_csv(candidates[0], sep="\t")
    assert not df.duplicated(["x", "y"]).any(), "Duplicate spot assignments"
    image = imread(args.image)
    x, y = df.x.to_numpy(), df.y.to_numpy()
    assert np.all((x >= 0) & (x < image.shape[0]) & (y >= 0) & (y < image.shape[1]))
    ids = df.Cellist.fillna(0).to_numpy()
    assert np.all(ids >= 0) and np.all(ids == ids.astype(np.uint32))
    labels = np.zeros(image.shape, dtype=np.uint32)
    labels[x, y] = ids.astype(np.uint32)
    source = pd.read_csv(args.gem, sep="\t")
    assert len(df) == len(source.drop_duplicates(["x", "y"])), "Missing expression spots"
    record_ids = labels[source.x.to_numpy(), source.y.to_numpy()]
    counts = source.iloc[:, 3].to_numpy(dtype=np.uint64)
    assigned_umis = int(counts[record_ids > 0].sum())
    cells = np.unique(ids[ids > 0])
    with h5py.File(result_dir / "ST19_Cellist_segmentation_cell_count.h5") as f:
        matrix_umis = float(f["matrix/data"][:].sum(dtype=np.float64))
        matrix_cells = int(f["matrix/shape"][1])
    assert matrix_umis == assigned_umis, (matrix_umis, assigned_umis)
    assert matrix_cells == len(cells), (matrix_cells, len(cells))
    nuclei = sparse.load_npz(output / "nuclei/ST19_Watershed_nucleus_matrix.npz").toarray().astype(np.uint32)
    np.savez_compressed(output / "labels.npz", cells=labels, nuclei=nuclei)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    axes[0].imshow(image.T, cmap="gray")
    axes[0].contour(find_boundaries(nuclei).T, levels=[0.5], colors=["cyan"], linewidths=0.25)
    axes[0].set_title("Hematoxylin + watershed nuclei")
    palette = ListedColormap(["black", *list(plt.get_cmap("tab20").colors)[:19]])
    axes[1].imshow(np.where(labels.T > 0, labels.T % 19 + 1, 0), cmap=palette, vmin=0, vmax=19)
    axes[1].set_title("Cellist: expression spot assignments")
    axes[2].imshow(image.T, cmap="gray")
    assigned = ids > 0
    axes[2].scatter(x[assigned][::4], y[assigned][::4], c=palette((ids[assigned][::4] % 19 + 1).astype(int)),
                    s=0.1, alpha=0.5, rasterized=True)
    axes[2].set_title("Assigned spots over hematoxylin")
    for ax in axes:
        ax.set_axis_off()
    fig.savefig(output / "segmentation_qa.png", dpi=160)
    plt.close(fig)
    report = dict(state="segmented", cell_count=len(cells), expression_spots=len(df),
                  assigned_spots=int(assigned.sum()), assigned_spot_fraction=float(assigned.mean()),
                  source_records=len(source), source_umis=int(counts.sum()), assigned_umis=assigned_umis,
                  assigned_umi_fraction=assigned_umis/int(counts.sum()), matrix_umis=matrix_umis,
                  initial_nuclei=int(np.count_nonzero(np.unique(nuclei))),
                  axis_order="x,y", result_dir=str(result_dir),
                  representation="Cellist labels at observed RNA spots; zero means unassigned or no RNA",
                  input_gem=str(Path(args.gem).resolve()), input_image=str(Path(args.image).resolve()))
    save_json(output / "completed.json", report)
    print(json.dumps(report, indent=2), flush=True)


def shared_watershed(ws, args, nuclei):
    """Format a crop of common whole-slide seeds as official Cellist inputs."""
    import numpy as np
    import pandas as pd
    from scipy import sparse
    from skimage.measure import regionprops_table
    from tifffile import memmap
    from Cellist.IO import gem_to_mat
    output = Path(args.output)
    tile = json.loads((output / "prepared.json").read_text())
    common = memmap(Path(args.shared_nuclei) / "nuclei.tif", mode="r")
    ox, oy = tile["origin_x"], tile["origin_y"]
    w, h = tile["width"]+2*tile["halo"], tile["height"]+2*tile["halo"]
    region = np.zeros((w, h), dtype=np.uint32)
    x0, y0 = max(ox, 0), max(oy, 0)
    x1, y1 = min(ox+w, common.shape[0]), min(oy+h, common.shape[1])
    region[x0-ox:x1-ox, y0-oy:y1-oy] = common[x0:x1, y0:y1]
    identities = np.unique(np.r_[np.uint32(0), region.ravel()])
    segmented = np.searchsorted(identities, region).astype(np.uint32)
    np.save(output / "nucleus_ids.npy", identities)
    nuclei.mkdir(exist_ok=True)
    sparse.save_npz(nuclei / "ST19_Watershed_nucleus_matrix.npz", sparse.csc_matrix(segmented))
    props = regionprops_table(segmented, properties=["label", "area", "centroid", "equivalent_diameter_area"])
    pd.DataFrame(props).to_csv(nuclei / "ST19_Watershed_nucleus_property.txt", sep="\t", index=False)
    gem = pd.read_csv(args.gem, sep="\t")
    coords = gem[["x", "y"]].drop_duplicates()
    sizes = np.bincount(segmented[coords.x.to_numpy(), coords.y.to_numpy()])
    if not np.any(sizes[1:] >= 20):
        # Cellist's eligibility test requires >=20 observed nuclear spots in
        # a slice. If no nucleus has 20 in the whole input, no slice can pass.
        np.savez_compressed(output / "labels.npz", cells=np.zeros_like(segmented), nuclei=segmented)
        save_json(output / "completed.json", dict(state="no_eligible_nuclei", cell_count=0,
                  expression_spots=len(coords), source_records=len(gem),
                  source_umis=int(gem.iloc[:, 3].sum()), assigned_spots=0, assigned_umis=0,
                  reason="No nucleus has the 20 observed RNA spots required by Cellist"))
        return
    gem = gem_to_mat(gem, str(nuclei / "ST19_bin1.h5"), countname=gem.columns[3])
    coord = gem[["x", "y", "x_y"]].drop_duplicates(["x_y"])
    nucleus_coords = ws.write_segmentation_coord(segmented, coord, str(nuclei), "ST19")
    ws.write_segmentation_cell(nucleus_coords, gem, "ST19", str(nuclei))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["inspect", "watershed", "seg", "qa"])
    parser.add_argument("--gem", default=str(ROOT / "prepared/ST19_dense_1200/expression.tsv"))
    parser.add_argument("--image", default=str(ROOT / "prepared/ST19_dense_1200/hematoxylin_registered.tif"))
    parser.add_argument("--output", default=str(ROOT / "runs/ST19_cellist/pilot"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resolution", type=float, default=0.5)
    parser.add_argument("--shared-nuclei")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run on the allocated compute node")
    if args.stage == "inspect":
        inspect_inputs()
        return
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.stage == "qa":
        quality_check(args)
        return
    ws, seg = load_core()
    start = time.monotonic()
    nuclei = output / "nuclei"
    if args.stage == "watershed":
        if args.shared_nuclei:
            shared_watershed(ws, args, nuclei)
        else:
            ws.Watershed(platform="barcoding", gem_path=args.gem, img_path=args.image,
                     out_dir=str(nuclei), out_prefix="ST19", min_distance=6,
                     no_local_threshold=False, expansion=False, expansion_dist=8)
    else:
        seg.Cellist(platform="barcoding", resolution=args.resolution, nucleus_seg_method="Watershed",
                    props_file=str(nuclei / "ST19_Watershed_nucleus_property.txt"),
                    nucleus_count_h5_file=str(nuclei / "ST19_Watershed_segmentation_cell_count.h5"),
                    nucleus_coord_file=str(nuclei / "ST19_Watershed_nucleus_coord.txt"),
                    all_spot_count_h5_file=str(nuclei / "ST19_bin1.h5"), spot_expr_file=args.gem,
                    patch_data_dir=None, num_workers=args.workers, alpha=0.8, sigma=1.0, beta=10,
                    gene_use="HVG", max_dist=15, two_step=False, cyto=False,
                    max_dist_s1_scale=0.5, noise_prop_s1=0.4, noise_prop=0.25, neigh_dist=2.5,
                    out_dir=str(output / "segmentation"), out_prefix="ST19")
    save_json(output / f"{args.stage}.done.json", dict(elapsed_seconds=time.monotonic()-start,
              parent_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              max_child_peak_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss))


if __name__ == "__main__":
    main()
