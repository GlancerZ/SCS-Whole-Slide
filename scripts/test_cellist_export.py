"""Exercise seam identity, unique spot ownership, and exact count aggregation."""
import json
from pathlib import Path
import tempfile

import anndata as ad
import h5py
import numpy as np

from cellist_cpu_stage import save_json
from cellist_merge import merge
from scs_whole_merge import Union


def main():
    union = Union()
    assert union.join(("a", 1), ("b", 2))
    assert not union.join(("a", 3), ("b", 2)), "Never merge distinct nuclei from one tile"
    with tempfile.TemporaryDirectory() as temp:
        base = Path(temp)
        tiles = []
        dtype = np.dtype([("x", "u2"), ("y", "u2"), ("gene", "u2"), ("count", "u1")])
        records = [np.array([(1199, 52, 0, 2), (1199, 52, 1, 3), (10, 10, 0, 1)], dtype=dtype),
                   np.array([(1200, 52, 0, 5), (1201, 52, 1, 7), (1210, 100, 1, 11), (1211, 100, 1, 13)], dtype=dtype)]
        with h5py.File(base / "spatial_index.h5", "w") as f:
            f.attrs["complete"] = True
            f.create_dataset("genes", data=np.array([(b"G1", b"Gene1"), (b"G2", b"Gene2")],
                                                    dtype=[("geneID", "S10"), ("geneName", "S10")]))
            for ix in range(2):
                tile = dict(id=f"x{ix:02d}_y00", ix=ix, iy=0, x=ix*1200, y=0, width=1200, height=1200,
                            halo=60, origin_x=ix*1200-60, origin_y=-60,
                            records=len(records[ix]), umis=int(records[ix]["count"].sum()))
                tiles.append(tile)
                directory = base / "tiles" / tile["id"]
                directory.mkdir(parents=True)
                nuclei = np.zeros((1320, 1320), dtype=np.uint32)
                labels = nuclei.copy()
                ox, oy = tile["origin_x"], tile["origin_y"]
                nuclei[1196-ox:1204-ox, 50-oy:60-oy] = ix+1
                for gx in [1199, 1200, 1201]:
                    labels[gx-ox, 52-oy] = ix+1
                if ix:
                    nuclei[1210-ox:1215-ox, 100-oy:105-oy] = 3
                    labels[1210-ox:1212-ox, 100-oy] = 3
                np.savez_compressed(directory / "labels.npz", cells=labels, nuclei=nuclei)
                save_json(directory / "completed.json", dict(state="segmented"))
                f.create_dataset(tile["id"], data=records[ix])
        save_json(base / "manifest.json", dict(tiles=tiles, total_records=7, total_umis=42,
                  shape=[2400, 1200], method_scope="synthetic seam test"))
        # Missing work must block completion.
        marker = base / "tiles/x01_y00/completed.json"
        marker.unlink()
        try:
            merge(base)
        except RuntimeError as error:
            assert "unresolved" in str(error)
        else:
            raise AssertionError("An incomplete slide was exported")
        save_json(marker, dict(state="segmented"))
        merge(base)
        result = ad.read_h5ad(base / "merged/cell_counts.h5ad")
        np.testing.assert_array_equal(result.X.toarray(), [[7, 10], [0, 24]])
        report = json.loads((base / "merged/completed.json").read_text())
        assert report["source_umis"] == 42 and report["assigned_umis"] == 41
        assert report["observed_spots"] == 6 and report["assigned_spots"] == 5
        assert report["cell_count"] == 2
        with h5py.File(base / "merged/cell_labels.h5") as f:
            assert f["cell_labels"][1199, 52] == f["cell_labels"][1200, 52] == 1
            assert f["cell_labels"][1210, 100] == 2
        with h5py.File(base / "merged/spot_to_cell.h5") as f:
            xy = [(int(x), int(y)) for group in f.values() for x, y in zip(group["x"][:], group["y"][:])]
            assert len(xy) == len(set(xy)) == 6
        # Shared atlas identity must give the same correct matrix without an
        # overlap heuristic, even when local nucleus numbers differ.
        np.save(base / "tiles/x00_y00/nucleus_ids.npy", np.array([0, 100], dtype=np.uint32))
        np.save(base / "tiles/x01_y00/nucleus_ids.npy", np.array([0, 0, 100, 200], dtype=np.uint32))
        manifest = json.loads((base / "manifest.json").read_text())
        manifest["shared_nuclei"] = "synthetic common atlas"
        save_json(base / "manifest.json", manifest)
        merge(base)
        result = ad.read_h5ad(base / "merged/cell_counts.h5ad")
        np.testing.assert_array_equal(result.X.toarray(), [[7, 10], [0, 24]])
        report = json.loads((base / "merged/completed.json").read_text())
        assert report["identity_source"] == "whole-slide shared nucleus atlas"
        # A measured core with no eligible nucleus must remain in denominators
        # and spot export without creating an artificial cell.
        empty = dict(id="x02_y00", ix=2, iy=0, x=2400, y=0, width=1200, height=1200,
                     halo=60, origin_x=2340, origin_y=-60, records=1, umis=17)
        directory = base / "tiles/x02_y00"
        directory.mkdir()
        nuclei = np.zeros((1320, 1320), dtype=np.uint32)
        nuclei[69:72, 99:102] = 1
        np.savez_compressed(directory / "labels.npz", cells=np.zeros_like(nuclei), nuclei=nuclei)
        np.save(directory / "nucleus_ids.npy", np.array([0, 300], dtype=np.uint32))
        save_json(directory / "completed.json", dict(state="no_eligible_nuclei"))
        with h5py.File(base / "spatial_index.h5", "a") as f:
            f.create_dataset(empty["id"], data=np.array([(2410, 40, 0, 17)], dtype=dtype))
        manifest["tiles"].append(empty)
        manifest.update(shape=[3600, 1200], total_records=8, total_umis=59)
        save_json(base / "manifest.json", manifest)
        merge(base)
        report = json.loads((base / "merged/completed.json").read_text())
        assert report["source_umis"] == 59 and report["assigned_umis"] == 41
        assert report["observed_spots"] == 7 and report["cell_count"] == 2
        assert report["tiles_without_eligible_nuclei"] == 1
    print("PASS: incomplete-export guard, seam identity, no label collapse, unique ownership, exact per-gene UMI conservation")


if __name__ == "__main__":
    main()
