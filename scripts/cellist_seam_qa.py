"""Describe agreement of two independently segmented overlapping ST19 tiles."""
import json
import numpy as np
from cellist_cpu_stage import ROOT, save_json
from cellist_merge import arrays
from scs_whole_merge import overlap_matches


def main():
    base = ROOT / "runs/ST19_cellist/whole_shared_nuclei_tissue"
    manifest = json.loads((base / "manifest.json").read_text())
    tiles = {tile["id"]: tile for tile in manifest["tiles"]}
    a, b = tiles["x10_y11"], tiles["x11_y11"]
    la, na = arrays(base, a)
    lb, nb = arrays(base, b)
    x0, y0 = max(a["origin_x"], b["origin_x"]), max(a["origin_y"], b["origin_y"])
    x1 = min(a["origin_x"]+la.shape[0], b["origin_x"]+lb.shape[0])
    y1 = min(a["origin_y"]+la.shape[1], b["origin_y"]+lb.shape[1])
    def crop(tile, array):
        return array[x0-tile["origin_x"]:x1-tile["origin_x"], y0-tile["origin_y"]:y1-tile["origin_y"]]
    la, lb, na, nb = crop(a, la), crop(b, lb), crop(a, na), crop(b, nb)
    matches = overlap_matches(na, nb)
    mapping = np.zeros(int(max(na.max(), la.max()))+1, dtype=np.uint32)
    for ia, ib, _, _ in matches:
        mapping[ia] = ib
    both = (la > 0) & (lb > 0)
    assert both.any() and matches
    ida = np.load(base / "tiles" / a["id"] / "nucleus_ids.npy")
    idb = np.load(base / "tiles" / b["id"] / "nucleus_ids.npy")
    same = ida[la[both]] == idb[lb[both]]
    assert np.array_equal(ida[na], idb[nb]), "The common nucleus atlas must agree exactly"
    report = dict(tile_a=a["id"], tile_b=b["id"], overlap_shape=list(la.shape),
                  nuclei_a=int(np.count_nonzero(np.unique(na))), nuclei_b=int(np.count_nonzero(np.unique(nb))),
                  mutual_best_nucleus_matches=len(matches), spots_assigned_by_both=int(both.sum()),
                  matched_cell_identity_fraction=float(same.mean()),
                  note="Descriptive seam agreement, not segmentation accuracy against manual truth")
    save_json(ROOT / "runs/ST19_cellist/seam_qa_shared_nuclei_tissue.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
