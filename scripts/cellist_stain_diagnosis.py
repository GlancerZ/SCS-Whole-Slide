import json
import h5py
import numpy as np
from skimage.color import rgb2hed
from skimage.filters import threshold_multiotsu
from cellist_cpu_stage import ROOT, save_json
from scs_whole_prepare import registered_rgb

rgb = registered_rgb(0, 0, 23520, 23520)
hem = rgb2hed(rgb)[..., 0].copy()
with h5py.File(ROOT / "ST19/visualization/visualization/A05956D4.tissue.gef") as f:
    coarse = f["wholeExp/bin100"][:]["MIDcount"] > 0
tissue = np.repeat(np.repeat(coarse, 50, axis=0), 50, axis=1)[:hem.shape[0], :hem.shape[1]]
quantiles = [1, 25, 50, 75, 95, 99, 99.5, 99.9]
report = dict(quantiles=quantiles,
              full=np.percentile(hem[hem > 0], quantiles).tolist(),
              expression_tissue=np.percentile(hem[tissue & (hem > 0)], quantiles).tolist(),
              pilot=np.percentile(hem[6300:6900, 6800:7400], quantiles).tolist())
low, high = np.percentile(hem[tissue & (hem > 0)], [1, 99.5])
image = (np.clip((hem-low)/(high-low), 0, 1)*255).astype(np.uint8)
report["tissue_normalized_otsu"] = threshold_multiotsu(image[tissue], 3).tolist()
report["tissue_fraction"] = float(tissue.mean())
save_json(ROOT / "runs/ST19_cellist/stain_diagnosis.json", report)
print(json.dumps(report, indent=2), flush=True)
