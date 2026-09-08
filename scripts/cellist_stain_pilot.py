import json
import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.color import rgb2hed
from skimage.feature import peak_local_max
from skimage.filters import threshold_multiotsu, threshold_local
from skimage.measure import label
from skimage.morphology import remove_small_holes
from skimage.segmentation import watershed
from tifffile import imread
from cellist_cpu_stage import ROOT, save_json
from scs_whole_prepare import registered_rgb

diagnosis = json.loads((ROOT / "runs/ST19_cellist/stain_diagnosis.json").read_text())
low, high = diagnosis["expression_tissue"][0], diagnosis["expression_tissue"][6]
hem = rgb2hed(registered_rgb(12600, 13600, 1200, 1200))[..., 0]
candidate = cv2.resize((np.clip((hem-low)/(high-low), 0, 1)*255).astype(np.uint8), (1200, 1200), interpolation=cv2.INTER_CUBIC)
original = imread(ROOT / "prepared/ST19_dense_1200/hematoxylin_registered.tif")
coords = pd.read_csv(ROOT / "prepared/ST19_dense_1200/expression.tsv", sep="\t", usecols=["x", "y"]).drop_duplicates()
reports = {}
for name, image, cutoff in [("original_pilot", original, threshold_multiotsu(original, 3)[0]),
                             ("tissue_global", candidate, diagnosis["tissue_normalized_otsu"][0]),
                             ("tissue_local", candidate, threshold_multiotsu(candidate, 3)[0])]:
    mask = remove_small_holes((image > cutoff) & (image > threshold_local(image, block_size=51)))
    distance = ndi.distance_transform_edt(mask)
    peaks = peak_local_max(distance, min_distance=6, exclude_border=False)
    marked = np.zeros(image.shape, dtype=bool)
    marked[tuple(peaks.T)] = True
    nuclei = watershed(-distance, label(marked), mask=mask)
    sizes = np.bincount(nuclei[coords.x.to_numpy(), coords.y.to_numpy()])
    reports[name] = dict(cutoff=int(cutoff), nuclei=int(nuclei.max()), eligible_nuclei=int((sizes[1:] >= 20).sum()),
                         foreground_fraction=float(mask.mean()), median_nucleus_pixels=float(np.median(np.bincount(nuclei.ravel())[1:])))
save_json(ROOT / "runs/ST19_cellist/stain_pilot_comparison.json", reports)
print(json.dumps(reports, indent=2), flush=True)
