"""Read-only coordinate/annotation audit before benchmarking a second platform."""
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from scipy.io import loadmat
from skimage.draw import polygon
import tifffile

root=Path('runs/SCS_paper_benchmark_v1/seqfish_source')
sys.path.insert(0,str(root/'vendor'))
from roifile import ImagejRoi

z=zipfile.ZipFile(root/'seqFISH_NIH3T3_point_locations.zip')
tot=loadmat(io.BytesIO(z.read('RNA_locations_run_1.mat')))['tot']
rois=zipfile.ZipFile(root/'ROIs_Experiment1_NIH3T3.zip')
images=zipfile.ZipFile(root/'DAPI_experiment1.zip')
report=[]
for fov in (0,1):
    mask=np.zeros((2048,2048),np.int32)
    files=sorted(n for n in rois.namelist() if n.startswith(f'ALL_Roi/RoiSet_Pos{fov}/') and n.endswith('.roi'))
    for label,name in enumerate(files,1):
        roi=ImagejRoi.frombytes(rois.read(name))
        xy=roi.coordinates()
        rr,cc=polygon(xy[:,1],xy[:,0],shape=mask.shape)
        mask[rr,cc]=label
    xy=np.concatenate([v for v in tot[fov].flat if v.size])[:,:2]
    orientations=[]
    for swap in (False,True):
        for shift in (0,-1):
            p=xy[:,::-1] if swap else xy
            p=np.floor(p+shift).astype(int)
            valid=(p>=0).all(1)&(p<2048).all(1)
            fraction=float((mask[p[valid,1],p[valid,0]]>0).mean())
            orientations.append(dict(swap_xy=swap,shift_pixels=shift,inside_roi_fraction=fraction))
    with tifffile.TiffFile(io.BytesIO(images.read(f'final_background_experiment1/MMStack_Pos{fov}.ome.tif'))) as t:
        info=dict(series=[dict(shape=s.shape,axes=s.axes,dtype=str(s.dtype)) for s in t.series],
                  image_metadata=t.pages[0].description)
    report.append(dict(fov=fov,points=len(xy),roi_count=len(files),orientations=orientations,
                       available_cell_slots=tot.shape[1],nonempty_cell_slots=sum(any(v.size for v in row) for row in tot[fov]),
                       image=info))
(root/'coordinate_audit.json').write_text(json.dumps(report,indent=2))
for r in report:
    print(json.dumps({k:v for k,v in r.items() if k!='image'}),flush=True)
