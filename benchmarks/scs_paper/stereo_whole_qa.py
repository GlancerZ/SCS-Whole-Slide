"""Small, data-derived registration overlays for manual pipeline QA."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def panel(stain,rna,title,size=(600,600)):
    s=cv2.resize(stain.astype(np.float32),size,interpolation=cv2.INTER_AREA)
    r=cv2.resize(rna.astype(np.float32),size,interpolation=cv2.INTER_AREA)
    s=np.clip(s/max(float(np.percentile(s,99.5)),1.),0,1)
    r=np.log1p(r)
    r=np.clip(r/max(float(np.percentile(r,99.5)),1e-6),0,1)
    rgb=np.stack((r,s,np.zeros_like(s)),axis=-1)
    bgr=(rgb[:,:,::-1]*255).astype(np.uint8).copy()
    cv2.putText(bgr,title,(12,25),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
    return bgr


def main(root):
    original=np.load(root/'stain_source.npy',mmap_mode='r')
    aligned=np.load(root/'stain_aligned.npy',mmap_mode='r')
    rna=np.load(root/'pixel_umi.npy',mmap_mode='r')
    alignment=json.loads((root/'patch_alignment_completed.json').read_text())
    tiles=alignment['patches']
    rows=[np.concatenate((panel(original,rna,'Whole RNA box - before'),
                          panel(aligned,rna,'Whole RNA box - after')),axis=1)]
    # Representative high-RNA, median-RNA and greatest-transform patches.
    tissue=[t for t in tiles if t['umis']>0 and t['nuclei']>0]
    ordered=sorted(tissue,key=lambda t:t['umis'])
    selected=[ordered[-1],ordered[len(ordered)//2],max(tissue,key=lambda t:t.get('max_corner_displacement',0))]
    for tile in selected:
        r,c,h,w=[tile[k] for k in ['row','col','height','width']]
        sl=np.s_[r:r+h,c:c+w]
        rows.append(np.concatenate((panel(original[sl],rna[sl],tile['id']+' before'),
                                    panel(aligned[sl],rna[sl],tile['id']+' after')),axis=1))
    cv2.imwrite(str(root/'alignment_qa.png'),np.concatenate(rows,axis=0))
    print(json.dumps(dict(patches=len(tiles),no_nuclei=sum(t['nuclei']==0 for t in tiles),
        nuclei=alignment['nuclei'],max_displacement=max(t.get('max_corner_displacement',0) for t in tiles),
        numeric_guard_passed=alignment['numeric_guard_passed'],preview_patches=[t['id'] for t in selected]),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    main(p.parse_args().root)
