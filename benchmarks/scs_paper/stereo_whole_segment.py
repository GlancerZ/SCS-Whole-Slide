"""Apply identical original SCS postprocessing and export full-coordinate masks.

Independent original patches retain distinct cell IDs at patch seams. This is
explicit, not an unvalidated cross-patch cell merge. Missing patches are a hard
failure, never silently represented as successfully segmented background.
"""
import argparse
import gzip
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys

import anndata as ad
import h5py
import numpy as np
from scipy import sparse

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from benchmarks.scs_paper.stereo_whole_prepare import write_json
from benchmarks.scs_paper.evaluate import segment


def segment_patch(job):
    root,method,tile=job
    data=root/'segmentation_patches'/tile['id']
    out=data/method
    if (out/'segmentation_completed.json').exists():
        return tile['id']
    data.mkdir(parents=True,exist_ok=True)
    (data/'data').mkdir(exist_ok=True)
    out.mkdir(exist_ok=True)
    r,c,h,w=[tile[k] for k in ['row','col','height','width']]
    nuclei=np.array(np.load(root/'nuclei.npy',mmap_mode='r')[r:r+h,c:c+w])
    # Compact nucleus labels locally before SCS Python postprocessing.
    local_ids=np.unique(nuclei[nuclei>0])
    if len(local_ids):
        nuclei=np.where(nuclei>0,np.searchsorted(local_ids,nuclei)+1,0).astype(np.int32)
    if not (data/'data/spots0:0:0:0.h5ad').exists():
        stain=np.array(np.load(root/'stain_aligned.npy',mmap_mode='r')[r:r+h,c:c+w])
        a=ad.AnnData(sparse.csr_matrix((h,w),dtype=np.int32),
            layers={'stain':stain,'watershed_labels':nuclei})
        a.write_h5ad(data/'data/spots0:0:0:0.h5ad')
    write_json(data/'prepared.json',dict(bin_size=3,radius_for_postprocessing=15,
        origin=[r,c],shape=[h,w],parent=str(root.resolve())))
    start,stop=tile['sample_start'],tile['sample_stop']
    if start==stop:
        np.savez_compressed(out/'segmentation.npz',cells=np.zeros((h,w),np.uint32),nuclei=nuclei)
        write_json(out/'segmentation_completed.json',dict(status='no valid 50-token centres',
            cells=0,covered=False,reason='insufficient occupied neighbours'))
        return tile['id']
    prediction=root/method
    completion=json.loads((prediction/'inference_completed.json').read_text())
    if completion['world_size']!=1:
        raise ValueError('Two-dataset export expects one training worker per dataset')
    coords=np.load(root/'coords.npy',mmap_mode='r')[start:stop]-[r,c]
    logits=np.load(prediction/'logits_rank0.npy',mmap_mode='r')[start:stop]
    binary=np.load(prediction/'foreground_rank0.npy',mmap_mode='r')[start:stop]
    with (out/'spot_prediction.txt').open('w') as f:
        for xy,dl,b in zip(coords,logits,binary):
            f.write(f'{xy[0]}\t{xy[1]}\t{b:.8g}\t'+':'.join(f'{v:.8g}' for v in dl)+'\n')
    segment(data,method,radius=15)
    return tile['id']


def export(root,method,tiles):
    out=root/method/'whole_slide'
    out.mkdir(exist_ok=True)
    if (out/'completed.json').exists():
        return
    meta=json.loads((root/'prepared.json').read_text())
    index_path=root/'indexed.json'
    source_shape=json.loads(index_path.read_text()).get('source_image_shape') if index_path.exists() else None
    patches=[]
    for tile in tiles:
        marker=root/'segmentation_patches'/tile['id']/method/'segmentation_completed.json'
        if not marker.exists():
            raise ValueError(f'Missing segmentation for {tile["id"]}; refuse full export')
        patches.append(json.loads(marker.read_text()))
    count=0
    total_area=0
    with h5py.File(out/'cell_labels.building.h5','w') as f, gzip.open(out/'cell_stats.csv.gz','wt') as stats:
        ds=f.create_dataset('cell_labels',shape=tuple(meta['shape']),dtype='u4',chunks=True,
            compression='gzip',compression_opts=1)
        source_ds=None
        if source_shape:
            source_ds=f.create_dataset('source_image_cell_labels',shape=tuple(source_shape),dtype='u4',
                chunks=True,compression='gzip',compression_opts=1,fillvalue=0)
            source_ds.attrs['outside_RNA_bounding_box']='unmeasured; zero is not a validated background label'
        stats.write('cell_id,area_pixels,centroid_x,centroid_y,source_patch\n')
        for tile in tiles:
            r,c,h,w=[tile[k] for k in ['row','col','height','width']]
            with np.load(root/'segmentation_patches'/tile['id']/method/'segmentation.npz') as z:
                labels=z['cells']
            ids=np.unique(labels[labels>0])
            compact=np.where(labels>0,np.searchsorted(ids,labels)+1,0).astype(np.uint32)
            global_labels=np.where(compact>0,compact+count,0).astype(np.uint32)
            ds[r:r+h,c:c+w]=global_labels
            if source_ds is not None:
                sx0,sy0=r+meta['origin'][0],c+meta['origin'][1]
                source_ds[sx0:sx0+h,sy0:sy0+w]=global_labels
            xx,yy=np.nonzero(compact)
            area=np.bincount(compact[xx,yy],minlength=len(ids)+1)
            sx=np.bincount(compact[xx,yy],weights=xx+r+meta['origin'][0],minlength=len(ids)+1)
            sy=np.bincount(compact[xx,yy],weights=yy+c+meta['origin'][1],minlength=len(ids)+1)
            for k in range(1,len(ids)+1):
                stats.write(f'{count+k},{area[k]},{sx[k]/area[k]},{sy[k]/area[k]},{tile["id"]}\n')
            count+=len(ids); total_area+=len(xx)
        f.attrs['axis_order']='x,y'
        f.attrs['source_origin']=meta['origin']
        f.attrs['coordinates']='array indices + source_origin = original GEM coordinates'
        f.attrs['all_patches_accounted_for']=True
        f.attrs['cell_count']=count
        f.attrs['patch_seams']='independent original patches; cells crossing seams may be split'
    (out/'cell_labels.building.h5').replace(out/'cell_labels.h5')
    # Mapping at every model centre, including cell_id=0 for unassigned centres.
    coords=np.load(root/'coords.npy',mmap_mode='r')
    with h5py.File(out/'cell_labels.h5') as f, gzip.open(out/'spot2cell.tsv.gz','wt',compresslevel=3) as mapping:
        mapping.write('x\ty\tcell_id\n')
        ds=f['cell_labels']
        for tile in tiles:
            r,c,h,w=[tile[k] for k in ['row','col','height','width']]
            labels=ds[r:r+h,c:c+w]
            xy=coords[tile['sample_start']:tile['sample_stop']]
            ids=labels[xy[:,0]-r,xy[:,1]-c]
            for point,ident in zip(xy,ids):
                mapping.write(f'{point[0]+meta["origin"][0]}\t{point[1]+meta["origin"][1]}\t{ident}\n')
    write_json(out/'completed.json',dict(cells=count,assigned_pixels=total_area,
        all_patches=len(tiles),patches_without_valid_centres=sum(p.get('covered') is False for p in patches),
        model_centres=meta['centres'],omitted_occupied_bins=meta.get('omitted_occupied_bins'),
        mapping_scope='all valid model centres; not an every-molecule assignment table',
        mask_scope='entire RNA bounding box; unsupported regions explicitly recorded',
        source_image_canvas=source_shape,
        patch_seams='not merged across original SCS patch boundaries'))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=ROOT/'runs/SCS_stereo_whole_v1')
    p.add_argument('--method',default='genept_all_scale4')
    p.add_argument('--workers',type=int,default=8)
    args=p.parse_args()
    if 'Cellist_full_genept_v1' in args.root.parts and os.environ.get('SLURM_JOB_GPUS'):
        raise RuntimeError('This benchmark reserves postprocessing for its separate CPU allocation; use cellist_full_genept/evaluate.py')
    if not (args.root/args.method/'inference_completed.json').exists():
        raise ValueError('Full inference is not complete')
    tiles=json.loads((args.root/'patch_ranges.json').read_text())
    with mp.get_context('spawn').Pool(args.workers) as pool:
        for tile in pool.imap_unordered(segment_patch,[(args.root,args.method,t) for t in tiles]):
            print('POSTPROCESS',tile,flush=True)
    export(args.root,args.method,tiles)
