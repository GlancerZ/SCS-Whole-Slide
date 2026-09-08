"""Import all genes from the already audited SCS Seq-Scope liver source.

Preserve the exact existing registered stain and nuclei. All-gene occupancy
expands the graph relative to the earlier 2000-HVG pilot, explicitly recorded.
"""
import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np
from scipy import sparse

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from benchmarks.scs_paper.stereo_whole_prepare import write_json, sha, pack


def prepare(source, root, genept):
    root.mkdir(parents=True,exist_ok=True)
    if (root/'prepared.json').exists():
        return
    old=json.loads((source/'prepared.json').read_text())
    genes=json.loads((source/'all_genes.json').read_text())
    shape=old['shape']; grid=old['bin_shape']
    rna=sparse.load_npz(source/'rna_pixels.npz').tocoo()
    r,c=np.divmod(rna.row,shape[1])
    x=sparse.csr_matrix((rna.data,((r//3)*grid[1]+c//3,rna.col)),
        shape=(int(np.prod(grid)),len(genes)),dtype=np.int64)
    if int(x.sum())!=old['total_umi']:
        raise AssertionError('All-gene liver counts are not conserved')
    sparse.save_npz(root/'all_expression.npz',x)
    write_json(root/'genes.json',genes)
    with h5py.File(source/'data/spots0:0:0:0.h5ad') as h:
        labels=h['layers/watershed_labels'][:].astype(np.int32)
        stain=h['layers/stain'][:]
    np.save(root/'nuclei.npy',labels)
    np.save(root/'stain_aligned.npy',stain)
    write_json(root/'indexed.json',dict(dataset='SCS Seq-Scope mouse liver',
        shape=shape, origin=old['origin'], genes=len(genes), umis=old['total_umi'],
        gem_sha256=old['archive_sha256'],original_preparation=str(source.resolve()),
        original_preparation_sha256=sha(source/'prepared.json')))
    write_json(root/'expression_indexed.json',dict(bin_size=3,bin_shape=grid,
        nnz=x.nnz,occupied_bins=int(np.count_nonzero(np.diff(x.indptr)))))
    write_json(root/'patch_alignment_completed.json',dict(numeric_guard_passed=True,
        method='Reuse exact existing prealigned SCS Seq-Scope image and nucleus labels; no extra alignment',
        patches=[dict(id=source.name,row=0,col=0,height=shape[0],width=shape[1],
                      nucleus_id_offset=0,nuclei=int(labels.max()))]))
    pack(root,genept)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--tile',required=True,choices=['2104','2105','2106','2107'])
    p.add_argument('--genept',type=Path,default=ROOT/'runs/ST19_shared_6000/genept_assets/GenePT_emebdding_v2/GenePT_gene_embedding_ada_text.pickle')
    a=p.parse_args()
    prepare(ROOT/'runs/SCS_paper_benchmark_v1'/a.tile,ROOT/'runs/SCS_stereo_whole_v1/liver'/a.tile,a.genept)
