"""Independent integrity checks, with source paths retained for reproducibility."""
import json
from pathlib import Path
import sys

import numpy as np
from scipy import sparse

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from optimizations.scs_streaming.genept import load_genept_embeddings

root=ROOT/'runs/SCS_paper_benchmark_v1'
lookup,dim=load_genept_embeddings(ROOT/'runs/ST19_shared_6000/genept_assets/GenePT_emebdding_v2/GenePT_gene_embedding_ada_text.pickle')
rows=[]
for tile in ['2104','2105','2106','2107','seqfish_rep1_fov0','seqfish_rep1_fov1']:
    data=root/tile
    meta=json.loads((data/'prepared.json').read_text())
    s=np.load(data/'samples.npz')
    x=sparse.load_npz(data/'expression.npz')
    rna=sparse.load_npz(data/'rna_pixels.npz')
    nb=s['neighbors'];centres=nb[:,0];split=s['split']
    assert len(np.unique(centres))==len(centres)
    assert nb.min()>=0 and nb.max()<x.shape[0]
    assert (np.diff(x.indptr)[nb]>0).all()
    for i in range(0,len(nb),8192):
        assert (np.diff(np.sort(nb[i:i+8192],axis=1),axis=1)>0).all()
    parts=[centres[split==i] for i in (0,1,2)]
    for i,j in [(0,1),(0,2),(1,2)]: assert not np.intersect1d(parts[i],parts[j]).size
    assert int(rna.sum())==meta['total_umi']
    genes=json.loads((data/'genes.json').read_text())
    mapped=np.array([g.upper() in lookup for g in genes])
    sums=np.asarray(x.sum(0)).ravel()
    raw_nonzero=np.diff(x.indptr)>0
    mapped_nonzero=np.diff(x[:,mapped].tocsr().indptr)>0
    r=dict(tile=tile,checks='passed',centres=len(centres),train=len(parts[0]),validation=len(parts[1]),test=len(parts[2]),
           mapped_genes=int(mapped.sum()),total_genes=len(genes),mapped_umi_fraction=float(sums[mapped].sum()/sums.sum()),
           nonempty_input_bins=int(raw_nonzero.sum()),bins_empty_after_genept_mapping=int((raw_nonzero&~mapped_nonzero).sum()),
           dimension=dim,umi_sum_recomputed=int(rna.sum()))
    missing=np.flatnonzero(~mapped)
    ranking=missing[np.argsort(sums[missing])[::-1]][:10]
    r['top_missing_by_input_umi']=[dict(gene=genes[i],umi=int(sums[i])) for i in ranking]
    rows.append(r)
    print(json.dumps(r),flush=True)
(root/'integrity_audit.json').write_text(json.dumps(rows,indent=2))
