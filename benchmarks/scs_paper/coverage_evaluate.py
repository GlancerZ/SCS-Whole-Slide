"""Separate coverage-control result; does not replace the locked main benchmark."""
import json
from pathlib import Path
import sys
import numpy as np
from scipy import sparse

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from benchmarks.scs_paper.evaluate import segment,paired_consistency

data=ROOT/'runs/SCS_paper_benchmark_v1/2104'
segment(data,'raw_matched_scale4')
rna=sparse.load_npz(data/'rna_pixels.npz')
nuclei=np.load(data/'scs_reference'/'segmentation.npz')['nuclei']
result=[]
for first,second in [('raw_matched_scale4','raw_scale4'),('genept_scale4','raw_matched_scale4')]:
    a=np.load(data/first/'segmentation.npz')['cells']
    b=np.load(data/second/'segmentation.npz')['cells']
    summary,rows=paired_consistency(nuclei,a,b,rna)
    result.append(dict(first=first,second=second,summary=summary,rows=rows))
(data/'coverage_ablation.json').write_text(json.dumps(result,indent=2))
