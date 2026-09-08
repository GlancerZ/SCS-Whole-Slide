"""Independent molecule-count / contingency-table audit and shared-support previews."""
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'runs/Cellist_full_genept_v1'


def audit():
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_GPUS'):
        raise RuntimeError('CPU allocation required')
    checks=[]
    for fov in (0,1):
        data=BASE/'datasets'/f'seqfish_rep1_fov{fov}'
        summary=json.loads((data/'comparison_completed.json').read_text())
        ground=np.load(data/'molecule_ground_truth.npz');xy=ground['grid_xy'];gt=ground['manual_cell']
        source=data/'cellist_inputs'/f'fov{fov}'
        coords=np.load(source/'coords.npy');rna=sparse.load_npz(source/'rna.npz')
        weights=np.asarray(rna.sum(1)).ravel();stain=np.load(source/'stain.npy');nuclei=np.load(source/'nuclei.npy')
        # The sparse RNA matrix was constructed in row=y, col=x convention.
        flat=xy[:,1]*stain.shape[1]+xy[:,0]
        counts=np.bincount(flat,minlength=stain.size).reshape(stain.shape)
        np.testing.assert_array_equal(counts[coords[:,0],coords[:,1]],weights)
        modelcoords=np.load(data/'coords.npy');prob=np.load(data/'genept_all_scale4/foreground_rank0.npy')
        key=modelcoords[:,0]*stain.shape[1]+modelcoords[:,1]
        order=np.argsort(key);idx=np.searchsorted(key[order],flat)
        valid=idx<len(key)
        valid[valid]&=key[order][idx[valid]]==flat[valid]
        per_molecule=np.full(len(gt),np.nan);per_molecule[valid]=prob[order[idx[valid]]]
        check=dict(dataset=data.name,coordinate_count_conservation='pass',molecules=len(gt),
            molecules_with_model_prediction=int(valid.sum()),
            fraction_RNA_in_shared_nuclei=float(np.mean(nuclei[xy[:,1],xy[:,0]]>0)),
            fraction_RNA_foreground_probability_gt01=float(np.mean(per_molecule>.1)),
            fraction_RNA_foreground_probability_gt05=float(np.mean(per_molecule>.5)),methods={})
        fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
        axes[0].imshow(stain,cmap='gray',vmin=0,vmax=255)
        axes[0].contour(nuclei>0,levels=[.5],colors=['cyan'],linewidths=.4)
        axes[0].set_title(f'FOV {fov}: DAPI + shared nuclei')
        for panel,(method,folder) in enumerate([('cellist','cellist_paper'),('full_genept_scale4','genept_paperpost_scale4')],1):
            mask=np.load(data/'segmentation_patches'/f'fov{fov}'/folder/'segmentation.npz')['cells']
            predicted=mask[xy[:,1],xy[:,0]]
            table=pd.crosstab(pd.Series(gt,name='manual'),pd.Series(predicted,name='predicted'))
            area_gt=table.sum(1);area_pred=table.sum(0)
            nonzero=table.drop(columns=[0],errors='ignore')
            g2p=nonzero.idxmax(1);p2g=nonzero.idxmax(0)
            scores=[];matches=0
            for g in table.index:
                p=g2p[g]
                if p2g[p]==g and table.loc[g,p]>0:
                    matches+=1
                    intersection=table.loc[g,p]
                    scores.append(intersection/(area_gt[g]+area_pred[p]-intersection))
                else:scores.append(0.)
            expected=summary['transcript_iou'][method]
            np.testing.assert_allclose(scores,expected['all_gt_ious'],rtol=0,atol=1e-12)
            assert int(np.sum(predicted>0))==summary['methods'][method]['assigned_umis']
            check['methods'][method]=dict(contingency_iou='pass',assigned_molecule_total='pass',
                all_gt_mean=float(np.mean(scores)),matches=matches,assigned_fraction=float(np.mean(predicted>0)))
            axes[panel].imshow(stain,cmap='gray',vmin=0,vmax=255)
            labels=mask[coords[:,0],coords[:,1]]
            unique=np.unique(labels[labels>0]);colors=np.random.RandomState(99).uniform(.2,1,(len(unique)+1,3));colors[0]=[.65,.65,.65]
            compact=np.searchsorted(np.r_[0,unique],labels)
            axes[panel].scatter(coords[:,1],coords[:,0],c=compact,cmap=ListedColormap(colors),vmin=0,vmax=len(unique),s=.8,alpha=.8,linewidths=0,rasterized=True)
            axes[panel].set_title(f"{'Cellist' if method=='cellist' else 'Full GenePT 4x'}: assigned RNA {np.mean(predicted>0):.1%}")
        for ax in axes:
            ax.set_xlim(-.5,stain.shape[1]-.5);ax.set_ylim(stain.shape[0]-.5,-.5);ax.set_aspect('equal');ax.set_xticks([]);ax.set_yticks([])
        fig.supxlabel('Same observed RNA coordinates; gray = unassigned, colors = predicted instances. 5 native pixels / grid.',fontsize=9)
        fig.savefig(BASE/f'seqfish_fov{fov}_shared_support.png',dpi=180)
        plt.close(fig)
        checks.append(check)
    (BASE/'seqfish_independent_audit.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks),flush=True)


if __name__=='__main__':audit()
