"""Synthetic smoke tests, not biological benchmark results."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np
import torch
from benchmarks.scs_paper.stereo_whole_train import SparseCache, make, per_example_loss


class TrainTests(unittest.TestCase):
    def test_loss_denominator(self):
        dl=torch.zeros(3,16); bl=torch.zeros(3)
        loss=per_example_loss(dl,bl,torch.tensor([0,1,0]),torch.tensor([1.,1.,0.]))
        np.testing.assert_allclose(loss.numpy(),[np.log(16)+np.log(2)]*2+[np.log(2)],rtol=1e-6)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA smoke test')
    def test_sparse_forward_backward_and_resume(self):
        with tempfile.TemporaryDirectory(prefix='scs-whole-smoke-',dir=os.environ.get('SLURM_TMPDIR')) as tmp:
            root=Path(tmp)
            rng=np.random.RandomState(3)
            n=80
            arrays=dict(expression_data=np.ones(n*3,np.float32),
                expression_indices=np.tile([0,1,2],n).astype(np.int32),
                expression_indptr=np.arange(0,n*3+1,3,dtype=np.int64),
                neighbors=(np.arange(n)[:,None]+np.arange(50)[None,:])%n,
                directions=np.arange(n)%16,foreground=(np.arange(n)%3!=0).astype(np.uint8),
                split=np.array([0]*64+[1]*8+[2]*8,np.int8),
                gene_embeddings=rng.normal(size=(3,1536)).astype(np.float32)/np.sqrt(1536))
            for key,value in arrays.items():
                np.save(root/f'{key}.npy',value)
            (root/'prepared.json').write_text(json.dumps(dict(complete=True,bin_shape=[8,10],
                neighbors_sha256='synthetic_fixture',split_sha256='synthetic_fixture',genept={'synthetic':True})))
            cache=SparseCache(root,torch.device('cuda',0))
            model,opts,_=make(cache,3812)
            x,pos,d,b=cache.batch(np.arange(4))
            with torch.autocast('cuda',dtype=torch.bfloat16):
                dl,bl=model(x,pos)
                loss=per_example_loss(dl,bl,d,b).mean()
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(torch.isfinite(model.expression_projection.weight.grad).all())
            del cache,model,opts,x,pos,d,b,dl,bl,loss
            torch.cuda.empty_cache()
            command=['torchrun','--standalone','--nproc_per_node=1',
                'benchmarks/scs_paper/stereo_whole_train.py','--root',str(root),
                '--epochs','1','--batch-per-gpu','8','--stop-after-steps','2']
            subprocess.run(command,check=True)
            ck=root/'genept_all_scale4/latest.pt'
            first=torch.load(ck,map_location='cpu',weights_only=False)
            self.assertEqual(first['next_step'],2)
            subprocess.run(command+['--resume'],check=True)
            second=torch.load(ck,map_location='cpu',weights_only=False)
            self.assertEqual(second['next_step'],4)
            self.assertFalse(torch.equal(first['model']['direction_head.weight'],second['model']['direction_head.weight']))
            self.assertFalse((root/'genept_all_scale4/training_completed.json').exists())


if __name__=='__main__':
    unittest.main()
