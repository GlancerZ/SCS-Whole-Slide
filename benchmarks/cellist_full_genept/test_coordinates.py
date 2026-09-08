import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from benchmarks.scs_paper.stereo_whole_train import SparseCache


class CoordinateTests(unittest.TestCase):
    def test_seqfish_grid_is_not_multiplied_by_three(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            arrays=dict(expression_data=np.array([2,3,4,5],np.float32),expression_indices=np.zeros(4,np.int32),
                expression_indptr=np.arange(5,dtype=np.int64),neighbors=np.array([[1,0,3]],np.int32),
                directions=np.array([2]),foreground=np.array([1]),split=np.array([0]),gene_embeddings=np.array([[1,0]],np.float32))
            for k,v in arrays.items():np.save(root/f'{k}.npy',v)
            (root/'prepared.json').write_text(json.dumps(dict(complete=True,bin_shape=[2,2],bin_size=1)))
            cache=SparseCache(root,torch.device('cpu'))
            _,position,_,_=cache.batch([0])
            np.testing.assert_array_equal(position.numpy(),[[[0,0],[0,-1],[1,0]]])


if __name__=='__main__':unittest.main()
