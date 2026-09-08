import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scipy import sparse
from .evaluate import evaluate_patch, CELLIST, POST_METHOD


class EmptyNucleusEvaluationTests(unittest.TestCase):
    def test_unassigned_RNA_patch_keeps_full_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            data=Path(tmp)/'stereo_mouse_brain';patch={'id':'edge'}
            source=data/'cellist_inputs'/'edge';source.mkdir(parents=True)
            coords=np.array([[0,0],[1,1],[2,2]],dtype=np.int32)
            np.save(source/'coords.npy',coords);np.save(source/'nuclei.npy',np.zeros((3,3),np.uint32))
            sparse.save_npz(source/'rna.npz',sparse.csr_matrix([[3,0,1],[0,2,0],[1,0,4]]))
            (source/'prepared.json').write_text(json.dumps(dict(source_umis=11,rna_sha256='synthetic',nuclei_sha256='synthetic')))
            (data/'comparison_input.json').write_text(json.dumps(dict(resolution_um=.5)))
            first=data/'segmentation_patches'/'edge'/CELLIST;first.mkdir(parents=True)
            (first/'segmentation_completed.json').write_text('{}')
            np.save(first/'assignments.npy',np.zeros(3,np.uint32))
            second=data/'segmentation_patches'/'edge'/POST_METHOD;second.mkdir(parents=True)
            np.savez(second/'segmentation.npz',cells=np.zeros((3,3),np.uint32))
            result=evaluate_patch((data,patch))
            self.assertEqual(result['source_umis'],11)
            self.assertEqual(result['source_spots'],3)
            self.assertEqual(result['purity']['eligible_nuclei'],0)
            for method in result['methods'].values():
                self.assertEqual(method['cells'],0)
                self.assertEqual(method['assigned_umis'],0)


if __name__=='__main__':unittest.main()
