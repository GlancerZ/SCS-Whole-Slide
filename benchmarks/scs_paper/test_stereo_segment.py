import gzip
import json
import tempfile
from pathlib import Path
import unittest

import h5py
import numpy as np
from benchmarks.scs_paper.stereo_whole_segment import segment_patch, export


class SegmentationTests(unittest.TestCase):
    def test_prediction_through_full_coordinate_export(self):
        with tempfile.TemporaryDirectory(prefix='scs-segment-test-') as tmp:
            root=Path(tmp); method='synthetic'; (root/method).mkdir()
            shape=(60,60)
            labels=np.zeros(shape,np.int32);labels[23:37,23:37]=1
            stain=(labels>0).astype(np.uint8)*200
            coords=np.stack(np.meshgrid(np.arange(0,60,3),np.arange(0,60,3),indexing='ij'),-1).reshape(-1,2)
            np.save(root/'nuclei.npy',labels);np.save(root/'stain_aligned.npy',stain)
            np.save(root/'coords.npy',coords)
            delta=np.array([30,30])-coords
            angle=np.mod(np.arctan2(delta[:,0],delta[:,1]),2*np.pi)
            logits=np.zeros((len(coords),16),np.float32)
            logits[np.arange(len(coords)),np.floor(angle*16/(2*np.pi)).astype(int)]=3
            np.save(root/method/'logits_rank0.npy',logits)
            np.save(root/method/'foreground_rank0.npy',(np.linalg.norm(delta,axis=1)<21).astype(np.float32))
            (root/method/'inference_completed.json').write_text(json.dumps(dict(world_size=1)))
            (root/'prepared.json').write_text(json.dumps(dict(shape=shape,origin=[3225,6175],
                centres=len(coords),omitted_occupied_bins=0)))
            (root/'indexed.json').write_text(json.dumps(dict(source_image_shape=[3300,6300])))
            tile=dict(id='test',row=0,col=0,height=60,width=60,sample_start=0,sample_stop=len(coords))
            segment_patch((root,method,tile))
            export(root,method,[tile])
            out=root/method/'whole_slide'
            with h5py.File(out/'cell_labels.h5') as h:
                self.assertEqual(h['cell_labels'].shape,shape)
                self.assertTrue(h.attrs['all_patches_accounted_for'])
                np.testing.assert_array_equal(h.attrs['source_origin'],[3225,6175])
                self.assertEqual(h['source_image_cell_labels'].shape,(3300,6300))
                np.testing.assert_array_equal(h['source_image_cell_labels'][3225:3285,6175:6235],h['cell_labels'][:])
            with gzip.open(out/'spot2cell.tsv.gz','rt') as f:
                rows=f.read().splitlines()
            self.assertEqual(len(rows),len(coords)+1)
            self.assertEqual(rows[1].split('\t')[:2],['3225','6175'])
            self.assertTrue((out/'completed.json').exists())


if __name__=='__main__':
    unittest.main()
