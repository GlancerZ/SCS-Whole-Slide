import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from scipy import sparse
from .prepare import write_rna


class PrepareTests(unittest.TestCase):
    def test_full_counts_duplicates_and_coordinates_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            frame=pd.DataFrame({'geneID':['C','A','A','B'], 'x':[2,0,0,1], 'y':[1,2,2,0], 'MIDCounts':[999,200,100,1]})
            labels=np.array([[0,0,1],[2,0,0],[0,3,0]],np.uint32)
            write_rna(out,frame,['A','B','C'],[3,3],labels,np.zeros((3,3),np.uint8))
            x=sparse.load_npz(out/'rna.npz')
            np.testing.assert_array_equal(x.toarray(),[[300,0,0],[0,1,0],[0,0,999]])
            np.testing.assert_array_equal(np.load(out/'coords.npy'),[[0,2],[1,0],[2,1]])
            self.assertEqual(json.loads((out/'prepared.json').read_text())['source_umis'],1300)

    def test_missing_gene_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame=pd.DataFrame({'geneID':['missing'],'x':[0],'y':[0],'MIDCounts':[1]})
            with self.assertRaises(ValueError):
                write_rna(Path(tmp),frame,['A'],[1,1],np.zeros((1,1),np.uint32),np.zeros((1,1),np.uint8))


if __name__=='__main__': unittest.main()
