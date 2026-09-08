import unittest
import numpy as np
from benchmarks.scs_paper.prepare import neighbors_for, ring_offsets


class TestNeighborhoods(unittest.TestCase):
    def test_exact_ring_reference(self):
        rng=np.random.RandomState(1)
        shape=(13,16)
        occupied=rng.rand(np.prod(shape))>.25
        expected=[]
        for center in np.flatnonzero(occupied):
            r,c=divmod(center,shape[1])
            nb=[center]
            for dr,dc in ring_offsets():
                rr,cc=r+dr,c+dc
                if 0<=rr<shape[0] and 0<=cc<shape[1] and occupied[rr*shape[1]+cc]:
                    nb.append(rr*shape[1]+cc)
                if len(nb)==50: break
            if len(nb)==50: expected.append(nb)
        np.testing.assert_array_equal(neighbors_for(occupied,shape),expected)

    def test_insufficient_neighbors(self):
        self.assertEqual(neighbors_for(np.ones(4,bool),(2,2)).shape,(0,50))


if __name__ == '__main__':
    unittest.main()
