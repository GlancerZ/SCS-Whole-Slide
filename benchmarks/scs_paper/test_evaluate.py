import unittest
import numpy as np
from scipy import sparse
from benchmarks.scs_paper.evaluate import paired_consistency, match_nuclei, instance_iou


class TestEvaluation(unittest.TestCase):
    def test_paired_population_and_int64_counts(self):
        # One nucleus: shared pixel 0, first-only 1, second-only 2.
        nu=np.array([[1,0,0,0]])
        a=np.array([[3,3,0,0]])
        b=np.array([[8,0,8,0]])
        x=sparse.csr_matrix(np.array([[200,100,0],[400,200,0],[0,100,200],[5,0,0]],np.int64))
        summary,rows=paired_consistency(nu,a,b,x)
        self.assertEqual(summary['eligible_nuclei'],1)
        self.assertEqual(rows[0]['umi_unique_first'],600)
        self.assertAlmostEqual(rows[0]['correlation_first'],1)
        self.assertAlmostEqual(rows[0]['correlation_second'],-1)
        rev,_=paired_consistency(nu,b,a,x)
        self.assertAlmostEqual(summary['mean_paired_difference_first_minus_second'],
                               -rev['mean_paired_difference_first_minus_second'])

    def test_identical_masks_are_ineligible_not_perfect_score(self):
        nu=np.array([[1,0]])
        a=np.array([[1,1]])
        x=sparse.csr_matrix(np.array([[200,1],[200,1]],np.int64))
        summary,rows=paired_consistency(nu,a,a,x)
        self.assertEqual(summary['eligible_nuclei'],0)
        self.assertIsNone(summary['correlation_first_mean'])

    def test_matching_uses_maximum_nuclear_overlap(self):
        self.assertEqual(match_nuclei(np.array([[1,1,1,2]]),np.array([[4,5,5,0]])),{1:5})

    def test_iou_perfect_missed_and_merged(self):
        gt=np.array([[1,1,2,2]],np.uint32)
        self.assertEqual(instance_iou(gt,gt)['mean_iou'],1.)
        self.assertEqual(instance_iou(gt,np.zeros_like(gt))['mean_iou'],0.)
        m=instance_iou(gt,np.ones_like(gt))
        self.assertEqual(m['mean_iou'],.25)
        self.assertEqual(m['gt_recall_iou50'],.5)


if __name__=='__main__':
    unittest.main()
