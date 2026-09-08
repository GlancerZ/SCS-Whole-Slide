import unittest
import numpy as np
from scipy import sparse
from .metrics import row_pearson, transcript_iou, cross_correlation, random_correlations, fano_median, paired_purity


class MetricsTests(unittest.TestCase):
    def test_sparse_pearson_matches_independent_dense(self):
        a=np.array([[1,0,4,0],[0,2,0,7],[1,1,1,1]],float)
        b=np.array([[0,0,3,2],[8,0,1,0],[0,2,3,4]],float)
        actual=row_pearson(sparse.csr_matrix(a),sparse.csr_matrix(b))
        for i in range(2): self.assertAlmostEqual(actual[i],np.corrcoef(a[i],b[i])[0,1])
        self.assertTrue(np.isnan(actual[2]))

    def test_iou_cannot_hide_missed_or_merged_cells(self):
        gt=np.array([1,1,2,2,3,3])
        missed=transcript_iou(gt,np.array([9,9,8,8,0,0]))
        self.assertEqual(missed['matched_only_mean'],1)
        self.assertAlmostEqual(missed['all_gt_mean'],2/3)
        merged=transcript_iou(gt,np.array([9,9,9,9,8,8]))
        self.assertEqual(merged['mutual_matches'],2)
        self.assertAlmostEqual(merged['all_gt_mean'],.5)

    def test_identical_masks_have_no_unique_region(self):
        x=sparse.csr_matrix([[3,1,0],[0,2,9],[1,0,3]])
        result=cross_correlation(x,np.array([1,1,2]),np.array([1,1,2]))
        self.assertEqual(len(result['target_cell']),0)

    def test_shared_splits_are_repeatable_and_counts_conserved(self):
        x=sparse.csr_matrix(np.arange(60).reshape(20,3)%7)
        coords=np.stack([np.arange(20),np.zeros(20)],1)
        labels=np.array([0]*2+[10]*9+[20]*9)
        a=random_correlations(x,coords,labels,3)
        b=random_correlations(x,coords,labels,3)
        np.testing.assert_equal(a['random_correlation'],b['random_correlation'])
        self.assertEqual(a['n_umis'].sum(),x[2:].sum())
        self.assertEqual(a['n_spots'].sum(),18)

    def test_fano_sample_variance(self):
        x=np.array([[1,0,4],[2,0,6],[3,0,8]],float)
        self.assertAlmostEqual(fano_median(sparse.csr_matrix(x)),np.median(x[:,[0,2]].var(0,ddof=1)/x[:,[0,2]].mean(0)))

    def test_RNA_without_nuclei_is_accounted_but_not_given_purity(self):
        x=sparse.csr_matrix([[2,0,1],[0,3,0]])
        coords=np.array([[1,1],[2,2]]);zero=np.zeros(2,dtype=int)
        rows,meta=paired_purity(x,coords,zero,np.zeros((1,2)),zero,np.ones(2,dtype=int),.5)
        self.assertEqual(rows,[])
        self.assertEqual(meta['eligible_nuclei'],0)
        correlations=random_correlations(x,coords,zero)
        self.assertEqual(len(correlations['cell_id']),0)
        self.assertEqual(len(cross_correlation(x,zero,zero)['target_cell']),0)


if __name__=='__main__': unittest.main()
