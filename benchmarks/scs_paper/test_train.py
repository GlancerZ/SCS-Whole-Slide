import unittest
import numpy as np
import torch
from benchmarks.scs_paper.train import CountGenePT, metrics
from optimizations.scs_streaming.torch_model import ModelConfig


class TestBenchmark(unittest.TestCase):
    def test_online_pooling_and_gradients(self):
        torch.manual_seed(12)
        table = torch.randn(7, 16)
        table[2] = 0
        cfg = ModelConfig(input_dim=16, n_genes=7, n_neighbors=3, scale=1,
                          expression_scale=4, expression_encoding='genept')
        model = CountGenePT(cfg, table)
        counts = torch.tensor([[[2.,0,5,0,3,0,0],[0,0,9,0,0,0,0],[0,1,0,0,0,0,0]]])
        actual = model.project_expression(counts)
        k = ((counts > 0) & (table.square().sum(-1)>0)).sum(-1,keepdim=True)
        pooled = (counts @ table) / k.clamp_min(1)
        expected = (model.expression_projection(pooled)+(k>0)*model.spot_bias)*4
        torch.testing.assert_close(actual,expected)
        torch.testing.assert_close(actual[:,1],torch.zeros_like(actual[:,1]))
        actual.sum().backward()
        grad = model.expression_projection.weight.grad.clone()
        model.zero_grad()
        expected.sum().backward()
        torch.testing.assert_close(grad,model.expression_projection.weight.grad)

    def test_metrics_denominators(self):
        d=np.eye(16)[[1,3,9,9]]
        m=metrics(d,np.array([.9,.2,.8,.1]),np.array([1,2,0,0]),np.array([1,1,0,0]))
        self.assertEqual(m['direction_accuracy'],.5)
        self.assertEqual(m['foreground_balanced_accuracy'],.5)
        self.assertEqual(np.asarray(m['direction_confusion']).sum(),2)

    def test_no_background_is_not_perfect_specificity(self):
        m=metrics(np.eye(16)[[1]],np.array([.9]),np.array([1]),np.array([1]))
        self.assertIsNone(m['background_specificity'])
        self.assertIsNone(m['foreground_balanced_accuracy'])


if __name__ == '__main__':
    unittest.main()
