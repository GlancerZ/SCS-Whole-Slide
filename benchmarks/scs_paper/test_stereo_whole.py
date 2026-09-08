"""CPU guards for the whole-slide source geometry and upstream alignment."""
import unittest
import numpy as np
from benchmarks.scs_paper.stereo_whole_prepare import normalized_to_pixel, original_aligner


class Geometry(unittest.TestCase):
    def test_identity_pixel_transform(self):
        theta=np.array([[1.,0.,0.],[0.,1.,0.]])
        for shape in [(1200,1200),(391,718)]:
            np.testing.assert_allclose(normalized_to_pixel(theta,shape),theta,atol=1e-12)

    def test_pixel_translation(self):
        theta=np.array([[1.,0.,.1],[0.,1.,-.2]])
        np.testing.assert_allclose(normalized_to_pixel(theta,(100,200)),
            [[1.,0.,10.],[0.,1.,-10.]],atol=1e-12)

    def test_upstream_identity_warp(self):
        cls,digest=original_aligner()
        image=np.random.RandomState(1).rand(19,31).astype(np.float32)
        result=cls.transform(image,dict(theta=np.array([[1.,0.,0.],[0.,1.,0.]],np.float32)))
        # Float32 affine_grid/grid_sample has a few micro-units of identity
        # interpolation roundoff (observed upstream CPU max 2.38e-6).
        np.testing.assert_allclose(result,image,atol=1e-5)
        self.assertEqual(len(digest),64)

    def test_upstream_loss(self):
        import torch
        cls,_=original_aligner()
        image=np.arange(1,145).reshape(12,12)
        model=cls(image,image)
        reference=torch.tensor(image/image.max(),dtype=torch.float32)[None,None]
        expected=-((reference+1)*reference*reference).mean()
        self.assertAlmostEqual(float(model.loss(model())),float(expected),places=6)


if __name__=='__main__':
    unittest.main()
