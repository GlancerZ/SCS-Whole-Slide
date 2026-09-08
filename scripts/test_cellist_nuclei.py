"""Verify memory/performance adaptations preserve the watershed definition."""
import numpy as np
from skimage._shared.coord import ensure_spacing
from skimage.filters import threshold_multiotsu, threshold_local


def main():
    rng = np.random.default_rng(71)
    image = rng.integers(0, 256, (256, 256), dtype=np.uint8)
    threshold = threshold_multiotsu(image, 3)[0]
    local_threshold = threshold_local(image, block_size=51, offset=0)
    masked = image.copy()
    masked[image <= threshold] = 0
    original = (image > local_threshold) & (masked > local_threshold)
    adapted = (image > threshold) & (image > local_threshold)
    np.testing.assert_array_equal(original, adapted)
    for coords in [rng.integers(0, 600, (5000, 2)), np.indices((50, 50)).reshape(2, -1).T]:
        np.testing.assert_array_equal(ensure_spacing(coords, spacing=6, p_norm=np.inf),
                                      ensure_spacing(coords, spacing=6, p_norm=np.inf, min_split_size=None))
    print("PASS: exact threshold equivalence and identical ordered peak selection with single-batch spacing")


if __name__ == "__main__":
    main()
