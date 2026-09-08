"""NumPy 1.x/2.x compatible adapter without changing the live TF pipeline."""
import zipfile
import numpy as np
from optimizations.scs_streaming.data import ArrayStore as StreamingArrayStore


def read_header(stream):
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        return np.lib.format.read_array_header_1_0(stream)
    if version == (2, 0):
        return np.lib.format.read_array_header_2_0(stream)
    raise ValueError(f"Unsupported NPY version {version}; SCS numeric arrays use v1/v2")


class ArrayStore(StreamingArrayStore):
    def header(self, key):
        with zipfile.ZipFile(self.path(key)) as z, z.open(key + ".npy") as stream:
            return read_header(stream)
