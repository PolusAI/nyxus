# Example for docs/source/installation.rst: checking an installation.
# The page shows the lines between the "example" markers. Everything after them checks the
# result, so tests/python/test_2d_docs_examples_mechanics.py can run the script as written.

# example-start
import numpy as np
import nyxus
from nyxus import Nyxus

print("Nyxus version:", nyxus.__version__)

nyx = Nyxus(["MEAN"])
print("Arrow IPC and Parquet output:", nyx.arrow_is_enabled())
print("CUDA GPU usable:", nyxus.gpu_is_available(id(nyx)))

intensity = np.array([[1, 2], [3, 4]], dtype=np.uint16)
features = nyx.featurize(intensity, np.ones_like(intensity))
print("MEAN of 1, 2, 3, 4:", features["MEAN"].iloc[0])
# example-end

assert isinstance(nyxus.__version__, str) and nyxus.__version__
assert isinstance(nyx.arrow_is_enabled(), bool)
assert isinstance(nyxus.gpu_is_available(id(nyx)), bool)
assert features.shape[0] == 1
assert features["MEAN"].iloc[0] == 2.5
