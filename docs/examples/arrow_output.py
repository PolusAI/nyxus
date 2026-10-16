# Example for docs/source/cmdline_and_examples.rst: writing results to an Arrow IPC file.
# Needs a Nyxus build with Arrow support (the PyPI and conda packages have it).
# The page shows the lines between the "example" markers.

# example-start
import numpy as np
import pyarrow as pa
from nyxus import Nyxus

intens = np.array([[[1, 4, 4, 1, 1], [1, 4, 6, 1, 1], [4, 1, 6, 4, 1], [4, 4, 6, 4, 1]]])
seg = np.ones_like(intens)

nyx = Nyxus(["*ALL_INTENSITY*"])
arrow_path = nyx.featurize(intens, seg, output_type="arrowipc", output_path="results")
print(arrow_path)

with pa.memory_map(arrow_path) as source:
    table = pa.ipc.open_file(source).read_all()
print(table.num_rows, table.column_names[:4])
# example-end

assert table.num_rows == 1
assert table.column_names[:4] == ["intensity_image", "mask_image", "phys_unit", "ROI_label"]
