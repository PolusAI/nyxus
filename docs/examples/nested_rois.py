# Example for docs/source/cmdline_and_examples.rst: features of child ROIs and the
# parent/child relations between two label channels. The page shows the lines between the
# "example" markers. The input images are copied from the test data that the example names.
import os
import shutil

data = os.environ["NYXUS_DOCS_DATA"]  # tests/python/data in the repository
for sub in ("int", "seg"):
    os.makedirs(sub, exist_ok=True)
    for c in (0, 1):
        shutil.copy(os.path.join(data, sub, f"p0_y1_r1_c{c}.ome.tif"), sub)

# example-start
from nyxus import Nyxus, Nested

# Channel 0 holds the child ROIs, channel 1 the parent ROIs
nyx = Nyxus(["*ALL_INTENSITY*"])
child_features = nyx.featurize_directory("int", "seg", file_pattern=r"p[0-9]_y[0-9]_r[0-9]_c0\.ome\.tif")
print(child_features[["mask_image", "ROI_label", "MEAN"]].head())

nest = Nested()
relations = nest.find_relations("seg", r"p.*_c1\.ome\.tif", r"p.*_c0\.ome\.tif")
print(relations.columns.tolist())
# example-end

assert len(child_features) == 66
assert relations.columns.tolist() == ["Image", "Parent_Label", "Child_Label"]
