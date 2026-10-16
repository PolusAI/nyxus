# Example for docs/source/imagequality.rst: image quality features of whole images.
# The page shows the lines between the "example" markers.
import os

import numpy as np
import tifffile

rng = np.random.default_rng(0)
os.makedirs("data/int", exist_ok=True)
sharp = rng.integers(0, 4000, (128, 128), dtype=np.uint16)
blurred = ((sharp[:-1, :-1].astype(np.uint32) + sharp[1:, :-1] + sharp[:-1, 1:] + sharp[1:, 1:]) // 4).astype(np.uint16)
tifffile.imwrite("data/int/sharp.ome.tif", sharp)
tifffile.imwrite("data/int/blurred.ome.tif", blurred)

# example-start
from nyxus import ImageQuality

# The intensity directory doubles as the label directory: each image is one ROI
imq = ImageQuality(["*ALL_IMQ*"])
features = imq.featurize_directory("data/int", "data/int")
print(features[["intensity_image", "FOCUS_SCORE", "MAX_SATURATION", "SHARPNESS"]])
# example-end

focus = {os.path.basename(n.replace("\\", "/")): v for n, v in zip(features["intensity_image"], features["FOCUS_SCORE"])}
assert focus["sharp.ome.tif"] > focus["blurred.ome.tif"]
assert features["MAX_SATURATION"].between(0, 1).all()
