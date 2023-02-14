import nyxus
import pytest
import time
import numpy as np
from pathlib import Path
from test_data import intens, seg

# cpu gabor
cpu_nyx = nyxus.Nyxus(["*ALL_GLCM*"])
#if (nyxus.gpu_is_available()):
#    cpu_nyx.using_gpu(False)

names = ["test_name_1", "test_name_2"]

cpu_features = cpu_nyx.featurize_memory(intens, seg, names, names)
print(cpu_features)
                
        