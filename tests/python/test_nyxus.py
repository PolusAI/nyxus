import nyxus
import pytest
import time
import numpy as np
from pathlib import Path
from test_download_data import download

intens = np.array([[[1, 4, 4, 1, 1],
        [1, 4, 6, 1, 1],
        [4, 1, 6, 4, 1],
        [4, 4, 6, 4, 1]],
                   
        [[0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]]       
                   
                   ])

seg = np.array([[1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1]])

# cpu gabor
cpu_nyx = nyxus.Nyxus(["*ALL_INTENSITY*"])
#if (nyxus.gpu_is_available()):
#    cpu_nyx.using_gpu(False)
print()
print(intens)
print()
print(seg)
print()
cpu_features = cpu_nyx.featurize_memory(intens, seg)
#print("finished features")
print(cpu_features)
                
        