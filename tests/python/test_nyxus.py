import nyxus
import pytest
import time
import numpy as np
from pathlib import Path
from test_data import intens, seg


class TestImport():
    def test_import(self):
        assert nyxus.__name__ == "nyxus" 
        
class TestNyxus():
        PATH = PATH = Path(__file__).with_name('data')

        def test_gabor_gpu(self):
            # cpu gabor
            cpu_nyx = nyxus.Nyxus(["GABOR"])
            if (nyxus.gpu_is_available()):
                cpu_nyx.using_gpu(False)
            cpu_features = cpu_nyx.featurize_memory(intens, seg)

            if (nyxus.gpu_is_available()):
                # gpu gabor
                gpu_nyx = nyxus.Nyxus(["GABOR"], using_gpu=0)
                gpu_nyx.using_gpu(True)
                gpu_features = gpu_nyx.featurize_memory(intens, seg)
                
                assert gpu_features.equals(cpu_features)
            else:
                print("Gpu not available")
                assert True
                        
        def test_in_memory(self):
            
            cpu_nyx = nyxus.Nyxus(["*ALL_GLCM*"], ibsi=True)
            
            names = ["test_name_1", "test_name_2", "test_name_3", "test_name_4"]

            cpu_features = cpu_nyx.featurize_memory(intens, seg, names, names)

            print(cpu_features)

            means = cpu_features.mean(numeric_only=True)

            mean_values = means.tolist()

            mean_values.pop(0)

            averaged_results = []

            i = 0
            while (i < len(mean_values)):
                averaged_results.append(sum(mean_values[i:i+4])/4)
                i += 4
                
            print(averaged_results)
                
            # check IBSI values
            assert pytest.approx(averaged_results[0], 0.01) == 0.368 # angular2ndmoment
            assert pytest.approx(averaged_results[1], 0.01) == 5.28 # contrast 
            assert pytest.approx(averaged_results[2], 0.01) == -0.0121 # correlation
            assert pytest.approx(averaged_results[9], 0.01) == 1.40 # difference entropy
            assert pytest.approx(averaged_results[11], 0.1) == 2.90 # difference variance#

                
        