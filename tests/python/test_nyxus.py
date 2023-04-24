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
            cpu_features = cpu_nyx.featurize(intens, seg)
            
            assert cpu_nyx.error_message == ''

            if (nyxus.gpu_is_available()):
                # gpu gabor
                gpu_nyx = nyxus.Nyxus(["GABOR"], using_gpu=0)
                gpu_nyx.using_gpu(True)
                gpu_features = gpu_nyx.featurize(intens, seg)
                
                assert gpu_features.equals(cpu_features)
            else:
                print("Gpu not available")
                assert True

        def test_gabor_customization (self):
            nyx = nyxus.Nyxus (["GABOR"])
            assert nyx is not None

            # test ability to digest valid parameters
            try:
                nyx.set_gabor_feature_params(kersize=16)
                nyx.set_gabor_feature_params(gamma=0.1)
                nyx.set_gabor_feature_params(sig2lam=0.8)
                nyx.set_gabor_feature_params(f0=0.1)
                nyx.set_gabor_feature_params(theta=1.5708)
                nyx.set_gabor_feature_params(thold=0.025)
                nyx.set_gabor_feature_params(freqs=[1])
                nyx.set_gabor_feature_params(freqs=[1,2,4,8,16,32,64])
            except Exception as exc:
                assert False, f"set_gabor_feature_params(valid argument) raised an exception {exc}"

            # test ability to intercept invalid values
            with pytest.raises (Exception):
                nyx.set_gabor_feature_params(kersize=16.789)
            with pytest.raises (Exception):
                nyx.set_gabor_feature_params(gamma="notAnumber")
            with pytest.raises (Exception):
                nyx.set_gabor_feature_params(sig2lam="notAnumber")
                nyx.set_gabor_feature_params(f0="notAnumber")
                nyx.set_gabor_feature_params(theta="notAnumber")
                nyx.set_gabor_feature_params(thold="notAnumber")
            with pytest.raises (Exception):
                nyx.set_gabor_feature_params(freqs=["notAnumber"])
            with pytest.raises (Exception):
                nyx.set_gabor_feature_params(freqs="notAList")
        
        def test_get_default_params(self):
            
            nyx = nyxus.Nyxus (["*ALL*"])
            assert nyx is not None
            
            params = nyx.get_params()
            
            result = {'coarse_gray_depth': 256, 
                      'features': ['*ALL*'], 
                      'gabor_f0': 0.1, 
                      'gabor_freqs': [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0], 
                      'gabor_gamma': 0.1, 
                      'gabor_kersize': 16, 
                      'gabor_sig2lam': 0.8, 
                      'gabor_theta': 45.0, 
                      'gabor_thold': 0.025, 
                      'ibsi': 0, 
                      'n_loader_threads': 1, 
                      'n_feature_calc_threads': 4, 
                      'neighbor_distance': 5, 
                      'pixels_per_micron': 1.0}
            
            for key in params:
                
                if (isinstance(params[key], float)):
                    assert params[key] == pytest.approx(result[key])
                else:
                    assert params[key] == pytest.approx(result[key])                
        
        def test_set_params(self):
            
            nyx = nyxus.Nyxus (["*ALL*"])
            assert nyx is not None

            new_values ={'coarse_gray_depth': 512, 
                        'features': ['*ALL*'], 
                        'gabor_f0': 0.1, 
                        'gabor_freqs': [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0], 
                        'gabor_gamma': 0.1, 
                        'gabor_kersize': 16, 
                        'gabor_sig2lam': 0.8, 
                        'gabor_theta': 45.0, 
                        'gabor_thold': 0.025, 
                        'ibsi': 0, 
                        'n_loader_threads': 1, 
                        'n_feature_calc_threads': 4, 
                        'neighbor_distance': 5, 
                        'pixels_per_micron': 1.0}
            
            nyx.set_params(
                **new_values
            )
            
            params = nyx.get_params()
            
            for key in params:
                    
                if (isinstance(params[key], float)):
                    assert params[key] == pytest.approx(new_values[key])
                else:
                    assert params[key] == pytest.approx(new_values[key])   

        def test_set_single_param(self):
            
            nyx = nyxus.Nyxus (["*ALL*"])
            assert nyx is not None
            
            nyx.set_params(coarse_gray_depth = 125)
            
            params = nyx.get_params()
            
            result = {'coarse_gray_depth': 125, 
                      'features': ['*ALL*'], 
                      'gabor_f0': 0.1, 
                      'gabor_freqs': [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0], 
                      'gabor_gamma': 0.1, 
                      'gabor_kersize': 16, 
                      'gabor_sig2lam': 0.8, 
                      'gabor_theta': 45.0, 
                      'gabor_thold': 0.025, 
                      'ibsi': 0, 
                      'n_loader_threads': 1, 
                      'n_feature_calc_threads': 4, 
                      'neighbor_distance': 5, 
                      'pixels_per_micron': 1.0}
            
            for key in params:
                
                if (isinstance(params[key], float)):
                    assert params[key] == pytest.approx(result[key])
                else:
                    assert params[key] == pytest.approx(result[key])
            
        
        def test_set_environment_all(self):
            
            nyx = nyxus.Nyxus (["*ALL*"])
            assert nyx is not None
            
            nyx.set_environment_params(
                features = ["GABOR"],
                neighbor_distance = 2,
                pixels_per_micron = 2,
                coarse_gray_depth = 2,
                n_feature_calc_threads = 2,
                n_loader_threads = 2,
                using_gpu = 0
            )
            
            params = nyx.get_params()
            
            result = {'coarse_gray_depth': 2, 
                      'features': ['GABOR'], 
                      'gabor_f0': 0.1, 
                      'gabor_freqs': [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0], 
                      'gabor_gamma': 0.1, 
                      'gabor_kersize': 16, 
                      'gabor_sig2lam': 0.8, 
                      'gabor_theta': 45.0, 
                      'gabor_thold': 0.025, 
                      'ibsi': 0, 
                      'n_loader_threads': 2, 
                      'n_feature_calc_threads': 2, 
                      'neighbor_distance': 2, 
                      'pixels_per_micron': 2}
            
            for key in params:
                
                if (isinstance(params[key], float)):
                    assert params[key] == pytest.approx(result[key])
                else:
                    assert params[key] == pytest.approx(result[key])     
            
        def test_set_environment_all(self):
            
            nyx = nyxus.Nyxus (["*ALL*"])
            assert nyx is not None
            
            nyx.set_environment_params(features = ["GABOR"])
            
            params = nyx.get_params()
            
            result = {'coarse_gray_depth': 256, 
                      'features': ['GABOR'], 
                      'gabor_f0': 0.1, 
                      'gabor_freqs': [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0], 
                      'gabor_gamma': 0.1, 
                      'gabor_kersize': 16, 
                      'gabor_sig2lam': 0.8, 
                      'gabor_theta': 45.0, 
                      'gabor_thold': 0.025, 
                      'ibsi': 0, 
                      'n_loader_threads': 1, 
                      'n_feature_calc_threads': 4, 
                      'neighbor_distance': 5, 
                      'pixels_per_micron': 1.0}
            
            for key in params:
                
                if (isinstance(params[key], float)):
                    assert params[key] == pytest.approx(result[key])
                else:
                    assert params[key] == pytest.approx(result[key]) 
                    
        def test_constructor_with_gabor(self):
            
            nyx = nyxus.Nyxus (
                ["*ALL*"],
                gabor_kersize = 1,
                gabor_gamma = 1,
                gabor_sig2lam = 1,
                gabor_f0 = 1,
                gabor_theta = 1,
                gabor_thold = 1,
                gabor_freqs = [1,1,1,1,1])
            
            assert nyx is not None
            
            params = nyx.get_params()
            
            result = {'coarse_gray_depth': 256, 
                      'features': ['*ALL*'], 
                      'gabor_f0': 1, 
                      'gabor_freqs': [1, 1, 1, 1, 1], 
                      'gabor_gamma': 1, 
                      'gabor_kersize': 1, 
                      'gabor_sig2lam': 1, 
                      'gabor_theta': 1, 
                      'gabor_thold': 1, 
                      'ibsi': 0, 
                      'n_loader_threads': 1, 
                      'n_feature_calc_threads': 4, 
                      'neighbor_distance': 5, 
                      'pixels_per_micron': 1.0}
            
            for key in params:
                
                if (isinstance(params[key], float)):
                    assert params[key] == pytest.approx(result[key])
                else:
                    assert params[key] == pytest.approx(result[key])    
            
                
        def test_in_memory_2d(self):
                
            cpu_nyx = nyxus.Nyxus(["*ALL_GLCM*"], ibsi=True)
            
            names = ["test_name_1"]

            cpu_features = cpu_nyx.featurize(intens[0], seg[0], names, names)
            
            assert cpu_nyx.error_message == ''
            
        def test_in_memory_3d(self):
            
            cpu_nyx = nyxus.Nyxus(["*ALL_GLCM*"], ibsi=True)
            
            names = ["test_name_1", "test_name_2", "test_name_3", "test_name_4"]

            cpu_features = cpu_nyx.featurize(intens, seg, names, names)
            
            assert cpu_nyx.error_message == ''

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

                
        