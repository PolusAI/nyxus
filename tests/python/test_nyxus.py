import pyarrow as pa
import pyarrow.parquet as pq
import nyxus
import pytest
import numpy as np
import math
from pathlib import Path
import pathlib
from test_data import intens, seg
import shutil

from test_tissuenet_data import tissuenet_int, tissuenet_seg

class TestImport():
    def test_import(self):
        assert nyxus.__name__ == "nyxus"  
        
class TestNyxus():     
        def test_featurize_all(self):
            path = str(pathlib.Path(__file__).parent.resolve())
            
            data_path = path + '/data/'
            
            nyx = nyxus.Nyxus (["*ALL*"])
            assert nyx is not None
            
            directory_features = nyx.featurize_directory(data_path + 'int/', data_path + 'seg/')
            directory_features.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
            
            files_features = nyx.featurize_files(
                [data_path + 'int/p0_y1_r1_c0.ome.tif', data_path + 'int/p0_y1_r1_c1.ome.tif'],
                [data_path + 'seg/p0_y1_r1_c0.ome.tif', data_path + 'seg/p0_y1_r1_c1.ome.tif'],
                single_roi=False,
            )
            
            files_features.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
            
            directory_columns = directory_features.columns
            files_columns = files_features.columns
            
            assert len(directory_columns) == len(files_columns)
            
            files_not_equal = []
            
            for col in directory_columns:
                directory_list = directory_features[col].tolist()
                files_list = files_features[col].tolist()
                
                for directory_val, files_val in zip(directory_list, files_list):
                    if not directory_val == pytest.approx(files_val, rel=1e-5, abs=1e-5):
                        files_not_equal.append(col)
                        break
            
            assert len(files_not_equal) == 0
            
        def test_featurize_montage(self):
            
            path = str(pathlib.Path(__file__).parent.resolve())
            
            data_path = path + '/data/'
            
            nyx = nyxus.Nyxus (["*ALL_INTENSITY*"])
            assert nyx is not None
            
            montage_features = nyx.featurize(tissuenet_int, tissuenet_seg, intensity_names=['p0_y1_r1_c0.ome.tif', 'p0_y1_r1_c1.ome.tif'], label_names=['p0_y1_r1_c0.ome.tif', 'p0_y1_r1_c1.ome.tif'])
            directory_features = nyx.featurize_directory(data_path + 'int/', data_path + 'seg/')
            
            montage_features.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
            directory_features.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
            
            directory_columns = directory_features.columns
            montage_columns = montage_features.columns
            
            assert len(directory_columns) == len(montage_columns)
            
            montage_not_equal = []
            
            for col in directory_columns:
                directory_list = directory_features[col].tolist()
                montage_list = montage_features[col].tolist()
                
                for directory_val, montage_val in zip(directory_list, montage_list):
                    if not directory_val == pytest.approx(montage_val, rel=1e-4, abs=1e-4): # higher error tolerance for COVERED_IMAGE_INTENSITY_RANGE
                        montage_not_equal.append(col)
                        break
            
            assert len(montage_not_equal) == 0
            
        @pytest.mark.gpu
        def test_gpu_available(self):
            
            nyx = nyxus.Nyxus(["*ALL*"])
            
            assert nyx.gpu_is_available == True, f"GPU assertion failed. If running tests on a machine without GPU, use the flag \"-m \"not gpu\"\"."
            
        @pytest.mark.skip_ci
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
                nyx.set_gabor_feature_params(thold=0.025)
                nyx.set_gabor_feature_params(freqs=[1], thetas=[30])
                nyx.set_gabor_feature_params(freqs=[1,2,4,8,16,32,64], thetas=[15, 30, 45, 75, 90, 105, 115])
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
                nyx.set_gabor_feature_params(thetas="notAnumber")
                nyx.set_gabor_feature_params(thold="notAnumber")
            with pytest.raises (Exception):
                nyx.set_gabor_feature_params(freqs=["notAnumber"])
            with pytest.raises (Exception):
                nyx.set_gabor_feature_params(freqs="notAList")
        
        def test_get_default_params(self):
            
            nyx = nyxus.Nyxus (["*ALL*"])
            
            assert nyx is not None
            
            # actual
            a = nyx.get_params()
            
            # expected
            e = {
                'coarse_gray_depth': 256,
                'features': ['*ALL*'],
                'gabor_f0': 0.1,
                'gabor_freqs': [4.0, 16.0, 32.0, 64.0],
                'gabor_thetas': [0.0, 45.0, 90.0, 135.0],
                'gabor_gamma': 0.1,
                'gabor_kersize': 16,
                'gabor_sig2lam': 0.8,
                'gabor_thold': 0.025,
                'ibsi': 0,
                'n_feature_calc_threads': 4,
                'n_loader_threads': 1, 
                'neighbor_distance': 5, 
                'pixels_per_micron': 1.0,
                'dynamic_range': 10000,
                'min_intensity': 0.0,
                'max_intensity': 1.0
                }
            
            for key in a:
                
                if (isinstance(a[key], float)):
                    assert a[key] == pytest.approx(e[key])
                else:
                    assert a[key] == e[key]

        def test_get_params(self):
            
            nyx = nyxus.Nyxus (["*ALL*"])
            assert nyx is not None
            
            params = nyx.get_params('coarse_gray_depth', 'features', 'gabor_f0')
            
            result = {'coarse_gray_depth': 256, 
                      'features': ['*ALL*'], 
                      'gabor_f0': 0.1}
            
            assert len(params) == 3
            
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
                        'gabor_thetas': [0, 30, 60, 90, 120, 150, 180],
                        'gabor_gamma': 0.1, 
                        'gabor_kersize': 16, 
                        'gabor_sig2lam': 0.8, 
                        'gabor_thold': 0.025, 
                        'ibsi': 0, 
                        'n_loader_threads': 1, 
                        'n_feature_calc_threads': 4, 
                        'neighbor_distance': 5, 
                        'pixels_per_micron': 1.0,
                        'dynamic_range': 100,
                        'min_intensity': 0.5,
                        'max_intensity': 0.7
                        }
            
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
            nyx.set_params (coarse_gray_depth = 125)
            actual = nyx.get_params()
            expected = {'coarse_gray_depth': 125}
            assert actual['coarse_gray_depth'] == expected['coarse_gray_depth']
            
        
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
                dynamic_range = 1000,
                min_intensity = 0.1,
                max_intensity = 0.9
            )

            # actual
            a = nyx.get_params()

            # expected
            e = {
                'features': ["GABOR"],
                'neighbor_distance': 2,
                'pixels_per_micron': 2,
                'coarse_gray_depth': 2,
                'n_feature_calc_threads': 2,
                'n_loader_threads': 2
            }
                
            # compare
            for key in e:
                if (isinstance(e[key], float)):
                    assert e[key] == pytest.approx(a[key])
                else:
                    assert e[key] == a[key]
  
        def test_constructor_with_gabor(self):
            
            nyx = nyxus.Nyxus (
                ["*ALL*"],
                gabor_kersize = 1,
                gabor_gamma = 1,
                gabor_sig2lam = 1,
                gabor_f0 = 1,
                gabor_thold = 1,
                gabor_thetas = [10, 20, 30, 40, 50],
                gabor_freqs = [1, 2, 3, 4, 5])
            
            assert nyx is not None
            
            params = nyx.get_params()
            
            result = {'coarse_gray_depth': 256, 
                      'features': ['*ALL*'], 
                      'gabor_f0': 1, 
                      'gabor_freqs': [1, 2, 3, 4, 5], 
                      'gabor_gamma': 1, 
                      'gabor_kersize': 1, 
                      'gabor_sig2lam': 1, 
                      'gabor_thetas': [10, 20, 30, 40, 50], 
                      'gabor_thold': 1, 
                      'ibsi': 0, 
                      'n_loader_threads': 1, 
                      'n_feature_calc_threads': 4, 
                      'neighbor_distance': 5, 
                      'pixels_per_micron': 1.0,
                      'dynamic_range': 10000,
                      'min_intensity': 0.0,
                      'max_intensity': 1.0}
            
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
            
            cpu_nyx = nyxus.Nyxus([
                "GLCM_ASM", "GLCM_CONTRAST", "GLCM_CORRELATION", "GLCM_DIFAVE", 
                "GLCM_DIFENTRO", "GLCM_DIFVAR", "GLCM_ENERGY", "GLCM_ENTROPY",
                "GLCM_HOM1", "GLCM_INFOMEAS1", "GLCM_INFOMEAS2", "GLCM_IDM", 
                "GLCM_SUMAVERAGE", "GLCM_SUMENTROPY", "GLCM_SUMVARIANCE", "GLCM_VARIANCE"], 
            ibsi=True)
            
            names = ["test_name_1", "test_name_2", "test_name_3", "test_name_4"]

            cpu_features = cpu_nyx.featurize(intens, seg, names, names)
            
            assert cpu_nyx.error_message == ''

            means = cpu_features.mean(numeric_only=True)

            mean_values = means.tolist()

            mean_values.pop(0)

            averaged_results = []

            i = 0
            while (i < len(mean_values)):
                averaged_results.append(sum(mean_values[i:i+4])/4)
                i += 4
                
            # check IBSI values
            assert pytest.approx(averaged_results[0], 0.01) == 0.368 # angular2ndmoment
            assert pytest.approx(averaged_results[1], 0.01) == 5.28 # contrast 
            assert pytest.approx(averaged_results[2], 0.01) == -0.0121 # correlation
            assert pytest.approx(averaged_results[4], 0.01) == 1.40 # difference entropy
            assert pytest.approx(averaged_results[5], 0.1) == 2.90 # difference variance#
            
        @pytest.mark.arrow        
        def test_parquet_writer(self):
            
            nyx = nyxus.Nyxus (["*ALL*"])
            assert nyx is not None
            
            features = nyx.featurize(intens, seg)

            parquet_file = nyx.featurize(intens, seg, output_type="parquet")
            
            open_parquet_file = pq.ParquetFile(parquet_file)
            
            parquet_df = open_parquet_file.read().to_pandas()

            # Read the Parquet file into a Pandas DataFrame
            pd_columns = list(features.columns)

            arrow_columns = list(parquet_df.columns)
                
            assert len(pd_columns) == len(arrow_columns)
                
            for column in pd_columns:
                column_list = features[column].tolist()
                arrow_list = parquet_df[column].tolist()
                
                assert (len(column_list) == len(arrow_list))
                
                for j in range(len(column_list)):
                    feature_value = column_list[j]
                    arrow_value = arrow_list[j]
                    
                    #skip nan values
                    if (isinstance(feature_value, (int, float)) and math.isnan(feature_value)):
                        if (not math.isnan(arrow_value)):
                            assert False

                        continue
                    assert feature_value == pytest.approx(arrow_value, rel=1e-6, abs=1e-6)
            
            open_parquet_file.close()
            Path(parquet_file).unlink()
            
        @pytest.mark.arrow        
        def test_parquet_writer_file_naming(self, tmp_path):
        
            nyx = nyxus.Nyxus (["*ALL*"])
            assert nyx is not None
            
            features = nyx.featurize(intens, seg)
            output_dir = tmp_path/"TestNyxusOut"
            output_dir.mkdir()
            
            parquet_file = nyx.featurize(intens, seg, output_type="parquet", output_path=str(output_dir/"test_parquet"))

            output_file = Path(parquet_file)
            assert output_file.is_file()

            assert parquet_file == str(output_file)

            # Read the Parquet file into a Pandas DataFrame
            file = pq.ParquetFile(str(output_file))
            parquet_df = file.read().to_pandas()
            pd_columns = list(features.columns)

            arrow_columns = list(parquet_df.columns)
            
            assert len(pd_columns) == len(arrow_columns)
                
            for column in pd_columns:
                column_list = features[column].tolist()
                arrow_list = parquet_df[column].tolist()
                
                assert (len(column_list) == len(arrow_list))
                
                for j in range(len(column_list)):
                    
                    feature_value = column_list[j]
                    arrow_value = arrow_list[j]
                    
                    #skip nan values
                    if (isinstance(feature_value, (int, float)) and math.isnan(feature_value)):
                        if (not math.isnan(arrow_value)):
                            assert False

                        continue
                    assert feature_value == pytest.approx(arrow_value, rel=1e-6, abs=1e-6)
            
            file.close()

            shutil.rmtree(output_dir)

        @pytest.mark.arrow
        def test_make_arrow_ipc(self):
            
            nyx = nyxus.Nyxus (["*ALL*"])
            assert nyx is not None
            
            features = nyx.featurize(intens, seg)
            
            arrow_path = nyx.featurize(intens, seg, output_type="arrowipc")

            with pa.memory_map(arrow_path, 'rb') as source:
                arrow_pd = pa.ipc.open_file(source).read_all().to_pandas()
            
                pd_columns = list(features.columns)
                arrow_columns = list(arrow_pd.columns)
                
                assert len(pd_columns) == len(arrow_columns)
                    
                for column in pd_columns:
                    
                    column_list = features[column].tolist()
                    arrow_list = arrow_pd[column].tolist()
                    
                    assert len(column_list) == len(arrow_list)
                    
                    for j in range(len(column_list)):
                        feature_value = column_list[j]
                        arrow_value = arrow_list[j]
                        
                        #skip nan values
                        if (isinstance(feature_value, (int, float)) and math.isnan(feature_value)):
                            if (not math.isnan(arrow_value)):
                                assert False

                            continue
                        assert feature_value == pytest.approx(arrow_value, rel=1e-6, abs=1e-6)

            
            path = nyx.get_arrow_ipc_file()
            assert path == arrow_path
            
            Path(arrow_path).unlink()

        
        @pytest.mark.arrow
        def test_arrow_ipc(self, tmp_path):
            
            nyx = nyxus.Nyxus (["*ALL*"])
            assert nyx is not None
            output_dir = tmp_path/"TestNyxusOut"
            output_dir.mkdir()
            arrow_path = nyx.featurize(intens, seg, output_type="arrowipc", output_path=str(output_dir))

            features = nyx.featurize(intens, seg)
            
            with pa.memory_map(arrow_path, 'rb') as source:
                arrow_pd = pa.ipc.open_file(source).read_all().to_pandas()
            
                pd_columns = list(features.columns)
                arrow_columns = list(arrow_pd.columns)
                
                assert len(pd_columns) == len(arrow_columns)
                    


                for column in pd_columns:
                    
                    column_list = features[column].tolist()
                    arrow_list = arrow_pd[column].tolist()
                    
                    assert len(column_list) == len(arrow_list)
                    
                    for j in range(len(column_list)):
                        feature_value = column_list[j]
                        arrow_value = arrow_list[j]
                        
                        #skip nan values
                        if (isinstance(feature_value, (int, float)) and math.isnan(feature_value)):
                            if (not math.isnan(arrow_value)):
                                assert False

                            continue
                        assert feature_value == pytest.approx(arrow_value, rel=1e-6, abs=1e-6)
            
            shutil.rmtree(output_dir)

                        
        @pytest.mark.arrow
        def test_arrow_ipc_file_naming(self, tmp_path):
            
            nyx = nyxus.Nyxus (["*ALL*"])
            assert nyx is not None
            output_dir = tmp_path/"TestNyxusOut"
            output_dir.mkdir()
            output_file = output_dir/"test_nyxus.arrow"
            arrow_path = nyx.featurize(intens, seg, output_type="arrowipc", output_path=str(output_file))
            assert output_file.is_file()
            assert arrow_path == str(output_file)

            features = nyx.featurize(intens, seg)
            
            with pa.memory_map(arrow_path, 'rb') as source:
                arrow_pd = pa.ipc.open_file(source).read_all().to_pandas()
            
                pd_columns = list(features.columns)
                arrow_columns = list(arrow_pd.columns)
                
                assert len(pd_columns) == len(arrow_columns)
                    
                for column in pd_columns:
                    
                    column_list = features[column].tolist()
                    arrow_list = arrow_pd[column].tolist()
                    
                    assert len(column_list) == len(arrow_list)
                    
                    for j in range(len(column_list)):
                        feature_value = column_list[j]
                        arrow_value = arrow_list[j]
                        
                        #skip nan values
                        if (isinstance(feature_value, (int, float)) and math.isnan(feature_value)):
                            if (not math.isnan(arrow_value)):
                                assert False

                            continue

                        assert feature_value == pytest.approx(arrow_value, rel=1e-6, abs=1e-6)
            
            shutil.rmtree(output_dir)

        @pytest.mark.arrow
        def test_arrow_ipc_no_path(self):
            
            nyx = nyxus.Nyxus (["*ALL*"])
            assert nyx is not None
            
            arrow_path = nyx.featurize(intens, seg, output_type="arrowipc")
            
            assert arrow_path == 'NyxusFeatures.arrow'

            features = nyx.featurize(intens, seg)
        
            with pa.memory_map(arrow_path, 'rb') as source:
                arrow_pd = pa.ipc.open_file(source).read_all().to_pandas()
            
                pd_columns = list(features.columns)
                arrow_columns = list(arrow_pd.columns)
                
                assert len(pd_columns) == len(arrow_columns)
                    

                for column in pd_columns:
                    
                    column_list = features[column].tolist()
                    arrow_list = arrow_pd[column].tolist()
                    
                    assert len(column_list) == len(arrow_list)
                    
                    for j in range(len(column_list)):
                        feature_value = column_list[j]
                        arrow_value = arrow_list[j]
                        
                        #skip nan values
                        if (isinstance(feature_value, (int, float)) and math.isnan(feature_value)):
                            if (not math.isnan(arrow_value)):
                                assert False

                            continue
                        assert feature_value == pytest.approx(arrow_value, rel=1e-6, abs=1e-6)
            Path(arrow_path).unlink()
                        
        @pytest.mark.arrow         
        def test_arrow_ipc_path(self):
            
            nyx = nyxus.Nyxus (["*ALL*"])
            assert nyx is not None
            
            arrow_path = nyx.featurize(intens, seg, output_type="arrowipc")

            assert arrow_path == 'NyxusFeatures.arrow'           

