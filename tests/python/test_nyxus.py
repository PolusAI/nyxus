import nyxus
import pytest
import time
from pathlib import Path
from test_download_data import download

def test_import():
    assert nyxus.__name__ == "nyxus" 
    
def download_datasets(urls, dir_names):
    for i in range(len(urls)):
        download(urls[i], dir_names[i])
        

class TestImport():
    def test_import(self):
        assert nyxus.__name__ == "nyxus" 
        
class TestNyxus():
    PATH = PATH = Path(__file__).with_name('data')

    def test_gabor_gpu_validate(self):
            urls = ['https://github.com/stardist/stardist/releases/download/0.1.0/dsb2018.zip']
            dir_names = ['dsb2018']
            
            download_datasets(urls, dir_names)
            
            for dset in dir_names:
                
                intens = str(self.PATH/dset/"train/images")
                seg = str(self.PATH/dset/"train/masks")
                
                # cpu gabor
                cpu_nyx = nyxus.Nyxus(["GABOR"])
                if (nyxus.gpu_is_available()):
                    cpu_nyx.using_gpu(False)
                cpu_features = cpu_nyx.featurize_directory(intens, seg)

                if (nyxus.gpu_is_available()):
                    # gpu gabor
                    gpu_nyx = nyxus.Nyxus(["GABOR"], using_gpu=0)
                    gpu_nyx.using_gpu(True)
                    gpu_features = gpu_nyx.featurize_directory(intens, seg)
                    
                    assert gpu_features.equals(cpu_features)
                else:
                    print("Gpu not available")
                    assert True
    
    @pytest.mark.parametrize("use_fastloop",[True, False])
    def test_intensity_benchmark(self,use_fastloop,benchmark):
        
        urls = ['https://github.com/stardist/stardist/releases/download/0.1.0/dsb2018.zip']
        dir_names = ['dsb2018']
        
        download_datasets(urls, dir_names)
        
        for dset in dir_names:
            
            intens = str(self.PATH/dset/"train/images")
            seg = str(self.PATH/dset/"train/masks")
            
            # cpu gabor
            cpu_nyx = nyxus.Nyxus(["*ALL_MORPHOLOGY*"])
            cpu_nyx.featurize_directory
            cpu_features = benchmark(cpu_nyx.featurize_directory,intens, seg, ".*", use_fastloop)
            
    def test_intensity_validate(self):
        
        urls = ['https://github.com/stardist/stardist/releases/download/0.1.0/dsb2018.zip']
        dir_names = ['dsb2018']
        
        download_datasets(urls, dir_names)
        
        for dset in dir_names:
            
            intens = str(self.PATH/dset/"train/images")
            seg = str(self.PATH/dset/"train/masks")
            
            # cpu gabor
            cpu_nyx = nyxus.Nyxus(["*ALL_MORPHOLOGY*"])
            cpu_features = cpu_nyx.featurize_directory(intens, seg, ".*", False)
            
            fast_features = cpu_nyx.featurize_directory(intens, seg, ".*", True)
            
            assert cpu_features.equals(fast_features)

    # For benchmarking gabor features            
    # @pytest.mark.parametrize("use_gpu",[True, False])
    # def test_gabor_benchmark(self, use_gpu, benchmark):
        
    #     urls = ['https://github.com/stardist/stardist/releases/download/0.1.0/dsb2018.zip']
    #     dir_names = ['dsb2018']
        
    #     download_datasets(urls, dir_names)
        
    #     for dset in dir_names:
            
    #         intens = str(self.PATH/dset/"train/images")
    #         seg = str(self.PATH/dset/"train/masks")
            
    #         # cpu gabor
    #         cpu_nyx = nyxus.Nyxus(["GABOR"])
    #         cpu_nyx.using_gpu(use_gpu)
    #         cpu_features = benchmark(cpu_nyx.featurize_directory,intens, seg)
        