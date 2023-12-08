import bfio
from test_data import intens, seg
import numpy as np

def write_test_data(path: str):
    # intensity and segmentation paths
    int_path = path +'int/'
    seg_path = path + 'seg/'

    # get in-memory test data and give image names
    files = {'image1.ome.tif': [intens[0], seg[0]], 
            'image2.ome.tif': [intens[1], seg[1]], 
            'image3.ome.tif': [intens[2], seg[2]], 
            'image4.ome.tif': [intens[3], seg[3]]}

    # write test data
    for file in files:
        # write intensity image
        with bfio.BioWriter(int_path + file) as bw:
                int_image = files[file][0]

                bw.shape = int_image.shape[:2]
                print(bw.shape)
                bw.dtype = np.uint8

                bw[:] = int_image[:, :]
        
        # write segmentation image
        with bfio.BioWriter(seg_path + file) as bw:
                seg_image = files[file][1]

                bw.shape = seg_image.shape[:2]
                print(bw.shape)
                bw.dtype = np.uint8
                

                bw[:] = int_image[:, :]

