import sys
import numpy as np
from . import nyx_backend

def isalive():
    print ("Responding")
    return True
    
def isalive2(arg):
    print ("isalive2 received " + str(arg))
    return True    

def isalive3():
    print ("Responding with a numpy array:")
    f = np.array([[1,2,3], [4,2,5]])
    return f    

def backend_is_alive (label_dir, intensity_dir):
    assert (bool(label_dir and not label_dir.isspace()))
    assert (bool(intensity_dir and not intensity_dir.isspace()))
    f = nyx_backend.backend_is_alive_imp (label_dir, intensity_dir)
    print (type(f))
    print(f)    
    return f

def pixel_intensity_stats (label_dir, intensity_dir):
    """Calculate mean, median, min, max, range, standard_deviation, skewness, kurtosis, 
    mean_absolute_deviation, energy, root_mean_squared, entropy, mode, uniformity, 
    P10, P25, P75, P90, interquartile_range, robust_mean_absolute_deviation, 
    weighted_centroid_y, weighted_centroid_x 

    Parameters
    ----------
    label_dir : path to the label images directory, type string
    intensity_dir : path to the intensity images directory, type string

    Returns
    -------
    numpy.ndarray, type float - 2-dimensional array of N calculated features 
        for each of L unique labels in shape 
        `[[f1, f2, ... fN], [f1, f2, ... fN], ... [f1, f2, ... fN]]`.
    """

    assert (bool(label_dir and not label_dir.isspace()))
    assert (bool(intensity_dir and not intensity_dir.isspace()))
    f = nyx_backend.calc_pixel_intensity_stats (label_dir, intensity_dir)
    return f

def bounding_box (label_dir, intensity_dir):
    """Calculate ROIs' bounding box x, y, width, and height  

    Parameters
    ----------
    label_dir : path to the label images directory, type string
    intensity_dir : path to the intensity images directory, type string

    Returns
    -------
    numpy.ndarray, type float - 2-dimensional array of N calculated features 
        for each of L unique labels in shape 
        `[[f1, f2, ... fN], [f1, f2, ... fN], ... [f1, f2, ... fN]]`.
    """

    assert (bool(label_dir and not label_dir.isspace()))
    assert (bool(intensity_dir and not intensity_dir.isspace()))
    f = nyx_backend.calc_bounding_box (label_dir, intensity_dir)
    return f


def feret (label_dir, intensity_dir):
    """Calculate ROIs' Feret diameters, angles, and statistics  

    Parameters
    ----------
    label_dir : path to the label images directory, type string
    intensity_dir : path to the intensity images directory, type string

    Returns
    -------
    numpy.ndarray, type float - 2-dimensional array of N calculated features 
            MIN_FERET_DIAMETER
            MAX_FERET_DIAMETER
            MIN_FERET_ANGLE
            MAX_FERET_ANGLE
            STAT_FERET_DIAM_MIN
            STAT_FERET_DIAM_MAX
            STAT_FERET_DIAM_MEAN
            STAT_FERET_DIAM_MEDIAN
            STAT_FERET_DIAM_STDDEV
            STAT_FERET_DIAM_MODE
        for each of L unique labels in shape 
        `[[f1, f2, ... fN], [f1, f2, ... fN], ... [f1, f2, ... fN]]`.
    """

    assert (bool(label_dir and not label_dir.isspace()))
    assert (bool(intensity_dir and not intensity_dir.isspace()))
    f = nyx_backend.calc_feret (label_dir, intensity_dir)
    return f

def glrlm (label_dir, intensity_dir):
    """Calculate ROIs' GLRLM features  

    Parameters
    ----------
    label_dir : path to the label images directory, type string
    intensity_dir : path to the intensity images directory, type string

    Returns
    -------
    numpy.ndarray, type float - 2-dimensional array of N calculated features 
	        GLRLM_SRE,
	        GLRLM_LRE,
	        GLRLM_GLN,
	        GLRLM_GLNN,
	        GLRLM_RLN,
	        GLRLM_RLNN,
	        GLRLM_RP,
	        GLRLM_GLV,
	        GLRLM_RV,
	        GLRLM_RE,
	        GLRLM_LGLRE,
	        GLRLM_HGLRE,
	        GLRLM_SRLGLE,
	        GLRLM_SRHGLE,
	        GLRLM_LRLGLE,
	        GLRLM_LRHGLE
        for each of L unique labels in shape 
        `[[f1, f2, ... fN], [f1, f2, ... fN], ... [f1, f2, ... fN]]`.
    """

    assert (bool(label_dir and not label_dir.isspace()))
    assert (bool(intensity_dir and not intensity_dir.isspace()))
    f = nyx_backend.calc_glrlm (label_dir, intensity_dir)
    return f
