from .backend import gpu_available, get_gpu_props

def gpu_is_available():
    return gpu_available()

def get_gpu_properties():
    return get_gpu_props()