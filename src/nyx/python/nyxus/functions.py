from .backend import gpu_available_imp, get_gpu_props

def gpu_is_available():
    return gpu_available_imp()

def get_gpu_properties():
    return get_gpu_props()