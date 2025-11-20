from .backend import gpu_available_imp, get_gpu_props

def gpu_is_available (nyxus_instance_id):
    return gpu_available_imp (nyxus_instance_id)

def get_gpu_properties (nyxus_instance_id):
    return get_gpu_props (nyxus_instance_id)