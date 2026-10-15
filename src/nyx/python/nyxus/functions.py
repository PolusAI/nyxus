from .backend import gpu_available_imp, get_gpu_props

def gpu_is_available (nyxus_instance_id):
    r"""Report whether a CUDA GPU can be used.

    Parameters
    ----------
    nyxus_instance_id : int
        ``id()`` of a :class:`Nyxus`, :class:`Nyxus3D` or
        :class:`ImageQuality` instance.

    Returns
    -------
    bool
        True when this build has GPU support and a CUDA device is present.

    Examples
    --------
    >>> import nyxus
    >>> nyx = nyxus.Nyxus(["MEAN"])
    >>> available = nyxus.gpu_is_available(id(nyx))
    """
    return gpu_available_imp (nyxus_instance_id)

def get_gpu_properties (nyxus_instance_id):
    r"""Describe the CUDA GPU devices.

    Parameters
    ----------
    nyxus_instance_id : int
        ``id()`` of a :class:`Nyxus`, :class:`Nyxus3D` or
        :class:`ImageQuality` instance.

    Returns
    -------
    list of dict
        One dict of property name to value (both str) per device. Empty when
        this build has no GPU support.

    Examples
    --------
    >>> import nyxus
    >>> nyx = nyxus.Nyxus(["MEAN"])
    >>> props = nyxus.get_gpu_properties(id(nyx))
    >>> isinstance(props, list)
    True
    """
    return get_gpu_props (nyxus_instance_id)