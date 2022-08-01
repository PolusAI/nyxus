from .nyxus import Nyxus
from .nyxus import Nested
from .functions import gpu_is_available, get_gpu_properties

from . import _version
__version__ = _version.get_versions()['version']
