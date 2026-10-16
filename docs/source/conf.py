# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html
#
# The docs build without the compiled extension: the Python package is imported
# from the source tree and its pybind11 module, nyxus.backend, is mocked.

import os
import sys

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(_repo_root, 'src', 'nyx', 'python'))

# -- Project information -----------------------------------------------------

project = 'Nyxus'
author = 'Andriy Kharchenko'
copyright = 'Nyxus developers'

# The citation on the index page quotes the latest release through |release|. CITATION.cff
# records it, and release-please updates its version line on every release.
with open(os.path.join(_repo_root, 'CITATION.cff'), encoding='utf-8') as _f:
    release = next(line.split(':', 1)[1].split('#')[0].strip()
                   for line in _f if line.startswith('version:'))
version = '.'.join(release.split('.')[:2])


# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx_autodoc_typehints',
    'sphinx.ext.autosummary',
    'sphinx_design',
]

autodoc_mock_imports = ['nyxus.backend']

napoleon_use_param = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
add_module_names = False

autodoc_member_order = 'groupwise'
autodoc_typehints = 'description'

autoclass_content = 'both'  # class docstring followed by the __init__ docstring

autodoc_default_options = {
    'members': True,
    'inherited-members': True,
    'special-members': '__getitem__,__call__,__setitem__',
    'show-inheritance': True,
}

autosummary_generate = True

master_doc = 'index'

# -- Options for HTML output -------------------------------------------------

html_theme = 'pydata_sphinx_theme'
# The pages describe the development head, not a release, so the title carries no version number.
html_title = 'Nyxus'
html_theme_options = {
    'github_url': 'https://github.com/PolusAI/nyxus',
    'navbar_align': 'left',
    'show_toc_level': 2,
    'navigation_with_keys': False,
    'secondary_sidebar_items': ['page-toc'],
}

# -- Options for the link check ----------------------------------------------

# External sites time out or rate-limit now and then; retry before reporting a link as broken.
linkcheck_retries = 3
linkcheck_timeout = 30
linkcheck_rate_limit_timeout = 120
