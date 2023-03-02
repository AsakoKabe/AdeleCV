# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import sys
from datetime import datetime

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))
sys.path.append('..')

# -- Project information -----------------------------------------------------

project = "AdeleCV"
copyright = '{}, Denis Mamatin'.format(datetime.now().year)
author = "Denis Mamatin"


def get_version():
    sys.path.append('../adelecv')
    from __version__ import __version__ as version
    # sys.path.pop()
    return version


version = get_version()

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.coverage',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    "sphinx.ext.autosummary",
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    'autodocsumm',
    'nbsphinx',
]

intersphinx_mapping = {
    "rtd": ("https://docs.readthedocs.io/en/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

# -- Options for EPUB output
epub_show_urls = "footnote"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['**.ipynb_checkpoints']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

import sphinx_rtd_theme

# html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

import faculty_sphinx_theme

html_theme = "faculty_sphinx_theme"
html_logo = "logo.png"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# -- Extension configuration -------------------------------------------------

autodoc_inherit_docstrings = False
napoleon_google_docstring = True
napoleon_include_init_with_doc = True
napoleon_numpy_docstring = False

autodoc_mock_imports = [
    'torch',
    'tqdm',
    'numpy',
    'cv2',
    'segmentation_models_pytorch',
    'albumentations',
    'fiftyone',
    'pandas',
    'optuna',
    'pydantic'
]

autoclass_content = 'both'
autodoc_typehints = 'description'


# --- Work around to make autoclass signatures not (*args, **kwargs) ----------

class FakeSignature:
    def __getattribute__(self, *args):
        raise ValueError


def f(app, obj, bound_method):
    if "__new__" in obj.__name__:
        obj.__signature__ = FakeSignature()


def setup(app):
    app.connect('autodoc-before-process-signature', f)


# Custom configuration --------------------------------------------------------

autodoc_member_order = 'bysource'

# Include example --------------------------------------------------------
import shutil
shutil.copy('../example/example_api.ipynb', './example_api.ipynb')
