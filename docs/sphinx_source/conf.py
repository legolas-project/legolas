# Configuration file for the Sphinx documentation builder.

# -- Path setup --------------------------------------------------------------
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath("../../post_processing/pylbo"))

# -- Version setup -----------------------------------------------------------
version_filepath = Path("../pylbo_version.txt").resolve()
VERSION = None
RELEASE = None
with open(version_filepath) as f:
    for line in f.readlines():
        if line.strip().startswith("version:"):
            VERSION = line.split(":")[-1].strip().strip('"')
        elif line.strip().startswith("release:"):
            RELEASE = line.split(":")[-1].strip().strip('"')

# -- Project information -----------------------------------------------------
project = "Pylbo"
copyright = "Niels Claes, Jordi De Jonghe and Rony Keppens (2020)"
author = "Niels Claes, Jordi De Jonghe and Rony Keppens"
version = f"{VERSION} - {RELEASE}"
release = RELEASE


# -- General configuration ---------------------------------------------------
extensions = [
    "autoapi.extension",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "numpydoc",
    "sphinx_rtd_theme",
]
numpydoc_show_inherited_class_members = True
napoleon_include_private_with_doc = True
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/", None),
}

# autoAPI settings
autoapi_type = "python"
autoapi_dirs = ["../../post_processing/pylbo/"]
autoapi_options = [
    "members",  # children of objects
    "undoc-members",  # include no-docstring members
    "private-members",  # include _
    # 'special-members',           # include __
    "show-inheritance",  # list of classes below base class
    # 'show-inheritance-diagram',  # needs graphviz
    "show-module-summary",  # passes to autosummary
    "imported-members",  # objects from top-level package
]

exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "logo_only": False,
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": True,
}
github_url = "https://github.com/n-claes/legolas/tree/master/post_processing/pylbo"
