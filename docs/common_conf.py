# Configuration file for the Sphinx documentation builder.

import os
import sys

# Define common settings here
project = 'LlamaFactory'
copyright = '2024, LlamaFactory Team'
author = 'LlamaFactory Team'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'myst_parser',
]

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'sphinx_rtd_theme'
# html_static_path = ['_static'] # Handled in individual conf.py if needed

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]
myst_heading_anchors = 3
