# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))


# -- Project information -----------------------------------------------------

project = "zerohertzLib"
copyright = "2023, Zerohertz"
author = "Zerohertz"

# The full version, including alpha/beta/rc tags
import zerohertzLib as zz

release = zz.__version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.duration",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinxcontrib.jquery",
    "sphinxcontrib.gtagjs",
    "sphinxext.opengraph",
    "sphinx_favicon",
    "sphinx_copybutton",
    "sphinx_paramlinks",
    "myst_parser",
]
autodoc_typehints = "none"
todo_include_todos = True
gtagjs_ids = [
    "G-ZCW0CR8M8X",
]
# add_module_names = False

# Add any paths that contain templates here, relative to this directory.
# templates_path = ["_templates"]

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_theme = "furo"
html_baseurl = "https://zerohertz.github.io/zerohertzLib/"
html_static_path = ["_static"]
html_logo = "_static/logo.png"

favicons = [
    {"href": "favicon.png"},
    {"href": "favicon-16x16.png"},
    {"href": "favicon-32x32.png"},
    {
        "rel": "apple-touch-icon",
        "href": "favicon.png",
    },
]

# https://sphinx-themes.org/sample-sites/furo/
# https://pradyunsg.me/furo/quickstart/
# https://github.com/a-r-j/graphein/blob/master/docs/source/conf.py
# https://github.com/gpchelkin/pip/blob/bc45b93eb679963d23f100be4ebb5d0f1568ceee/docs/html/conf.py
# https://github.com/MacHu-GWU/cottonformation-project/blob/f5253fe8b53ae53ea9517bd366e776f207371b56/docs/source/conf.py
html_css_files = [
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/fontawesome.min.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/solid.min.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/brands.min.css",
]
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#800a0a",
        "color-brand-content": "#800a0a",
    },
    "dark_css_variables": {
        "color-brand-primary": "#800a0a",
        "color-brand-content": "#800a0a",
    },
    "source_repository": "https://github.com/Zerohertz/zerohertzLib/",
    "source_branch": "master",
    "source_directory": "sphinx/source/",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/Zerohertz",
            "html": "",
            "class": "fa-brands fa-solid fa-github fa-2x",
        },
    ],
}

ogp_site_url = "https://zerohertz.github.io/zerohertzLib/"
ogp_description_length = 200
ogp_site_name = f"Zerohertz's Library {release} Documents"
ogp_image = "_static/og.png"
ogp_type = "website"


# -- Options for sphinx_rtd_theme -------------------------------------------------

# html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
# https://sphinx-rtd-theme.readthedocs.io/en/stable/configuring.html
# html_theme_options = {
#     "analytics_id": "G-ZCW0CR8M8X",
#     "analytics_anonymize_ip": False,
#     "logo_only": True,
#     "style_nav_header_background": "#000000",
# }
