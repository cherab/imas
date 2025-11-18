"""The configuration for this package documentation."""

from datetime import date

from packaging.version import parse

from cherab.imas import __version__

# -- Project information -----------------------------------------------------
project = "CHERAB-IMAS"
author = "CHERAB Team"
copyright = f"2023-{date.today().year}, {author}"
version_obj = parse(__version__)
release = version_obj.public

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx.ext.napoleon",
    "sphinx_api_relink",
    "sphinx_copybutton",
    "sphinx_codeautolink",
    "sphinx_design",
    "sphinx_github_style",
    "sphinx_immaterial",
    "IPython.sphinxext.ipython_console_highlighting",
    "myst_parser",
    "nbsphinx",
]

default_role = "obj"

# autodoc config
autodoc_member_order = "bysource"
autodoc_typehints = "signature"
autodoc_type_aliases = {
    "ArrayLike": "ArrayLike",
}

# autosummary config
autosummary_generate = True
autosummary_generate_overwrite = True
autosummary_imported_members = True
autosummary_ignore_module_all = False

# napoleon config
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_attr_annotations = True

# todo config
todo_include_todos = True

# Strip prompt text when copying code blocks with copy button
copybutton_exclude = ".linenos, .gp"
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
]

templates_path = ["_templates"]

# myst-parser config
myst_enable_extensions = [
    "amsmath",
    "attrs_inline",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]
myst_url_schemes = {
    "http": None,
    "https": None,
    "mailto": None,
    "ftp": None,
    "wiki": "https://en.wikipedia.org/wiki/{{path}}#{{fragment}}",
    "doi": "https://doi.org/{{path}}",
}

# -- HTML output ------------------------------------------------------------
html_theme = "sphinx_immaterial"
html_title = f"{project} v{release}"
html_theme_options = {
    "repo_url": "https://github.com/cherab/imas",
    "repo_name": "CHERAB-IMAS",
    "edit_uri": "blob/master/docs/source",
    "icon": {
        "repo": "fontawesome/brands/github",
    },
    "features": [
        "navigation.expand",
        # "navigation.tabs",
        # "navigation.tabs.sticky",
        # "toc.integrate",
        "navigation.sections",
        # "navigation.instant",
        # "header.autohide",
        "navigation.top",
        "navigation.footer",
        # "navigation.tracking",
        # "search.highlight",
        "search.share",
        "search.suggest",
        "toc.follow",
        "toc.sticky",
        "content.tabs.link",
        # "content.code.copy",
        # "content.action.edit",
        # "content.action.view",
        "content.tooltips",
        "announce.dismiss",
    ],
    # "toc_title_is_page_title": True,
    # "globaltoc_collapse": True,
    "palette": [
        {
            "media": "(prefers-color-scheme: light)",
            "scheme": "default",
            "primary": "indigo",
            "toggle": {
                "icon": "material/lightbulb-outline",
                "name": "Switch to dark mode",
            },
        },
        {
            "media": "(prefers-color-scheme: dark)",
            "scheme": "slate",
            "primary": "indigo",
            "toggle": {
                "icon": "material/lightbulb",
                "name": "Switch to light mode",
            },
        },
    ],
}

# Shorten Table Of Contents in API documentation
object_description_options = [
    (".*", dict(include_fields_in_toc=False)),
    (".*parameter", dict(include_in_toc=False)),
]

# -- Intersphinx configuration -----------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "raysect": ("http://www.raysect.org", None),
    "cherab": ("https://www.cherab.info", None),
    "imas-python": ("https://imas-python.readthedocs.io/en/stable/", None),
    "rich": ("https://rich.readthedocs.io/en/stable/", None),
    "pooch": ("https://www.fatiando.org/pooch/latest/", None),
}

intersphinx_timeout = 10

# -- Sphinx GitHub Style configuration ----------------------------------------
linkcode_blob = "master" if version_obj.is_devrelease else f"v{version_obj.public}"
linkcode_url = "https://github.com/cherab/imas"
linkcode_link_text = "Source"

# -- NBSphinx configuration ---------------------------------------------------
# nbsphinx_execute = "never"
nbsphinx_prolog = r"""
{% set docname = 'docs/' + env.doc2path(env.docname, base=None)|string %}

.. raw:: html

    <div class="admonition note">
      This page was generated from
      <a class="reference external" href="https://github.com/cherab/imas/blob/{{ env.config.linkcode_blob|e }}/{{ docname|e }}">{{ docname|e }}</a>.
      <br />
      <a href="{{ env.docname.split('/')|last|e + '.ipynb' }}" class="reference download internal" download>Download notebook</a>.
      <script>
        if (document.location.host) {
          let nbviewer_link = document.createElement('a');
          nbviewer_link.setAttribute('href',
            'https://nbviewer.org/url' +
            (window.location.protocol == 'https:' ? 's/' : '/') +
            window.location.host +
            window.location.pathname.slice(0, -4) +
            'ipynb');
          nbviewer_link.innerHTML = 'Or view it on <em>nbviewer</em>';
          nbviewer_link.classList.add('reference');
          nbviewer_link.classList.add('external');
          document.currentScript.replaceWith(nbviewer_link, '.');
        }
      </script>
    </div>

.. raw:: latex

    \nbsphinxstartnotebook{\scriptsize\noindent\strut
    \textcolor{gray}{The following section was generated from
    \sphinxcode{\sphinxupquote{\strut {{ docname | escape_latex }}}} \dotfill}}
"""
