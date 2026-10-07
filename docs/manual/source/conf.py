"""Sphinx configuration for the community developer manual."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "_ext"))

project = "MicroIR Developer Manual"
author = "DSLCompiler contributors"
copyright = "2026, DSLCompiler contributors"
version = "main"
release = "main"

extensions = ["sphinx.ext.autosectionlabel", "source_links"]
autosectionlabel_prefix_document = True
autosectionlabel_maxdepth = 2
source_suffix = {".rst": "restructuredtext"}
root_doc = "index"
exclude_patterns = ["_build", "_ext", "design", "reviews", "superpowers"]
nitpicky = True
highlight_language = "text"
pygments_style = "sphinx"

html_theme = "furo"
html_title = project
html_static_path = ["_static"]
html_css_files = ["manual.css"]
html_show_sourcelink = True
html_theme_options = {
    "source_repository": "https://github.com/skg7on/DSLCompiler/",
    "source_branch": "main",
    "source_directory": "docs/manual/source/",
    "light_css_variables": {
        "color-brand-primary": "#235c86",
        "color-brand-content": "#235c86",
    },
    "dark_css_variables": {
        "color-brand-primary": "#79bce8",
        "color-brand-content": "#79bce8",
    },
}


def setup(app):
    from mlir_lexer import MLIRLexer

    app.add_lexer("mlir", MLIRLexer)
