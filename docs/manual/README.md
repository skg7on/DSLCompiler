# MicroIR developer manual

Published at **<https://skg7on.github.io/DSLCompiler/>**, updated automatically
from `main`.

Open [`html/index.html`](html/index.html) in a browser. The HTML directory includes search, navigation, styles, the architecture diagram, and downloadable examples. It can be copied or served independently.

| Location | Purpose |
|---|---|
| [`source/index.rst`](source/index.rst) | Manual entry point and table of contents |
| [`source/`](source/) | RST guides, tool references, and tutorials |
| [`source/examples/`](source/examples/) | Checked-in MLIR tutorial inputs |
| [`source/conf.py`](source/conf.py) | Sphinx configuration |
| [`requirements.txt`](requirements.txt) | Pinned documentation dependencies; Python 3.12+ |
| [`html/`](html/) | Generated HTML checked into Git |

From the repository root, rebuild after editing sources:

```bash
python3 -m venv build/docs-venv
build/docs-venv/bin/python -m pip install -r docs/manual/requirements.txt
build/docs-venv/bin/python -m sphinx -b html -n -W --keep-going \
  -d build/docs/doctrees docs/manual/source docs/manual/html
```

Commit source changes and their regenerated HTML together. Sphinx's cache stays in ignored `build/docs/doctrees/`. The optional CMake `docs-html` target uses the same source and output directories. See [`source/building-docs.rst`](source/building-docs.rst) for checks and maintenance instructions.

Preview locally with `python3 -m http.server 8000 --bind 127.0.0.1 --directory docs/manual/html`, then open `http://127.0.0.1:8000/`.
