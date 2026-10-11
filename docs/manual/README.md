# MicroIR developer manual

Published at **<https://skg7on.github.io/DSLCompiler/>**, rebuilt from `main` by
CI on every merge. The repository holds sources only — there is no committed
HTML copy to keep in sync, and none to open from a plain checkout.

| Location | Purpose |
|---|---|
| [`source/index.rst`](source/index.rst) | Manual entry point and table of contents |
| [`source/`](source/) | RST guides, tool references, and tutorials |
| [`source/examples/`](source/examples/) | Checked-in MLIR tutorial inputs |
| [`source/conf.py`](source/conf.py) | Sphinx configuration |
| [`requirements.txt`](requirements.txt) | Pinned documentation dependencies; Python 3.12+ |

To read the manual, use the published site. To build it locally after editing
sources:

```bash
python3 -m venv build/docs-venv
build/docs-venv/bin/python -m pip install -r docs/manual/requirements.txt
build/docs-venv/bin/python -m sphinx -b html -n -W --keep-going \
  -d build/docs/doctrees docs/manual/source build/docs/html
```

The output is a self-contained directory in the ignored build tree; Sphinx's
cache stays in `build/docs/doctrees/`. The optional CMake `docs-html` target
builds the same way. See [`source/building-docs.rst`](source/building-docs.rst)
for checks and maintenance instructions.

Preview locally with `python3 -m http.server 8000 --bind 127.0.0.1 --directory build/docs/html`, then open `http://127.0.0.1:8000/`.
