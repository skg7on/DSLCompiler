Build and maintain this manual
==============================

The community manual is written in native reStructuredText and built with
`Sphinx <https://www.sphinx-doc.org/en/master/usage/quickstart.html>`__ and the
`Furo theme <https://pradyunsg.me/furo/quickstart/>`__. You can build it without
LLVM, MLIR, or any compiler binaries. Python **3.12 or newer** is required by
the pinned Sphinx version.

Build HTML
----------

From the repository root or your isolated worktree:

.. code-block:: bash

   python3 -m venv build/docs-venv
   build/docs-venv/bin/python -m pip install -r docs/requirements.txt
   build/docs-venv/bin/python -m sphinx -b html -n -W --keep-going \
     docs build/docs/html

Open ``build/docs/html/index.html`` in your browser. The complete
``build/docs/html/`` directory is the portable manual: retain its assets,
search index, and download directory when copying or sharing it. The theme
and architecture diagram are bundled locally; reading the pages needs no CDN.
Links to repository source and historical design documents open GitHub.

To preview through HTTP, run:

.. code-block:: bash

   python3 -m http.server 8000 --bind 127.0.0.1 --directory build/docs/html

Then open ``http://127.0.0.1:8000/``. Use Ctrl-C to stop the preview server.

The strict build enables nitpicky reference checking (``-n``) and treats
warnings as errors (``-W``). ``--keep-going`` collects the remaining diagnostics
so one run can show all broken references. A successful build validates the
manual's document/section links, downloads, images, and repository source paths;
it does not execute the documented compiler commands.

If you already configured a project build, the optional CMake target uses the
same strict command:

.. code-block:: bash

   cmake -S . -B build \
     -DLLK_DOCS_PYTHON="$PWD/build/docs-venv/bin/python"
   cmake --build build --target docs-html

Install :source:`docs/requirements.txt` first. ``LLK_DOCS_PYTHON`` defaults to
``python3``; Sphinx is needed only when you invoke ``docs-html``. The ordinary
compiler build and ``check-llk`` do not install or require Sphinx.

Edit a page
-----------

Use an isolated worktree as described in :doc:`contributing`. Edit the ``.rst``
page beside :source:`docs/conf.py`, then build HTML and inspect the page.
The older design, review, and implementation-plan documents remain Markdown
reference material linked from the manual. The repository README remains the
entry point for GitHub readers.

Use the `Sphinx reStructuredText guide
<https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`__
for syntax. These conventions keep the manual navigable and checkable:

* Add each new reader-facing page to a ``toctree`` and to ``DOCS`` in
  :source:`test/Docs/check_doc_references.py`.
* Use ``:doc:`` for manual pages and ``:ref:`` for explicitly labeled sections.
  Absolute manual targets start at ``docs/``, for example ``/tools/compiler``.
* Use ``:source:`` for repository files or directories outside the manual.
  Targets are repository-relative; the build checks that they exist before
  creating a GitHub source link.
* Use ``:download:`` for checked-in example inputs and the tutorial runner.
  Downloads are copied into the HTML build. Use ``literalinclude`` to display
  a complete fixture without duplicating its source.
* Use ``code-block`` with a language such as ``bash``, ``mlir``, or ``yaml``.
  Put shell commands in ``bash``, ``sh``, or ``shell`` blocks so the CLI flag
  checker can inspect them.
* Use ``list-table`` for option and capability tables, and an admonition for a
  result that requires special interpretation. Keep claims tied to the reviewed
  source revision; a repair plan is not an implemented feature.

Verify commands and artifacts
-----------------------------

After building the compiler tools, run the registered documentation checks:

.. code-block:: bash

   ctest --test-dir build -R 'DocReferences|CommunityTutorials' \
     --output-on-failure

``DocReferences`` checks paths and CLI flag names against the built tools'
help output. ``CommunityTutorials`` runs the companion workflows and checks
their resulting artifacts. The checker itself has focused regressions:

.. code-block:: bash

   python3 -m unittest discover -s test/Docs -p 'test_*.py'

The Sphinx source-link extension also checks incremental invalidation. Run its
regression using the documentation environment:

.. code-block:: bash

   build/docs-venv/bin/python -m unittest discover \
     -s docs/_ext/tests -p 'test_*.py'

For a behavior or build-system change, also run ``check-llk`` as described in
:doc:`contributing`. Report skips separately from passes.

Continuous integration
----------------------

The :source:`documentation workflow <.github/workflows/docs.yml>` installs
the pinned dependencies, runs checker and Sphinx extension regressions, and
builds the manual with
warnings treated as errors. Each successful run uploads a ``developer-manual``
artifact containing the HTML directory. The compiler CI separately runs
``DocReferences`` and ``CommunityTutorials`` against freshly built binaries.

An HTML artifact can be served by any static host. The workflow builds and
packages it; site publication is a separate repository-maintainer decision.
