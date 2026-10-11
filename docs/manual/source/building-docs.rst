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

Only sources are tracked. ``docs/manual/source/`` holds the reStructuredText
pages, configuration, and examples; the rendered HTML is produced by CI and
published, so there is no committed copy to open or update.

.. code-block:: bash

   python3 -m venv build/docs-venv
   build/docs-venv/bin/python -m pip install -r docs/manual/requirements.txt
   build/docs-venv/bin/python -m sphinx -b html -n -W --keep-going \
     -d build/docs/doctrees docs/manual/source build/docs/html

Open ``build/docs/html/index.html`` in your browser. The generated directory is
the portable manual: retain its assets, search index, and download directory
when copying or sharing it. It lives in the ignored build tree, so it is never
committed. The theme and architecture diagram are bundled locally; reading the
pages needs no CDN. Links to repository source and historical design documents
open GitHub.

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

Install :source:`docs/manual/requirements.txt` first. ``LLK_DOCS_PYTHON`` defaults to
``python3``; Sphinx is needed only when you invoke ``docs-html``. The ordinary
compiler build and ``check-llk`` do not install or require Sphinx.

Edit a page
-----------

Use an isolated worktree as described in :doc:`contributing`. Edit the ``.rst``
page beside :source:`docs/manual/source/conf.py`, then build HTML and inspect the page.
The older design, review, and implementation-plan documents remain Markdown
reference material linked from the manual. The repository README remains the
entry point for GitHub readers.

Edit sources only; there is no generated HTML to commit. Keep Sphinx's
environment cache under ignored ``build/docs/doctrees/`` and the output under
``build/docs/html/``, both inside the build tree. Documentation CI renders the
manual with warnings treated as errors on every pull request and publishes it on
merge, so a source page is the single input to both the check and the live site.

Use the `Sphinx reStructuredText guide
<https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`__
for syntax. These conventions keep the manual navigable and checkable:

* Add each new reader-facing page to a ``toctree`` and to ``DOCS`` in
  :source:`test/Docs/check_doc_references.py`.
* Use ``:doc:`` for manual pages and ``:ref:`` for explicitly labeled sections.
  Absolute manual targets start at ``docs/manual/source/``, for example ``/tools/compiler``.
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
     -s docs/manual/source/_ext/tests -p 'test_*.py'

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

Published site
--------------

The manual is served from GitHub Pages at
``https://skg7on.github.io/DSLCompiler/``. On every push to ``main`` the
workflow deploys the directory it just built to the ``github-pages``
environment; pull requests build and verify the same way but never publish.

Merging is what changes the live site: the deployed tree is the one the ``html``
job just built from the sources in that commit. No HTML is committed, so there
is nothing to regenerate or keep in sync — editing a source page and merging it
is the whole release step.

Publication needs the repository's Pages source set to **GitHub Actions**
(Settings → Pages). Without it the ``deploy`` job fails while ``html`` still
passes, so a missing setting never blocks the build or a pull request. Once the
source is set, the deployed URL is reported as the ``github-pages`` environment
URL on the workflow run.

Any HTML artifact remains servable by any other static host; Pages is the
project's chosen one.
