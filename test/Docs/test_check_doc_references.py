#!/usr/bin/env python3
"""Regression tests for documentation references and runnable shell examples."""

import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


CHECKER_PATH = Path(__file__).with_name("check_doc_references.py")
SPEC = importlib.util.spec_from_file_location("check_doc_references", CHECKER_PATH)
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


class DocReferencesTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.write("docs/index.rst", "Manual\n======\n")
        self.write("docs/tools/compiler.rst", "Compiler\n========\n")
        self.write("docs/examples/input.mlir", "module {}\n")
        self.write("tools/llk-opt/main.cpp", "// source\n")
        self.tool_paths = {
            name: self.make_tool(name) for name in
            ("llk-opt", "llk-compile", "llk-tune", "micro-perf", "llk-bench")
        }

    def write(self, path, contents):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(contents)
        return target

    def make_tool(self, name):
        path = self.write("bin/" + name, textwrap.dedent("""\
            #!{python}
            import sys
            if sys.argv[1:] != ["--help"]:
                sys.exit(2)
            print("--help --good --threads --reps -M -N -K")
            """).format(python=sys.executable))
        path.chmod(0o755)
        return str(path)

    def path_problems(self, text, doc="docs/tools/compiler.rst"):
        problems = []
        checker.check_paths(str(self.root), doc, text, problems)
        return problems

    def flag_problems(self, text, tool_paths=None):
        problems = []
        checker.check_flags(str(self.root), "docs/tools/compiler.rst", text,
                            self.tool_paths if tool_paths is None else tool_paths,
                            problems)
        return problems

    def test_doc_roles_resolve_absolute_and_relative_documents(self):
        text = ":doc:`/index` :doc:`Compiler <compiler>` :doc:`../index`"
        self.assertEqual([], self.path_problems(text))
        problems = self.path_problems(":doc:`Missing <../missing>`")
        self.assertEqual(1, len(problems))
        self.assertIn("../missing", problems[0])

    def test_download_roles_resolve_from_page_or_docs_root(self):
        text = (":download:`Input <../examples/input.mlir>` "
                ":download:`/examples/input.mlir`")
        self.assertEqual([], self.path_problems(text))
        problems = self.path_problems(":download:`Absent </examples/missing.mlir>`")
        self.assertEqual(1, len(problems))
        self.assertIn("missing.mlir", problems[0])

    def test_source_roles_check_files_and_directories_and_strip_anchors(self):
        text = (":source:`Implementation <tools/llk-opt/main.cpp#L1>` "
                ":source:`tools/llk-opt/`")
        self.assertEqual([], self.path_problems(text))
        problems = self.path_problems(":source:`Missing <tools/deleted.cpp#L10>`")
        self.assertEqual(1, len(problems))
        self.assertIn("tools/deleted.cpp", problems[0])

    def test_rst_images_and_figures_must_exist(self):
        self.write("docs/images/pipeline.svg", "<svg/>\n")
        text = (".. image:: /images/pipeline.svg\n\n"
                ".. figure:: ../images/pipeline.svg\n\n"
                ".. image:: https://example.com/remote.svg\n")
        self.assertEqual([], self.path_problems(text))
        problems = self.path_problems(".. figure:: ../images/missing.svg\n")
        self.assertEqual(1, len(problems))
        self.assertIn("missing.svg", problems[0])

    def test_rst_literals_preserve_repository_path_checks(self):
        self.assertEqual([], self.path_problems("``tools/llk-opt/main.cpp``"))
        problems = self.path_problems("``test/deleted.mlir``")
        self.assertEqual(1, len(problems))
        self.assertIn("test/deleted.mlir", problems[0])

    def test_rst_bad_flags_are_checked_in_each_shell_language(self):
        for language in ("bash", "sh", "shell"):
            with self.subTest(language=language):
                text = ".. code-block:: %s\n\n   build/llk-opt --bad\n" % language
                problems = self.flag_problems(text)
                self.assertEqual(1, len(problems))
                self.assertIn("--bad", problems[0])

    def test_indented_rst_shell_blocks_join_continuations(self):
        text = ("* Run the optimizer:\n\n"
                "  .. code-block:: bash\n"
                "     :caption: A multiline command\n\n"
                "     build/llk-opt --good \\\n"
                "       --bad input.mlir\n\n"
                "Outside the example, --bad is prose.\n")
        problems = self.flag_problems(text)
        self.assertEqual(1, len(problems))
        self.assertIn("--bad", problems[0])

    def test_shell_block_boundaries_do_not_join_separate_examples(self):
        text = (".. code-block:: bash\n\n   build/llk-opt --good\n\n"
                ".. code-block:: mlir\n\n   build/llk-opt --bad\n\n"
                ".. code-block:: sh\n\n   build/llk-opt --bad\n")
        problems = self.flag_problems(text)
        self.assertEqual(1, len(problems))
        self.assertIn("--bad", problems[0])

    def test_llk_bench_flags_are_checked(self):
        text = ".. code-block:: sh\n\n   build/llk-bench --bad\n"
        problems = self.flag_problems(text)
        self.assertEqual(1, len(problems))
        self.assertIn("llk-bench", problems[0])
        self.assertIn("--bad", problems[0])

    def test_single_dash_benchmark_options_are_checked(self):
        text = (".. code-block:: sh\n\n"
                "   build/llk-bench -M=2 -N=16 -K=16 --threads=1 -Bad=2\n")
        problems = self.flag_problems(text)
        self.assertEqual(1, len(problems))
        self.assertIn("-Bad", problems[0])

    def test_missing_tool_is_reported_for_a_documented_rst_command(self):
        paths = dict(self.tool_paths)
        del paths["llk-bench"]
        problems = self.flag_problems(
            ".. code-block:: shell\n\n   build/llk-bench --help\n", paths)
        self.assertEqual(1, len(problems))
        self.assertIn("tool not found", problems[0])
        self.assertIn("llk-bench", problems[0])

    def test_legacy_markdown_checks_remain_active(self):
        text = ("[source](../tools/llk-opt/main.cpp#L1) `tools/llk-opt/main.cpp`\n"
                "```bash\n"
                "build/llk-opt --good \\\n"
                "  --bad\n"
                "```\n")
        self.assertEqual([], self.path_problems(text, "docs/legacy.md"))
        problems = self.flag_problems(text)
        self.assertEqual(1, len(problems))
        self.assertIn("--bad", problems[0])
        self.assertEqual(1, len(self.path_problems("[missing](missing.md)", "README.md")))

    def test_cli_checks_rst_corpus_and_infers_sibling_benchmark(self):
        docs = [
            "README.md", "docs/index.rst", "docs/getting-started.rst",
            "docs/concepts.rst", "docs/architecture.rst", "docs/features.rst",
            "docs/contributing.rst", "docs/building-docs.rst",
            "docs/examples/index.rst", "docs/tools/index.rst",
            "docs/tools/compiler.rst", "docs/tools/optimizer.rst",
            "docs/tools/performance.rst", "docs/tools/tuning.rst",
            "docs/tools/benchmark.rst", "docs/tutorials/index.rst",
            "docs/tutorials/01-first-kernel.rst", "docs/tutorials/02-mapping.rst",
            "docs/tutorials/03-performance.rst", "docs/tutorials/04-tuning.rst",
            "docs/design/micro-ir-mapping-workflow.md",
            "docs/reviews/issue67-final-acceptance.md",
        ]
        for doc in docs:
            self.write(doc, "")
        command = [sys.executable, str(CHECKER_PATH), str(self.root)] + [
            self.tool_paths[name] for name in
            ("llk-opt", "llk-compile", "llk-tune", "micro-perf")
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(0, result.returncode, result.stderr)
        self.write("docs/tools/benchmark.rst",
                   ".. code-block:: sh\n\n   build/llk-bench --bad\n")
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(1, result.returncode)
        self.assertIn("--bad", result.stderr)
        self.assertIn("llk-bench", result.stderr)


if __name__ == "__main__":
    unittest.main()
